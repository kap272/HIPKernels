#include<assert.h>
#include <cstddef>  // for std::size_t
#include <array>
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_runtime.h>
#include<assert.h>
#include<stdio.h>

constexpr const int BLOCK = 128;
constexpr const int TILE_WIDTH = 16;
constexpr const int TILES_PER_BLOCK = BLOCK/TILE_WIDTH;

// // get C_ij where
// // C := A @ B_T
// // C_ij = A_row_i * B_T_col_j
// //      = A_row_i * B_row_j
// // A m by k
// // B n by k
// // both column major format
// float get_cell_value(float* A, float* B, int m, int k, int n) {

// } 

float* generate_random_vector(int n) {
    float* vec = (float*) malloc(sizeof(float) * n); 
    for (int i = 0; i < n; i++) {
        vec[i] = rand() % 10;
    }
    return vec;
}

template<typename T> T* generate_random_matrix(int rows, int cols) {
    T* A = (T*) malloc(sizeof(T) * rows * cols);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            A[i * cols + j] = (T) rand(); //% (__hip_fp8_e4m3_fnuz) 10.0;
        }
    }
    return A;
}

// __hip_fp8_e4m3_fnuz* generate_random_matrix(int rows, int cols) {
//     __hip_fp8_e4m3_fnuz* A = (__hip_fp8_e4m3_fnuz*) malloc(sizeof(__hip_fp8_e4m3_fnuz) * rows * cols);
//     for (int i = 0; i < rows; i++) {
//         for (int j = 0; j < cols; j++) {
//             A[i * cols + j] = (__hip_fp8_e4m3_fnuz) rand(); //% (__hip_fp8_e4m3_fnuz) 10.0;
//         }
//     }
//     return A;
// }

template<typename T> T* allocate_matrix(int rows, int cols) {
    return (T*) malloc(sizeof(__hip_fp8_e4m3_fnuz) * rows * cols);
}

void print_vector(float* v, int n) {
    for (int i = 0; i < n; i++) {
        printf("%f\n", v[i]);
    }
    printf("\n");
}    

void print_matrix(float* M, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%f ", M[cols * i + j]);
        }
        printf("\n");
    }
    printf("\n");
}

template<typename T> bool matrices_are_equal(T* A, T *B, int A_rows, int A_cols, int B_rows, int B_cols) {
    if (A_rows != B_rows || A_cols != B_cols) {
        return false;
    }

    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_cols; j++) {
            //if (A[i * A_cols + j] != B[i * B_cols + j]) {
            if (A[j * A_cols + i] != B[j * B_cols + i]) {
                return false;
            }
        }
    }

    return true;

}

void mat_mul(float* A, float* B, float* C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            float C_val = 0.0;
            for (int l = 0; l < A_cols; l++) {
                C_val += A[i * A_cols + l] * B[l * B_cols + j];
            }
            C[i * B_cols + j] = C_val;
        }
    }
}


typedef void (*mat_mul_func)(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, __hip_bfloat16* c, int m, int n, int k); 

void hipMalloc_wrapped(float* addr, int size) {

}

void hipMemcpy_wrapped(float* addr, int size) {

}

void hipDeviceSynchronize_wrapped() {
    hipError_t err = hipGetLastError();
    if (err != hipSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", hipGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

float compare_mat_mul_kernels(mat_mul_func f, mat_mul_func g, int m, int k, int n, bool print_results, dim3 f_grid_dims, dim3 f_block_dims, dim3 g_grid_dims, dim3 g_block_dims) {
    // // A m-by-k matrix in column major format
    // // B k-by-n matrix in column major format (repressented as a n-by-k)
    // // C m-by-n matrix in row major format
    printf("starting test\n");
    int A_size = sizeof(__hip_fp8_e4m3_fnuz) * m * k;
    int B_size = sizeof(__hip_fp8_e4m3_fnuz) * n * k;
    int C_size = sizeof(__hip_bfloat16) * m * n;
    int alpha_size = sizeof(float) * m * k/128;
    int beta_size = sizeof(float) * n * k/128;

    __hip_fp8_e4m3_fnuz* A = generate_random_matrix<__hip_fp8_e4m3_fnuz>(m, k); 
    __hip_fp8_e4m3_fnuz* B = generate_random_matrix<__hip_fp8_e4m3_fnuz>(n, k); 


    __hip_fp8_e4m3_fnuz* alpha = generate_random_matrix<__hip_fp8_e4m3_fnuz>(m, k/128);
    __hip_fp8_e4m3_fnuz* beta = generate_random_matrix<__hip_fp8_e4m3_fnuz>(n, k/128);


    // establish ground truth
    // print_matrix(A, A_rows, A_cols);
    // print_matrix(B, A_cols, B_cols);

    // run kernel f
    __hip_bfloat16* C_h_f = allocate_matrix<__hip_bfloat16>(m, n);

    __hip_fp8_e4m3_fnuz* A_d;
    __hip_fp8_e4m3_fnuz* B_d;
    float* alpha_d;
    float* beta_d;
    __hip_bfloat16* C_d_f = allocate_matrix<__hip_bfloat16>(m, n);

    hipMalloc(&A_d, A_size);
    hipMalloc(&B_d, B_size); 
    hipMalloc(&C_d_f, C_size); 
    hipMalloc(&alpha_d, alpha_size); 
    hipMalloc(&beta_d, beta_size); 

    hipMemcpy(A_d, A, A_size, hipMemcpyHostToDevice); 
    hipMemcpy(B_d, B, B_size, hipMemcpyHostToDevice); 
    hipMemcpy(alpha_d, alpha, alpha_size, hipMemcpyHostToDevice); 
    hipMemcpy(beta_d, beta, beta_size, hipMemcpyHostToDevice); 

    float f_time;
    hipEvent_t f_start, f_stop;    
    hipEventCreate(&f_start);
    hipEventCreate(&f_stop);
    hipEventRecord(f_start, 0);

    // TODO: a better way to determine kernel launch dimensions
    f<<<f_grid_dims, f_block_dims>>>(A_d, B_d, alpha_d, beta_d, C_d_f, m, n, k);
    
    hipEventRecord(f_stop, 0);
    hipDeviceSynchronize_wrapped();
    hipEventElapsedTime(&f_time, f_start, f_stop);

    hipMemcpy(C_h_f, C_d_f, C_size, hipMemcpyDeviceToHost); 
    hipFree(C_d_f);

    // then knl g
    __hip_bfloat16* C_h_g = (__hip_bfloat16*) malloc(sizeof(__hip_fp8_e4m3_fnuz) * m * n); 
    __hip_bfloat16* C_d_g = (__hip_bfloat16*) malloc(sizeof(__hip_fp8_e4m3_fnuz) * m  * n); 
    printf("%p %p \n", &C_h_g, &C_h_f); //0x7ffeee88cf78 0x7ffeee88cf38
    hipMalloc(&C_d_g, C_size);

    // TODO fix timing
    float g_time;
    hipEvent_t g_start, g_stop;    
    hipEventCreate(&g_start);
    hipEventCreate(&g_stop);
    hipEventRecord(g_start, 0);

    // TODO: a better way to determine kernel launch dimensions
    g<<<g_grid_dims, g_block_dims>>>(A_d, B_d, alpha_d, beta_d, C_d_g, m, n, k);

    hipEventRecord(g_stop, 0);
    hipDeviceSynchronize_wrapped();
    hipEventElapsedTime(&g_time, g_start, g_stop);

    hipMemcpy(C_h_g, C_d_g, C_size, hipMemcpyDeviceToHost); 

    if (matrices_are_equal<__hip_bfloat16>(C_h_f, C_h_g, m, n, m, n)) {
        printf("results match \n");
    } else {
        printf("results dont match \n");
    }
    

    if (!matrices_are_equal(C_h_f, C_h_g, m, n, m, n)) {
        printf("kernel g is wrong!\n");
    }
    hipFree(C_d_g);
    if (print_results) {
        int R = 0;
        printf("g result: \n");

        for (int row = 0; row < R; row++) {
            for (int col = 0; col < R; col++) {
                printf("%f ", C_h_g[row * n + col]);
            }
            printf("\n");
        }

        printf("f result: \n");

        for (int row = 0; row < R; row++) {
            for (int col = 0; col < R; col++) {
                printf("%f ", C_h_f[row * n + col]);
            }
            printf("\n");
        }

        int miss = 0;
        int total = 0;
        int zero = 0;
        for (int row = 0; row < m; row++) {
            for (int col = 0; col < n; col++) {
                if (C_h_f[row * n + col] != C_h_g[row * n + col]) {
                   miss += 1; 
                   if (C_h_g[row * n + col] == (__hip_fp8_e4m3_fnuz) 0.0) {
                       zero += 1;
                   }
                }
                total += 1;
            }
        }
        printf("miss: %d zero: %d total: %d \n", miss, zero, total);
        printf("miss rate : %f \n", miss/(float)total);
        printf("zero rate : %f \n ", zero/(float)total);
    }

    printf("kernel f run time: %f\n", f_time);
    printf("kernel g run time: %f\n", g_time);

    // hipFree(A_d);
    // hipFree(B_d);
    // free(A);
    // free(B);
    // free(C_h_f);
    // free(C_h_g);

    return ((g_time - f_time)/f_time);
}

// // A, B, C, as, bs, C, m, n, k
// // __global__ void mat_mul_ref(const float* a, const float* b, const float* as, const float* bs, float* c, int m, int n, int k) {
                   
// //     // Your implementation here
// //     int cx = threadIdx.x + blockDim.x * blockIdx.x;
// //     int cy = threadIdx.y + blockDim.y * blockIdx.y;
// //     if(cx >= m || cy >= n) return;
    
// //     int sn = (n + BLOCK - 1) / BLOCK;
    
// //     float result = 0;
// //     // split loop into an outer loop over different blocks, and an inner loop within one block.
// //     // we can assume k % BLOCK == 0.
// //     for(int i = 0; i < k; i += BLOCK) {
// //         // block results accumulates the inner product across a single block.
// //         // within each block, scales are constant, so we can lift the scaling 
// //         // outside of the inner loop.
// //         float block_result = 0;
// //         for(int ii = 0; ii < BLOCK; ++ii) {
// //             // load input matrix elements and convert to float for computations
// //             float av = (float)a[cx + (i + ii) * m];
// //             float bv = (float)b[cy + (i + ii) * n];
// //             block_result += av * bv; 
// //         }
        
// //         // before we can go to the next block, scale the result of the current block
// //         // and accumulate to final result
// //         // note the different indexing into as and bs
// //         result += block_result ;//* as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
// //     }
    
// //     // finally, write the result as bf16
// //     c[cx * n + cy] = result; 
// // }

__global__ void custom_kernel_bak(const float* a, const float* b, const float* as, const float* bs, float* c, int m, int n, int k) {
    // Your implementation here
    int row = threadIdx.x + blockDim.x * blockIdx.x;
    int col = threadIdx.y + blockDim.y * blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    if(row >= m || col >= n) {
        return;
    };
    
    int sn = (n + BLOCK - 1) / BLOCK;
    
    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    float result = 0.0;

    // go across the inner dimension (horizonatlly across A and horizontally across B in increments of BLOCK

    for (int inner_dim_idx = 0; inner_dim_idx < k; inner_dim_idx += BLOCK) {
        float block_result = 0.0;
        for (int tile_num = 0; tile_num < TILES_PER_BLOCK; tile_num++) {
            A_tile[tx][ty] = a[row + m * (inner_dim_idx  + (tile_num * TILE_WIDTH) + ty)];
            B_tile[ty][tx] = b[col + n * (inner_dim_idx  + (tile_num * TILE_WIDTH )+ tx)];
            __syncthreads();
            for (int k = 0; k < TILE_WIDTH; k++) {
                block_result += A_tile[tx][k] * B_tile[ty][k];
            }
            __syncthreads();
        } 
        result += block_result;//  * as[row + inner_dim_idx/BLOCK * m] * bs[col/BLOCK + inner_dim_idx/BLOCK * sn];
        __syncthreads();
    } 
    c[row* n + col] = result;
}


__global__ void custom_kernel_bak_bak(const __hip_fp8_e4m3_fnuz* A, const __hip_fp8_e4m3_fnuz* B, const float* as, const float* bs, __hip_bfloat16* C, int m, int n, int k) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int cx = bx * TILE_WIDTH + tx; 
    int cy = by * TILE_WIDTH + ty; 

    if (cx >= m || cy >= n) {
        return;
    }

    float C_val = 0.0;

    __shared__ float A_tile[TILE_WIDTH][TILE_WIDTH];
    __shared__ float B_tile[TILE_WIDTH][TILE_WIDTH];

    for (int i = 0; i < k/TILE_WIDTH; i++) { 
        // fill the tile with a row of A and a row of B
        // we need row cx of A and row cy of B
        // fill one row of the tile with one column slice
        A_tile[ty][tx] = A[cx + (i * TILE_WIDTH + ty ) * m]; //A[tx][ty];
        B_tile[ty][tx] = B[tx + (i * TILE_WIDTH + cy ) * m];
        __syncthreads();

//        for (int q = 0; q < TILE_WIDTH; q++) {
//            for (int w = 0; w < TILE_WIDTH; w++) {
//                A_tile[q][w] == A[];
//
//            }
//        }

        for (int j = 0; j < TILE_WIDTH; j++) {
            //C_val += A_tile[j][ty] * B_tile[j][tx];
            C_val += A_tile[j][tx] * B_tile[j][ty];
        }
        __syncthreads();
    }

    C[cx * n + cy] = C_val;

}

__global__ void custom_kernel(const float* A, const float* B, const float* as, const float* bs, float* C, int m, int n, int k) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int cx = bx * TILE_WIDTH + tx; 
    int cy = by * TILE_WIDTH + ty; 

    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x);

    float A_val = 0.0;
    float B_val = 0.0;
    float C_val = 0.0;

    //if (cx == 0 && cy == 0) {
    //    printf("%d %d %d %d", blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
    //}
    for (int i = 0; i < k; i += blockDim.x) {
        if (lane_id < 16) {
            A_val = A[cx + m * (i + lane_id%8)];    
        } else {
            B_val = B[cy + n * (i + lane_id%8)];    
        }
            if (i == 0 && cx == 0 && cy == 0) {
                
                //for (int A_tile_idx = 0; A_tile_idx < 16; A_tile_idx++) {
                //    printf("%f ", __shfl_sync(0xffffffff, A_val, A_tile_idx));
                //} 

                printf("\n");
                
                for (int A_row = 0; A_row < 2; A_row++) {
                    for (int A_col = 0; A_col < 8; A_col++) {
                        printf("%f ", A[A_row + A_col * m]);
                }
                printf("\n");
            }
            }

        for (int j = 0; j < blockIdx.x; j++) {
            //if (i == 0) {
            //    printf("%f %f \n", __shfl_sync(0xffffffff, A_val, j) , __shfl_sync(0xffffffff, B_val, j + 16));
            //}
            //C_val += __shfl_sync(0xffffffff, A_val, j) * __shfl_sync(0xffffffff, B_val, j + 16);
        }

    }

    C[cx * n + cy] = C_val;
}

__global__ void mat_mul_ref(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, __hip_bfloat16* c, int m, int n, int k) {
                   
    // Your implementation here
    int cx = threadIdx.x + blockDim.x * blockIdx.x;
    int cy = threadIdx.y + blockDim.y * blockIdx.y;
    if(cx >= m || cy >= n) return;
    
    int sn = (n + BLOCK - 1) / BLOCK;
    
    float result = 0;
    // split loop into an outer loop over different blocks, and an inner loop within one block.
    // we can assume k % BLOCK == 0.
    for(int i = 0; i < k; i += BLOCK) {
        // block results accumulates the inner product across a single block.
        // within each block, scales are constant, so we can lift the scaling 
        // outside of the inner loop.
        float block_result = 0;
        for(int ii = 0; ii < BLOCK; ++ii) {
            // load input matrix elements and convert to float for computations
            float av = (float)a[cx + (i + ii) * m];
            float bv = (float)b[cy + (i + ii) * n];
            block_result += av * bv; 
        }
        
        // before we can go to the next block, scale the result of the current block
        // and accumulate to final result
        // note the different indexing into as and bs
        result += block_result * as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
    }
    
    // finally, write the result as bf16
    c[cx * n + cy] = (__hip_bfloat16)result;
}

__global__ void mat_mul(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, __hip_bfloat16* c, int m, int n, int k) {}


int main() {

    printf("here !!!!");
    int m = 64;
    float diff = 0.0;
    int runs = 5;
    float inc= 0.0;

    for (int i = 0; i < runs; i++) {
        printf("here !!!!");
        inc = compare_mat_mul_kernels(mat_mul_ref, custom_kernel_bak_bak, m, m, m, true, dim3(m/TILE_WIDTH,m/TILE_WIDTH ), dim3(TILE_WIDTH, TILE_WIDTH), dim3(m/8, m/4), dim3(8, 4));  
        if (i != 0) {
          diff += inc;
        }
    }
      printf("\n%f\n", diff/runs);
    printf("done");
}
