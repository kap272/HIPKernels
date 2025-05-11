#include<assert.h>
#include<stdio.h>
#ifdef __HIPCC__
#include <hip/amd_detail/amd_hip_fp8.h>
#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_runtime.h>
#endif



// get C_ij where
// C := A @ B_T
// C_ij = A_row_i * B_T_col_j
//      = A_row_i * B_row_j
// A m by k
// B n by k
// both column major format
float get_cell_value(float* A, float* B, int m, int k, int n) {

} 


float* generate_random_vector(int n) {
    float* vec = (float*) malloc(sizeof(float) * n); 
    for (int i = 0; i < n; i++) {
        vec[i] = rand() % 10;
    }
    return vec;
}

float* allocate_matrix(int rows, int cols) {
    float* vec = (float*) malloc(sizeof(float) * rows * cols); 
    return vec;
}

float* generate_random_matrix(int rows, int cols) {
    float* M = (float*) malloc(sizeof(float) * rows * cols); 
    for (int i = 0; i < rows * cols; i++) {
        M[i] = rand() % 10;
    }
    return M;

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

bool matrices_are_equal(float* A, float *B, int A_rows, int A_cols, int B_rows, int B_cols) {
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


typedef void (*mat_mul_func)(const float* a, const float* b, const float* as, const float* bs, float* c, int m, int n, int k); 

void cudaMalloc_wrapped(float* addr, int size) {

}

void cudaMemcpy_wrapped(float* addr, int size) {

}

void cudaDeviceSynchronize_wrapped() {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }

    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "Kernel execution failed: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

float compare_mat_mul_kernels(mat_mul_func f, mat_mul_func g, int m, int k, int n, bool print_results, dim3 f_grid_dims, dim3 f_block_dims, dim3 g_grid_dims, dim3 g_block_dims) {
    // A m-by-k matrix in column major format
    // B k-by-n matrix in column major format (repressented as a n-by-k)
    // C m-by-n matrix in row major format
    int A_size = sizeof(float) * m * k;
    int B_size = sizeof(float) * n * k;
    int C_size = sizeof(float) * m * n;
    int alpha_size = sizeof(float) * m * k/128;
    int beta_size = sizeof(float) * n * k/128;

    float* A = generate_random_matrix(m, k); 
    float* B = generate_random_matrix(n, k); 


    float* alpha = generate_random_matrix(m, k/128);
    float* beta = generate_random_matrix(n, k/128);


    // establish ground truth
    // print_matrix(A, A_rows, A_cols);
    // print_matrix(B, A_cols, B_cols);

    // run kernel f
    float* C_h_f = allocate_matrix(m, n);

    float* A_d;
    float* B_d;
    float* alpha_d;
    float* beta_d;
    float* C_d_f = allocate_matrix(m, n);

    cudaMalloc(&A_d, A_size);
    cudaMalloc(&B_d, B_size); 
    cudaMalloc(&C_d_f, C_size); 
    cudaMalloc(&alpha_d, alpha_size); 
    cudaMalloc(&beta_d, beta_size); 

    cudaMemcpy(A_d, A, A_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(B_d, B, B_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(alpha_d, alpha, alpha_size, cudaMemcpyHostToDevice); 
    cudaMemcpy(beta_d, beta, beta_size, cudaMemcpyHostToDevice); 

    float f_time;
    cudaEvent_t f_start, f_stop;    
    cudaEventCreate(&f_start);
    cudaEventCreate(&f_stop);
    cudaEventRecord(f_start, 0);

    // TODO: a better way to determine kernel launch dimensions
    f<<<f_grid_dims, f_block_dims>>>(A_d, B_d, alpha_d, beta_d, C_d_f, m, n, k);
    
    cudaEventRecord(f_stop, 0);
    cudaDeviceSynchronize_wrapped();
    cudaEventElapsedTime(&f_time, f_start, f_stop);

    cudaMemcpy(C_h_f, C_d_f, C_size, cudaMemcpyDeviceToHost); 
    cudaFree(C_d_f);

    // then knl g
    float* C_h_g = (float*) malloc(sizeof(float) * m * n); 
    float* C_d_g = (float*) malloc(sizeof(float) * m  * n); 
    printf("%p %p \n", &C_h_g, &C_h_f); //0x7ffeee88cf78 0x7ffeee88cf38
    cudaMalloc(&C_d_g, C_size);

    // TODO fix timing
    float g_time;
    cudaEvent_t g_start, g_stop;    
    cudaEventCreate(&g_start);
    cudaEventCreate(&g_stop);
    cudaEventRecord(g_start, 0);

    // TODO: a better way to determine kernel launch dimensions
    g<<<g_grid_dims, g_block_dims>>>(A_d, B_d, alpha_d, beta_d, C_d_g, m, n, k);

    cudaEventRecord(g_stop, 0);
    cudaDeviceSynchronize_wrapped();
    cudaEventElapsedTime(&g_time, g_start, g_stop);

    cudaMemcpy(C_h_g, C_d_g, C_size, cudaMemcpyDeviceToHost); 

    if (matrices_are_equal(C_h_f, C_h_g, m, n, m, n)) {
        printf("results match \n");
    } else {
        printf("results dont match \n");
    }
    

    if (!matrices_are_equal(C_h_f, C_h_g, m, n, m, n)) {
        printf("kernel g is wrong!\n");
    }
    cudaFree(C_d_g);
    if (print_results) {
//        printf("g result: \n");
        print_matrix(C_h_g, m, n);
    }

    printf("kernel f run time: %f\n", f_time);
    printf("kernel g run time: %f\n", g_time);
    cudaFree(A_d);
    cudaFree(B_d);
    free(A);
    free(B);

    return ((g_time - f_time)/f_time);
}
constexpr const int BLOCK = 128;
constexpr const int TILE_WIDTH = 16;
constexpr const int TILES_PER_BLOCK = BLOCK/TILE_WIDTH;

// A, B, C, as, bs, C, m, n, k
__global__ void mat_mul_ref(const float* a, const float* b, const float* as, const float* bs, float* c, int m, int n, int k) {
                   
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
        result += block_result ;//* as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
    }
    
    // finally, write the result as bf16
    c[cx * n + cy] = result; 
}

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


__global__ void custom_kernel_bak_bak(const float* A, const float* B, const float* as, const float* bs, float* C, int m, int n, int k) {
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

    int lane_id = threadIdx.x & 0x1f;

    float A_val = 0.0;
    float B_val = 0.0;
    float C_val = 0.0;

    // fill up two 4-by-4 cells
    //
    // A_00 A_01 A_02 A_03
    // A_10 A_11 A_12 A_23
    // A_20 A_01 A_22 A_23
    // A_30 A_31 A_32 A_33
    //
    // lane_id < 16 -> lane_id/4 + lane_id%4 
    //


    for (int i = 0; i < k/TILE_WIDTH; i++) {
        if (lane_id < 16) {
            A_val = A[cx + (i * 4 + lane_id%4)*k];
        } else {
            B_val = B[cy + (i * 4 + lane_id%4)*k];
        }

        for (int s = 0; s < 16; s++) {
            for (int q = 0; q < 16; q++) {
                C_val += __shfl_sync(0xffffffff, A_val, s%4) * __shfl_sync(0xffffffff, B_val, q%4);                    }
            } 
        }

    C[cx * n + cy] = C_val;
}

int main() {
      int m = 1024;
      float diff = 0.0;
      int runs = 2;
      for (int i = 0; i < runs; i++) {
         diff += compare_mat_mul_kernels(mat_mul_ref, custom_kernel, m, m, m, false, dim3(m/TILE_WIDTH,m/TILE_WIDTH ), dim3(TILE_WIDTH, TILE_WIDTH), dim3(m/TILE_WIDTH, m/TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH)); 
      }
      printf("\n%f\n", diff/runs);
      diff = 0.0;
     // for (int i = 0; i < runs; i++) {
     //     diff += compare_mat_mul_kernels(mat_mul_ref, custom_kernel, m, m, m, false, dim3(m/TILE_WIDTH,m/TILE_WIDTH ), dim3(TILE_WIDTH, TILE_WIDTH), dim3(m/TILE_WIDTH, m/TILE_WIDTH), dim3(TILE_WIDTH, TILE_WIDTH)); 
     // }
     // printf("\n%f\n", diff/runs);
}

