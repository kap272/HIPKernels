
#include <hip/amd_detail/amd_hip_fp8.h>

#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>
#include <cstdlib>
#define M 16
#define N 16
#define K 4

#define BLOCK 128


std::mt19937 gen(42);
 
float dot_product(__hip_fp8_e4m3_fnuz* A, __hip_fp8_e4m3_fnuz* B, int A_row, int B_row, int A_cols) {
    float res = 0.0;

    for (int i = 0; i < A_cols; i++) {
        res += (float) A[A_row + i * A_cols] * (float) B[B_row = i * A_cols];
    }

    return res;
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
        for(int ii = 0; ii < min(k, BLOCK); ++ii) {
            // load input matrix elements and convert to float for computations
            float av = (float)a[cx + (i + ii) * m];
            float bv = (float)b[cy + (i + ii) * n];
            block_result += av * bv;
        }

        // before we can go to the next block, scale the result of the current block
        // and accumulate to final result
        // note the different indexing into as and bs

//         printf("i %d as row %d as col %d A row %d B row %d \n", i, cx, i/BLOCK, cy, cx);
        result += block_result * as[cx + i/BLOCK * m] * bs[cy/BLOCK + i/BLOCK * sn];
//         result += block_result; 
    }

    // finally, write the result as bf16
    c[cx * n + cy] = (__hip_bfloat16)result;
}

__global__ void sgemm_16x16x4(const float *A, const float *B, float *D) {
  using float4 = __attribute__( (__vector_size__(K * sizeof(float)) )) float;
  float4 dmn = {0};
 
  int mk = threadIdx.y + K * threadIdx.x;
  int kn = threadIdx.x + N * threadIdx.y;
 
  float amk = A[mk];
  float bkn = B[kn];
  __syncthreads();
  dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
 
  for (int i = 0; i < 4; ++i) {
    const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
    D[idx] = dmn[i];
  }
}

__device__ inline uint32_t pack_f8_f32(__hip_fp8_e4m3_fnuz a0, __hip_fp8_e4m3_fnuz a1, __hip_fp8_e4m3_fnuz a2, __hip_fp8_e4m3_fnuz a3) {
    return (uint32_t(a3) << 24) |
           (uint32_t(a2) << 16) |
           (uint32_t(a1) << 8)  |
           (uint32_t(a0));
}


__global__ void sgemm_16x16x4_e4m3_bak(const __hip_fp8_e4m3_fnuz* A, const __hip_fp8_e4m3_fnuz* B, const float* as, const float* bs, __hip_bfloat16* C, int A_rows, int B_rows, int B_cols) {
  using float4 = __attribute__( (__vector_size__(K * sizeof(float)) )) float;
  float4 dmn = {0};

  int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 64;

  // determine the upper left corner of the tiles in the input
  // ths is the offset against which to map lane id to matrix indices
  int upper_left_x = blockDim.x * blockIdx.x;
  int upper_left_y = blockDim.y * blockIdx.y * 4;

  // within the tile & with respect to the offset
  // the mapping goes lane_id -> A[lane_id%16][lane_id//16]
  //                  lane_id -> B_T[lane_id//16][lane_id%16] or  
  //                  lane_id -> B[lane_id%16][lane_id//16]
  int A_row = upper_left_y + lane_id%16;
  int B_row = upper_left_x + lane_id%16;

  if (A_row >= A_rows || B_row >= B_rows) {
      return;
  }

  for (int i = 0; i < B_cols; i += 4) {
      int A_col = (i) + lane_id/16;
      int B_col =  A_col; 

      float A_val = (float) A[A_row + A_col * A_rows]; 
      float B_val = (float) B[B_row + B_col * B_rows]; 

      dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(A_val, B_val, dmn, 0, 0, 0);
      
      for (int p = 0; p < 4; ++p) {
            // lane_id -> C[4 * lane_id/16 + i][lane_id % 16] for i = 0, 1, 2, 3
            int C_row = upper_left_y + (4 * (lane_id/16)) + p;
            int C_col = upper_left_x + (lane_id % 16);

            if (C_row < A_rows && C_col < B_rows) {
                C[C_row * B_rows + C_col] = (__hip_bfloat16) dmn[p];
            }
          }
      }
}

__global__ void sgemm_16x16x4_e4m3(const __hip_fp8_e4m3_fnuz* A, const __hip_fp8_e4m3_fnuz* B, const float* alpha, const float* beta, __hip_bfloat16* C, int A_rows, int B_rows, int B_cols) {
    using float4 = __attribute__( (__vector_size__(K * sizeof(float)) )) float;
    float4 C_frags[4] = {0.0f, 0.0f, 0.0f, 0.0f} ;

    int thread_num =  threadIdx.y * blockDim.x + threadIdx.x;
    int lane_id = thread_num%64;
    int wf_id = thread_num/64;

    int upper_left_x = blockDim.x * blockIdx.x ;
    int upper_left_y = 4*blockDim.y * blockIdx.y ;

    int A_row = (upper_left_y + lane_id%16);
    int B_row = upper_left_x + lane_id%16;

    int sn = (B_rows + 128 - 1)/128;


    __shared__  __hip_fp8_e4m3_fnuz A_tile[32][32];
    __shared__ __hip_fp8_e4m3_fnuz B_tile[32][32];

    float a ;
    float b;

    for (int i = 0; i < B_cols; i += 32) {
        int A_col = i + thread_num%32; 
        int B_col = A_col;

        for (int row = 0; row < 4; row ++) {
            int A_tile_row = (4 * (thread_num/32)) + row; 
            int A_tile_col = thread_num%32; 
            int A_row = ((A_tile_row) + upper_left_y) ; 
            int B_row = ((A_tile_row) + upper_left_x) ; 

            A_tile[A_tile_row][thread_num%32] = A[A_row + A_col * A_rows]; 
            B_tile[A_tile_row][thread_num%32] = B[B_row + B_col * B_rows]; 
        
        }

        __syncthreads();

        for (int subtile = 0; subtile < 8; subtile++) {
            int a_row_offset = 16 * (wf_id / 2);
            int b_row_offset = 16 * (wf_id % 2);
            int col_offset   =  (subtile * 4);

            // recover indices in A 
            //
            // row: a_row_offset + lane_id % 16 + upper_left_x (cx)
            // col: col_offset + lane_id / 16

            // recover indices in B 
            //
            // row: b_row_offset + lane_id % 16 + upper_left_y (cy)
            // col: col_offset + lane_id / 16

            float A_val = (float) A_tile[a_row_offset + lane_id % 16][col_offset + lane_id / 16];
            float B_val = (float) B_tile[b_row_offset + lane_id % 16][col_offset + lane_id /16];

//             if (i%128 == 0) {
                float a = alpha[a_row_offset + lane_id % 16 + upper_left_y + i/128 * A_rows];
                float b = beta[(b_row_offset + lane_id % 16 + upper_left_x)/128 + i/128 * sn];
//             }

            C_frags[wf_id] = __builtin_amdgcn_mfma_f32_16x16x4f32((float) a * A_val, (float) b * B_val, C_frags[wf_id], 0, 0, 0);
          }

          __syncthreads();
        
    }

    for (int p = 0; p < 4; ++p) {
        int wf_row_offset = 16 * (wf_id / 2);
        int wf_col_offset = 16 * (wf_id % 2);
        int C_row = upper_left_y + wf_row_offset + (4 * (lane_id / 16)) + p;
        int C_col = upper_left_x + wf_col_offset + (lane_id % 16);

        if (C_row < A_rows && C_col < B_rows && wf_id < 4 && p < 4) {
            C[C_row * B_rows + C_col] = (__hip_bfloat16) C_frags[wf_id][p];
        }
    }
}

__global__ void custom_kernel_bak(const float* A, const float* B, const float* as, const float* bs, float* C, int m, int n, int k) {
    int cx = threadIdx.x + blockDim.x * blockIdx.x;
    
    int cy = threadIdx.y + blockDim.y * blockIdx.y;

    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 64;

    if (cx >= m || cy >= n) {
        return;
    }

    using float4 = __attribute__( (__vector_size__(K * sizeof(float)) )) float;
    float4 cmn = {0};

    int idx = lane_id/4 + lane_id%4;
    int upper_left_x = blockDim.x * blockIdx.x;
    int upper_left_y = blockDim.y * blockIdx.y;

    int A_row = upper_left_y + lane_id%16;
    int A_col = upper_left_x + lane_id/16;

    int B_row = upper_left_y + lane_id/16;
    int B_col = upper_left_x + lane_id%16;
    
    int A_idx = A_row * 4 + A_col;
    int B_idx = B_row * 16 + B_col;


    float amk = A[A_idx];
    float bkn = B[B_idx];

    if (cx == 1 && cy == 1) {
        printf("%f %f \n", A[0], B[0]);
        printf("%d %d \n", upper_left_y, upper_left_x);
        printf("lane id: %d, A row, col: %d, %d, A idx: %d, A val: %f\n", lane_id, A_row + 1, A_col + 1, A_idx, amk);
        printf("             B row, col: %d, %d, B idx: %d, B val: %f\n\n",         B_row + 1, B_col + 1, B_idx, bkn);
    }

    cmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, cmn, 0, 0, 0);

    for (int i = 0; i < 4; ++i) {
    const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
    C[idx] = cmn[i];
    }

}

__global__ void custom_kernel_bak_bak(const float* A, const float* B, const float* as, const float* bs, float* C, int m, int n, int k) {
    int cx = threadIdx.x + blockIdx.x * blockDim.x; 
    int cy = threadIdx.y + blockIdx.y * blockDim.y; 
    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 64;

    using float4 = __attribute__( (__vector_size__(K * sizeof(float)) )) float;

    float4 cmn = {0};

    if (cx >= m || cy >= n) {
        return;
    }

    __shared__ float A_tile[16][4];
    __shared__ float B_tile[4][16];

    int upper_left_x = 0; //= blockDim.x * blockIdx.x;
    int upper_left_y = 0; //= blockDim.y * blockIdx.y;

    int A_row = 0;//  = upper_left_y + lane_id%16;
    int A_col = 0;//upper_left_x + lane_id/16;

    int B_row = 0; //upper_left_y + lane_id/16;
    int B_col = 0; //upper_left_x + lane_id%16;
    
    int A_idx = 0;//A_row * 4 + A_col;
    int B_idx = 0;//B_row * 16 + B_col;


    float amk  = 0.0; //= A[A_idx];
    float bkn = 0.0;//= B[B_idx];

    for (int i = 0; i < k/4; i++) {
        upper_left_x = blockDim.x * blockIdx.x + 4 * i;
        upper_left_y = blockDim.y * blockIdx.y + 4 * i;

        A_row = upper_left_y + lane_id%16;
        A_col = upper_left_x + lane_id/16;
        B_row = upper_left_y + lane_id/16;
        B_col = upper_left_x + lane_id%16;

        A_idx = A_row * n + A_col;
        B_idx = B_row * k + B_col;

        amk = A[A_idx];
        bkn = B[B_idx];
         
        A_tile[lane_id%16][lane_id/16] = A[A_idx];
        B_tile[lane_id/16][lane_id%16] = B[B_idx];
         if (cx == 0 && cy == 0 && i == 0) {
            
                printf("A tile \n");
            for (int i = 0; i < 16; i++) {
                for (int j = 0 ; j < 4; j++) {
                        printf("%f ", A_tile[i][j]);
                    }
                printf("\n");
            }       __syncthreads();
         }

        cmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, cmn, 0, 0, 0);
        __syncthreads();
        if (cx == 0 && cy == 0 && i == 0) {
            
                printf("A tile \n");
            for (int i = 0; i < 16; i++) {
                for (int j = 0 ; j < 4; j++) {
                        printf("%f ", A_tile[i][j]);
                    }
                printf("\n");
            }
                printf("\n");

                printf("B tile \n");
            for (int i = 0; i < 4; i++) {
                for (int j = 0 ; j < 16; j++) {
                        printf("%f ", B_tile[i][j]);
                    }
                printf("\n");
            }
                printf("\n");
                printf("cmn \n");
            for (int i = 0; i < 4; i++) {
                printf("%f \n", cmn[i]); 
                    
            }
                printf("\n");


            printf("%d %d \n", upper_left_y, upper_left_x);
            printf("lane id: %d, A row, col: %d, %d, A idx: %d, A val: %f\n", lane_id, A_row + 1, A_col + 1, A_idx, amk);
            printf("             B row, col: %d, %d, B idx: %d, B val: %f\n\n",         B_row + 1, B_col + 1, B_idx, bkn);
        }

        // TODO unroll
        for (int i = 0; i < 4; ++i) {
            const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
            C[idx] = cmn[i];
    //        C[(upper_left_y * (lane_id/4) + i) * m + (upper_left_x + lane_id%16)] = cmn[i];
            //const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
            //C[cx * n + ] = cmn[i];
        }

    }
        
}
//__device__ void run_mfma_fp8(float4 &acc, uint32_t a_packed, uint32_t b_packed) {
//    asm volatile(
//        "v_mfma_f32_16x16x64_fp8_e4m3 %0, %1, %2, %0\n"
//        : "=v"(acc)               // output
//        : "v"(a_packed),          // input A
//          "v"(b_packed),          // input B
//          "0"(acc)                // initial accumulator
//    );
//}


__global__ void custom_kernel(const float* A, const float* B, const float* as, const float* bs, float* C, int m, int n, int k) {
    int cx = threadIdx.x + blockDim.x * blockIdx.x;
    int cy = threadIdx.y + blockDim.y * blockIdx.y;

    int lane_id = (threadIdx.y * blockDim.x + threadIdx.x) % 64;

    if (cx >= m || cy >= n) {
        return;
    }

    using float4 = __attribute__( (__vector_size__(K * sizeof(float)) )) float;
    float4 cmn = {0};

    int idx = lane_id/4 + lane_id%4;
    int upper_left_x = blockDim.x * blockIdx.x;
    int upper_left_y = blockDim.y * blockIdx.y;

    int A_row = upper_left_y + lane_id%16;
    int A_col = upper_left_x + lane_id/16;

    int B_row = upper_left_y + lane_id/16;
    int B_col = upper_left_x + lane_id%16;
    
    int A_idx = A_row * 4 + A_col;
    int B_idx = B_row * 16 + B_col;


    float amk = A[A_idx];
    float bkn = B[B_idx];

    cmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, cmn, 0, 0, 0);

    printf("lane id %d A row %d A col %d A val %f \n", lane_id, A_row + 1, A_col + 1, amk);
    printf("lane id %d B row %d B col %d B val %f \n\n", lane_id, B_row + 1, B_col + 1 , bkn);
    if (lane_id == 6) {
        printf("%f %f \n", A[0], B[0]);
        printf("%d %d \n", upper_left_y, upper_left_x);
        printf("lane id: %d, A row, col: %d, %d, A idx: %d, A val: %f\n", lane_id, A_row + 1, A_col + 1, A_idx, amk);
        printf("             B row, col: %d, %d, B idx: %d, B val: %f\n\n",         B_row + 1, B_col + 1, B_idx, bkn);

        for (int i = 0; i < 4; ++i) {
            printf("%f \n", cmn[i]);

        }
    }


    for (int i = 0; i < 4; ++i) {
//        const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
//        C[idx] += cmn[i];
//
        int c_row = upper_left_y + 4 * (lane_id/16) + i;
        int c_col = upper_left_x + lane_id%16;
        
        printf("lane id %d upper_left_y %d upper_left_x %d C row %d C col %d curr C val %f cmn val %f \n", lane_id,
                upper_left_y + 1, 
                upper_left_x + 1, 
                1 + c_row, 
                1 + c_col,
                C[c_row * k + c_col],
                cmn[i]);

        C[c_row * k + c_col] = cmn[i];
        }

    }

template<typename T> void print_matrix(T* A, int rows, int cols) {
    float total = 0;
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", static_cast<float>(A[i + j * rows]));
            total += static_cast<float>(A[i + j * rows]);
        }
        printf("\n");
    }
    printf("\n");
    printf("total: %f \n", total);
}
template<typename T> bool matrices_are_equal(T* A, T *B, int A_rows, int A_cols, int B_rows, int B_cols) {
    if (A_rows != B_rows || A_cols != B_cols) {
        return false;
    }

    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < A_cols; j++) {
            //if (A[i * A_cols + j] != B[i * B_cols + j]) {
            if (A[j * A_cols + i] != B[j * B_cols + i]) {
                printf("failed on row %d col i %d\n", i, j);
                return false;
            }
        }
    }

    return true;

}

template<typename T, typename U > void mat_mul(T* A, T* B, U* C, int A_rows, int A_cols, int B_cols) {
    for (int i = 0; i < A_rows; i++) {
        for (int j = 0; j < B_cols; j++) {
            U C_val = 0.0;
            for (int l = 0; l < A_cols; l++) {
                C_val += (U) A[i * A_cols + l] * B[l * B_cols + j];
            }
            C[i * B_cols + j] = C_val;
        }
    }
}

template<typename T>
std::vector<T> generate_random_vector(int rows, int cols) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};  
    std::uniform_int_distribution<int> dist {1, 10};

    auto gen = [&](){ return static_cast<T>(dist(mersenne_engine)); };

    std::vector<T> vec(rows * cols);
    std::generate(vec.begin(), vec.end(), gen);

    return vec;
}


typedef void (*mat_mul_func)(const __hip_fp8_e4m3_fnuz* a, const __hip_fp8_e4m3_fnuz* b, const float* as, const float* bs, __hip_bfloat16* c, int m, int n, int k);

float test_mat_mul_knl(mat_mul_func f, int A_rows, int A_cols, int B_cols, dim3 grid_dims, dim3 block_dims, bool print) {
    const size_t A_size = sizeof(__hip_fp8_e4m3_fnuz) * A_rows * A_cols;
    const size_t B_size = sizeof(__hip_fp8_e4m3_fnuz) * A_cols * B_cols;
    const size_t C_size = sizeof(__hip_bfloat16) * A_rows * B_cols;

    std::vector<__hip_fp8_e4m3_fnuz> A_h = generate_random_vector<__hip_fp8_e4m3_fnuz>(A_rows, A_cols); // (size_A, 1.0f); 
    std::vector<__hip_fp8_e4m3_fnuz> B_h = generate_random_vector<__hip_fp8_e4m3_fnuz>(B_cols, A_cols);
    std::vector<__hip_bfloat16> C_h(A_rows * B_cols, (__hip_bfloat16) 0.0f);
    std::vector<__hip_bfloat16> C(A_rows * B_cols, (__hip_bfloat16) 0.0f);
    
    std::vector<float> alpha = generate_random_vector<float>(1, 1);
    std::vector<float> beta = generate_random_vector<float>(1, 1);
   

    printf("alpha \n");
    for (int x = 0; x < 10; x++) {
        printf("%f " , alpha[x]);
    }
    printf("\n");

    if (print) {
        print_matrix<__hip_fp8_e4m3_fnuz>(A_h.data(), A_rows, A_cols);
        printf("\n");
        printf("\n");
        print_matrix<__hip_fp8_e4m3_fnuz>(B_h.data(), A_cols, B_cols);
        printf("\n");
        printf("\n");
    }

    __hip_fp8_e4m3_fnuz *A_d, *B_d;
    __hip_bfloat16 *C_d;
    hipMalloc(&A_d, A_size) ;
    hipMalloc(&B_d, B_size);
    hipMalloc(&C_d, C_size);

    hipMemcpy(A_d, A_h.data(), A_size, hipMemcpyHostToDevice);
    hipMemcpy(B_d, B_h.data(), B_size, hipMemcpyHostToDevice);

    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start, 0);
    hipLaunchKernelGGL(sgemm_16x16x4_e4m3, grid_dims, block_dims, 0, 0, A_d, B_d, alpha.data(), beta.data(), C_d, A_rows, A_cols, B_cols);
    hipEventRecord(stop, 0);
    hipEventSynchronize(stop);
    float ms = 0;
    hipEventElapsedTime(&ms, start, stop);
    hipEventDestroy(start);
    hipEventDestroy(stop);

    hipMemcpy(C_h.data(), C_d, C_size, hipMemcpyDeviceToHost);

    if (print) {
        print_matrix<__hip_bfloat16>(C_h.data(), A_rows, B_cols);
    }
    mat_mul<__hip_fp8_e4m3_fnuz, __hip_bfloat16>(A_h.data(), B_h.data(), C.data(), A_rows, A_cols, B_cols);

    if (print) {
        printf("\n");
        printf("\n");
        print_matrix(C.data(), A_rows, B_cols);
        printf("\n");
        printf("\n");
    }

    //assert(matrices_are_equal(C.data(), C_h.data(), A_rows, B_cols, A_rows, B_cols));
    printf("done \n");
    hipFree(A_d); 
    A_d = nullptr;
    hipFree(B_d); 
    B_d = nullptr;
    hipFree(C_d); 
    C_d = nullptr;

    printf("time knl : %f \n", ms);
    return ms;
} 

float compare_mat_mul_knls(mat_mul_func f, mat_mul_func g, int A_rows, int B_rows, int B_cols, dim3 f_grid_dims, dim3 f_block_dims, dim3 g_grid_dims, dim3 g_block_dims, bool print, int runs) {
    const size_t A_size = sizeof(__hip_fp8_e4m3_fnuz) * A_rows * B_cols;
    const size_t B_size = sizeof(__hip_fp8_e4m3_fnuz) * B_rows * B_cols;
    const size_t C_size = sizeof(__hip_bfloat16) * A_rows * B_cols;
    const size_t alpha_size = sizeof(float) * A_rows * B_cols/128;
    const size_t beta_size = sizeof(float) * B_rows/128 * B_cols/128;

    float ms_f_total = 0.0;
    float ms_g_total = 0.0;

    for (int i = 0; i < runs + 1; i++) {
        // TODO: factor out mallocs
        printf("A rows %d  A cols %d \n", A_rows, B_cols);
        std::vector<__hip_fp8_e4m3_fnuz> A_h = generate_random_vector<__hip_fp8_e4m3_fnuz>(A_rows, B_cols); // (size_A, 1.0f); 
        printf("B rows %d  B cols %d \n", B_rows, B_cols);
        std::vector<__hip_fp8_e4m3_fnuz> B_h = generate_random_vector<__hip_fp8_e4m3_fnuz>(B_rows, B_cols);
        std::vector<__hip_bfloat16> C_h_f(A_rows * B_cols, (__hip_bfloat16) 0.0f);
        std::vector<__hip_bfloat16> C_h_g(A_rows * B_cols, (__hip_bfloat16) 0.0f);
        
        std::vector<float> alpha = generate_random_vector<float>(A_rows, B_cols/128);
        std::vector<float> beta = generate_random_vector<float>(B_rows/128, B_cols/128);
           printf("alpha \n");
    for (int x = 0; x < 10; x++) {
        printf("%f " , alpha[x]);
    }
    printf("\n"); 
//            print_matrix<float>(alpha.data(), A_rows, B_cols/128);
       if (print) {
           printf("A: \n");
           print_matrix<__hip_fp8_e4m3_fnuz>(A_h.data(), A_rows, B_cols);
           printf("B: \n");
            print_matrix<__hip_fp8_e4m3_fnuz>(B_h.data(), B_rows, B_cols);
           printf("\n");
       }

        __hip_fp8_e4m3_fnuz *A_d, *B_d;
        __hip_bfloat16 *C_d_f, *C_d_g;
        
        float* alpha_d; 
        float* beta_d;

        hipMalloc(&alpha_d, alpha_size);
        hipMalloc(&beta_d, beta_size);

        hipMalloc(&A_d, A_size) ;
        hipMalloc(&B_d, B_size);
        hipMalloc(&C_d_f, C_size);
        hipMalloc(&C_d_g, C_size);

        hipMemcpy(A_d, A_h.data(), A_size, hipMemcpyHostToDevice);
        hipMemcpy(B_d, B_h.data(), B_size, hipMemcpyHostToDevice);
        hipMemcpy(alpha_d, alpha.data(), alpha_size, hipMemcpyHostToDevice);
        hipMemcpy(beta_d, beta.data(), beta_size, hipMemcpyHostToDevice);

        hipEvent_t f_start, f_stop;
        hipEventCreate(&f_start);
        hipEventCreate(&f_stop);
        hipEventRecord(f_start, 0);
        //                                                                                                     m       n       k
        //hipLaunchKernelGGL(f, f_grid_dims, f_block_dims, 0, 0, A_d, B_d, alpha.data(), beta.data(), C_d_f, A_rows, A_cols, A_cols );
       hipLaunchKernelGGL(f, f_grid_dims, f_block_dims, 0, 0, A_d, B_d, alpha_d, beta_d, C_d_f, A_rows, B_rows, B_cols);
        hipEventRecord(f_stop, 0);
        hipEventSynchronize(f_stop);
        float ms_f = 0.0;
        // ignore first run
        hipEventElapsedTime(&ms_f, f_start, f_stop);
        if (i != 0) {
            ms_f_total += ms_f;
        }
        hipEventDestroy(f_start);
        hipEventDestroy(f_stop);

        hipMemcpy(C_h_f.data(), C_d_f, C_size, hipMemcpyDeviceToHost);

       if (print) {
           printf("C f \n");
           print_matrix<__hip_bfloat16>(C_h_f.data(), A_rows, B_cols);
       }

        hipEvent_t g_start, g_stop;
        hipEventCreate(&g_start);
        hipEventCreate(&g_stop);
        hipEventRecord(g_start, 0);
        hipLaunchKernelGGL(g, g_grid_dims, g_block_dims, 0, 0, A_d, B_d, alpha_d, beta_d, C_d_g, A_rows, B_rows, B_cols);
        hipEventRecord(g_stop, 0);
        hipEventSynchronize(g_stop);
        float ms_g = 0;
        hipEventElapsedTime(&ms_g, g_start, g_stop);
        if (i != 0) {
            ms_g_total += ms_g;
        }
        hipEventDestroy(g_start);
        hipEventDestroy(g_stop);

        hipMemcpy(C_h_g.data(), C_d_g, C_size, hipMemcpyDeviceToHost);

       if (print) {
           printf("C g \n");
           print_matrix<__hip_bfloat16>(C_h_g.data(), A_rows, B_cols);
       }

        if (!matrices_are_equal<__hip_bfloat16>(C_h_f.data(), C_h_g.data(), A_rows, B_rows, A_rows, B_rows)) {

            printf("kernel g is wrong \n");
            float wrong = 0.0;
            float total = 0.0;
            float zero = 0.0;

            int zeros_found = 0;

            for (int q = 0; q < A_rows; q++) {
                for (int w = 0; w < B_rows; w++) {
                    if (C_h_f[q * B_rows + w] != C_h_g[q * B_rows + w]) {
                        wrong += 1.0;
                        if (wrong < 10) {
                            printf("C with worng value row %d col %d f: %f g: %f \n", q, w, (float) C_h_f[q * B_rows + w], (float) C_h_g[q * B_rows + w]);
                        }
                        if ((float) C_h_g[q * B_rows + w] == 0.0) {
                            zero += 1;
                            zeros_found += 1;
                            if (zeros_found < 10) {
                                printf("C with zero row %d col %d \n", q, w);
                            }
                        }
                    }
                    total += 1.0;
            }
            }
            
            printf("error rate: wrong: %f total: %f %f \n", wrong, total, wrong/total);
            printf("zero rate  %f \n", zero/total);

            if (print) {
                for (int q = 0; q < A_rows; q++) {
                    for (int w = 0; w < B_rows; w++) {
                        if (C_h_f[q * B_cols + w] != C_h_g[q * B_cols + w]) {
                            printf("(%d, %d) F f: %f g: %f", q, w, (float) C_h_f[q * B_cols + w], (float) C_h_g[q * B_cols + w]);
                            printf("F ");
                        } else {
                            printf("P ");
                        }

                }

                        printf("\n");
                }
            }
        }
        printf("done \n");
        hipFree(A_d); 
        A_d = nullptr;
        hipFree(B_d); 
        B_d = nullptr;
        hipFree(C_d_f); 
        hipFree(C_d_g); 
        C_d_f = nullptr;
        C_d_g = nullptr;

    }

    float f_avg = ms_f_total/(float)runs;
    float g_avg = ms_g_total/(float)runs;
    
    printf("f time %f g time %f \n", f_avg, g_avg);

    return (g_avg - f_avg)/f_avg;
} 

float test_mfma(mat_mul_func f, int A_rows, int A_cols, int B_cols, dim3 grid_dims, dim3 block_dims, bool print) {
    const size_t A_size = sizeof(__hip_fp8_e4m3_fnuz) * A_rows * A_cols;
    const size_t B_size = sizeof(__hip_fp8_e4m3_fnuz) * A_cols * B_cols;
    const size_t C_size = sizeof(__hip_bfloat16) * A_rows * B_cols;

    int runs = 10;
    float total_time = 0.0;
    for (int i = 0; i < runs; i++) {
        total_time += test_mat_mul_knl(f, A_rows, A_cols, B_cols, grid_dims, block_dims, false);         
    }

    return total_time/runs;
}

int main() {
// 1024    1536    7168      8.63
// 1024    4608    7168     25.89
// 6144    1536    7168     51.78
// 6144    4608    7168    155.30
// 1024    7168     256      3.17
// 6144    7168     256     17.27


    //int A_rows = 1024;
    //int A_cols = 1600;
    //int B_cols = 7200;
    //
    //6144    4608    7168
//    (1023, 700) F f: 48896.000000 g: 49408.000000F
    int A_rows = 1024;
    int B_rows = 1536;
    int B_cols = 7168;
//    int B_rows = 1600;
//    int B_cols = 7200;
    

    dim3 f_grid(A_rows/16, B_rows/16);
    dim3 f_block(16, 16);

    int g_grid_len = 32;
    dim3 g_grid(B_rows/g_grid_len, A_rows/g_grid_len);
    dim3 g_block(g_grid_len, g_grid_len/4);

//     int g_grid_len = 16;
//     dim3 g_grid(B_rows/g_grid_len, A_rows/g_grid_len);
//     dim3 g_block(g_grid_len, g_grid_len/4);
    float t = compare_mat_mul_knls(mat_mul_ref, sgemm_16x16x4_e4m3, A_rows, B_rows, B_cols, f_grid, f_block, g_grid, g_block, false, 10);
   // float t = compare_mat_mul_knls(mat_mul_ref, sgemm_16x16x4_e4m3_bak, A_rows, B_rows, B_cols, f_grid, f_block, g_grid, g_block, false, 5);
    
    printf(" delta %f \n", t);
}

