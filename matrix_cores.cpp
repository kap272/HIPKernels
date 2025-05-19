#include <hip/amd_detail/amd_hip_fp8.h>

#include <hip/amd_detail/amd_hip_bf16.h>
#include <hip/hip_runtime.h>
#include <random>
#include <iterator>
#include <algorithm>
#include <iostream>
#include <vector>
#include <cassert>

#define M 16
#define N 16
#define K 4

#define BLOCK 128
 
 
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

__global__ void sgemm_16x16x4(const float *A, const float *B, float *D)
{
  using float4 = __attribute__( (__vector_size__(K * sizeof(float)) )) float;
  float4 dmn = {0};
 
  int mk = threadIdx.y + K * threadIdx.x;
  int kn = threadIdx.x + N * threadIdx.y;
 
  float amk = A[mk];
  float bkn = B[kn];
  dmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, dmn, 0, 0, 0);
 
  for (int i = 0; i < 4; ++i) {
    const int idx = threadIdx.x + i * N + threadIdx.y * 4 * N;
    D[idx] = dmn[i];
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

__global__ void custom_kernel(const float* A, const float* B, const float* as, const float* bs, float* C, int m, int n, int k) {
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
        __syncthreads();

        if (cx == 2 && cy == 2) {
            printf("%f %f \n", A[0], B[0]);
            printf("%d %d \n", upper_left_y, upper_left_x);
            printf("lane id: %d, A row, col: %d, %d, A idx: %d, A val: %f\n", lane_id, A_row + 1, A_col + 1, A_idx, amk);
            printf("             B row, col: %d, %d, B idx: %d, B val: %f\n\n",         B_row + 1, B_col + 1, B_idx, bkn);
        }
        cmn = __builtin_amdgcn_mfma_f32_16x16x4f32(amk, bkn, cmn, 0, 0, 0);
        __syncthreads();
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
void print_matrix(float* A, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.1f ", A[cols * i + j]);
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

template<typename T> std::vector<T> generate_random_vector(int rows, int cols) {
    std::random_device rnd_device;
    std::mt19937 mersenne_engine {rnd_device()};  
    std::uniform_int_distribution<int> dist {1, 10};
    
    auto gen = [&](){ return (T) dist(mersenne_engine); };
    std::vector<T> vec(sizeof(T) * rows * cols);
    std::generate(vec.begin(), vec.end(), gen);
    
    return vec; 
}

int main() {
    int A_rows = 32;
    int A_cols = 32;
    int B_cols = 32;
    const size_t size_A = sizeof(float) * A_rows * A_cols;
    const size_t size_B = sizeof(float) * A_cols * B_cols;
    const size_t size_D = sizeof(float) * A_rows * B_cols;

//    std::vector<float> h_A(size_A, 1.0f); 
    std::vector<float> h_A = generate_random_vector<float>(A_rows, A_cols); // (size_A, 1.0f); 
    std::vector<float> h_B = generate_random_vector<float>(A_cols, B_cols);
    std::vector<float> h_D = generate_random_vector<float>(A_rows, B_cols);
    std::vector<float> C(size_D, 0.0f);
    
    std::vector<float> alpha = generate_random_vector<float>(1, 1);
    std::vector<float> beta = generate_random_vector<float>(1, 1);
    
    print_matrix(h_A.data(), A_rows, A_cols);
    printf("\n");
    printf("\n");
    print_matrix(h_B.data(), A_cols, B_cols);
    printf("\n");
    printf("\n");

    float *d_A, *d_B, *d_D;
    hipMalloc(&d_A, size_A * sizeof(float));
    hipMalloc(&d_B, size_B * sizeof(float));
    hipMalloc(&d_D, size_D * sizeof(float));

    hipMemcpy(d_A, h_A.data(), size_A * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(d_B, h_B.data(), size_B * sizeof(float), hipMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid(A_rows/16, B_cols/16);
    hipLaunchKernelGGL(custom_kernel, grid, block, 0, 0, d_A, d_B, alpha.data(), beta.data(),  d_D, A_rows, A_cols, B_cols);
    hipDeviceSynchronize();

    hipMemcpy(h_D.data(), d_D, size_D * sizeof(float), hipMemcpyDeviceToHost);

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << h_D[i * N + j] << " ";
        }
        std::cout << "\n";
    }
    mat_mul(h_A.data(), h_B.data(), C.data(), A_rows, A_cols, B_cols);

    printf("\n");
    printf("\n");
    print_matrix(C.data(), A_rows, B_cols);
    printf("\n");
    printf("\n");

    assert(matrices_are_equal(C.data(), h_D.data(), A_rows, B_cols, A_rows, B_cols));

    printf("done \n");

    hipFree(d_A);
    hipFree(d_B);
    hipFree(d_D);

    return 0;
}

