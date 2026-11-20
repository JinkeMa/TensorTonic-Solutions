#include <cuda_runtime.h>

__global__ void matrix_transpose_kernel(const float* A, float* B, int M, int N) {
    // Write code here
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(y < M && x < N)
    {
        B[x *M +y] = A[y*N +x];
    }
}

extern "C" void solve(const float* A, float* B, int M, int N) {
    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (M + 15) / 16);
    matrix_transpose_kernel<<<blocks, threads>>>(A, B, M, N);
    cudaDeviceSynchronize();
}
