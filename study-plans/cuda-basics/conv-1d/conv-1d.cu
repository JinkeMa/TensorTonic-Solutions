#include <cuda_runtime.h>

__global__ void conv1d_kernel(const float* input, const float* kernel, float* output, int N, int kN) {
    // Write code here
    int outN = N - kN + 1;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int start_i = tid;
    float o=0;
    if(tid < outN){
        for(int i = 0; i<kN; ++i){
            o += input[start_i + i] * kernel[i];
        }        
    }
    output[tid] = o;
}

extern "C" void solve(const float* input, const float* kernel, float* output, int N, int kN) {
    int outN = N - kN + 1;
    int threads = 256;
    dim3 blocks((outN + 255) / 256);
    conv1d_kernel<<<blocks, threads>>>(input, kernel, output, N, kN);
    cudaDeviceSynchronize();
}
