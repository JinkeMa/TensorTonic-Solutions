#include <cuda_runtime.h>

__global__ void conv2d_kernel(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    // Write code here
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    float o = 0;
    if (tidx < outW && tidy < outH){
        for (int j = 0; j < kH; ++j){
            for(int i = 0; i < kW; ++i){
                o += input[(tidy+j) * W + tidx + i] * kernel[j * kW + i];
            }
        }
        output[tidy * outW + tidx] = o;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int H, int W, int kH, int kW) {
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    dim3 threads(16, 16);
    dim3 blocks((outW + 15) / 16, (outH + 15) / 16);
    conv2d_kernel<<<blocks, threads>>>(input, kernel, output, H, W, kH, kW);
    cudaDeviceSynchronize();
}
