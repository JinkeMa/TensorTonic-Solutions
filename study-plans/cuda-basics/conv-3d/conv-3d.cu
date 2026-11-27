#include <cuda_runtime.h>

__global__ void conv3d_kernel(const float* input, const float* kernel, float* output, int D, int H, int W, int kD, int kH, int kW) {
    // Write code here
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;

    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    int tidz = blockIdx.z * blockDim.z + threadIdx.z;

    float o =0;
    if(tidx < outW && tidy < outH && tidz < outD){
        for(int d=0;d<kD;++d){
            for(int i =0;i<kH;++i){
                for(int j=0;j<kW;++j){                    
                    o += input[(tidz +d)* H * W + (tidy + i) * W + tidx + j] * kernel[d * kH * kW + i * kW + j];
                }
            }
        }
        output[tidz*outH * outW + tidy * outW + tidx] = o;
    }
}

extern "C" void solve(const float* input, const float* kernel, float* output, int D, int H, int W, int kD, int kH, int kW) {
    int outD = D - kD + 1;
    int outH = H - kH + 1;
    int outW = W - kW + 1;
    dim3 threads(8, 8, 8);
    dim3 blocks((outW + 7) / 8, (outH + 7) / 8, (outD + 7) / 8);
    conv3d_kernel<<<blocks, threads>>>(input, kernel, output, D, H, W, kD, kH, kW);
    cudaDeviceSynchronize();
}
