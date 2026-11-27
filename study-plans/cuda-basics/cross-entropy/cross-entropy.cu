#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void cross_entropy_row_kernel(const float* logits, const int* target, float* partial, int B, int C) {
    // Write code here
    __shared__ float maxs[256];
    __shared__ float ls[256];
    
    int tid = threadIdx.x;
    float max_ = -FLT_MAX;
    float ls_ = 0;
    for(int i = tid; i < C; i += blockDim.x){
        float max_new = fmaxf(logits[blockIdx.x * C + i],max_);
        ls_ = ls_ * expf(max_-max_new) + expf(logits[blockIdx.x * C + i]-max_new);
        max_ = max_new;
    }
    maxs[tid] = max_;
    ls[tid] = ls_;
    __syncthreads();
    for(int st = blockDim.x >> 1; st > 0; st >>= 1){
        if(tid < st) {
            float maxa = maxs[tid];
            float maxb = maxs[tid + st];
            float la = ls[tid];
            float lb = ls[tid + st];
            float max_new = fmaxf(maxa, maxb);
            ls[tid] = la * expf(maxa - max_new) + lb * expf(maxb - max_new);
            maxs[tid] = max_new;
            __syncthreads();
        }
    }
    // m = maxs[0]
    int cls_id = target[blockIdx.x];
    if(tid ==0) {
        float log_softmax = logits[blockIdx.x * C + cls_id] - maxs[0] - __logf(ls[0]);
        //partial[blockIdx.x] = __fdividef(log_softmax,-B);
        float loss_partial = __fdividef(log_softmax,-B);
        atomicAdd(partial,loss_partial);
    }
}

__global__ void cross_entropy_finalize_kernel(float* loss, int B) {
    // Write code here
}

extern "C" void solve(const float* logits, const int* target, float* loss, int B, int C) {
    cudaMemset(loss, 0, sizeof(float));

    int threads = 256;
    dim3 blocks(B);
    cross_entropy_row_kernel<<<blocks, threads>>>(logits, target, loss, B, C);
    cross_entropy_finalize_kernel<<<1, 1>>>(loss, B);
    cudaDeviceSynchronize();
}