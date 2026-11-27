#include <cuda_runtime.h>
#include <math.h>
#include <float.h>

__global__ void scores_kernel(const float* Q, const float* K, float* scores, int N, int D) {
    // Write code here
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;
    // Q * K.T
    float scale =sqrtf(static_cast<float>(D));
    float score = 0;
    if(tidx < N && tidy < N){
        for(int i =0; i < D; ++i){
            score += Q[tidy * D + i] * K[tidx * D + i];
        }
        score = __fdividef(score,scale);
        scores[tidy * N + tidx] = score;
    }
}

__global__ void softmax_rows_kernel(float* scores, int N) {
    // Write code here
    __shared__ float maxs[256];
    __shared__ float ls[256];

    int tid = threadIdx.x;
    
    float max_ = -FLT_MAX;
    float l_ = 0;

    for (int i = tid; i < N; i+=blockDim.x){
        float max_new = fmaxf(scores[i + blockIdx.x * N],max_);
        l_ = l_ * __expf(max_ - max_new) + __expf(scores[i + blockIdx.x * N] - max_new);
        max_ = max_new;
    }
    ls[tid] = l_;
    maxs[tid] = max_;
    __syncthreads();

    for(int st = blockDim.x>>1; st > 0; st>>=1){
        if(tid < st){
            float maxa = maxs[tid];
            float maxb = maxs[tid + st];
            float la = ls[tid];
            float lb = ls[tid + st];
            float max_new = fmaxf(maxa,maxb);            
            ls[tid] = la * __expf(maxa - max_new) + lb * __expf(maxb - max_new);
            maxs[tid] = max_new;
            
        }
        __syncthreads();
    }
    // softmax
    for (int i = tid; i < N; i+=blockDim.x){
        scores[i + blockIdx.x * N] = __fdividef(__expf(scores[i + blockIdx.x * N] - maxs[0]), ls[0]);
    }
}

__global__ void av_kernel(const float* attn, const float* V, float* output, int N, int D) {
    // Write code here
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    int tidy = blockIdx.y * blockDim.y + threadIdx.y;

    if(tidx < D && tidy < N){
        float o = 0;
        for(int i = 0; i < N; ++i){
            o +=attn[tidy * N + i] * V[i * D + tidx];
        }
        output[tidy * D + tidx] = o;
    }    
}

extern "C" void solve(const float* Q, const float* K, const float* V, float* output, int N, int D) {
    float* scores;
    cudaMalloc(&scores, (size_t)N * N * sizeof(float));

    dim3 sThreads(16, 16);
    dim3 sBlocks((N + 15) / 16, (N + 15) / 16);
    scores_kernel<<<sBlocks, sThreads>>>(Q, K, scores, N, D);

    softmax_rows_kernel<<<N, 256>>>(scores, N);

    dim3 oThreads(16, 16);
    dim3 oBlocks((D + 15) / 16, (N + 15) / 16);
    av_kernel<<<oBlocks, oThreads>>>(scores, V, output, N, D);

    cudaDeviceSynchronize();
    cudaFree(scores);
}