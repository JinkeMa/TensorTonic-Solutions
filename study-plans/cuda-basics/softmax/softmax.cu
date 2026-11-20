#include <cuda_runtime.h>
#include <float.h>

__global__ void reduce_max(const float* input, float* max_, int N)
{
    __shared__ float maxs[256];
    float tmax = -FLT_MAX;
    for (int i = threadIdx.x; i < N; i+=blockDim.x) 
    {
        tmax = fmaxf(tmax,input[i]);
    };
    maxs[threadIdx.x] = tmax;
    __syncthreads();
    for (int st = blockDim.x >> 1; st >0; st >>= 1)
    {
        if (threadIdx.x < st) maxs[threadIdx.x] = fmaxf(maxs[threadIdx.x],maxs[threadIdx.x + st]);
    }
    __syncthreads();
    if (threadIdx.x == 0) 
        *max_ = maxs[0];
}

__global__ void exp_and_sum(const float* input, float* output, const float*max_,  float* sum_, int N)
{
    //
    __shared__ float sum[256];
    float m = *max_;
    float local_sum = 0;
    for (int i = threadIdx.x; i < N; i+=blockDim.x) 
    {
        float xx = expf(input[i]-m);
        output[i] = xx;
        local_sum += xx;
    }
    sum[threadIdx.x] = local_sum;
    __syncthreads();
    for (int st = blockDim.x >> 1; st > 0; st >>= 1)
    {
        if (threadIdx.x < st) sum[threadIdx.x] += sum[threadIdx.x + st];
    }
    __syncthreads();
    if(threadIdx.x ==0 )
        *sum_ = sum[0];
}

__global__ void normalize(float* output,float*sum_,int N)
{
    int tidx = blockIdx.x * blockDim.x + threadIdx.x;
    if(tidx < N)
        output[tidx] = output[tidx] / *sum_;
}


extern "C" void solve(const float* input, float* output, int N) {

    float* max_;
    float* exp_sum;
    cudaMalloc(&max_, sizeof(float));
    cudaMalloc(&exp_sum, sizeof(float));
    reduce_max<<<1,256>>>(input, max_, N);
    exp_and_sum<<<1,256>>>(input,output,max_,exp_sum,N);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    normalize<<<blocks, threads>>>(output, exp_sum,N);
    cudaDeviceSynchronize();
}