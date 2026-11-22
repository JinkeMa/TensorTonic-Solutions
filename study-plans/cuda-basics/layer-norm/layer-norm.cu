#include <cuda_runtime.h>
#include <math.h>

__global__ void get_mu_(const float* input, float* mu_,int N)
{
    // one row per block
    __shared__ float mu[256];
    int tid = threadIdx.x;
    float sum = 0;
    for(int i = tid; i<N; i+=blockDim.x) sum += input[i + blockIdx.x * N];    
    mu[tid] = sum;
    __syncthreads();
    for(int st = blockDim.x >> 1; st > 0; st >>= 1)
    {
         if(tid < st) mu[tid] += mu[tid + st];
        __syncthreads();
    }
    if(tid==0) mu_[blockIdx.x] = mu[0] / N;
}

__global__ void get_sigma_(const float* input, const float*mu_, float* sigma_, int N)
{
    __shared__ float sig[256];
    int tid = threadIdx.x;
    float sum = 0;
    
    for(int i = tid; i<N; i+=blockDim.x)
    {
        float subt = input[i + blockIdx.x * N]-mu_[blockIdx.x];
        sum += subt * subt;   
    }
    sig[tid] = sum;
    __syncthreads();
    for(int st = blockDim.x >> 1; st > 0; st >>= 1)
    {
         if(tid < st) sig[tid] += sig[tid + st];
         __syncthreads();
    }
    if(tid==0) sigma_[blockIdx.x] = sig[0] / N;
}

__global__ void layer_norm_kernel(const float* input, const float* gamma, const float* beta, float* output, float*mu_, float*sigma_, int N, float eps) {
    // Write code here
    int tid = threadIdx.x;
    int block_stride = blockIdx.x*N;
    for(int i =tid; i< N; i+=blockDim.x) 
    {
        output[i + block_stride] = gamma[i] * (input[i + block_stride] - mu_[blockIdx.x])/(sqrtf(sigma_[blockIdx.x]+eps)) + beta[i];
    }
    
}

extern "C" void solve(const float* input, const float* gamma, const float* beta, float* output, int M, int N, float eps) {
    int threads = 256;
    dim3 blocks(M);
    
    float* mu_;
    float* sigma_;
    cudaMalloc(&mu_,sizeof(float)*M);
    cudaMalloc(&sigma_,sizeof(float)*M);
    get_mu_<<<blocks,256>>>(input,mu_,N);
    get_sigma_<<<blocks,256>>>(input,mu_,sigma_,N);
    layer_norm_kernel<<<blocks, threads>>>(input, gamma, beta, output, mu_,sigma_, N, eps);
    cudaDeviceSynchronize();
}
