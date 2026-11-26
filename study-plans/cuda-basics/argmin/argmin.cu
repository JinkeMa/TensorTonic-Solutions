#include <cuda_runtime.h>
#include <float.h>

// each block get its own min value and corresponding index
__global__ void argmin_kernel(const float* input, float* block_vals, int* block_idxs, int N) {
    // Write code here
    __shared__ float vals[256];
    __shared__ float idxs[256];
    int tid = threadIdx.x;
    int g_tid = blockIdx.x * blockDim.x + tid;
    idxs[tid] = g_tid;
    if (g_tid < N){
        vals[tid] = input[g_tid];
    }
    else
        vals[tid] = FLT_MAX;
    
    __syncthreads();
    for(int st = blockDim.x >> 1; st >0; st >>=1){
        if(tid < st){
            float v1 = vals[tid];
            float v2 = vals[tid+st];
            int i1 = idxs[tid];
            int i2 = idxs[tid+st];
            if(v2 < v1 || (v2==v1 && i2 < i1)) {
                idxs[tid] = i2;
                vals[tid] = v2;
            }
        }
        __syncthreads();
    }
    if(tid==0) {
        block_vals[blockIdx.x] = vals[0];
        block_idxs[blockIdx.x] = idxs[0];
    }
}

// use the first thread block to do reducing
__global__ void argmin_finalize_kernel(const float* block_vals, const int* block_idxs, int* result, int num_blocks) {
    // Write code here
    __shared__ float mins_[256];
    __shared__ float idxs_[256];

    int tid = threadIdx.x;
    float min_v = FLT_MAX;
    int min_idx = INT_MAX;
    for(int i = tid; i < num_blocks; i += blockDim.x){
        float v = block_vals[i];
        int id = block_idxs[i];
        if ( v < min_v || (v == min_v &&id<min_idx))  {
            min_v = v;
            min_idx = id;
        }
    }
    mins_[tid] =min_v;
    idxs_[tid] =min_idx;
    __syncthreads();
    for(int st = blockDim.x >> 1; st >0; st>>=1){
        if(tid < st){
            float v1 = mins_[tid];
            float v2 = mins_[tid+st];
            int id1 = idxs_[tid];
            int id2 = idxs_[tid+st];
            if(v2 < v1 || (v2==v1 && id2 < id1)){
                mins_[tid] = v2;
                idxs_[tid] = id2;
            }
        }
        __syncthreads();
    }
    if(tid ==0) result[0] = idxs_[0];
}

extern "C" void solve(const float* input, int* result, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    float* block_vals = nullptr;
    int* block_idxs = nullptr;
    cudaMalloc(&block_vals, blocks * sizeof(float));
    cudaMalloc(&block_idxs, blocks * sizeof(int));

    argmin_kernel<<<blocks, threads>>>(input, block_vals, block_idxs, N);
    argmin_finalize_kernel<<<1, threads>>>(block_vals, block_idxs, result, blocks);
    cudaDeviceSynchronize();

    cudaFree(block_vals);
    cudaFree(block_idxs);
}
