#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <float.h>


__device__ __forceinline__ float loadfp32(const void* p, int dtype_code){
    if (dtype_code == 0) return *(const float*)p;
    else if(dtype_code == 1) return __half2float(*(const half*)p);
    else if(dtype_code == 2) return __bfloat162float(*(const nv_bfloat16*)p);
}
__device__ __forceinline__ void fromfp32(float p, void* o, int dtype_code){
    if (dtype_code == 0)  *(float*)o = p;
    else if(dtype_code == 1)  *(__half*)o = __float2half_rn(p);
    else if(dtype_code == 2)  *(__nv_bfloat16*)o = __float2bfloat16(p);
}

__global__ void online_softmax_kernel(
    const void* input,
    void* output,
    int rows,
    int cols,
    long long input_row_stride,
    long long input_col_stride,
    long long output_row_stride,
    long long output_col_stride,
    int dtype_code
) {
    //
    int szb = dtype_code == 0 ? 4 : 2;
    extern __shared__  float smem[];
    float* max_s = smem;
    float* l_s = smem + blockDim.x;
    //
    float max_val = -FLT_MAX;
    float l_val  = 0;
    int tid = threadIdx.x;
    int g_tid = blockIdx.x * blockDim.x + threadIdx.x;
    // reduce #1
    const unsigned char* row = (const unsigned char*)input + blockIdx.x * input_row_stride * szb;
    for(int i = tid;i < cols; i+=blockDim.x){
        float v = loadfp32(row + i * input_col_stride * szb, dtype_code);
        float max_new = fmaxf(v, max_val);
        //
        float l_new = expf(max_val - max_new)*l_val + expf(v - max_new);

        max_val = max_new;
        l_val = l_new;
    }
    max_s[tid] = max_val;
    l_s[tid] = l_val;
    __syncthreads();

    for(int st = blockDim.x >> 1; st > 0; st >>= 1){
        if(tid < st){
            float max1 = max_s[tid];
            float max2 = max_s[tid + st];
            float l_1 = l_s[tid];
            float l_2 = l_s[tid + st];
            float max_new = fmaxf(max1,max2);
            float left_scale = max1 == -FLT_MAX ? 0.0f : expf(max1 - max_new);
            float right_scale = max2 == -FLT_MAX ? 0.0f : expf(max2 - max_new);
            float l_new = left_scale * l_1 + right_scale * l_2;
            max_s[tid] = max_new;
            l_s[tid] = l_new;
        }
        __syncthreads();
    }

    // output
    unsigned char* row_o = (unsigned char*)output + blockIdx.x * output_row_stride * szb;
    for (int j = tid; j < cols; j+=blockDim.x){
        float v_input = loadfp32(row + j * input_col_stride * szb, dtype_code);
        float p = __fdividef(expf(v_input - max_s[0]), l_s[0]);        
        fromfp32(p, row_o + j * output_col_stride * szb, dtype_code);
    }
    
}

extern "C" void solve(
    const void* input,
    void* output,
    int rows,
    int cols,
    long long input_row_stride,
    long long input_col_stride,
    long long output_row_stride,
    long long output_col_stride,
    int block_size,
    int dtype_code
) {
    if (
        rows < 0 ||
        cols < 0 ||
        block_size < 32 ||
        block_size > 1024 ||
        (block_size & (block_size - 1)) != 0 ||
        dtype_code < 0 ||
        dtype_code > 2
    ) {
        cudaDeviceSynchronize();
        return;
    }
    if (rows == 0 || cols == 0) {
        cudaDeviceSynchronize();
        return;
    }
    size_t shared_bytes = 2 * block_size * sizeof(float);
    online_softmax_kernel<<<rows, block_size, shared_bytes>>>(
        input,
        output,
        rows,
        cols,
        input_row_stride,
        input_col_stride,
        output_row_stride,
        output_col_stride,
        dtype_code
    );
    cudaDeviceSynchronize();
}
