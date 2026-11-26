`input_row_stride` 是**行方向的步长**：从第 `i` 行到第 `i+1` 行，指针要跳多少个元素。它让 kernel 能处理**不连续存储**的矩阵。
## 元素定位公式

```
// 逻辑上访问 input[i][j]
const char* p = (const char*)input
              + (i * input_row_stride + j * input_col_stride) * elem_size;
```

`output_row_stride` / `output_col_stride` 同理，用于输出。输入和输出分开传 stride，是因为**两者布局可能不同**（比如输入是转置视图，输出是连续写）。
## 为什么需要它

如果只支持连续矩阵，`row_stride` 恒等于 `cols`、`col_stride` 恒等于 1，那这两个参数就是废的。传进来是因为**输入往往是某个大 tensor 的视图**：



场景                                                                  row_stride                 col_stride

连续 `rows×cols`                                                    `cols`                         `1`

从 `R×C` 大矩阵取前 `rows×cols` 子块               `C`（≠ cols）                  `1`

转置视图（逻辑 `rows×cols`，物理列主序）          `1`                     `rows` 或原矩阵行数

每隔 k 列取样                                                          `C`                            `K`

沿 batch 维切片                                             `C`（原矩阵 cols）            `1`



核心价值：**上游不用做 **`contiguous()`** 拷贝**，零拷贝直接进 kernel。这是 PyTorch / NumPy 视图语义的 C 接口映射。
## 为什么是 ,`long long`

两个原因：

- **超大偏移**：`i * row_stride` 在元素数超过 2^31 的 tensor 上会溢出 `int`（比如 50000×50000 的矩阵，行偏移就到 2.5e9）；
- **允许负 stride**：翻转视图（`x[::-1]`）的 stride 是负的，指针往前退。所以不能是无符号类型。
## 单位约定：元素，不是字节

这是最容易踩的坑。stride 通常是**元素个数**，所以字节偏移要乘 `elem_size`：

```
const size_t esz = (dtype_code == 0) ? 4 : 2;   // fp32 / fp16,bf16
const char* row = (const char*)input + (size_t)blockIdx.x * input_row_stride * esz;
float v = ((const float*)row)[j * input_col_stride];
```

如果约定成字节数，那 `dtype_code` 就不用参与了，公式也更简单——但那样跨 dtype 复用同一个 stride 值就不方便。**两种约定都有人用，接口文档必须写死一个**，否则调用方和 kernel 各按各的理解算，结果就是错位读取。
## 在原始 kernel 里的典型用法

`&lt;&lt;&lt;rows, block_size&gt;&gt;&gt;` 说明**一个 block 处理一行**：

```
const int row = blockIdx.x;                              // 哪一行
const char* base = (const char*)input
                 + (size_t)row * input_row_stride * esz; // 该行起点
// 然后 block 内各线程按 col_stride 取列
for (int j = threadIdx.x; j &lt; cols; j += blockDim.x) {
    float v = ((const float*)base)[j * input_col_stride];
    ...
}
```
## 这个函数里没校验的风险

原始 `solve` 对 stride **一个字节都没检查**，能出的事：

- `row_stride`** 传 0**：所有行读到同一行数据，结果全错但不报错（除非是有意的广播）；
- `row_stride &lt; cols`：行与行重叠，读到脏数据；
- **负 stride + 空指针边界**：指针算术退到 `input` 之前，非法访存；
- `long long`** 乘 **`esz`** 溢出**：`row * row_stride * esz` 在极端情况下仍可能溢出 `size_t`，乘法前应该检查；
- **输出 stride 与输入不匹配**：写越界，静默踩坏别的 tensor。

工程上至少该加一条：`row_stride != 0 &amp;&amp; col_stride != 0`，以及 `abs(row_stride) &gt;= cols * abs(col_stride)`（连续或带 padding 的前提）。
## 一句话总结

`input_row_stride`** 告诉 kernel“换到下一行要跳多远”，让 kernel 能直接吃非连续视图、省掉上游的 contiguous 拷贝**



---

---

---

---

---

---

---



CUDA 中常见的浮点类型可分为**常规计算类型**和**低精度／矩阵计算格式**。一般逐元素计算优先用 `float`；需要更高精度时用 `double`；训练或矩阵乘法中才常考虑低精度类型。

- 单精度  `float`  GPU 上常用的通用浮点类型.
- 双精度  `double`  需要更高数值精度时使用；不同 GPU 的双精度吞吐能力差异较大.
- 半精度  `__half`  16 位浮点，常用于减少存储和带宽开销、加速矩阵运算；通常可转成 `float` 做累加.
- Brain Float 16  `__nv_bfloat16` | 16 位格式，动态范围与 `float` 相近、精度较低；适用于支持它的 Tensor Core 矩阵运算。[Bfloat16]([https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=229](https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=229))
- FP8 | E4M3、E5M2  更低精度的矩阵运算输入格式，主要用于支持 FP8 的 Tensor Core 和相关库。[FP8 Tensor Cores]([https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=519](https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=519)) |
- TF32  `precision::tf32`  Tensor Core 矩阵运算格式；数据通常仍以 `float` 存放，使用前转换到 TF32 精度，不是日常标量计算类型。[TF32]([https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=229](https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=229))
- 四精度  `__float128`  特定平台和较新计算能力设备上的选项；受主机编译器与设备能力限制，不是一般 CUDA 程序的常用类型。[Quad Precision]([https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=439](https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=439))


### 常规写法,`float`, 和 ,`double`

它们像普通 C++ 浮点数一样使用：

```
c__global__ void calc(const float* x, float* y) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
y[i] = x[i] *x[i] + 1.0f;
}
```

`float` 的常见数学函数带 `f` 后缀，例如 `fmaxf(a, b)sqrtf(a)double` 常用 `fmax(a, b)sqrt(a)`。CUDA 指南列出了两种类型各自的数学函数精度信息。[Float Math]([https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=422)[Double](https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=422)[Double) Math]([https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=425](https://www.alphaxiv.org/abs/626c6fb3-2bc2-49c5-a243-c22bb07467e0v1?page=425))
### 半精度和 BF16：低精度存储，常用 ,`float`, 累加

头文件分别是 `&lt;cuda_fp16.h&gt;` 和 `&lt;cuda_bf16.h&gt;`。一种常见模式是输入、输出用低精度，计算累加用 `float`：

```
#include &lt;cuda_fp16.h&gt; 
__global__ void halfExample(const __half* x, __half* y) {
int i = blockIdx.x * blockDim.x + threadIdx.x;
float v = __half2float(x[i]);
y[i] = __float2half_rn(v * v);}
```



BF16 的转换模式类似：

```
#include &lt;cuda_bf16.h&gt;
float v = __bfloat162float(x[i]);
y[i] = __float2bfloat16(v * v);
```