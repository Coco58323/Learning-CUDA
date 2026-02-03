# FlashAttention CUDA 实现详解

## 目录
1. [背景知识](#1-背景知识)
2. [标准 Attention 回顾](#2-标准-attention-回顾)
3. [FlashAttention 核心思想](#3-flashattention-核心思想)
4. [CUDA 实现详解](#4-cuda-实现详解)
5. [源码解析](#5-源码解析)
6. [优化技术](#6-优化技术)
7. [调试经验](#7-调试经验)

---

## 1. 背景知识

### 1.1 CUDA 执行模型
```
GPU
├── Grid (所有 blocks)
│   ├── Block 0 (一组 threads)
│   │   ├── Warp 0 (32 threads, 锁步执行)
│   │   ├── Warp 1
│   │   └── ...
│   ├── Block 1
│   └── ...
```

**关键概念**：
- **Warp**: 32 个线程同时执行相同指令（SIMT）
- **Shared Memory**: Block 内线程共享，低延迟（~5 cycles）
- **Global Memory**: 所有线程可访问，高延迟（~400 cycles）
- **Registers**: 每个线程私有，最快

### 1.2 内存带宽瓶颈

| 内存类型 | 带宽 (H100) | 延迟 |
|---------|------------|------|
| Register | ~19 TB/s | 1 cycle |
| Shared | ~19 TB/s | ~5 cycles |
| L2 Cache | ~4 TB/s | ~30 cycles |
| HBM | ~3 TB/s | ~400 cycles |

**FlashAttention 的核心目标**：减少 HBM 访问，用计算换内存

---

## 2. 标准 Attention 回顾

### 2.1 公式
```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

其中：
- Q: Query, shape [seq_len, d_k]
- K: Key, shape [seq_len, d_k]
- V: Value, shape [seq_len, d_v]
- d_k: head dimension

### 2.2 标准实现的内存问题

```python
# 标准实现 (PyTorch)
scores = Q @ K.T / sqrt(d_k)    # [N, N] - 占用 O(N²) 内存!
scores = softmax(scores, dim=-1) # [N, N]
output = scores @ V              # [N, d_v]
```

**问题**：N=4096 时，scores 矩阵需要 64MB (float32)

### 2.3 Softmax 详解

对于向量 x = [x_1, x_2, ..., x_n]：

```
softmax(x_i) = exp(x_i) / Σ exp(x_j)
```

**数值稳定版本**（避免 exp 溢出）：
```
m = max(x)
softmax(x_i) = exp(x_i - m) / Σ exp(x_j - m)
```

---

## 3. FlashAttention 核心思想

### 3.1 分块计算 (Tiling)

将 N×N 的 attention 矩阵分成小块处理：

```
        K (seq_len)
      ┌───┬───┬───┐
      │ 0 │ 1 │ 2 │  <- K blocks
  Q   ├───┼───┼───┤
(seq) │ 0 │ 1 │ 2 │
      ├───┼───┼───┤
      │ 0 │ 1 │ 2 │
      └───┴───┴───┘
```

每次只计算一个 block，不需要存储整个 N×N 矩阵。

### 3.2 Online Softmax 算法

**问题**：Softmax 需要知道所有元素才能计算 max 和 sum。如何分块计算？

**解决方案**：维护 running statistics，动态更新

```python
# 初始化
m_i = -inf  # running max
l_i = 0     # running sum of exp
o_i = 0     # running weighted sum (未归一化的 output)

# 处理每个 K-V block
for j in range(num_blocks):
    # 1. 计算当前 block 的 scores
    S_j = Q @ K_j.T / sqrt(d)
    
    # 2. 找到当前 block 的 max
    m_j = max(S_j)
    
    # 3. 更新 running max
    m_new = max(m_i, m_j)
    
    # 4. 缩放之前的累积值
    # 因为 max 变了，之前计算的 exp 需要调整
    scale = exp(m_i - m_new)
    l_i = l_i * scale
    o_i = o_i * scale
    
    # 5. 计算当前 block 的贡献
    P_j = exp(S_j - m_new)  # [block_m, block_n]
    l_i = l_i + sum(P_j)
    o_i = o_i + P_j @ V_j
    
    # 6. 更新 running max
    m_i = m_new

# 最终归一化
output = o_i / l_i
```

### 3.3 为什么需要缩放？

假设处理到第 j 个 block 时，之前的 max 是 m_i，新的 max 是 m_new。

之前计算的 exp 值基于 m_i：
```
exp(x - m_i)
```

现在需要基于 m_new：
```
exp(x - m_new) = exp(x - m_i) * exp(m_i - m_new)
                              ↑ scale factor
```

---

## 4. CUDA 实现详解

### 4.1 并行策略

```cpp
// 每个 block 处理一个 query position
dim3 grid(batch_size, query_heads, target_seq_len);
dim3 block(128);  // 128 threads per block
```

**工作分配**：
- 每个 CUDA block → 一个 (batch, head, query_pos)
- Block 内 128 threads → 并行处理 head_dim 维度和 key positions

### 4.2 Shared Memory 布局

```cpp
extern __shared__ char smem[];
float* s_q = (float*)smem;          // [head_dim] - Q 向量
float* s_scores = s_q + head_dim;   // [64] - 当前处理的 scores
float* s_out = s_scores + 64;       // [head_dim] - output 累加器
```

**为什么用 Shared Memory？**
- Q 向量被所有 key positions 使用 → 避免重复从 global memory 读取
- Scores 需要做 reduction → Shared memory 支持线程间通信

### 4.3 Warp Shuffle 归约

**传统 shared memory 归约**：
```cpp
// 需要多次 __syncthreads()
for (int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) sdata[tid] += sdata[tid + s];
    __syncthreads();  // 昂贵！
}
```

**Warp Shuffle 归约** (无需 shared memory)：
```cpp
// Warp 内 32 个线程直接通信
for (int offset = 16; offset > 0; offset >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, offset);
}
// val 现在包含 32 个线程值的和
```

**`__shfl_xor_sync` 工作原理**：
```
offset=16: thread 0 与 thread 16 交换
           thread 1 与 thread 17 交换
           ...
offset=8:  thread 0 与 thread 8 交换
           ...
offset=4:  thread 0 与 thread 4 交换
           ...
offset=2:  thread 0 与 thread 2 交换
offset=1:  thread 0 与 thread 1 交换

结果: 所有 32 个线程都有相同的归约结果
```

### 4.4 GQA (Grouped Query Attention)

```cpp
// query_heads = 32, kv_heads = 8
// 每 4 个 query head 共享 1 个 kv head
int kv_head_idx = head_idx * kv_heads / query_heads;
// head_idx=0,1,2,3 → kv_head_idx=0
// head_idx=4,5,6,7 → kv_head_idx=1
// ...
```

### 4.5 Causal Masking

```cpp
// 防止 query 看到未来的 key
if (is_causal && src_pos > tgt_pos) {
    score = -INFINITY;  // softmax 后变成 0
}
```

---

## 5. 源码解析

### 5.1 主要函数

```cpp
template <typename T>
__global__ void flashAttentionKernel(
    const T* Q, const T* K, const T* V, T* O,
    int batch_size, int tgt_seq_len, int src_seq_len,
    int query_heads, int kv_heads, int head_dim,
    float scale, bool is_causal
);
```

### 5.2 主循环结构

```cpp
// 1. 加载 Q 到 shared memory
for (int d = tid; d < head_dim; d += blockDim.x) {
    s_q[d] = (float)q_ptr[d];
}
__syncthreads();

// 2. 初始化 running statistics
m_i = -INFINITY;
l_i = 0;
memset(s_out, 0, head_dim * sizeof(float));

// 3. 处理每个 key block
for (int block_start = 0; block_start < src_seq_len; block_start += BLOCK_SIZE) {
    // 3.1 计算 Q @ K^T
    // 3.2 找 max (parallel reduction)
    // 3.3 计算 exp
    // 3.4 求 sum (parallel reduction)
    // 3.5 缩放并累加 output
}

// 4. 归一化并写出
for (int d = tid; d < head_dim; d += blockDim.x) {
    o_ptr[d] = (T)(s_out[d] / l_i);
}
```

---

## 6. 优化技术

### 6.1 Memory Coalescing

**好**：连续线程访问连续内存地址
```cpp
// threads 0,1,2,3 访问 data[0,1,2,3]
data[tid]
```

**差**：跨步访问
```cpp
// threads 0,1,2,3 访问 data[0,128,256,384]
data[tid * stride]
```

### 6.2 Bank Conflict

Shared Memory 分为 32 个 bank，每个 bank 4 bytes。

**无冲突**：不同线程访问不同 bank
```cpp
sdata[tid]  // thread 0 → bank 0, thread 1 → bank 1, ...
```

**有冲突**：多个线程访问同一 bank
```cpp
sdata[tid * 32]  // 所有线程都访问 bank 0！
```

### 6.3 FMA (Fused Multiply-Add)

```cpp
// 两条指令
a = b * c;
d = a + e;

// 一条 FMA 指令 (更快，更精确)
d = __fmaf_rn(b, c, e);  // d = b * c + e
```

---

## 7. 调试经验

### 7.1 精度问题

**症状**：测试失败，误差约 1e-5

**可能原因**：
1. 浮点累加顺序不同
2. FMA vs 分离的乘加
3. expf vs exp2f

**尝试过的方法**：
| 方法 | 结果 |
|------|------|
| Kahan summation | ❌ 更差 |
| 累加顺序反转 | ❌ 更差 |
| exp2f | ❌ 更差 |
| Parallel reduction | ✅ 保持 |

### 7.2 内存访问错误

```
Runtime error: an illegal memory access was encountered
```

**常见原因**：
1. Shared memory 大小不够
2. 数组越界
3. 指针计算错误

**调试方法**：
```bash
compute-sanitizer ./test_kernels
```

### 7.3 竞态条件

**症状**：结果随机错误

**原因**：缺少 `__syncthreads()`

**规则**：
- 读取其他线程写入的 shared memory 前必须同步
- `__syncthreads()` 是 block 级别屏障

---

## 附录：FlashAttention 源码结构

```
flash-attention/csrc/flash_attn/src/
├── flash_fwd_kernel.h      # 前向核心逻辑
├── softmax.h               # Online Softmax
├── utils.h                 # Allreduce, MaxOp, SumOp
├── mask.h                  # Causal masking
├── kernel_traits.h         # Kernel 配置
└── flash_fwd_launch_template.h  # Kernel launch
```

### 关键代码片段

**Allreduce (warp shuffle)**：
```cpp
template<int THREADS>
struct Allreduce {
    template<typename T, typename Operator>
    static __device__ T run(T x, Operator &op) {
        constexpr int OFFSET = THREADS / 2;
        x = op(x, __shfl_xor_sync(0xffffffff, x, OFFSET));
        return Allreduce<OFFSET>::run(x, op);
    }
};
```

**Online Softmax 更新**：
```cpp
// softmax.h: softmax_rescale_o()
float scores_scale = exp2f((scores_max_prev - scores_max_cur) * softmax_scale_log2);
row_sum *= scores_scale;
acc_o *= scores_scale;
```

---

## 8. Triton vs CUDA 对比

### 8.1 Triton 实现解析

```python
@triton.jit
def attention_forward_kernel(Q, K, V, O, ...):
    # 获取当前处理的位置
    pid_qs = tl.program_id(0)  # query sequence block
    pid_bh = tl.program_id(1)  # batch * head
    
    # 加载 Q block
    q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
    
    # 初始化 running statistics
    old_max = tl.full([BLOCK_SIZE_M], -inf)
    lse_new = tl.full([BLOCK_SIZE_M], -inf)  # log-sum-exp
    acc = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_D])
    
    # 遍历 K/V blocks
    for start_n in range(0, end_k, BLOCK_SIZE_N):
        # 1. 计算 QK^T
        qk = tl.dot(q, tl.trans(k)) * scale
        
        # 2. Causal mask
        qk = tl.where(offs_qs[:, None] >= offs_n[None, :], qk, -inf)
        
        # 3. 找新 max
        new_max = tl.maximum(tl.max(qk, 1), old_max)
        
        # 4. 计算 softmax
        p_i = tl.exp(qk - new_max[:, None])
        
        # 5. 缩放之前的累积
        o_scale = tl.exp(old_max - new_max)
        acc = acc * o_scale[:, None]
        
        # 6. 累加当前 block
        acc += tl.dot(p_i.to(v.dtype), v)
        
        # 7. 更新 statistics
        old_max = new_max
        l_i = tl.exp(lse_new - new_max) + tl.sum(p_i, axis=1)
        lse_new = new_max + tl.log(l_i)
    
    # 最终归一化
    acc_o_scale = tl.exp(old_max - lse_new)
    acc = acc * acc_o_scale[:, None]
```

### 8.2 关键差异

| 方面 | Triton | CUDA |
|-----|--------|------|
| **编程模型** | Block-level，自动向量化 | Thread-level，手动管理 |
| **内存访问** | `tl.load/tl.store` 自动优化 | 需要手动考虑 coalescing |
| **矩阵乘法** | `tl.dot` (编译器优化) | 手动展开或调用 cuBLAS |
| **Reduction** | `tl.max`, `tl.sum` | 手动 warp shuffle |
| **同步** | 自动管理 | 手动 `__syncthreads()` |

### 8.3 LSE (Log-Sum-Exp) 详解

#### 问题：数值溢出

标准 Softmax 计算：
```python
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

**问题**：当 x 很大时，`exp(x)` 会溢出（float32 最大约 3.4e38，exp(89) 就溢出了）

#### 解决方案 1：减去 max（标准技巧）

```python
m = max(x)
softmax(x_i) = exp(x_i - m) / sum(exp(x_j - m))
```

这样 `exp(x_i - m) <= 1`，不会溢出。

#### 问题：Online Softmax 中的累积

在 Online Softmax 中，我们需要跟踪 `sum of exp`：
```python
sum_i = sum(exp(x_j - m_i))  # 相对于当前 max m_i 的 sum
```

当 max 更新时，需要调整：
```python
m_new = max(m_old, m_current_block)
# 缩放之前的 sum
sum_new = sum_old * exp(m_old - m_new) + sum_current_block
```

**问题**：如果 `sum_old` 很大，乘以 `exp(m_old - m_new)` 后可能溢出或精度损失。

#### 解决方案 2：存储 log(sum) 而不是 sum

**核心思想**：存储 `lse = m + log(sum)`（log-sum-exp）

```
lse = m + log(sum(exp(x - m)))
    = m + log(sum of exp values relative to m)
```

#### LSE 的性质

1. **从 lse 恢复 sum**：
   ```
   sum = exp(lse - m)
   ```
   
2. **更新公式推导**：
   
   假设当前状态：`lse_old = m_old + log(sum_old)`
   
   新 block 来了，更新 max：`m_new = max(m_old, block_max)`
   
   新的 sum（相对于 m_new）：
   ```
   sum_new = sum_old * exp(m_old - m_new) + sum_current_block
           = exp(lse_old - m_old) * exp(m_old - m_new) + sum_current_block
           = exp(lse_old - m_new) + sum_current_block
   ```
   
   新的 lse：
   ```
   lse_new = m_new + log(sum_new)
           = m_new + log(exp(lse_old - m_new) + sum_current_block)
   ```

3. **最终归一化**：
   
   Softmax 输出 = unnormalized_output / sum
   
   用 lse 表示：
   ```
   output = unnormalized_output * exp(m - lse)
   ```
   
   因为 `exp(m - lse) = exp(m) / exp(lse) = exp(m) / (exp(m) * sum) = 1/sum`

#### 完整流程（Triton 风格）

```python
# 初始化
m_old = -inf
lse = -inf  # 表示 sum = 0

for each block:
    # 计算 scores 和 max
    scores = Q @ K_block^T / sqrt(d)
    m_block = max(scores)
    m_new = max(m_old, m_block)
    
    # 计算 exp (相对于 m_new)
    p = exp(scores - m_new)
    
    # 缩放之前的输出
    o_scale = exp(m_old - m_new)
    output *= o_scale
    
    # 累加当前 block
    output += p @ V_block
    
    # 更新 LSE
    # l_i = exp(lse - m_new) + sum(p)
    l_i = exp(lse - m_new) + sum(p)
    lse = m_new + log(l_i)
    m_old = m_new

# 最终归一化
final_scale = exp(m_old - lse)  # = 1 / sum
output *= final_scale
```

#### 为什么 LSE 更稳定？

| 方法 | 存储 | 可能的数值范围 |
|-----|------|--------------|
| 直接存 sum | `sum_i` | 可能非常大（如 1e30） |
| 存 LSE | `m + log(sum)` | 通常在合理范围内 |

**关键**：`log` 压缩了数值范围，避免了大数的累积。

#### CUDA 实现

```cpp
// 初始化
float m_old = -INFINITY;
float lse = -INFINITY;

// 处理每个 block 后
float l_i = expf(lse - m_new) + block_sum;
lse = m_new + logf(l_i);
m_old = m_new;

// 最终归一化
float final_scale = expf(m_old - lse);
for (int d = tid; d < head_dim; d += blockDim.x) {
    output[d] = s_out[d] * final_scale;
}
```

### 8.4 Triton 优势

1. **开发效率**：代码简洁，类似 NumPy
2. **自动优化**：内存访问、向量化
3. **可移植性**：相同代码可跑 AMD/NVIDIA

### 8.5 CUDA 优势

1. **精细控制**：可以优化每个细节
2. **成熟生态**：更多库和工具
3. **性能上限**：手动优化可能更快

---

## 9. 完整对照：Triton → CUDA

### Triton 代码片段
```python
# Triton: 加载 Q
q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)
```

### 对应 CUDA
```cpp
// CUDA: 加载 Q 到 shared memory
for (int d = tid; d < head_dim; d += blockDim.x) {
    s_q[d] = (float)q_ptr[d];
}
__syncthreads();
```

---

### Triton 代码片段
```python
# Triton: 矩阵乘法
qk = tl.dot(q, tl.trans(k)) * scale
```

### 对应 CUDA
```cpp
// CUDA: 手动计算 dot product
float score = 0.0f;
for (int d = 0; d < head_dim; d++) {
    score += s_q[d] * (float)k_ptr[d];
}
score *= scale;
```

---

### Triton 代码片段
```python
# Triton: 找 max
new_max = tl.maximum(tl.max(qk, 1), old_max)
```

### 对应 CUDA
```cpp
// CUDA: warp shuffle reduction
float local_max = -INFINITY;
for (int i = tid; i < block_len; i += blockDim.x) {
    local_max = fmaxf(local_max, s_scores[i]);
}
// Warp reduction
for (int offset = 16; offset > 0; offset >>= 1) {
    local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
}
// Cross-warp reduction...
```

---

## 参考资料

1. [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
2. [FlashAttention-2 Paper](https://arxiv.org/abs/2307.08691)
3. [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
4. [Triton Documentation](https://triton-lang.org/)
5. [PyTorch scaled_dot_product_attention](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
