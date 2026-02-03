#include <vector>
#include <cuda_fp16.h>
#include <cmath>
#include <limits>
#include <algorithm>

#include "../tester/utils.h"

// ============================================================================
// TRACE IMPLEMENTATION
// ============================================================================

/**
 * @brief CUDA kernel for computing partial sums of diagonal elements.
 * 
 * Each thread computes one diagonal element and adds it to a shared memory
 * reduction buffer. Block-level reduction is performed, then atomic add
 * to global result.
 */
template <typename T>
__global__ void traceKernel(const T* d_input, T* d_result, size_t rows, size_t cols, size_t diag_len) {
    // Shared memory for block-level reduction
    extern __shared__ char shared_mem[];
    T* sdata = reinterpret_cast<T*>(shared_mem);
    
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread loads one diagonal element (if within bounds)
    T val = T(0);
    if (idx < diag_len) {
        // Diagonal element at (idx, idx) in row-major layout: idx * cols + idx
        val = d_input[idx * cols + idx];
    }
    sdata[tid] = val;
    __syncthreads();
    
    // Block-level parallel reduction
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = sdata[tid] + sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Thread 0 writes block result to global memory
    if (tid == 0) {
        atomicAdd(d_result, sdata[0]);
    }
}

/**
 * @brief Computes the trace of a matrix.
 *
 * The trace of a matrix is defined as the sum of its diagonal elements.
 * This function expects a flattened row-major matrix stored in a
 * std::vector. If the matrix is not square, the trace will sum up
 * elements along the main diagonal up to the smaller of rows or cols.
 *
 * @tparam T The numeric type of matrix elements (e.g., float, int).
 * @param h_input A flattened matrix of size rows * cols.
 * @param rows Number of rows in the matrix.
 * @param cols Number of columns in the matrix.
 * @return The trace (sum of diagonal values) of the matrix.
 */
template <typename T>
T trace(const std::vector<T>& h_input, size_t rows, size_t cols) {
    // Handle edge cases
    if (rows == 0 || cols == 0 || h_input.empty()) {
        return T(0);
    }
    
    size_t diag_len = std::min(rows, cols);
    size_t input_size = rows * cols;
    
    // Allocate device memory
    T* d_input = nullptr;
    T* d_result = nullptr;
    RUNTIME_CHECK(cudaMalloc(&d_input, input_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_result, sizeof(T)));
    
    // Copy input to device and initialize result to 0
    RUNTIME_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemset(d_result, 0, sizeof(T)));
    
    // Launch kernel
    const int blockSize = 256;
    const int numBlocks = (diag_len + blockSize - 1) / blockSize;
    size_t sharedMemSize = blockSize * sizeof(T);
    
    traceKernel<T><<<numBlocks, blockSize, sharedMemSize>>>(d_input, d_result, rows, cols, diag_len);
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    T result;
    RUNTIME_CHECK(cudaMemcpy(&result, d_result, sizeof(T), cudaMemcpyDeviceToHost));
    
    // Free device memory
    RUNTIME_CHECK(cudaFree(d_input));
    RUNTIME_CHECK(cudaFree(d_result));
    
    return result;
}

// ============================================================================
// FLASH ATTENTION IMPLEMENTATION
// ============================================================================

/**
 * @brief Flash Attention kernel with tiling for memory efficiency.
 * 
 * Implements the Flash Attention algorithm with support for:
 * - Grouped Query Attention (GQA): query_heads can be multiple of kv_heads
 * - Causal masking: masks future positions
 * 
 * Each block handles one (batch, query_head, query_position) combination.
 */
template <typename T>
__global__ void flashAttentionKernel(
    const T* __restrict__ Q,      // [batch_size, tgt_seq_len, query_heads, head_dim]
    const T* __restrict__ K,      // [batch_size, src_seq_len, kv_heads, head_dim]
    const T* __restrict__ V,      // [batch_size, src_seq_len, kv_heads, head_dim]
    T* __restrict__ O,            // [batch_size, tgt_seq_len, query_heads, head_dim]
    int batch_size,
    int tgt_seq_len,
    int src_seq_len,
    int query_heads,
    int kv_heads,
    int head_dim,
    float scale,
    bool is_causal
) {
    // Each block handles one (batch, head, tgt_pos) combination
    int batch_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int tgt_pos = blockIdx.z;
    
    if (batch_idx >= batch_size || head_idx >= query_heads || tgt_pos >= tgt_seq_len) {
        return;
    }
    
    // GQA: map query head to corresponding KV head
    // Each KV head serves (query_heads / kv_heads) query heads
    int kv_head_idx = head_idx * kv_heads / query_heads;
    
    int tid = threadIdx.x;
    
    // Pointers to current Q row
    // Q layout: [batch_size, tgt_seq_len, query_heads, head_dim]
    const T* q_ptr = Q + batch_idx * tgt_seq_len * query_heads * head_dim
                       + tgt_pos * query_heads * head_dim
                       + head_idx * head_dim;
    
    // K, V layout: [batch_size, src_seq_len, kv_heads, head_dim]
    const T* k_base = K + batch_idx * src_seq_len * kv_heads * head_dim
                        + kv_head_idx * head_dim;
    const T* v_base = V + batch_idx * src_seq_len * kv_heads * head_dim
                        + kv_head_idx * head_dim;
    
    // Output pointer
    T* o_ptr = O + batch_idx * tgt_seq_len * query_heads * head_dim
                 + tgt_pos * query_heads * head_dim
                 + head_idx * head_dim;
    
    // Shared memory layout for Online Softmax:
    // s_q: [head_dim] - Q vector
    // s_scores: [BLOCK_SIZE] - scores for current block (64 positions max)
    // s_out: [head_dim] - output accumulator
    const int BLOCK_SIZE = 64;
    
    extern __shared__ char smem[];
    float* s_q = reinterpret_cast<float*>(smem);        // [head_dim]
    float* s_scores = s_q + head_dim;                   // [BLOCK_SIZE]
    float* s_out = s_scores + BLOCK_SIZE;               // [head_dim]
    
    // Load Q vector into shared memory
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_q[d] = static_cast<float>(q_ptr[d]);
    }
    __syncthreads();
    
    // ========================================================================
    // Online Softmax: Process sequence in blocks, update running max/sum/output
    // Based on FlashAttention algorithm
    // ========================================================================
    
    // Initialize output accumulator
    for (int d = tid; d < head_dim; d += blockDim.x) {
        s_out[d] = 0.0f;
    }
    __syncthreads();
    
    // Running statistics (shared across block)
    // FlashAttention stores sum directly during loop, computes LSE at the end
    __shared__ float s_max_old, s_max_new, s_running_sum, s_o_scale;
    
    // Initialize
    if (tid == 0) {
        s_max_old = -INFINITY;
        s_running_sum = 0.0f;
    }
    __syncthreads();
    
    for (int block_start = 0; block_start < src_seq_len; block_start += BLOCK_SIZE) {
        int block_end = min(block_start + BLOCK_SIZE, src_seq_len);
        
        // Step 1: Compute attention scores for this block
        for (int src_pos = block_start + tid; src_pos < block_end; src_pos += blockDim.x) {
            if (is_causal && src_pos > tgt_pos) {
                s_scores[src_pos - block_start] = -INFINITY;
            } else {
                const T* k_ptr = k_base + src_pos * kv_heads * head_dim;
                float score = 0.0f;
                for (int d = 0; d < head_dim; d++) {
                    score += s_q[d] * static_cast<float>(k_ptr[d]);
                }
                s_scores[src_pos - block_start] = score * scale;
            }
        }
        __syncthreads();
        
        // Step 2: Find max in this block (parallel reduction)
        float local_max = -INFINITY;
        for (int i = tid; i < block_end - block_start; i += blockDim.x) {
            local_max = fmaxf(local_max, s_scores[i]);
        }
        // Warp-level reduction using shuffle
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, offset));
        }
        // Cross-warp reduction via shared memory
        __shared__ float warp_max[4];
        int warp_id = tid / 32;
        int lane_id = tid % 32;
        // Initialize warp_max to -INFINITY to handle inactive warps
        if (tid < 4) {
            warp_max[tid] = -INFINITY;
        }
        __syncthreads();
        if (lane_id == 0) {
            warp_max[warp_id] = local_max;
        }
        __syncthreads();
        if (tid == 0) {
            float block_max = warp_max[0];
            for (int i = 1; i < 4; i++) {
                block_max = fmaxf(block_max, warp_max[i]);
            }
            s_max_new = fmaxf(s_max_old, block_max);
        }
        __syncthreads();
        
        // Step 3: Compute exp(score - new_max)
        // Handle -INFINITY case to avoid NaN (like FlashAttention)
        
        // Compute o_scale in thread 0 and share via shared memory
        if (tid == 0) {
            // If both are -inf, scale should be 1 (nothing to scale)
            // If old is -inf and new is not, scale should be 0 (no previous contribution)
            if (s_max_old == -INFINITY) {
                s_o_scale = 0.0f;  // No previous contribution
            } else if (s_max_new == -INFINITY) {
                s_o_scale = 1.0f;  // Keep previous (shouldn't happen)
            } else {
                s_o_scale = expf(s_max_old - s_max_new);
            }
        }
        __syncthreads();
        
        for (int i = tid; i < block_end - block_start; i += blockDim.x) {
            float score = s_scores[i];
            // Handle -inf scores (from causal mask)
            if (score == -INFINITY) {
                s_scores[i] = 0.0f;
            } else {
                s_scores[i] = expf(score - s_max_new);
            }
        }
        __syncthreads();
        
        // Step 4: Compute sum of this block (parallel reduction)
        float local_sum = 0.0f;
        for (int i = tid; i < block_end - block_start; i += blockDim.x) {
            local_sum += s_scores[i];
        }
        // Warp-level reduction
        for (int offset = 16; offset > 0; offset >>= 1) {
            local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
        }
        // Cross-warp reduction
        __shared__ float warp_sum[4];
        // Initialize warp_sum to 0 to handle inactive warps
        if (tid < 4) {
            warp_sum[tid] = 0.0f;
        }
        __syncthreads();
        if (lane_id == 0) {
            warp_sum[warp_id] = local_sum;
        }
        __syncthreads();
        
        // Step 5: Scale output and update running sum (like FlashAttention)
        if (tid == 0) {
            float block_sum = 0.0f;
            for (int i = 0; i < 4; i++) {
                block_sum += warp_sum[i];
            }
            
            // Update running sum: sum_new = sum_old * o_scale + block_sum
            s_running_sum = s_running_sum * s_o_scale + block_sum;
            s_max_old = s_max_new;
        }
        __syncthreads();
        
        // Scale previous output and add current block's contribution
        for (int d = tid; d < head_dim; d += blockDim.x) {
            // Scale previous output
            s_out[d] *= s_o_scale;
            
            // Add current block's contribution
            float contrib = 0.0f;
            for (int i = 0; i < block_end - block_start; i++) {
                int src_pos = block_start + i;
                const T* v_ptr = v_base + src_pos * kv_heads * head_dim;
                contrib += s_scores[i] * static_cast<float>(v_ptr[d]);
            }
            s_out[d] += contrib;
        }
        __syncthreads();
    }
    
    // Final normalization: output = unnormalized_output / sum
    // Handle sum == 0 or NaN (like FlashAttention)
    float inv_sum = (s_running_sum == 0.0f || s_running_sum != s_running_sum) 
                    ? 1.0f : 1.0f / s_running_sum;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        o_ptr[d] = static_cast<T>(s_out[d] * inv_sum);
    }
}

/**
 * @brief Computes flash attention for given query, key, and value tensors.
 * 
 * @tparam T Data type (float) for input/output tensors
 * @param[in] h_q Query tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] h_k Key tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[in] h_v Value tensor of shape [batch_size, src_seq_len, kv_heads, head_dim]
 * @param[out] h_o Output attention tensor of shape [batch_size, tgt_seq_len, query_heads, head_dim]
 * @param[in] batch_size Batch dimension size
 * @param[in] target_seq_len Target sequence length
 * @param[in] src_seq_len Source sequence length  
 * @param[in] query_heads Number of query attention heads
 * @param[in] kv_heads Number of key/value heads (supports grouped query attention)
 * @param[in] head_dim Dimension size of each attention head
 * @param[in] is_causal Whether to apply causal masking
 */
template <typename T>
void flashAttention(const std::vector<T>& h_q, const std::vector<T>& h_k,
                    const std::vector<T>& h_v, std::vector<T>& h_o,
                    int batch_size, int target_seq_len, int src_seq_len, 
                    int query_heads, int kv_heads, int head_dim, bool is_causal) {
    // Calculate tensor sizes
    size_t q_size = batch_size * target_seq_len * query_heads * head_dim;
    size_t kv_size = batch_size * src_seq_len * kv_heads * head_dim;
    size_t o_size = q_size;
    
    // Ensure output vector has correct size
    h_o.resize(o_size);
    
    // Allocate device memory
    T *d_q, *d_k, *d_v, *d_o;
    RUNTIME_CHECK(cudaMalloc(&d_q, q_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_k, kv_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_v, kv_size * sizeof(T)));
    RUNTIME_CHECK(cudaMalloc(&d_o, o_size * sizeof(T)));
    
    // Copy inputs to device
    RUNTIME_CHECK(cudaMemcpy(d_q, h_q.data(), q_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_k, h_k.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
    RUNTIME_CHECK(cudaMemcpy(d_v, h_v.data(), kv_size * sizeof(T), cudaMemcpyHostToDevice));
    
    // Compute scale factor: 1 / sqrt(head_dim)
    float scale = 1.0f / sqrtf(static_cast<float>(head_dim));
    
    // Launch kernel
    // Grid: (batch_size, query_heads, target_seq_len)
    // Each block handles one output row
    dim3 grid(batch_size, query_heads, target_seq_len);
    int blockSize = 128;  // Threads per block
    
    // Shared memory: Q vector + scores block + output accumulator
    // Online Softmax only needs BLOCK_SIZE (64) for scores, not full src_seq_len
    const int BLOCK_SIZE = 64;
    size_t sharedMemSize = (head_dim + BLOCK_SIZE + head_dim) * sizeof(float);
    
    flashAttentionKernel<T><<<grid, blockSize, sharedMemSize>>>(
        d_q, d_k, d_v, d_o,
        batch_size, target_seq_len, src_seq_len,
        query_heads, kv_heads, head_dim,
        scale, is_causal
    );
    RUNTIME_CHECK(cudaGetLastError());
    RUNTIME_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    RUNTIME_CHECK(cudaMemcpy(h_o.data(), d_o, o_size * sizeof(T), cudaMemcpyDeviceToHost));
    
    // Free device memory
    RUNTIME_CHECK(cudaFree(d_q));
    RUNTIME_CHECK(cudaFree(d_k));
    RUNTIME_CHECK(cudaFree(d_v));
    RUNTIME_CHECK(cudaFree(d_o));
}

// *********************************************************************
// Explicit Template Instantiations (REQUIRED FOR LINKING WITH TESTER.O)
// DO NOT MODIFY THIS SECTION
// *********************************************************************
template int trace<int>(const std::vector<int>&, size_t, size_t);
template float trace<float>(const std::vector<float>&, size_t, size_t);
template void flashAttention<float>(const std::vector<float>&, const std::vector<float>&,
  const std::vector<float>&, std::vector<float>&,
  int, int, int, int, int, int, bool);
template void flashAttention<half>(const std::vector<half>&, const std::vector<half>&,
  const std::vector<half>&, std::vector<half>&,
  int, int, int, int, int, int, bool);
