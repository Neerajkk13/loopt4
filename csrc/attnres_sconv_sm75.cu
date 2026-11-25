#include "common_sm75.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace loopt4 {

// ============================================================================
// 1. FP32-Safe Fused Block Attention Residuals + Cross-Loop Message + RMSNorm
//    D = 2304 -> 256 threads * 9 elements/thread (exact division, zero tail)
// ============================================================================
__global__ void fused_attnres_msg_rmsnorm_kernel(
    half* __restrict__ out_normed,         // [M, 2304]
    half* __restrict__ out_mixed,          // [M, 2304] (optional, nullable)
    const half* __restrict__ block_sums,   // [num_blocks, M, 2304]
    const half* __restrict__ prev_msg,     // [M, 2304] (nullable on pass 1)
    const half* __restrict__ q_depth,      // [2304]
    const half* __restrict__ kappa,        // [2304] (nullable on pass 1)
    const half* __restrict__ gamma,        // [2304] (nullable on pass 1)
    const half* __restrict__ rms_weight,   // [2304]
    int M, int num_blocks, bool has_loop_msg, float eps
) {
    constexpr int D = 2304;
    const int tok = blockIdx.x;
    if (tok >= M) return;
    const int tid = threadIdx.x; // 0..255

    __shared__ float s_alpha[12];
    __shared__ float s_inv_rms[12];
    __shared__ float s_red[8];

    // Step 1: Depth-wise attention logits over stored block residual sums in FP32
    for (int b = 0; b < num_blocks; ++b) {
        const half* v_ptr = block_sums + (size_t(b) * M + tok) * D;
        float sq_sum = 0.0f;
        float dot_q  = 0.0f;
        #pragma unroll
        for (int k = 0; k < 9; ++k) {
            int i = tid + k * 256;
            float v = __half2float(v_ptr[i]);
            sq_sum += v * v;
            dot_q  += v * __half2float(q_depth[i]);
        }
        float tot_sq = block_reduce_sum<8>(sq_sum, s_red);
        float tot_dq = block_reduce_sum<8>(dot_q, s_red);
        if (tid == 0) {
            float inv_r = rsqrtf(tot_sq / float(D) + eps);
            s_inv_rms[b] = inv_r;
            s_alpha[b]   = (tot_dq * inv_r) * (1.0f / 48.0f); // sqrt(2304) == 48
        }
        __syncthreads();
    }

    // Softmax over active blocks
    if (tid == 0) {
        float mx = -1e30f;
        for (int b = 0; b < num_blocks; ++b) mx = fmaxf(mx, s_alpha[b]);
        float denom = 0.0f;
        for (int b = 0; b < num_blocks; ++b) {
            s_alpha[b] = expf(s_alpha[b] - mx);
            denom += s_alpha[b];
        }
        float inv_d = 1.0f / denom;
        for (int b = 0; b < num_blocks; ++b) s_alpha[b] *= inv_d;
    }
    __syncthreads();

    // Step 2: Weighted residual mix + cross-loop gate + final RMSNorm in FP32
    float reg_x[9];
    float sq_final = 0.0f;
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        int i = tid + k * 256;
        float acc = 0.0f;
        for (int b = 0; b < num_blocks; ++b) {
            float v = __half2float(block_sums[(size_t(b) * M + tok) * D + i]);
            acc += s_alpha[b] * (v * s_inv_rms[b]);
        }
        if (has_loop_msg) {
            float m = __half2float(prev_msg[tok * D + i]);
            acc = __half2float(kappa[i]) * acc + __half2float(gamma[i]) * m;
        }
        reg_x[k] = acc;
        if (out_mixed) out_mixed[tok * D + i] = __float2half(acc);
        sq_final += acc * acc;
    }

    float inv_final = rsqrtf(block_reduce_sum<8>(sq_final, s_red) / float(D) + eps);
    #pragma unroll
    for (int k = 0; k < 9; ++k) {
        int i = tid + k * 256;
        float y = reg_x[k] * inv_final * __half2float(rms_weight[i]);
        out_normed[tok * D + i] = __float2half(y);
    }
}

// ============================================================================
// 2. Causal 4-Tap Depthwise Short-Conv1D Step (handles M >= 1 tokens at once)
//    Updates circular 3-tap history state in-place.
//    x:      [M, C] FP16
//    weight: [C, 4] FP16  (tap 0 = oldest t-3 ... tap 3 = current t)
//    state:  [C, 3] FP16  (rolling history [t-3, t-2, t-1])
//    out:    [M, C] FP16
// ============================================================================
__global__ void short_conv4_step_kernel(
    const half* __restrict__ x,
    const half* __restrict__ weight,
    half* __restrict__ state,
    half* __restrict__ out,
    int M, int C, bool apply_silu
) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= C) return;

    float w0 = __half2float(weight[ch * 4 + 0]);
    float w1 = __half2float(weight[ch * 4 + 1]);
    float w2 = __half2float(weight[ch * 4 + 2]);
    float w3 = __half2float(weight[ch * 4 + 3]);

    float h0 = __half2float(state[ch * 3 + 0]); // t-3
    float h1 = __half2float(state[ch * 3 + 1]); // t-2
    float h2 = __half2float(state[ch * 3 + 2]); // t-1

    for (int t = 0; t < M; ++t) {
        float cur = __half2float(x[t * C + ch]);
        float y = w0 * h0 + w1 * h1 + w2 * h2 + w3 * cur;
        if (apply_silu) y = silu_f32(y);
        out[t * C + ch] = __float2half(y);
        h0 = h1;
        h1 = h2;
        h2 = cur;
    }

    state[ch * 3 + 0] = __float2half(h0);
    state[ch * 3 + 1] = __float2half(h1);
    state[ch * 3 + 2] = __float2half(h2);
}

// ============================================================================
// C++ Launchers
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor> launch_fused_attnres_msg_rmsnorm(
    torch::Tensor block_sums,
    c10::optional<torch::Tensor> prev_msg,
    torch::Tensor q_depth,
    c10::optional<torch::Tensor> kappa,
    c10::optional<torch::Tensor> gamma,
    torch::Tensor rms_weight,
    double eps
) {
    c10::cuda::CUDAGuard guard(block_sums.device());
    int num_blocks = block_sums.size(0);
    int M = block_sums.size(1);
    int D = block_sums.size(2);
    TORCH_CHECK(D == 2304, "fused_attnres_msg_rmsnorm expects D == 2304");

    auto out_normed = torch::empty({M, D}, block_sums.options());
    auto out_mixed  = torch::empty({M, D}, block_sums.options());

    bool has_msg = prev_msg.has_value() && kappa.has_value() && gamma.has_value();

    fused_attnres_msg_rmsnorm_kernel<<<M, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<half*>(out_normed.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out_mixed.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(block_sums.data_ptr<at::Half>()),
        has_msg ? reinterpret_cast<const half*>(prev_msg->data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<const half*>(q_depth.data_ptr<at::Half>()),
        has_msg ? reinterpret_cast<const half*>(kappa->data_ptr<at::Half>()) : nullptr,
        has_msg ? reinterpret_cast<const half*>(gamma->data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<const half*>(rms_weight.data_ptr<at::Half>()),
        M, num_blocks, has_msg, float(eps)
    );
    return {out_normed, out_mixed};
}

torch::Tensor launch_short_conv4_step(
    torch::Tensor x,
    torch::Tensor weight,
    torch::Tensor state,
    bool apply_silu
) {
    c10::cuda::CUDAGuard guard(x.device());
    int M = x.size(0);
    int C = x.size(1);
    auto out = torch::empty_like(x);
    int threads = 256;
    int blocks = (C + threads - 1) / threads;
    short_conv4_step_kernel<<<blocks, threads, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(weight.data_ptr<at::Half>()),
        reinterpret_cast<half*>(state.data_ptr<at::Half>()),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, C, apply_silu
    );
    return out;
}

} // namespace loopt4
