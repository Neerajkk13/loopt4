#include "common_sm75.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace loopt4 {

// ============================================================================
// 1. Fused QK L2 Clip + Value RMSNorm (no gamma) + Ring/Global KV Cache Write
//    Supports either host start_pos_arg or GPU pos_ptr (for CUDA Graph replay).
// ============================================================================
__global__ void qkv_norm_cache_write_kernel(
    half* __restrict__ q,
    const half* __restrict__ k,
    const half* __restrict__ v,
    half* __restrict__ k_cache,
    half* __restrict__ v_cache,
    int start_pos_arg,
    const int32_t* __restrict__ pos_ptr,
    int M, int max_slots,
    bool is_swa, float qk_clip_thresh, float rms_eps
) {
    const int start_pos = pos_ptr ? pos_ptr[0] : start_pos_arg;
    const int tok = blockIdx.x;
    const int blk = blockIdx.y;
    const int tid = threadIdx.x; // 128 threads -> 1 dim/thread
    __shared__ float s_red[4];

    if (blk < 24) {
        half* q_ptr = q + (tok * 24 + blk) * 128;
        float val = __half2float(q_ptr[tid]);
        float sq = block_reduce_sum<4>(val * val, s_red);
        float norm = sqrtf(sq + 1e-12f);
        if (norm > qk_clip_thresh) {
            val *= (qk_clip_thresh / norm);
            q_ptr[tid] = __float2half(val);
        }
    } else {
        const int kv_h = blk - 24;
        const int pos  = start_pos + tok;
        const int slot = is_swa ? (pos & (max_slots - 1)) : pos;

        const half* k_in = k + (tok * 8 + kv_h) * 128;
        const half* v_in = v + (tok * 8 + kv_h) * 128;

        float kv = __half2float(k_in[tid]);
        float k_sq = block_reduce_sum<4>(kv * kv, s_red);
        float k_norm = sqrtf(k_sq + 1e-12f);
        if (k_norm > qk_clip_thresh) kv *= (qk_clip_thresh / k_norm);
        k_cache[(size_t(slot) * 8 + kv_h) * 128 + tid] = __float2half(kv);

        float vv = __half2float(v_in[tid]);
        float v_sq = block_reduce_sum<4>(vv * vv, s_red);
        float v_inv_rms = rsqrtf(v_sq / 128.0f + rms_eps);
        v_cache[(size_t(slot) * 8 + kv_h) * 128 + tid] = __float2half(vv * v_inv_rms);
    }
}

// ============================================================================
// 2. Hybrid GQA Attention Kernel (No RoPE, Learned Relative Bias, Attention
//    Sink, Sigmoid Output Gate, Log Length Scaling, SWA-1024 or Full Global)
// ============================================================================
__global__ void hybrid_gqa_attn_sm75_kernel(
    const half* __restrict__ q,
    const half* __restrict__ gate,
    const half* __restrict__ k_cache,
    const half* __restrict__ v_cache,
    const half* __restrict__ rel_bias,
    const float* __restrict__ attn_sink,
    half* __restrict__ out,
    int start_pos_arg,
    const int32_t* __restrict__ pos_ptr,
    int M, int max_slots,
    bool is_swa, int window_size,
    float log_n_floor, float log_alpha
) {
    const int start_pos = pos_ptr ? pos_ptr[0] : start_pos_arg;
    const int q_head  = blockIdx.x;        // 0..23
    const int tok     = blockIdx.y;        // 0..M-1
    if (tok >= M) return;

    const int kv_head = q_head / 3;        // 3:1 GQA (24 Q / 8 KV)
    const int tid     = threadIdx.x;       // 0..127 -> 1 dim of head_dim=128
    __shared__ float s_red[4];

    const int cur_pos  = start_pos + tok;
    const int kv_begin = is_swa ? max(0, cur_pos - window_size + 1) : 0;
    const int kv_end   = cur_pos + 1;

    float scale = 1.0f / sqrtf(128.0f);
    if (!is_swa && float(cur_pos + 1) > log_n_floor) {
        scale *= (1.0f + log_alpha * logf(float(cur_pos + 1) / log_n_floor));
    }

    float q_val = __half2float(q[(tok * 24 + q_head) * 128 + tid]) * scale;

    float m_i   = attn_sink[q_head];
    float l_i   = 1.0f;
    float o_acc = 0.0f;

    const half* rel_ptr = rel_bias ? (rel_bias + q_head * 1024) : nullptr;

    for (int p = kv_begin; p < kv_end; ++p) {
        const int slot = is_swa ? (p & (max_slots - 1)) : p;
        const size_t kv_off = (size_t(slot) * 8 + kv_head) * 128 + tid;

        float k_val = __half2float(k_cache[kv_off]);
        float dot   = block_reduce_sum<4>(q_val * k_val, s_red);

        int dist = cur_pos - p;
        if (rel_ptr && dist < 1024) {
            dot += __half2float(rel_ptr[dist]);
        }

        float m_new = fmaxf(m_i, dot);
        float alpha = expf(m_i - m_new);
        float beta  = expf(dot - m_new);

        float v_val = __half2float(v_cache[kv_off]);
        o_acc = o_acc * alpha + beta * v_val;
        l_i   = l_i * alpha + beta;
        m_i   = m_new;
    }

    float o_norm = o_acc / fmaxf(l_i, 1e-12f);
    if (gate) {
        float g = sigmoid_f32(__half2float(gate[(tok * 24 + q_head) * 128 + tid]));
        o_norm *= g;
    }
    out[(tok * 24 + q_head) * 128 + tid] = __float2half(o_norm);
}

// ============================================================================
// C++ Launchers (Host start_pos + CUDA-Graph-safe GPU pos_buf)
// ============================================================================
static torch::Tensor _launch_hybrid_gqa_impl(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> gate,
    torch::Tensor k_cache, torch::Tensor v_cache, c10::optional<torch::Tensor> rel_bias,
    torch::Tensor attn_sink, int start_pos_arg, const int32_t* pos_ptr,
    bool is_swa, int64_t window_size, double qk_clip_thresh, double rms_eps,
    double log_n_floor, double log_alpha
) {
    c10::cuda::CUDAGuard guard(q.device());
    int M = q.size(0);
    int max_slots = k_cache.size(0);
    auto stream = at::cuda::getCurrentCUDAStream();

    qkv_norm_cache_write_kernel<<<dim3(M, 32), 128, 0, stream>>>(
        reinterpret_cast<half*>(q.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(k.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v.data_ptr<at::Half>()),
        reinterpret_cast<half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<half*>(v_cache.data_ptr<at::Half>()),
        start_pos_arg, pos_ptr, M, max_slots,
        is_swa, float(qk_clip_thresh), float(rms_eps)
    );

    auto out = torch::empty_like(q);
    hybrid_gqa_attn_sm75_kernel<<<dim3(24, M), 128, 0, stream>>>(
        reinterpret_cast<const half*>(q.data_ptr<at::Half>()),
        gate.has_value() ? reinterpret_cast<const half*>(gate->data_ptr<at::Half>()) : nullptr,
        reinterpret_cast<const half*>(k_cache.data_ptr<at::Half>()),
        reinterpret_cast<const half*>(v_cache.data_ptr<at::Half>()),
        rel_bias.has_value() ? reinterpret_cast<const half*>(rel_bias->data_ptr<at::Half>()) : nullptr,
        attn_sink.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        start_pos_arg, pos_ptr, M, max_slots,
        is_swa, int(window_size), float(log_n_floor), float(log_alpha)
    );
    return out;
}

torch::Tensor launch_hybrid_gqa_attn(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> gate,
    torch::Tensor k_cache, torch::Tensor v_cache, c10::optional<torch::Tensor> rel_bias,
    torch::Tensor attn_sink, int64_t start_pos, bool is_swa, int64_t window_size,
    double qk_clip_thresh, double rms_eps, double log_n_floor, double log_alpha
) {
    return _launch_hybrid_gqa_impl(
        q, k, v, gate, k_cache, v_cache, rel_bias, attn_sink,
        int(start_pos), nullptr, is_swa, window_size,
        qk_clip_thresh, rms_eps, log_n_floor, log_alpha
    );
}

torch::Tensor launch_hybrid_gqa_attn_pos_buf(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> gate,
    torch::Tensor k_cache, torch::Tensor v_cache, c10::optional<torch::Tensor> rel_bias,
    torch::Tensor attn_sink, torch::Tensor pos_buf, bool is_swa, int64_t window_size,
    double qk_clip_thresh, double rms_eps, double log_n_floor, double log_alpha
) {
    return _launch_hybrid_gqa_impl(
        q, k, v, gate, k_cache, v_cache, rel_bias, attn_sink,
        0, pos_buf.data_ptr<int32_t>(), is_swa, window_size,
        qk_clip_thresh, rms_eps, log_n_floor, log_alpha
    );
}

} // namespace loopt4
