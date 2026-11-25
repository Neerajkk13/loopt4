#include "common_sm75.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace loopt4 {

// ============================================================================
// 1. Sigmoid Top-8 MoE Router (128 experts, learned gate bias, route_scale=8.0)
// ============================================================================
__global__ void moe_router_top8_sigmoid_kernel(
    const half* __restrict__ logits,       // [M, 128]
    const half* __restrict__ gate_bias,    // [128] (optional, nullable)
    int32_t* __restrict__ topk_ids,        // [M, 8]
    float* __restrict__ topk_weights,      // [M, 8]
    float route_scale,
    int M
) {
    const int tok = blockIdx.x;
    if (tok >= M) return;
    const int tid = threadIdx.x; // 128 threads -> 1 thread per expert

    __shared__ float s_score[128];
    __shared__ float s_prob[128];

    float raw = __half2float(logits[tok * 128 + tid]);
    float p = sigmoid_f32(raw);
    float b = gate_bias ? __half2float(gate_bias[tid]) : 0.0f;
    s_prob[tid]  = p;
    s_score[tid] = p + b;
    __syncthreads();

    // Thread 0 performs exact top-8 selection over 128 scores in registers (~100 ns)
    if (tid == 0) {
        int best_idx[8];
        float best_prob[8];
        float sum_p = 0.0f;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            float max_s = -1e30f;
            int max_i = 0;
            for (int e = 0; e < 128; ++e) {
                float sc = s_score[e];
                if (sc > max_s) {
                    max_s = sc;
                    max_i = e;
                }
            }
            best_idx[k] = max_i;
            best_prob[k] = s_prob[max_i];
            sum_p += best_prob[k];
            s_score[max_i] = -1e30f;
        }
        float inv_sum = (sum_p > 1e-12f) ? (route_scale / sum_p) : 0.0f;
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            topk_ids[tok * 8 + k]     = best_idx[k];
            topk_weights[tok * 8 + k] = best_prob[k] * inv_sum;
        }
    }
}

// ============================================================================
// 2. Register-Level NVFP4 W4A16 MoE Gate+Up + Softcap SwiGLU (Layers 0-19)
//    x:          [M, D] FP16 (D=2304, divisible by 16 & 32)
//    w_gate_up:  [128, 2*I, D/2] uint8 packed (2 E2M1 nibbles/byte)
//    s_gate_up:  [128, 2*I, D/16] uint8 FP8 E4M3FN block-16 scales
//    outer_s:    [128] FP32 outer tensor scale (per expert)
//    out_inter:  [M, 8, I] FP16 (I=1152)
// Grid: (I / 4, M * 8), Block: 128 threads (4 warps -> 1 warp per output row)
// ============================================================================
__global__ void nvfp4_moe_gate_up_swiglu_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ w_gate_up,
    const uint8_t* __restrict__ s_gate_up,
    const float* __restrict__ outer_s,
    const int32_t* __restrict__ topk_ids,
    half* __restrict__ out_inter,
    int M, int D, int I,
    float gate_softcap, float up_softcap
) {
    const int pair_idx = blockIdx.y; // 0 .. M*8 - 1
    const int tok = pair_idx >> 3;
    const int k_slot = pair_idx & 7;
    if (tok >= M) return;

    const int expert = topk_ids[pair_idx];
    const int warp_id = threadIdx.x >> 5;   // 0..3
    const int lane    = threadIdx.x & 31;   // 0..31
    const int row     = blockIdx.x * 4 + warp_id;
    if (row >= I) return;

    const float lane_lut = init_e2m1_lane_lut();
    const float expert_outer = outer_s[expert];

    const half* x_row = x + tok * D;
    const size_t expert_w_stride = size_t(2 * I) * size_t(D >> 1);
    const size_t expert_s_stride = size_t(2 * I) * size_t(D >> 4);

    const uint8_t* w_g_ptr = w_gate_up + expert * expert_w_stride + size_t(row) * (D >> 1);
    const uint8_t* w_u_ptr = w_gate_up + expert * expert_w_stride + size_t(I + row) * (D >> 1);
    const uint8_t* s_g_ptr = s_gate_up + expert * expert_s_stride + size_t(row) * (D >> 4);
    const uint8_t* s_u_ptr = s_gate_up + expert * expert_s_stride + size_t(I + row) * (D >> 4);

    float acc_g = 0.0f;
    float acc_u = 0.0f;

    // Each thread processes 16 elements (1 NVFP4 scale block = 8 bytes) per step.
    // 32 lanes * 16 = 512 elements per outer iteration. D=2304 -> 4 full + 1 partial (256).
    const int num_blocks16 = D >> 4; // 144 blocks of 16
    for (int b = lane; b < ((num_blocks16 + 31) & ~31); b += 32) {
        bool valid = (b < num_blocks16);
        uint32_t wg0 = 0, wg1 = 0, wu0 = 0, wu1 = 0;
        float sg = 0.0f, su = 0.0f;
        half2 xv[8];

        if (valid) {
            const uint32_t* wg32 = reinterpret_cast<const uint32_t*>(w_g_ptr + b * 8);
            const uint32_t* wu32 = reinterpret_cast<const uint32_t*>(w_u_ptr + b * 8);
            wg0 = wg32[0]; wg1 = wg32[1];
            wu0 = wu32[0]; wu1 = wu32[1];
            sg = e4m3fn_to_float(s_g_ptr[b]);
            su = e4m3fn_to_float(s_u_ptr[b]);
            const half2* x2 = reinterpret_cast<const half2*>(x_row + b * 16);
            #pragma unroll
            for (int i = 0; i < 8; ++i) xv[i] = x2[i];
        } else {
            #pragma unroll
            for (int i = 0; i < 8; ++i) xv[i] = __float2half2_rn(0.0f);
        }

        float bg = 0.0f;
        float bu = 0.0f;
        // Unpack all 16 nibbles uniformly across all 32 lanes (safe __shfl_sync)
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 xf = __half22float2(xv[i]);
            uint32_t g_byte = (wg0 >> (i * 8)) & 0xFFu;
            uint32_t u_byte = (wu0 >> (i * 8)) & 0xFFu;
            bg += e2m1_lut_lookup(g_byte & 0xFu, lane_lut) * xf.x
                + e2m1_lut_lookup(g_byte >> 4,   lane_lut) * xf.y;
            bu += e2m1_lut_lookup(u_byte & 0xFu, lane_lut) * xf.x
                + e2m1_lut_lookup(u_byte >> 4,   lane_lut) * xf.y;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            float2 xf = __half22float2(xv[4 + i]);
            uint32_t g_byte = (wg1 >> (i * 8)) & 0xFFu;
            uint32_t u_byte = (wu1 >> (i * 8)) & 0xFFu;
            bg += e2m1_lut_lookup(g_byte & 0xFu, lane_lut) * xf.x
                + e2m1_lut_lookup(g_byte >> 4,   lane_lut) * xf.y;
            bu += e2m1_lut_lookup(u_byte & 0xFu, lane_lut) * xf.x
                + e2m1_lut_lookup(u_byte >> 4,   lane_lut) * xf.y;
        }
        acc_g += bg * sg;
        acc_u += bu * su;
    }

    acc_g = warp_reduce_sum(acc_g) * expert_outer;
    acc_u = warp_reduce_sum(acc_u) * expert_outer;

    if (lane == 0) {
        float g_val = silu_f32(softcap_f32(acc_g, gate_softcap));
        float u_val = softcap_f32(acc_u, up_softcap);
        out_inter[(tok * 8 + k_slot) * I + row] = __float2half(g_val * u_val);
    }
}

// ============================================================================
// 3. Register-Level NVFP4 W4A16 MoE Down + Weighted Sum across Top-8 (Layers 0-19)
//    inter:      [M, 8, I] FP16 (I=1152)
//    w_down:     [128, D, I/2] uint8 packed
//    s_down:     [128, D, I/16] uint8 FP8 E4M3FN
//    outer_s:    [128] FP32
//    out:        [M, D] FP16
// Grid: (D / 4, M), Block: 128 threads (4 warps -> 1 warp per output dim)
// ============================================================================
__global__ void nvfp4_moe_down_reduce_kernel(
    const half* __restrict__ inter,
    const uint8_t* __restrict__ w_down,
    const uint8_t* __restrict__ s_down,
    const float* __restrict__ outer_s,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    half* __restrict__ out,
    int M, int D, int I
) {
    const int tok = blockIdx.y;
    if (tok >= M) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * 4 + warp_id;
    if (row >= D) return;

    const float lane_lut = init_e2m1_lane_lut();
    const size_t expert_w_stride = size_t(D) * size_t(I >> 1);
    const size_t expert_s_stride = size_t(D) * size_t(I >> 4);
    const int num_blocks16 = I >> 4; // 72 blocks of 16
    const int padded_blocks = (num_blocks16 + 31) & ~31; // 96

    float total_acc = 0.0f;

    // Loop over all 8 active experts for this token & accumulate in FP32
    for (int k = 0; k < 8; ++k) {
        const int expert = topk_ids[tok * 8 + k];
        const float weight = topk_weights[tok * 8 + k] * outer_s[expert];
        const half* x_ptr = inter + (tok * 8 + k) * I;
        const uint8_t* w_ptr = w_down + expert * expert_w_stride + size_t(row) * (I >> 1);
        const uint8_t* s_ptr = s_down + expert * expert_s_stride + size_t(row) * (I >> 4);

        float e_acc = 0.0f;
        for (int b = lane; b < padded_blocks; b += 32) {
            bool valid = (b < num_blocks16);
            uint32_t w0 = 0, w1 = 0;
            float sc = 0.0f;
            half2 xv[8];
            if (valid) {
                const uint32_t* w32 = reinterpret_cast<const uint32_t*>(w_ptr + b * 8);
                w0 = w32[0]; w1 = w32[1];
                sc = e4m3fn_to_float(s_ptr[b]);
                const half2* x2 = reinterpret_cast<const half2*>(x_ptr + b * 16);
                #pragma unroll
                for (int i = 0; i < 8; ++i) xv[i] = x2[i];
            } else {
                #pragma unroll
                for (int i = 0; i < 8; ++i) xv[i] = __float2half2_rn(0.0f);
            }

            float blk = 0.0f;
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 xf = __half22float2(xv[i]);
                uint32_t byt = (w0 >> (i * 8)) & 0xFFu;
                blk += e2m1_lut_lookup(byt & 0xFu, lane_lut) * xf.x
                     + e2m1_lut_lookup(byt >> 4,   lane_lut) * xf.y;
            }
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                float2 xf = __half22float2(xv[4 + i]);
                uint32_t byt = (w1 >> (i * 8)) & 0xFFu;
                blk += e2m1_lut_lookup(byt & 0xFu, lane_lut) * xf.x
                     + e2m1_lut_lookup(byt >> 4,   lane_lut) * xf.y;
            }
            e_acc += blk * sc;
        }
        total_acc += warp_reduce_sum(e_acc) * weight;
    }

    if (lane == 0) {
        out[tok * D + row] = __float2half(total_acc);
    }
}

// ============================================================================
// 4. Register-Level MXFP8 W8A16 MoE Gate+Up + Softcap SwiGLU (Layers 20-23)
//    w_gate_up: [128, 2*I, D] uint8 FP8 E4M3FN
//    s_gate_up: [128, 2*I, D/32] uint8 UE8M0 block-32 scales
// ============================================================================
__global__ void mxfp8_moe_gate_up_swiglu_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ w_gate_up,
    const uint8_t* __restrict__ s_gate_up,
    const int32_t* __restrict__ topk_ids,
    half* __restrict__ out_inter,
    int M, int D, int I,
    float gate_softcap, float up_softcap
) {
    const int pair_idx = blockIdx.y;
    const int tok = pair_idx >> 3;
    const int k_slot = pair_idx & 7;
    if (tok >= M) return;

    const int expert = topk_ids[pair_idx];
    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * 4 + warp_id;
    if (row >= I) return;

    const half* x_row = x + tok * D;
    const size_t expert_w_stride = size_t(2 * I) * size_t(D);
    const size_t expert_s_stride = size_t(2 * I) * size_t(D >> 5);

    const uint8_t* w_g_ptr = w_gate_up + expert * expert_w_stride + size_t(row) * D;
    const uint8_t* w_u_ptr = w_gate_up + expert * expert_w_stride + size_t(I + row) * D;
    const uint8_t* s_g_ptr = s_gate_up + expert * expert_s_stride + size_t(row) * (D >> 5);
    const uint8_t* s_u_ptr = s_gate_up + expert * expert_s_stride + size_t(I + row) * (D >> 5);

    const int num_blocks32 = D >> 5; // 72 blocks of 32
    float acc_g = 0.0f;
    float acc_u = 0.0f;

    // Each lane handles 1 element per 32-element block -> warp reads 1 full block coalesced
    for (int b = 0; b < num_blocks32; ++b) {
        float sg = ue8m0_to_float(s_g_ptr[b]);
        float su = ue8m0_to_float(s_u_ptr[b]);
        int col = (b << 5) + lane;
        float xv = __half2float(x_row[col]);
        acc_g += (e4m3fn_to_float(w_g_ptr[col]) * sg) * xv;
        acc_u += (e4m3fn_to_float(w_u_ptr[col]) * su) * xv;
    }

    acc_g = warp_reduce_sum(acc_g);
    acc_u = warp_reduce_sum(acc_u);

    if (lane == 0) {
        float g_val = silu_f32(softcap_f32(acc_g, gate_softcap));
        float u_val = softcap_f32(acc_u, up_softcap);
        out_inter[(tok * 8 + k_slot) * I + row] = __float2half(g_val * u_val);
    }
}

// ============================================================================
// 5. Register-Level MXFP8 W8A16 MoE Down + Weighted Sum (Layers 20-23)
// ============================================================================
__global__ void mxfp8_moe_down_reduce_kernel(
    const half* __restrict__ inter,
    const uint8_t* __restrict__ w_down,
    const uint8_t* __restrict__ s_down,
    const int32_t* __restrict__ topk_ids,
    const float* __restrict__ topk_weights,
    half* __restrict__ out,
    int M, int D, int I
) {
    const int tok = blockIdx.y;
    if (tok >= M) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * 4 + warp_id;
    if (row >= D) return;

    const size_t expert_w_stride = size_t(D) * size_t(I);
    const size_t expert_s_stride = size_t(D) * size_t(I >> 5);
    const int num_blocks32 = I >> 5; // 36 blocks of 32

    float total_acc = 0.0f;
    for (int k = 0; k < 8; ++k) {
        const int expert = topk_ids[tok * 8 + k];
        const float weight = topk_weights[tok * 8 + k];
        const half* x_ptr = inter + (tok * 8 + k) * I;
        const uint8_t* w_ptr = w_down + expert * expert_w_stride + size_t(row) * I;
        const uint8_t* s_ptr = s_down + expert * expert_s_stride + size_t(row) * (I >> 5);

        float e_acc = 0.0f;
        for (int b = 0; b < num_blocks32; ++b) {
            float sc = ue8m0_to_float(s_ptr[b]);
            int col = (b << 5) + lane;
            e_acc += (e4m3fn_to_float(w_ptr[col]) * sc) * __half2float(x_ptr[col]);
        }
        total_acc += warp_reduce_sum(e_acc) * weight;
    }

    if (lane == 0) {
        out[tok * D + row] = __float2half(total_acc);
    }
}

// ============================================================================
// C++ Launchers
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor> launch_moe_router_top8(
    torch::Tensor logits,
    c10::optional<torch::Tensor> gate_bias,
    double route_scale
) {
    c10::cuda::CUDAGuard guard(logits.device());
    int M = logits.size(0);
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(logits.device());
    auto opts_f = torch::TensorOptions().dtype(torch::kFloat32).device(logits.device());
    auto topk_ids = torch::empty({M, 8}, opts_i);
    auto topk_w   = torch::empty({M, 8}, opts_f);

    const half* bias_ptr = gate_bias.has_value()
        ? reinterpret_cast<const half*>(gate_bias->data_ptr<at::Half>())
        : nullptr;

    moe_router_top8_sigmoid_kernel<<<M, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(logits.data_ptr<at::Half>()),
        bias_ptr,
        topk_ids.data_ptr<int32_t>(),
        topk_w.data_ptr<float>(),
        float(route_scale),
        M
    );
    return {topk_ids, topk_w};
}

torch::Tensor launch_nvfp4_moe_forward(
    torch::Tensor x,            // [M, D] fp16
    torch::Tensor w_gate_up,    // [128, 2*I, D/2] uint8
    torch::Tensor s_gate_up,    // [128, 2*I, D/16] uint8
    torch::Tensor outer_gu,     // [128] fp32
    torch::Tensor w_down,       // [128, D, I/2] uint8
    torch::Tensor s_down,       // [128, D, I/16] uint8
    torch::Tensor outer_down,   // [128] fp32
    torch::Tensor topk_ids,     // [M, 8] int32
    torch::Tensor topk_weights, // [M, 8] fp32
    double gate_softcap,
    double up_softcap
) {
    c10::cuda::CUDAGuard guard(x.device());
    int M = x.size(0);
    int D = x.size(1);
    int I = w_gate_up.size(1) / 2;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto inter = torch::empty({M, 8, I}, x.options());
    auto out   = torch::empty({M, D}, x.options());

    dim3 grid_gu((I + 3) / 4, M * 8);
    nvfp4_moe_gate_up_swiglu_kernel<<<grid_gu, 128, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        w_gate_up.data_ptr<uint8_t>(),
        s_gate_up.data_ptr<uint8_t>(),
        outer_gu.data_ptr<float>(),
        topk_ids.data_ptr<int32_t>(),
        reinterpret_cast<half*>(inter.data_ptr<at::Half>()),
        M, D, I, float(gate_softcap), float(up_softcap)
    );

    dim3 grid_dn((D + 3) / 4, M);
    nvfp4_moe_down_reduce_kernel<<<grid_dn, 128, 0, stream>>>(
        reinterpret_cast<const half*>(inter.data_ptr<at::Half>()),
        w_down.data_ptr<uint8_t>(),
        s_down.data_ptr<uint8_t>(),
        outer_down.data_ptr<float>(),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, D, I
    );
    return out;
}

torch::Tensor launch_mxfp8_moe_forward(
    torch::Tensor x,
    torch::Tensor w_gate_up,
    torch::Tensor s_gate_up,
    torch::Tensor w_down,
    torch::Tensor s_down,
    torch::Tensor topk_ids,
    torch::Tensor topk_weights,
    double gate_softcap,
    double up_softcap
) {
    c10::cuda::CUDAGuard guard(x.device());
    int M = x.size(0);
    int D = x.size(1);
    int I = w_gate_up.size(1) / 2;

    auto stream = at::cuda::getCurrentCUDAStream();
    auto inter = torch::empty({M, 8, I}, x.options());
    auto out   = torch::empty({M, D}, x.options());

    dim3 grid_gu((I + 3) / 4, M * 8);
    mxfp8_moe_gate_up_swiglu_kernel<<<grid_gu, 128, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        w_gate_up.data_ptr<uint8_t>(),
        s_gate_up.data_ptr<uint8_t>(),
        topk_ids.data_ptr<int32_t>(),
        reinterpret_cast<half*>(inter.data_ptr<at::Half>()),
        M, D, I, float(gate_softcap), float(up_softcap)
    );

    dim3 grid_dn((D + 3) / 4, M);
    mxfp8_moe_down_reduce_kernel<<<grid_dn, 128, 0, stream>>>(
        reinterpret_cast<const half*>(inter.data_ptr<at::Half>()),
        w_down.data_ptr<uint8_t>(),
        s_down.data_ptr<uint8_t>(),
        topk_ids.data_ptr<int32_t>(),
        topk_weights.data_ptr<float>(),
        reinterpret_cast<half*>(out.data_ptr<at::Half>()),
        M, D, I
    );
    return out;
}

} // namespace loopt4
