#include "common_sm75.cuh"
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

namespace loopt4 {

// ============================================================================
// 1. Sigmoid Top-8 Router (unchanged interface)
// ============================================================================
__global__ void moe_router_top8_sigmoid_kernel(
    const half* __restrict__ logits,
    const half* __restrict__ gate_bias,
    int32_t* __restrict__ topk_ids,
    float* __restrict__ topk_weights,
    float route_scale,
    int M
) {
    const int tok = blockIdx.x;
    if (tok >= M) return;
    const int tid = threadIdx.x;

    __shared__ float s_score[128];
    __shared__ float s_prob[128];

    float p = sigmoid_f32(__half2float(logits[tok * 128 + tid]));
    float b = gate_bias ? __half2float(gate_bias[tid]) : 0.0f;
    s_prob[tid]  = p;
    s_score[tid] = p + b;
    __syncthreads();

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
                if (sc > max_s) { max_s = sc; max_i = e; }
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
// 2. Zero-Sync GPU Expert Grouping (1 block, 128 threads, ~1.5 us, Graph-safe)
//    Groups all M*8 (tok, k_slot) pairs by unique active expert so weight
//    matrices are read ONCE per unique expert during M>1 verification/prefill.
// ============================================================================
__global__ void build_expert_groups_kernel(
    const int32_t* __restrict__ topk_ids, // [M * 8]
    int32_t* __restrict__ unique_experts, // [M * 8]
    int32_t* __restrict__ expert_offsets, // [M * 8 + 1]
    int32_t* __restrict__ pair_indices,   // [M * 8]
    int32_t* __restrict__ num_unique,     // [1]
    int total_pairs                       // M * 8 (<= 2048 for chunk=256)
) {
    const int tid = threadIdx.x; // 0..127 -> expert ID
    __shared__ int32_t s_topk[2048];
    __shared__ int32_t s_counts[128];
    __shared__ int32_t s_offsets[128];

    for (int i = tid; i < total_pairs; i += 128) {
        s_topk[i] = topk_ids[i];
    }
    __syncthreads();

    int cnt = 0;
    for (int i = 0; i < total_pairs; ++i) {
        if (s_topk[i] == tid) cnt++;
    }
    s_counts[tid] = cnt;
    __syncthreads();

    if (tid == 0) {
        int u = 0, off = 0;
        for (int e = 0; e < 128; ++e) {
            int c = s_counts[e];
            if (c > 0) {
                unique_experts[u] = e;
                expert_offsets[u] = off;
                s_offsets[e] = off;
                off += c;
                u++;
            }
        }
        expert_offsets[u] = off;
        num_unique[0] = u;
    }
    __syncthreads();

    if (cnt > 0) {
        int write_pos = s_offsets[tid];
        for (int i = 0; i < total_pairs; ++i) {
            if (s_topk[i] == tid) {
                pair_indices[write_pos++] = i;
            }
        }
    }
}

// ============================================================================
// 3. Vectorized Expert-Grouped NVFP4 Gate+Up + Softcap SwiGLU (Layers 0-19)
//    - Pure-ALU E2M1 decode (no __shfl_sync, no padded_blocks)
//    - uint2 (64-bit) weight loads + float4 (128-bit) activation loads
//    - Reads each expert weight ONCE across up to 8 tokens simultaneously
// ============================================================================
__global__ void nvfp4_moe_gate_up_grouped_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ w_gate_up,
    const uint8_t* __restrict__ s_gate_up,
    const float* __restrict__ outer_s,
    const int32_t* __restrict__ unique_experts,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ pair_indices,
    const int32_t* __restrict__ num_unique,
    half* __restrict__ out_inter,
    int D, int I,
    float gate_softcap, float up_softcap
) {
    const int u_idx = blockIdx.y;
    if (u_idx >= num_unique[0]) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * 4 + warp_id;
    if (row >= I) return;

    const int expert  = unique_experts[u_idx];
    const int p_start = expert_offsets[u_idx];
    const int p_end   = expert_offsets[u_idx + 1];
    const float expert_outer = outer_s[expert];

    const size_t expert_w_stride = size_t(2 * I) * size_t(D >> 1);
    const size_t expert_s_stride = size_t(2 * I) * size_t(D >> 4);
    const uint8_t* w_g_ptr = w_gate_up + expert * expert_w_stride + size_t(row) * (D >> 1);
    const uint8_t* w_u_ptr = w_gate_up + expert * expert_w_stride + size_t(I + row) * (D >> 1);
    const uint8_t* s_g_ptr = s_gate_up + expert * expert_s_stride + size_t(row) * (D >> 4);
    const uint8_t* s_u_ptr = s_gate_up + expert * expert_s_stride + size_t(I + row) * (D >> 4);
    const int num_blocks16 = D >> 4; // 144

    // Process tokens assigned to this expert in tiles of up to 8
    for (int base = p_start; base < p_end; base += 8) {
        const int count = min(8, p_end - base);
        int pairs[8];
        int toks[8];
        float acc_g[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float acc_u[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < count) {
                pairs[c] = pair_indices[base + c];
                toks[c]  = pairs[c] >> 3;
            }
        }

        // Exact loop without padding: b = lane, lane+32, ... < 144
        for (int b = lane; b < num_blocks16; b += 32) {
            // Single 64-bit vectorized load per matrix
            uint2 wg = *reinterpret_cast<const uint2*>(w_g_ptr + b * 8);
            uint2 wu = *reinterpret_cast<const uint2*>(w_u_ptr + b * 8);
            float sg = e4m3fn_to_float(s_g_ptr[b]);
            float su = e4m3fn_to_float(s_u_ptr[b]);

            // Decode 16 weights ONCE in registers (reused across all `count` tokens!)
            float fg[16], fu[16];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t bg0 = (wg.x >> (i * 8)) & 0xFFu;
                uint32_t bg1 = (wg.y >> (i * 8)) & 0xFFu;
                uint32_t bu0 = (wu.x >> (i * 8)) & 0xFFu;
                uint32_t bu1 = (wu.y >> (i * 8)) & 0xFFu;
                fg[2 * i]     = e2m1_to_float_alu(bg0 & 0xFu) * sg;
                fg[2 * i + 1] = e2m1_to_float_alu(bg0 >> 4)   * sg;
                fg[8 + 2 * i] = e2m1_to_float_alu(bg1 & 0xFu) * sg;
                fg[9 + 2 * i] = e2m1_to_float_alu(bg1 >> 4)   * sg;
                fu[2 * i]     = e2m1_to_float_alu(bu0 & 0xFu) * su;
                fu[2 * i + 1] = e2m1_to_float_alu(bu0 >> 4)   * su;
                fu[8 + 2 * i] = e2m1_to_float_alu(bu1 & 0xFu) * su;
                fu[9 + 2 * i] = e2m1_to_float_alu(bu1 >> 4)   * su;
            }

            // Two 128-bit activation loads per token (L1/L2 resident)
            for (int c = 0; c < count; ++c) {
                const half* xp = x + size_t(toks[c]) * D + b * 16;
                union { float4 f4[2]; half2 h2[8]; } xb;
                xb.f4[0] = *reinterpret_cast<const float4*>(xp);
                xb.f4[1] = *reinterpret_cast<const float4*>(xp + 8);
                float dg = 0.0f, du = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float2 xf = __half22float2(xb.h2[i]);
                    dg += fg[2 * i] * xf.x + fg[2 * i + 1] * xf.y;
                    du += fu[2 * i] * xf.x + fu[2 * i + 1] * xf.y;
                }
                acc_g[c] += dg;
                acc_u[c] += du;
            }
        }

        for (int c = 0; c < count; ++c) {
            float g_tot = warp_reduce_sum(acc_g[c]) * expert_outer;
            float u_tot = warp_reduce_sum(acc_u[c]) * expert_outer;
            if (lane == 0) {
                float g_val = silu_f32(softcap_f32(g_tot, gate_softcap));
                float u_val = softcap_f32(u_tot, up_softcap);
                out_inter[size_t(pairs[c]) * I + row] = __float2half(g_val * u_val);
            }
        }
    }
}

// ============================================================================
// 4. Vectorized Expert-Grouped NVFP4 Down + FP32 Atomic Reduce (Layers 0-19)
//    - 8x higher SM parallelism at M=1 (grid_y = num_unique instead of 1)
//    - Reads each expert w_down ONCE across all tokens hitting it at M>1
// ============================================================================
__global__ void nvfp4_moe_down_grouped_kernel(
    const half* __restrict__ inter,
    const uint8_t* __restrict__ w_down,
    const uint8_t* __restrict__ s_down,
    const float* __restrict__ outer_s,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ unique_experts,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ pair_indices,
    const int32_t* __restrict__ num_unique,
    float* __restrict__ out_fp32,
    int D, int I
) {
    const int u_idx = blockIdx.y;
    if (u_idx >= num_unique[0]) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * 4 + warp_id;
    if (row >= D) return;

    const int expert  = unique_experts[u_idx];
    const int p_start = expert_offsets[u_idx];
    const int p_end   = expert_offsets[u_idx + 1];
    const float expert_outer = outer_s[expert];

    const size_t expert_w_stride = size_t(D) * size_t(I >> 1);
    const size_t expert_s_stride = size_t(D) * size_t(I >> 4);
    const uint8_t* w_ptr = w_down + expert * expert_w_stride + size_t(row) * (I >> 1);
    const uint8_t* s_ptr = s_down + expert * expert_s_stride + size_t(row) * (I >> 4);
    const int num_blocks16 = I >> 4; // 72 (no wasted padding!)

    for (int base = p_start; base < p_end; base += 8) {
        const int count = min(8, p_end - base);
        int pairs[8], toks[8];
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};

        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < count) {
                pairs[c] = pair_indices[base + c];
                toks[c]  = pairs[c] >> 3;
            }
        }

        for (int b = lane; b < num_blocks16; b += 32) {
            uint2 wd = *reinterpret_cast<const uint2*>(w_ptr + b * 8);
            float sc = e4m3fn_to_float(s_ptr[b]);
            float fw[16];
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                uint32_t b0 = (wd.x >> (i * 8)) & 0xFFu;
                uint32_t b1 = (wd.y >> (i * 8)) & 0xFFu;
                fw[2 * i]     = e2m1_to_float_alu(b0 & 0xFu) * sc;
                fw[2 * i + 1] = e2m1_to_float_alu(b0 >> 4)   * sc;
                fw[8 + 2 * i] = e2m1_to_float_alu(b1 & 0xFu) * sc;
                fw[9 + 2 * i] = e2m1_to_float_alu(b1 >> 4)   * sc;
            }

            for (int c = 0; c < count; ++c) {
                const half* xp = inter + size_t(pairs[c]) * I + b * 16;
                union { float4 f4[2]; half2 h2[8]; } xb;
                xb.f4[0] = *reinterpret_cast<const float4*>(xp);
                xb.f4[1] = *reinterpret_cast<const float4*>(xp + 8);
                float d = 0.0f;
                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    float2 xf = __half22float2(xb.h2[i]);
                    d += fw[2 * i] * xf.x + fw[2 * i + 1] * xf.y;
                }
                acc[c] += d;
            }
        }

        for (int c = 0; c < count; ++c) {
            float sum = warp_reduce_sum(acc[c]) * (topk_weights[pairs[c]] * expert_outer);
            if (lane == 0) {
                atomicAdd(&out_fp32[size_t(toks[c]) * D + row], sum);
            }
        }
    }
}

// ============================================================================
// 5. Expert-Grouped MXFP8 Gate+Up & Down Kernels (Layers 20-23)
// ============================================================================
__global__ void mxfp8_moe_gate_up_grouped_kernel(
    const half* __restrict__ x,
    const uint8_t* __restrict__ w_gate_up,
    const uint8_t* __restrict__ s_gate_up,
    const int32_t* __restrict__ unique_experts,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ pair_indices,
    const int32_t* __restrict__ num_unique,
    half* __restrict__ out_inter,
    int D, int I,
    float gate_softcap, float up_softcap
) {
    const int u_idx = blockIdx.y;
    if (u_idx >= num_unique[0]) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * 4 + warp_id;
    if (row >= I) return;

    const int expert  = unique_experts[u_idx];
    const int p_start = expert_offsets[u_idx];
    const int p_end   = expert_offsets[u_idx + 1];

    const size_t expert_w_stride = size_t(2 * I) * size_t(D);
    const size_t expert_s_stride = size_t(2 * I) * size_t(D >> 5);
    const uint8_t* w_g_ptr = w_gate_up + expert * expert_w_stride + size_t(row) * D;
    const uint8_t* w_u_ptr = w_gate_up + expert * expert_w_stride + size_t(I + row) * D;
    const uint8_t* s_g_ptr = s_gate_up + expert * expert_s_stride + size_t(row) * (D >> 5);
    const uint8_t* s_u_ptr = s_gate_up + expert * expert_s_stride + size_t(I + row) * (D >> 5);
    const int num_blocks32 = D >> 5; // 72

    for (int base = p_start; base < p_end; base += 8) {
        const int count = min(8, p_end - base);
        int pairs[8], toks[8];
        float acc_g[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        float acc_u[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < count) { pairs[c] = pair_indices[base + c]; toks[c] = pairs[c] >> 3; }
        }

        for (int b = 0; b < num_blocks32; ++b) {
            int col = (b << 5) + lane;
            float wg_f = e4m3fn_to_float(w_g_ptr[col]) * ue8m0_to_float(s_g_ptr[b]);
            float wu_f = e4m3fn_to_float(w_u_ptr[col]) * ue8m0_to_float(s_u_ptr[b]);
            for (int c = 0; c < count; ++c) {
                float xv = __half2float(x[size_t(toks[c]) * D + col]);
                acc_g[c] += wg_f * xv;
                acc_u[c] += wu_f * xv;
            }
        }
        for (int c = 0; c < count; ++c) {
            float g_tot = warp_reduce_sum(acc_g[c]);
            float u_tot = warp_reduce_sum(acc_u[c]);
            if (lane == 0) {
                out_inter[size_t(pairs[c]) * I + row] = __float2half(
                    silu_f32(softcap_f32(g_tot, gate_softcap)) * softcap_f32(u_tot, up_softcap)
                );
            }
        }
    }
}

__global__ void mxfp8_moe_down_grouped_kernel(
    const half* __restrict__ inter,
    const uint8_t* __restrict__ w_down,
    const uint8_t* __restrict__ s_down,
    const float* __restrict__ topk_weights,
    const int32_t* __restrict__ unique_experts,
    const int32_t* __restrict__ expert_offsets,
    const int32_t* __restrict__ pair_indices,
    const int32_t* __restrict__ num_unique,
    float* __restrict__ out_fp32,
    int D, int I
) {
    const int u_idx = blockIdx.y;
    if (u_idx >= num_unique[0]) return;

    const int warp_id = threadIdx.x >> 5;
    const int lane    = threadIdx.x & 31;
    const int row     = blockIdx.x * 4 + warp_id;
    if (row >= D) return;

    const int expert  = unique_experts[u_idx];
    const int p_start = expert_offsets[u_idx];
    const int p_end   = expert_offsets[u_idx + 1];

    const size_t expert_w_stride = size_t(D) * size_t(I);
    const size_t expert_s_stride = size_t(D) * size_t(I >> 5);
    const uint8_t* w_ptr = w_down + expert * expert_w_stride + size_t(row) * I;
    const uint8_t* s_ptr = s_down + expert * expert_s_stride + size_t(row) * (I >> 5);
    const int num_blocks32 = I >> 5; // 36

    for (int base = p_start; base < p_end; base += 8) {
        const int count = min(8, p_end - base);
        int pairs[8], toks[8];
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            if (c < count) { pairs[c] = pair_indices[base + c]; toks[c] = pairs[c] >> 3; }
        }
        for (int b = 0; b < num_blocks32; ++b) {
            int col = (b << 5) + lane;
            float wf = e4m3fn_to_float(w_ptr[col]) * ue8m0_to_float(s_ptr[b]);
            for (int c = 0; c < count; ++c) {
                acc[c] += wf * __half2float(inter[size_t(pairs[c]) * I + col]);
            }
        }
        for (int c = 0; c < count; ++c) {
            float sum = warp_reduce_sum(acc[c]) * topk_weights[pairs[c]];
            if (lane == 0) {
                atomicAdd(&out_fp32[size_t(toks[c]) * D + row], sum);
            }
        }
    }
}

// ============================================================================
// C++ Launchers (identical Python signature — drop-in replacement!)
// ============================================================================
std::tuple<torch::Tensor, torch::Tensor> launch_moe_router_top8(
    torch::Tensor logits, c10::optional<torch::Tensor> gate_bias, double route_scale
) {
    c10::cuda::CUDAGuard guard(logits.device());
    int M = logits.size(0);
    auto topk_ids = torch::empty({M, 8}, torch::TensorOptions().dtype(torch::kInt32).device(logits.device()));
    auto topk_w   = torch::empty({M, 8}, torch::TensorOptions().dtype(torch::kFloat32).device(logits.device()));
    const half* bias_ptr = gate_bias.has_value()
        ? reinterpret_cast<const half*>(gate_bias->data_ptr<at::Half>()) : nullptr;

    moe_router_top8_sigmoid_kernel<<<M, 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const half*>(logits.data_ptr<at::Half>()), bias_ptr,
        topk_ids.data_ptr<int32_t>(), topk_w.data_ptr<float>(), float(route_scale), M
    );
    return {topk_ids, topk_w};
}

torch::Tensor launch_nvfp4_moe_forward(
    torch::Tensor x, torch::Tensor w_gate_up, torch::Tensor s_gate_up, torch::Tensor outer_gu,
    torch::Tensor w_down, torch::Tensor s_down, torch::Tensor outer_down,
    torch::Tensor topk_ids, torch::Tensor topk_weights, double gate_softcap, double up_softcap
) {
    c10::cuda::CUDAGuard guard(x.device());
    int M = x.size(0), D = x.size(1), I = w_gate_up.size(1) / 2;
    int total_pairs = M * 8;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(x.device());

    auto u_exp  = torch::empty({total_pairs}, opts_i);
    auto u_off  = torch::empty({total_pairs + 1}, opts_i);
    auto p_idx  = torch::empty({total_pairs}, opts_i);
    auto n_uniq = torch::empty({1}, opts_i);

    build_expert_groups_kernel<<<1, 128, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), u_exp.data_ptr<int32_t>(),
        u_off.data_ptr<int32_t>(), p_idx.data_ptr<int32_t>(),
        n_uniq.data_ptr<int32_t>(), total_pairs
    );

    auto inter    = torch::empty({M, 8, I}, x.options());
    auto out_fp32 = torch::zeros({M, D}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    // Graph-safe launch: grid_y = min(total_pairs, 128), unused blocks return in 1 cycle
    int max_grid_y = std::min(total_pairs, 128);
    dim3 grid_gu((I + 3) / 4, max_grid_y);
    nvfp4_moe_gate_up_grouped_kernel<<<grid_gu, 128, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        w_gate_up.data_ptr<uint8_t>(), s_gate_up.data_ptr<uint8_t>(), outer_gu.data_ptr<float>(),
        u_exp.data_ptr<int32_t>(), u_off.data_ptr<int32_t>(), p_idx.data_ptr<int32_t>(), n_uniq.data_ptr<int32_t>(),
        reinterpret_cast<half*>(inter.data_ptr<at::Half>()), D, I, float(gate_softcap), float(up_softcap)
    );

    dim3 grid_dn((D + 3) / 4, max_grid_y);
    nvfp4_moe_down_grouped_kernel<<<grid_dn, 128, 0, stream>>>(
        reinterpret_cast<const half*>(inter.data_ptr<at::Half>()),
        w_down.data_ptr<uint8_t>(), s_down.data_ptr<uint8_t>(), outer_down.data_ptr<float>(),
        topk_weights.data_ptr<float>(),
        u_exp.data_ptr<int32_t>(), u_off.data_ptr<int32_t>(), p_idx.data_ptr<int32_t>(), n_uniq.data_ptr<int32_t>(),
        out_fp32.data_ptr<float>(), D, I
    );
    return out_fp32.to(torch::kFloat16);
}

torch::Tensor launch_mxfp8_moe_forward(
    torch::Tensor x, torch::Tensor w_gate_up, torch::Tensor s_gate_up,
    torch::Tensor w_down, torch::Tensor s_down,
    torch::Tensor topk_ids, torch::Tensor topk_weights, double gate_softcap, double up_softcap
) {
    c10::cuda::CUDAGuard guard(x.device());
    int M = x.size(0), D = x.size(1), I = w_gate_up.size(1) / 2;
    int total_pairs = M * 8;
    auto stream = at::cuda::getCurrentCUDAStream();
    auto opts_i = torch::TensorOptions().dtype(torch::kInt32).device(x.device());

    auto u_exp  = torch::empty({total_pairs}, opts_i);
    auto u_off  = torch::empty({total_pairs + 1}, opts_i);
    auto p_idx  = torch::empty({total_pairs}, opts_i);
    auto n_uniq = torch::empty({1}, opts_i);

    build_expert_groups_kernel<<<1, 128, 0, stream>>>(
        topk_ids.data_ptr<int32_t>(), u_exp.data_ptr<int32_t>(),
        u_off.data_ptr<int32_t>(), p_idx.data_ptr<int32_t>(),
        n_uniq.data_ptr<int32_t>(), total_pairs
    );

    auto inter    = torch::empty({M, 8, I}, x.options());
    auto out_fp32 = torch::zeros({M, D}, torch::TensorOptions().dtype(torch::kFloat32).device(x.device()));

    int max_grid_y = std::min(total_pairs, 128);
    dim3 grid_gu((I + 3) / 4, max_grid_y);
    mxfp8_moe_gate_up_grouped_kernel<<<grid_gu, 128, 0, stream>>>(
        reinterpret_cast<const half*>(x.data_ptr<at::Half>()),
        w_gate_up.data_ptr<uint8_t>(), s_gate_up.data_ptr<uint8_t>(),
        u_exp.data_ptr<int32_t>(), u_off.data_ptr<int32_t>(), p_idx.data_ptr<int32_t>(), n_uniq.data_ptr<int32_t>(),
        reinterpret_cast<half*>(inter.data_ptr<at::Half>()), D, I, float(gate_softcap), float(up_softcap)
    );

    dim3 grid_dn((D + 3) / 4, max_grid_y);
    mxfp8_moe_down_grouped_kernel<<<grid_dn, 128, 0, stream>>>(
        reinterpret_cast<const half*>(inter.data_ptr<at::Half>()),
        w_down.data_ptr<uint8_t>(), s_down.data_ptr<uint8_t>(), topk_weights.data_ptr<float>(),
        u_exp.data_ptr<int32_t>(), u_off.data_ptr<int32_t>(), p_idx.data_ptr<int32_t>(), n_uniq.data_ptr<int32_t>(),
        out_fp32.data_ptr<float>(), D, I
    );
    return out_fp32.to(torch::kFloat16);
}

} // namespace loopt4
