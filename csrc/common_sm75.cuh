#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>
#include <math.h>

namespace loopt4 {

// Exact OCP NVFP4 E2M1 decode table (16 values) loaded into warp registers.
// Lanes 0..15 and 16..31 hold identical copies so warp-wide shuffles work with 0 bank conflicts.
__device__ __forceinline__ float init_e2m1_lane_lut() {
    static constexpr float kTable[16] = {
         0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
    };
    return kTable[threadIdx.x & 15];
}

__device__ __forceinline__ float e2m1_lut_lookup(uint32_t nibble, float lane_lut) {
    return __shfl_sync(0xFFFFFFFFu, lane_lut, nibble & 15u);
}

// Exact software FP8 E4M3FN -> FP32 converter for Turing (sm_75 has no native FP8).
__device__ __forceinline__ float e4m3fn_to_float(uint8_t x) {
    uint32_t s = (x >> 7) & 1u;
    uint32_t e = (x >> 3) & 0xFu;
    uint32_t m = x & 0x7u;
    if (e == 0u) {
        float v = float(m) * (1.0f / 512.0f);
        return s ? -v : v;
    }
    uint32_t f32 = (s << 31) | ((e + 120u) << 23) | (m << 20);
    return __uint_as_float(f32);
}

// Exact OCP MXFP8 UE8M0 -> FP32 scale converter (power-of-two scale, bias 127).
__device__ __forceinline__ float ue8m0_to_float(uint8_t u) {
    if (u == 0u) return 5.877471754111438e-39f; // 2^-127
    if (u == 255u) return 0.0f;                // NaN guard
    return __uint_as_float(uint32_t(u) << 23);
}

__device__ __forceinline__ float softcap_f32(float x, float cap) {
    return cap * tanhf(x / cap);
}

__device__ __forceinline__ float silu_f32(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float sigmoid_f32(float x) {
    return 1.0f / (1.0f + expf(-x));
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xFFFFFFFFu, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFFu, val, offset));
    }
    return val;
}

template <int NUM_WARPS>
__device__ __forceinline__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x & 31;
    int wid  = threadIdx.x >> 5;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (threadIdx.x < NUM_WARPS) ? smem[lane] : 0.0f;
    if (wid == 0) val = warp_reduce_sum(val);
    if (threadIdx.x == 0) smem[0] = val;
    __syncthreads();
    return smem[0];
}

} // namespace loopt4
