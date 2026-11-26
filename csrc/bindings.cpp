#include <torch/extension.h>
#include <cuda_runtime.h>

namespace loopt4 {

std::tuple<torch::Tensor, torch::Tensor> launch_moe_router_top8(
    torch::Tensor logits, c10::optional<torch::Tensor> gate_bias, double route_scale);

torch::Tensor launch_nvfp4_moe_forward(
    torch::Tensor x, torch::Tensor w_gate_up, torch::Tensor s_gate_up, torch::Tensor outer_gu,
    torch::Tensor w_down, torch::Tensor s_down, torch::Tensor outer_down,
    torch::Tensor topk_ids, torch::Tensor topk_weights, double gate_softcap, double up_softcap);

torch::Tensor launch_mxfp8_moe_forward(
    torch::Tensor x, torch::Tensor w_gate_up, torch::Tensor s_gate_up,
    torch::Tensor w_down, torch::Tensor s_down,
    torch::Tensor topk_ids, torch::Tensor topk_weights, double gate_softcap, double up_softcap);

std::tuple<torch::Tensor, torch::Tensor> launch_fused_attnres_msg_rmsnorm(
    torch::Tensor block_sums, c10::optional<torch::Tensor> prev_msg,
    torch::Tensor q_depth, c10::optional<torch::Tensor> kappa,
    c10::optional<torch::Tensor> gamma, torch::Tensor rms_weight, double eps);

torch::Tensor launch_short_conv4_step(
    torch::Tensor x, torch::Tensor weight, torch::Tensor state, bool apply_silu);

torch::Tensor launch_hybrid_gqa_attn(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> gate,
    torch::Tensor k_cache, torch::Tensor v_cache, c10::optional<torch::Tensor> rel_bias,
    torch::Tensor attn_sink, int64_t start_pos, bool is_swa, int64_t window_size,
    double qk_clip_thresh, double rms_eps, double log_n_floor, double log_alpha);

torch::Tensor launch_hybrid_gqa_attn_pos_buf(
    torch::Tensor q, torch::Tensor k, torch::Tensor v, c10::optional<torch::Tensor> gate,
    torch::Tensor k_cache, torch::Tensor v_cache, c10::optional<torch::Tensor> rel_bias,
    torch::Tensor attn_sink, torch::Tensor pos_buf, bool is_swa, int64_t window_size,
    double qk_clip_thresh, double rms_eps, double log_n_floor, double log_alpha);

bool enable_p2p_access(int dev0, int dev1) {
    int can01 = 0, can10 = 0;
    cudaDeviceCanAccessPeer(&can01, dev0, dev1);
    cudaDeviceCanAccessPeer(&can10, dev1, dev0);
    if (can01 && can10) {
        cudaSetDevice(dev0);
        cudaError_t e0 = cudaDeviceEnablePeerAccess(dev1, 0);
        if (e0 == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        cudaSetDevice(dev1);
        cudaError_t e1 = cudaDeviceEnablePeerAccess(dev0, 0);
        if (e1 == cudaErrorPeerAccessAlreadyEnabled) cudaGetLastError();
        return true;
    }
    return false;
}

} // namespace loopt4

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("moe_router_top8",           &loopt4::launch_moe_router_top8, "Top-8 sigmoid MoE router");
    m.def("nvfp4_moe_forward",         &loopt4::launch_nvfp4_moe_forward, "Turing NVFP4 W4A16 MoE");
    m.def("mxfp8_moe_forward",         &loopt4::launch_mxfp8_moe_forward, "Turing MXFP8 W8A16 MoE");
    m.def("fused_attnres_msg_rmsnorm", &loopt4::launch_fused_attnres_msg_rmsnorm, "FP32-safe AttnRes+Msg+RMSNorm");
    m.def("short_conv4_step",          &loopt4::launch_short_conv4_step, "4-tap causal short-conv1d");
    m.def("hybrid_gqa_attn",           &loopt4::launch_hybrid_gqa_attn, "Hybrid SWA-1024 / Global GQA");
    m.def("hybrid_gqa_attn_pos_buf",   &loopt4::launch_hybrid_gqa_attn_pos_buf, "CUDA-Graph-safe Hybrid GQA");
    m.def("enable_p2p_access",         &loopt4::enable_p2p_access, "Enable CUDA P2P between 2xT4");
}
