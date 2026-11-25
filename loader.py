import os
import json
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional
from .config import Inkling2MiniConfig

@dataclass
class LayerWeights:
    layer_idx: int
    device: torch.device
    is_nvfp4: bool

    # Block Attention Residuals & RMSNorm
    attn_q_depth: torch.Tensor        # [D] fp16
    attn_rms_w: torch.Tensor          # [D] fp16
    mlp_q_depth: torch.Tensor         # [D] fp16
    mlp_rms_w: torch.Tensor           # [D] fp16

    # Optional cross-loop rank-64 message gates (only layers 6..17, passes 2 & 3)
    msg_down: Optional[torch.Tensor] = None       # [64, D] fp16
    msg_up: Optional[torch.Tensor] = None         # [D, 64] fp16
    attn_kappa: Optional[List[torch.Tensor]] = None  # [pass-1] -> [D] fp16
    attn_gamma: Optional[List[torch.Tensor]] = None  # [pass-1] -> [D] fp16
    mlp_kappa: Optional[List[torch.Tensor]] = None
    mlp_gamma: Optional[List[torch.Tensor]] = None

    # Attention projections (Q, K, V, Gate, O) + 4 short-convs + rel bias + sink
    w_q: torch.Tensor = None          # [24*128, D] fp16
    w_k: torch.Tensor = None          # [8*128, D] fp16
    w_v: torch.Tensor = None          # [8*128, D] fp16
    w_gate: torch.Tensor = None       # [24*128, D] fp16
    w_o: torch.Tensor = None          # [D, 24*128] fp16
    rel_bias: torch.Tensor = None     # [24, 1024] fp16
    attn_sink: torch.Tensor = None    # [24] fp32

    k_sconv_w: torch.Tensor = None    # [8*128, 4] fp16
    v_sconv_w: torch.Tensor = None    # [8*128, 4] fp16
    attn_sconv_w: torch.Tensor = None # [D, 4] fp16
    mlp_sconv_w: torch.Tensor = None  # [D, 4] fp16

    # Router
    router_w: torch.Tensor = None     # [128, D] fp16
    router_bias: Optional[torch.Tensor] = None # [128] fp16

    # Quantized MoE Experts (NVFP4 for 0..19, MXFP8 for 20..23)
    w_gate_up: torch.Tensor = None    # uint8
    s_gate_up: torch.Tensor = None    # uint8
    outer_gu: Optional[torch.Tensor] = None  # [128] fp32 (NVFP4 only)
    w_down: torch.Tensor = None       # uint8
    s_down: torch.Tensor = None       # uint8
    outer_down: Optional[torch.Tensor] = None # [128] fp32 (NVFP4 only)


def _cast_fp16(t: torch.Tensor, dev: torch.device) -> torch.Tensor:
    return t.to(device=dev, dtype=torch.float16, non_blocking=True).contiguous()


def load_core_pinned_safetensors(
    checkpoint_dir: str,
    cfg: Inkling2MiniConfig,
    num_gpus: int = 2,
) -> Dict:
    """
    Streams tensors directly from mmap'd safetensors onto their target GPU
    (GPU 0 = Core 6..17, GPU 1 = Prelude/Coda/Emb/Head) to stay under Kaggle's 30GB RAM cap.
    """
    from safetensors import safe_open

    st_path = os.path.join(checkpoint_dir, "model-00001-of-00001.safetensors")
    if not os.path.exists(st_path):
        st_path = os.path.join(checkpoint_dir, "model.safetensors")

    io_dev = torch.device("cuda:1" if num_gpus >= 2 else "cuda:0")
    weights: Dict = {"layers": {}, "io_dev": io_dev}

    with safe_open(st_path, framework="pt", device="cpu") as f:
        keys = set(f.keys())

        # Auto-detect prefix ("language_model.model." vs "model.")
        prefix = "language_model.model." if any(k.startswith("language_model.model.") for k in keys) else "model."
        lm_prefix = "language_model." if prefix.startswith("language_model.") else ""

        def fetch(suffix: str, dev: torch.device, dtype=torch.float16) -> torch.Tensor:
            t = f.get_tensor(prefix + suffix)
            return t.to(device=dev, dtype=dtype).contiguous()

        def fetch_opt(suffix: str, dev: torch.device, dtype=torch.float16) -> Optional[torch.Tensor]:
            k = prefix + suffix
            if k in keys:
                return f.get_tensor(k).to(device=dev, dtype=dtype).contiguous()
            return None

        weights["embed_tokens"] = fetch("embed_tokens.weight", io_dev, torch.float16)
        weights["final_norm"]   = fetch("norm.weight", io_dev, torch.float16)
        lm_key = f"{lm_prefix}lm_head.weight" if f"{lm_prefix}lm_head.weight" in keys else "lm_head.weight"
        weights["lm_head"]      = f.get_tensor(lm_key).to(device=io_dev, dtype=torch.float16).contiguous()

        for l in range(cfg.num_hidden_layers):
            dev = torch.device(cfg.layer_device(l, num_gpus))
            lp = f"layers.{l}."
            is_nvfp4 = (l < cfg.nvfp4_layer_end)
            D = cfg.hidden_size

            lw = LayerWeights(
                layer_idx=l,
                device=dev,
                is_nvfp4=is_nvfp4,
                attn_q_depth=fetch_opt(lp + "attn_res.query", dev) or torch.zeros(D, device=dev, dtype=torch.float16),
                attn_rms_w=fetch(lp + "input_layernorm.weight", dev),
                mlp_q_depth=fetch_opt(lp + "mlp_res.query", dev) or torch.zeros(D, device=dev, dtype=torch.float16),
                mlp_rms_w=fetch(lp + "post_attention_layernorm.weight", dev),
                w_q=fetch(lp + "self_attn.q_proj.weight", dev),
                w_k=fetch(lp + "self_attn.k_proj.weight", dev),
                w_v=fetch(lp + "self_attn.v_proj.weight", dev),
                w_gate=fetch_opt(lp + "self_attn.g_proj.weight", dev) or torch.ones(cfg.num_attention_heads * cfg.head_dim, D, device=dev, dtype=torch.float16),
                w_o=fetch(lp + "self_attn.o_proj.weight", dev),
                rel_bias=fetch_opt(lp + "self_attn.rel_bias", dev) or torch.zeros(cfg.num_attention_heads, 1024, device=dev, dtype=torch.float16),
                attn_sink=fetch_opt(lp + "self_attn.attn_sink", dev, torch.float32) or torch.full((cfg.num_attention_heads,), -20.0, device=dev, dtype=torch.float32),
                k_sconv_w=fetch(lp + "self_attn.k_proj.sconv.conv1d.weight", dev).squeeze(1),
                v_sconv_w=fetch(lp + "self_attn.v_proj.sconv.conv1d.weight", dev).squeeze(1),
                attn_sconv_w=fetch(lp + "attn_sconv.weight", dev).squeeze(1),
                mlp_sconv_w=fetch(lp + "mlp_sconv.weight", dev).squeeze(1),
                router_w=fetch(lp + "mlp.gate.weight", dev),
                router_bias=fetch_opt(lp + "mlp.gate.bias", dev),
            )

            # Cross-loop message weights for core layers 6..17
            if cfg.core_start <= l < cfg.core_end:
                lw.msg_down = fetch_opt(lp + "loop_msg.down_proj.weight", dev) or torch.zeros(cfg.gate_rank, D, device=dev, dtype=torch.float16)
                lw.msg_up   = fetch_opt(lp + "loop_msg.up_proj.weight", dev) or torch.zeros(D, cfg.gate_rank, device=dev, dtype=torch.float16)
                lw.attn_kappa = [fetch_opt(lp + f"loop_msg.attn_kappa.{p}", dev) or torch.ones(D, device=dev, dtype=torch.float16) for p in range(cfg.num_passes - 1)]
                lw.attn_gamma = [fetch_opt(lp + f"loop_msg.attn_gamma.{p}", dev) or torch.zeros(D, device=dev, dtype=torch.float16) for p in range(cfg.num_passes - 1)]
                lw.mlp_kappa  = [fetch_opt(lp + f"loop_msg.mlp_kappa.{p}", dev) or torch.ones(D, device=dev, dtype=torch.float16) for p in range(cfg.num_passes - 1)]
                lw.mlp_gamma  = [fetch_opt(lp + f"loop_msg.mlp_gamma.{p}", dev) or torch.zeros(D, device=dev, dtype=torch.float16) for p in range(cfg.num_passes - 1)]

            # Load quantized expert tensors directly in raw uint8 (no dequantization on CPU!)
            lw.w_gate_up = fetch(lp + "mlp.experts.w13_weight", dev, torch.uint8)
            lw.s_gate_up = fetch(lp + "mlp.experts.w13_weight_scale", dev, torch.uint8)
            lw.w_down    = fetch(lp + "mlp.experts.w2_weight", dev, torch.uint8)
            lw.s_down    = fetch(lp + "mlp.experts.w2_weight_scale", dev, torch.uint8)
            if is_nvfp4:
                lw.outer_gu   = fetch_opt(lp + "mlp.experts.w13_weight_scale_2", dev, torch.float32) or torch.ones(cfg.num_experts, device=dev, dtype=torch.float32)
                lw.outer_down = fetch_opt(lp + "mlp.experts.w2_weight_scale_2", dev, torch.float32) or torch.ones(cfg.num_experts, device=dev, dtype=torch.float32)

            weights["layers"][l] = lw

    return weights


def create_synthetic_weights(cfg: Inkling2MiniConfig, num_gpus: int = 2) -> Dict:
    """
    Allocates exact-shape NVFP4/MXFP8 weights directly on 2xT4 for instant
    Kaggle benchmarking & numerical validation without downloading 20GB first.
    """
    io_dev = torch.device("cuda:1" if num_gpus >= 2 else "cuda:0")
    D, I, E = cfg.hidden_size, cfg.intermediate_size, cfg.num_experts
    weights: Dict = {
        "io_dev": io_dev,
        "embed_tokens": torch.randn(cfg.vocab_size, D, device=io_dev, dtype=torch.float16) * 0.02,
        "final_norm":   torch.ones(D, device=io_dev, dtype=torch.float16),
        "lm_head":      torch.randn(cfg.usable_vocab_size, D, device=io_dev, dtype=torch.float16) * 0.02,
        "layers": {},
    }

    for l in range(cfg.num_hidden_layers):
        dev = torch.device(cfg.layer_device(l, num_gpus))
        is_nvfp4 = (l < cfg.nvfp4_layer_end)
        lw = LayerWeights(
            layer_idx=l,
            device=dev,
            is_nvfp4=is_nvfp4,
            attn_q_depth=torch.randn(D, device=dev, dtype=torch.float16) * 0.05,
            attn_rms_w=torch.ones(D, device=dev, dtype=torch.float16),
            mlp_q_depth=torch.randn(D, device=dev, dtype=torch.float16) * 0.05,
            mlp_rms_w=torch.ones(D, device=dev, dtype=torch.float16),
            w_q=torch.randn(24 * 128, D, device=dev, dtype=torch.float16) * 0.02,
            w_k=torch.randn(8 * 128, D, device=dev, dtype=torch.float16) * 0.02,
            w_v=torch.randn(8 * 128, D, device=dev, dtype=torch.float16) * 0.02,
            w_gate=torch.randn(24 * 128, D, device=dev, dtype=torch.float16) * 0.02,
            w_o=torch.randn(D, 24 * 128, device=dev, dtype=torch.float16) * 0.02,
            rel_bias=torch.zeros(24, 1024, device=dev, dtype=torch.float16),
            attn_sink=torch.full((24,), -4.0, device=dev, dtype=torch.float32),
            k_sconv_w=torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev, dtype=torch.float16).repeat(8 * 128, 1),
            v_sconv_w=torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev, dtype=torch.float16).repeat(8 * 128, 1),
            attn_sconv_w=torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev, dtype=torch.float16).repeat(D, 1),
            mlp_sconv_w=torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev, dtype=torch.float16).repeat(D, 1),
            router_w=torch.randn(E, D, device=dev, dtype=torch.float16) * 0.02,
            router_bias=torch.zeros(E, device=dev, dtype=torch.float16),
        )
        if cfg.core_start <= l < cfg.core_end:
            lw.msg_down   = torch.randn(cfg.gate_rank, D, device=dev, dtype=torch.float16) * 0.02
            lw.msg_up     = torch.randn(D, cfg.gate_rank, device=dev, dtype=torch.float16) * 0.02
            lw.attn_kappa = [torch.ones(D, device=dev, dtype=torch.float16) for _ in range(cfg.num_passes - 1)]
            lw.attn_gamma = [torch.full((D,), 0.1, device=dev, dtype=torch.float16) for _ in range(cfg.num_passes - 1)]
            lw.mlp_kappa  = [torch.ones(D, device=dev, dtype=torch.float16) for _ in range(cfg.num_passes - 1)]
            lw.mlp_gamma  = [torch.full((D,), 0.1, device=dev, dtype=torch.float16) for _ in range(cfg.num_passes - 1)]

        if is_nvfp4:
            # 0x22 = (E2M1 1.0, 1.0), 0x38 = FP8 E4M3FN 1.0
            lw.w_gate_up  = torch.randint(0, 256, (E, 2 * I, D // 2), device=dev, dtype=torch.uint8)
            lw.s_gate_up  = torch.full((E, 2 * I, D // 16), 0x30, device=dev, dtype=torch.uint8)
            lw.outer_gu   = torch.full((E,), 0.02, device=dev, dtype=torch.float32)
            lw.w_down     = torch.randint(0, 256, (E, D, I // 2), device=dev, dtype=torch.uint8)
            lw.s_down     = torch.full((E, D, I // 16), 0x30, device=dev, dtype=torch.uint8)
            lw.outer_down = torch.full((E,), 0.02, device=dev, dtype=torch.float32)
        else:
            # MXFP8: E4M3FN weights + UE8M0 power-of-2 scales (121 -> 2^-6)
            lw.w_gate_up = torch.randint(0, 127, (E, 2 * I, D), device=dev, dtype=torch.uint8)
            lw.s_gate_up = torch.full((E, 2 * I, D // 32), 121, device=dev, dtype=torch.uint8)
            lw.w_down    = torch.randint(0, 127, (E, D, I), device=dev, dtype=torch.uint8)
            lw.s_down    = torch.full((E, D, I // 32), 121, device=dev, dtype=torch.uint8)

        weights["layers"][l] = lw

    return weights
