from dataclasses import dataclass, field
from typing import List, Set

@dataclass
class Inkling2MiniConfig:
    # Core text backbone (verified from thinkingmachines/Inkling-2-Mini config.json)
    hidden_size: int = 2304
    num_hidden_layers: int = 24
    vocab_size: int = 201024
    usable_vocab_size: int = 200058
    eos_token_id: int = 200006
    logit_softcap: float = 30.0

    # Hybrid GQA Attention
    num_attention_heads: int = 24
    num_key_value_heads: int = 8
    head_dim: int = 128
    window_size: int = 1024
    d_rel: int = 16
    rel_extent: int = 1024
    qk_l2_clip_threshold: float = 22.627416997969522
    rel_l2_clip_threshold: float = 8.0
    log_scaling_n_floor: float = 8192.0
    log_scaling_alpha: float = 0.1
    rms_norm_eps: float = 1e-6

    # Looped Core topology: layers 6..17 run 3x -> 48 visits total
    core_start: int = 6
    core_end: int = 18
    num_passes: int = 3
    gate_rank: int = 64
    block_size_attnres: int = 6

    # Sparse MoE (NVFP4 layers 0..19, MXFP8 layers 20..23)
    num_experts: int = 128
    top_k: int = 8
    intermediate_size: int = 1152
    route_scale: float = 8.0
    mlp_gate_softcap: float = 4.0
    mlp_up_softcap: float = 25.0
    nvfp4_layer_end: int = 20  # [0..19] NVFP4, [20..23] MXFP8

    # DSpark MTP Speculator (mtp.safetensors)
    mtp_num_layers: int = 3
    mtp_intermediate_size: int = 9216
    mtp_markov_rank: int = 256
    mtp_draft_steps: int = 5
    # Execution visits tapped for DSpark context
    mtp_target_visits: List[int] = field(default_factory=lambda: [5, 17, 29, 41])

    # Every 6th layer (5, 11, 17, 23) is full global attention; others are SWA-1024
    global_layer_ids: Set[int] = field(default_factory=lambda: {5, 11, 17, 23})

    # Runtime memory cap for the 8 global KV caches (tune for Kaggle 2xT4 headroom)
    max_global_seq_len: int = 32768

    @property
    def total_visits(self) -> int:
        return (self.core_start
                + (self.core_end - self.core_start) * self.num_passes
                + (self.num_hidden_layers - self.core_end))

    def layer_device(self, layer_idx: int, num_gpus: int) -> str:
        """Core-Pinned Asymmetric Split: Core (6..17) on GPU 0, I/O+Prelude+Coda on GPU 1."""
        if num_gpus < 2:
            return "cuda:0"
        return "cuda:0" if (self.core_start <= layer_idx < self.core_end) else "cuda:1"
