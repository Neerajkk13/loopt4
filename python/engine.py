import torch
import torch.nn.functional as F
import loopt4_ext
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .config import Inkling2MiniConfig
from .loader import LayerWeights
from .dspark import DSparkDrafter


@dataclass
class VisitState:
    """Per-visit (0..47) KV cache and 4x short-conv history states with O(1) rollback."""
    is_swa: bool
    k_cache: torch.Tensor        # [1024 or max_global, 8, 128] fp16
    v_cache: torch.Tensor
    k_sconv_state: torch.Tensor  # [8*128, 3] fp16
    v_sconv_state: torch.Tensor  # [8*128, 3] fp16
    attn_sconv_state: torch.Tensor # [D, 3] fp16
    mlp_sconv_state: torch.Tensor  # [D, 3] fp16

    # Tiny snapshot buffers for O(1) DSpark rollback
    snap_k_sconv: torch.Tensor = None
    snap_v_sconv: torch.Tensor = None
    snap_attn_sconv: torch.Tensor = None
    snap_mlp_sconv: torch.Tensor = None

    def snapshot_conv(self):
        self.snap_k_sconv.copy_(self.k_sconv_state)
        self.snap_v_sconv.copy_(self.v_sconv_state)
        self.snap_attn_sconv.copy_(self.attn_sconv_state)
        self.snap_mlp_sconv.copy_(self.mlp_sconv_state)

    def restore_conv(self):
        self.k_sconv_state.copy_(self.snap_k_sconv)
        self.v_sconv_state.copy_(self.snap_v_sconv)
        self.attn_sconv_state.copy_(self.snap_attn_sconv)
        self.mlp_sconv_state.copy_(self.snap_mlp_sconv)


class LoopT4Engine:
    """
    Core-Pinned Asymmetric 2xT4 Engine for Inkling 2 Mini.
    - GPU 1: Embeddings + Prelude (0..5) + Coda (18..23) + LM Head + DSpark
    - GPU 0: Looped Core (6..17 x 3 passes = 36 local layer visits)
    - Exactly 2 tiny PCIe peer-to-peer transfers per decode/verify step.
    """
    def __init__(self, cfg: Inkling2MiniConfig, weights: Dict, num_gpus: int = 2, enable_dspark: bool = True):
        self.cfg = cfg
        self.weights = weights
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.dev_core = torch.device("cuda:0")
        self.dev_io   = torch.device("cuda:1" if self.num_gpus >= 2 else "cuda:0")

        if self.num_gpus >= 2:
            self.p2p_enabled = loopt4_ext.enable_p2p_access(0, 1)
        else:
            self.p2p_enabled = False

        # Build the 48-step execution schedule: (visit_id, layer_idx, pass_idx)
        self.schedule: List[Tuple[int, int, int]] = []
        v = 0
        for l in range(0, cfg.core_start):
            self.schedule.append((v, l, 0)); v += 1
        for p in range(cfg.num_passes):
            for l in range(cfg.core_start, cfg.core_end):
                self.schedule.append((v, l, p)); v += 1
        for l in range(cfg.core_end, cfg.num_hidden_layers):
            self.schedule.append((v, l, 0)); v += 1

        # Allocate 48 VisitStates on their owning GPU
        self.states: Dict[int, VisitState] = {}
        D = cfg.hidden_size
        for visit_id, l, _ in self.schedule:
            dev = torch.device(cfg.layer_device(l, self.num_gpus))
            is_swa = (l not in cfg.global_layer_ids)
            slots = cfg.window_size if is_swa else cfg.max_global_seq_len
            st = VisitState(
                is_swa=is_swa,
                k_cache=torch.zeros(slots, 8, 128, device=dev, dtype=torch.float16),
                v_cache=torch.zeros(slots, 8, 128, device=dev, dtype=torch.float16),
                k_sconv_state=torch.zeros(8 * 128, 3, device=dev, dtype=torch.float16),
                v_sconv_state=torch.zeros(8 * 128, 3, device=dev, dtype=torch.float16),
                attn_sconv_state=torch.zeros(D, 3, device=dev, dtype=torch.float16),
                mlp_sconv_state=torch.zeros(D, 3, device=dev, dtype=torch.float16),
                snap_k_sconv=torch.zeros(8 * 128, 3, device=dev, dtype=torch.float16),
                snap_v_sconv=torch.zeros(8 * 128, 3, device=dev, dtype=torch.float16),
                snap_attn_sconv=torch.zeros(D, 3, device=dev, dtype=torch.float16),
                snap_mlp_sconv=torch.zeros(D, 3, device=dev, dtype=torch.float16),
            )
            self.states[visit_id] = st

        self.dspark = (
            DSparkDrafter(cfg, self.dev_io, weights["embed_tokens"], weights["lm_head"])
            if enable_dspark else None
        )
        self.pos = 0
        self._core_graph_cache: Dict[int, Tuple[torch.cuda.CUDAGraph, Dict[str, torch.Tensor]]] = {}

    def reset_cache(self):
        self.pos = 0
        for st in self.states.values():
            st.k_cache.zero_(); st.v_cache.zero_()
            st.k_sconv_state.zero_(); st.v_sconv_state.zero_()
            st.attn_sconv_state.zero_(); st.mlp_sconv_state.zero_()

    def _run_layer(
        self,
        lw: LayerWeights,
        st: VisitState,
        block_sums: List[torch.Tensor],
        partial_block: torch.Tensor,
        prev_msg: Optional[torch.Tensor],
        pass_idx: int,
        start_pos: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes 1 visit of a layer: AttnRes+Norm -> Hybrid GQA -> AttnRes+Norm -> MoE."""
        M = partial_block.size(0)
        stacked = torch.stack(block_sums + [partial_block], dim=0) # [B, M, D]

        use_msg = (prev_msg is not None and pass_idx > 0 and lw.attn_kappa is not None)
        x_norm, _ = loopt4_ext.fused_attnres_msg_rmsnorm(
            stacked,
            prev_msg if use_msg else None,
            lw.attn_q_depth,
            lw.attn_kappa[pass_idx - 1] if use_msg else None,
            lw.attn_gamma[pass_idx - 1] if use_msg else None,
            lw.attn_rms_w,
            self.cfg.rms_norm_eps,
        )

        # Q, K, V, Gate projections + K/V 4-tap short convolutions
        q = F.linear(x_norm, lw.w_q).view(M, 24, 128)
        k_raw = F.linear(x_norm, lw.w_k)
        v_raw = F.linear(x_norm, lw.w_v)
        k = loopt4_ext.short_conv4_step(k_raw, lw.k_sconv_w, st.k_sconv_state, False).view(M, 8, 128)
        v = loopt4_ext.short_conv4_step(v_raw, lw.v_sconv_w, st.v_sconv_state, False).view(M, 8, 128)
        g = F.linear(x_norm, lw.w_gate).view(M, 24, 128)

        attn_out = loopt4_ext.hybrid_gqa_attn(
            q, k, v, g,
            st.k_cache, st.v_cache,
            lw.rel_bias, lw.attn_sink,
            start_pos, st.is_swa, self.cfg.window_size,
            self.cfg.qk_l2_clip_threshold, self.cfg.rms_norm_eps,
            self.cfg.log_scaling_n_floor, self.cfg.log_scaling_alpha,
        )
        attn_proj = F.linear(attn_out.view(M, -1), lw.w_o)
        attn_proj = loopt4_ext.short_conv4_step(attn_proj, lw.attn_sconv_w, st.attn_sconv_state, False)
        partial_block = partial_block + attn_proj

        # MLP sublayer with Attention Residuals + cross-loop message
        stacked2 = torch.stack(block_sums + [partial_block], dim=0)
        x_mlp_norm, _ = loopt4_ext.fused_attnres_msg_rmsnorm(
            stacked2,
            prev_msg if use_msg else None,
            lw.mlp_q_depth,
            lw.mlp_kappa[pass_idx - 1] if use_msg else None,
            lw.mlp_gamma[pass_idx - 1] if use_msg else None,
            lw.mlp_rms_w,
            self.cfg.rms_norm_eps,
        )

        # Top-8 Sigmoid Router + Native NVFP4 / MXFP8 Turing MoE
        router_logits = F.linear(x_mlp_norm, lw.router_w)
        topk_ids, topk_w = loopt4_ext.moe_router_top8(router_logits, lw.router_bias, self.cfg.route_scale)

        if lw.is_nvfp4:
            moe_out = loopt4_ext.nvfp4_moe_forward(
                x_mlp_norm, lw.w_gate_up, lw.s_gate_up, lw.outer_gu,
                lw.w_down, lw.s_down, lw.outer_down,
                topk_ids, topk_w, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap,
            )
        else:
            moe_out = loopt4_ext.mxfp8_moe_forward(
                x_mlp_norm, lw.w_gate_up, lw.s_gate_up,
                lw.w_down, lw.s_down,
                topk_ids, topk_w, self.cfg.mlp_gate_softcap, self.cfg.mlp_up_softcap,
            )

        moe_out = loopt4_ext.short_conv4_step(moe_out, lw.mlp_sconv_w, st.mlp_sconv_state, False)
        partial_block = partial_block + moe_out

        # Compute rank-64 cross-loop message for next pass
        next_msg = None
        if lw.msg_down is not None and lw.msg_up is not None:
            next_msg = F.linear(F.linear(partial_block, lw.msg_down), lw.msg_up)

        return partial_block, next_msg

    @torch.inference_mode()
    def forward_tokens(self, token_ids: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Runs Prelude (GPU 1) -> Hop 1 -> 3x Looped Core (GPU 0) -> Hop 2 -> Coda + Logits (GPU 1).
        Returns: (logits [M, usable_vocab], 4 DSpark taps on GPU 1).
        """
        cfg = self.cfg
        toks_io = token_ids.to(self.dev_io, non_blocking=True)
        h0 = self.weights["embed_tokens"][toks_io] # [M, D]

        block_sums_io: List[torch.Tensor] = [h0]
        partial_io = torch.zeros_like(h0)
        dspark_taps: Dict[int, torch.Tensor] = {}

        # ---- Stage 1: Prelude (Layers 0..5) on GPU 1 ----
        for visit_id in range(0, cfg.core_start):
            _, l, p = self.schedule[visit_id]
            partial_io, _ = self._run_layer(
                self.weights["layers"][l], self.states[visit_id],
                block_sums_io, partial_io, None, p, start_pos
            )
            if visit_id in cfg.mtp_target_visits:
                dspark_taps[visit_id] = partial_io[-1:].clone()
        block_sums_io.append(partial_io) # Completed Block 1 (Layers 0..5)

        # ---- PCIe Hop 1 (GPU 1 -> GPU 0): only 2 block sums (~9 KB/tok) ----
        packed_to_core = torch.stack(block_sums_io, dim=0).to(self.dev_core, non_blocking=True)
        block_sums_core = [packed_to_core[0], packed_to_core[1]]
        partial_core = torch.zeros_like(block_sums_core[0])

        # ---- Stage 2: Looped Core (Layers 6..17 x 3 passes = 36 visits) on GPU 0 ----
        prev_pass_msgs: Dict[int, torch.Tensor] = {}
        core_visit_start = cfg.core_start
        core_visit_end   = cfg.core_start + (cfg.core_end - cfg.core_start) * cfg.num_passes
        core_taps: Dict[int, torch.Tensor] = {}

        for visit_id in range(core_visit_start, core_visit_end):
            _, l, p = self.schedule[visit_id]
            partial_core, next_msg = self._run_layer(
                self.weights["layers"][l], self.states[visit_id],
                block_sums_core, partial_core, prev_pass_msgs.get(l), p, start_pos
            )
            prev_pass_msgs[l] = next_msg
            if visit_id in cfg.mtp_target_visits:
                core_taps[visit_id] = partial_core[-1:].clone()
            # Every 6 layers finalize an Attention-Residual block sum
            if ((visit_id + 1) % cfg.block_size_attnres) == 0:
                block_sums_core.append(partial_core)
                partial_core = torch.zeros_like(partial_core)

        # ---- PCIe Hop 2 (GPU 0 -> GPU 1): coalesced transfer of block sums + taps ----
        core_payload = torch.stack(block_sums_core[2:], dim=0).to(self.dev_io, non_blocking=True)
        for i in range(core_payload.size(0)):
            block_sums_io.append(core_payload[i])
        for vid, tap in core_taps.items():
            dspark_taps[vid] = tap.to(self.dev_io, non_blocking=True)

        # ---- Stage 3: Coda (Layers 18..23, incl. MXFP8 20..23) on GPU 1 ----
        partial_io = torch.zeros_like(h0)
        for visit_id in range(core_visit_end, cfg.total_visits):
            _, l, p = self.schedule[visit_id]
            partial_io, _ = self._run_layer(
                self.weights["layers"][l], self.states[visit_id],
                block_sums_io, partial_io, None, p, start_pos
            )
            if visit_id in cfg.mtp_target_visits:
                dspark_taps[visit_id] = partial_io[-1:].clone()

        # Final RMSNorm (FP32 safe) + LM Head + Logit Softcap (30.0 over 200,058 usable IDs)
        final_h = block_sums_io[-1] + partial_io
        xf = final_h.float()
        x_norm = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(torch.float16)
        x_norm = x_norm * self.weights["final_norm"]
        logits = F.linear(x_norm, self.weights["lm_head"]).float()
        logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)

        ordered_taps = [dspark_taps.get(v, final_h[-1:]) for v in cfg.mtp_target_visits]
        return logits, ordered_taps

    @torch.inference_mode()
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 64, use_dspark: bool = True) -> List[int]:
        """End-to-end prefill + greedy decode (with optional DSpark speculative verification)."""
        self.reset_cache()
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long)

        # Chunked prefill (256 tokens/chunk keeps peak activation memory tiny on T4)
        logits, taps = None, None
        chunk = 256
        for i in range(0, len(prompt_ids), chunk):
            sub = prompt_t[i:i + chunk]
            logits, taps = self.forward_tokens(sub, self.pos)
            self.pos += sub.numel()

        next_tok = int(torch.argmax(logits[-1]).item())
        generated = [next_tok]

        while len(generated) < max_new_tokens and next_tok != self.cfg.eos_token_id:
            if use_dspark and self.dspark is not None:
                # 1. Snapshot all 48 short-conv states before speculative verify
                for st in self.states.values():
                    st.snapshot_conv()

                # 2. Draft K tokens locally on GPU 1
                k = min(self.cfg.mtp_draft_steps, max_new_tokens - len(generated))
                draft = self.dspark.draft_k_tokens(next_tok, taps, self.pos, k)
                verify_ids = torch.tensor([next_tok] + draft, dtype=torch.long)

                # 3. Single batched verification pass across 2xT4 (M = K + 1)
                v_logits, taps = self.forward_tokens(verify_ids, self.pos)
                preds = torch.argmax(v_logits, dim=-1).tolist()

                # 4. Greedy acceptance check + O(1) rollback on mismatch
                accepted = 0
                for j, d_tok in enumerate(draft):
                    if preds[j] == d_tok:
                        generated.append(d_tok)
                        accepted += 1
                    else:
                        generated.append(preds[j])
                        break
                else:
                    # All K draft tokens accepted -> also take bonus token preds[K]!
                    generated.append(preds[k])

                if accepted < k:
                    # Roll back short-conv states and replay only the accepted prefix cleanly
                    for st in self.states.values():
                        st.restore_conv()
                    replay_ids = torch.tensor([next_tok] + draft[:accepted], dtype=torch.long)
                    _, taps = self.forward_tokens(replay_ids, self.pos)
                    self.pos += accepted + 1
                else:
                    self.pos += k + 1

                next_tok = generated[-1]
            else:
                tok_t = torch.tensor([next_tok], dtype=torch.long)
                logits, taps = self.forward_tokens(tok_t, self.pos)
                self.pos += 1
                next_tok = int(torch.argmax(logits[-1]).item())
                generated.append(next_tok)

        return generated[:max_new_tokens]
