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
    k_cache: torch.Tensor           # [1024 or max_global, 8, 128] fp16
    v_cache: torch.Tensor
    k_sconv_state: torch.Tensor     # [8*128, 3] fp16
    v_sconv_state: torch.Tensor     # [8*128, 3] fp16
    attn_sconv_state: torch.Tensor  # [D, 3] fp16
    mlp_sconv_state: torch.Tensor   # [D, 3] fp16

    # Snapshot buffers for O(1) DSpark rollback & state-safe CUDA Graph capture
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
    Patched Core-Pinned Asymmetric 2xT4 Engine for Inkling 2 Mini.
    - GPU 1 (dev_io):   Embeddings + Prelude (0..5) + Coda (18..23) + LM Head + DSpark
    - GPU 0 (dev_core): Looped Core (6..17 x 3 passes = 36 local layer visits)
    - Exactly 2 event-ordered PCIe transfers per decode/verify step.
    """
    def __init__(
        self,
        cfg: Inkling2MiniConfig,
        weights: Dict,
        num_gpus: int = 2,
        enable_dspark: bool = True,
        enable_p2p: bool = False,
        use_cuda_graph: bool = True,
    ):
        self.cfg = cfg
        self.weights = weights
        self.num_gpus = min(num_gpus, torch.cuda.device_count())
        self.dev_core = torch.device("cuda:0")
        self.dev_io   = torch.device("cuda:1" if self.num_gpus >= 2 else "cuda:0")

        # Disabled by default to prevent Kaggle virtualized PCIe teardown glitches
        if enable_p2p and self.num_gpus >= 2:
            self.p2p_enabled = loopt4_ext.enable_p2p_access(0, 1)
        else:
            self.p2p_enabled = False

        # Fuse [w_q, w_k, w_v, w_gate] -> single contiguous [8192, D] tensor per layer
        for lw in weights["layers"].values():
            if not hasattr(lw, "w_qkvg") or lw.w_qkvg is None:
                lw.w_qkvg = torch.cat([lw.w_q, lw.w_k, lw.w_v, lw.w_gate], dim=0).contiguous()
                lw.w_q = lw.w_k = lw.w_v = lw.w_gate = None
        torch.cuda.empty_cache()

        # Check if compiled extension supports GPU pos_buf (required for CUDA Graphs)
        self._has_pos_buf_attn = hasattr(loopt4_ext, "hybrid_gqa_attn_pos_buf")
        self.use_cuda_graph = bool(use_cuda_graph and self._has_pos_buf_attn)
        if use_cuda_graph and not self._has_pos_buf_attn:
            print("[LoopT4] hybrid_gqa_attn_pos_buf not found in loopt4_ext -> running Core eager "
                  "(fused QKVG + zero-copy block_buf active).")

        # Build the 48-step execution schedule: (visit_id, layer_idx, pass_idx)
        self.schedule: List[Tuple[int, int, int]] = []
        v = 0
        for l in range(0, cfg.core_start):
            self.schedule.append((v, l, 0)); v += 1
        self.core_visit_start = v
        for p in range(cfg.num_passes):
            for l in range(cfg.core_start, cfg.core_end):
                self.schedule.append((v, l, p)); v += 1
        self.core_visit_end = v
        for l in range(cfg.core_end, cfg.num_hidden_layers):
            self.schedule.append((v, l, 0)); v += 1

        self.core_tap_visits = [
            vid for vid in cfg.mtp_target_visits
            if self.core_visit_start <= vid < self.core_visit_end
        ]
        self.core_tap_index = {vid: idx for idx, vid in enumerate(self.core_tap_visits)}

        # Allocate 48 VisitStates on their owning GPU
        self.states: Dict[int, VisitState] = {}
        D = cfg.hidden_size
        for visit_id, l, _ in self.schedule:
            dev = torch.device(cfg.layer_device(l, self.num_gpus))
            is_swa = (l not in cfg.global_layer_ids)
            slots = cfg.window_size if is_swa else cfg.max_global_seq_len
            self.states[visit_id] = VisitState(
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

        self.dspark = (
            DSparkDrafter(cfg, self.dev_io, weights["embed_tokens"], weights["lm_head"])
            if enable_dspark else None
        )
        self.pos = 0

        # Preallocated contiguous block_buf caches keyed by batch/token count M
        self._io_block_bufs: Dict[int, torch.Tensor] = {}
        self._core_block_bufs: Dict[int, torch.Tensor] = {}
        self._core_tap_bufs: Dict[int, torch.Tensor] = {}

        # CUDA Graph cache for GPU 0 core loop keyed by M (only captured for M <= 16)
        self._core_graphs: Dict[int, Tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = {}

        # Reusable cross-GPU stream events (zero CPU sync)
        self._ev_io_to_core = torch.cuda.Event()
        self._ev_core_to_io = torch.cuda.Event()

    def reset_cache(self):
        self.pos = 0
        for st in self.states.values():
            st.k_cache.zero_()
            st.v_cache.zero_()
            st.k_sconv_state.zero_()
            st.v_sconv_state.zero_()
            st.attn_sconv_state.zero_()
            st.mlp_sconv_state.zero_()

    def _get_io_block_buf(self, M: int) -> torch.Tensor:
        buf = self._io_block_bufs.get(M)
        if buf is None:
            buf = torch.empty((10, M, self.cfg.hidden_size), device=self.dev_io, dtype=torch.float16)
            self._io_block_bufs[M] = buf
        return buf

    def _get_core_bufs(self, M: int) -> Tuple[torch.Tensor, torch.Tensor]:
        buf = self._core_block_bufs.get(M)
        if buf is None:
            buf = torch.empty((10, M, self.cfg.hidden_size), device=self.dev_core, dtype=torch.float16)
            taps = torch.empty((max(1, len(self.core_tap_visits)), self.cfg.hidden_size),
                               device=self.dev_core, dtype=torch.float16)
            self._core_block_bufs[M] = buf
            self._core_tap_bufs[M] = taps
        return self._core_block_bufs[M], self._core_tap_bufs[M]

    def _run_layer(
        self,
        lw: LayerWeights,
        st: VisitState,
        block_buf: torch.Tensor,   # [max_blocks, M, D], active slice is [:num_completed + 1]
        num_completed: int,        # running partial sum lives in-place at block_buf[num_completed]
        prev_msg: Optional[torch.Tensor],
        pass_idx: int,
        start_pos_int: Optional[int],
        pos_buf: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        """Executes 1 layer visit completely in-place on block_buf[num_completed]."""
        active_blocks = block_buf[:num_completed + 1]  # Contiguous view (zero copy!)
        partial = block_buf[num_completed]             # [M, D] view
        M = partial.size(0)

        use_msg = (prev_msg is not None and pass_idx > 0 and lw.attn_kappa is not None)
        x_norm, _ = loopt4_ext.fused_attnres_msg_rmsnorm(
            active_blocks,
            prev_msg if use_msg else None,
            lw.attn_q_depth,
            lw.attn_kappa[pass_idx - 1] if use_msg else None,
            lw.attn_gamma[pass_idx - 1] if use_msg else None,
            lw.attn_rms_w,
            self.cfg.rms_norm_eps,
        )

        # Single fused QKV+Gate projection instead of 4 separate GEMVs
        qkvg = F.linear(x_norm, lw.w_qkvg)
        q_raw, k_raw, v_raw, g_raw = qkvg.split([3072, 1024, 1024, 3072], dim=-1)

        q = q_raw.contiguous().view(M, 24, 128)
        k = loopt4_ext.short_conv4_step(k_raw.contiguous(), lw.k_sconv_w, st.k_sconv_state, False).view(M, 8, 128)
        v = loopt4_ext.short_conv4_step(v_raw.contiguous(), lw.v_sconv_w, st.v_sconv_state, False).view(M, 8, 128)
        g = g_raw.contiguous().view(M, 24, 128)

        if pos_buf is not None:
            attn_out = loopt4_ext.hybrid_gqa_attn_pos_buf(
                q, k, v, g,
                st.k_cache, st.v_cache,
                lw.rel_bias, lw.attn_sink,
                pos_buf, st.is_swa, self.cfg.window_size,
                self.cfg.qk_l2_clip_threshold, self.cfg.rms_norm_eps,
                self.cfg.log_scaling_n_floor, self.cfg.log_scaling_alpha,
            )
        else:
            attn_out = loopt4_ext.hybrid_gqa_attn(
                q, k, v, g,
                st.k_cache, st.v_cache,
                lw.rel_bias, lw.attn_sink,
                start_pos_int, st.is_swa, self.cfg.window_size,
                self.cfg.qk_l2_clip_threshold, self.cfg.rms_norm_eps,
                self.cfg.log_scaling_n_floor, self.cfg.log_scaling_alpha,
            )

        attn_proj = F.linear(attn_out.view(M, -1), lw.w_o)
        attn_proj = loopt4_ext.short_conv4_step(attn_proj, lw.attn_sconv_w, st.attn_sconv_state, False)
        partial.add_(attn_proj)

        # MLP sublayer (active_blocks already reflects the updated partial!)
        x_mlp_norm, _ = loopt4_ext.fused_attnres_msg_rmsnorm(
            active_blocks,
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
        partial.add_(moe_out)

        # Rank-64 cross-loop message for next pass (layers 6..17 only)
        if lw.msg_down is not None and lw.msg_up is not None:
            return F.linear(F.linear(partial, lw.msg_down), lw.msg_up)
        return None

    def _run_core_eager(
        self,
        packed_in: torch.Tensor,          # [2, M, D] on dev_core
        start_pos_int: Optional[int],
        pos_buf: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Executes all 36 core visits (layers 6..17 x 3 passes) locally on GPU 0."""
        M = packed_in.size(1)
        block_buf, taps_out = self._get_core_bufs(M)
        block_buf[:2].copy_(packed_in)
        num_core = 2
        block_buf[num_core].zero_()

        prev_pass_msgs: Dict[int, torch.Tensor] = {}
        for visit_id in range(self.core_visit_start, self.core_visit_end):
            _, l, p = self.schedule[visit_id]
            next_msg = self._run_layer(
                self.weights["layers"][l],
                self.states[visit_id],
                block_buf,
                num_core,
                prev_pass_msgs.get(l),
                p,
                start_pos_int,
                pos_buf,
            )
            prev_pass_msgs[l] = next_msg

            t_idx = self.core_tap_index.get(visit_id)
            if t_idx is not None:
                taps_out[t_idx].copy_(block_buf[num_core][-1])

            if ((visit_id + 1) % self.cfg.block_size_attnres) == 0:
                num_core += 1
                if (visit_id + 1) < self.core_visit_end:
                    block_buf[num_core].zero_()

        # Completed core blocks live at block_buf[2:8] ([6, M, D])
        return block_buf[2:num_core], taps_out

    def _capture_core_graph(self, M: int):
        """State-preserving static CUDA Graph capture for GPU 0's 36-visit core loop."""
        with torch.cuda.device(self.dev_core):
            D = self.cfg.hidden_size
            s_in  = torch.zeros((2, M, D), device=self.dev_core, dtype=torch.float16)
            s_pos = torch.zeros((1,), device=self.dev_core, dtype=torch.int32)

            # Save live conv states + slots [0:M] of core KV caches so capture is 100% side-effect free
            saved_kv = []
            for vid in range(self.core_visit_start, self.core_visit_end):
                st = self.states[vid]
                st.snapshot_conv()
                saved_kv.append((st.k_cache[:M].clone(), st.v_cache[:M].clone()))

            side_stream = torch.cuda.Stream(device=self.dev_core)
            side_stream.wait_stream(torch.cuda.current_stream(self.dev_core))
            with torch.cuda.stream(side_stream):
                for _ in range(3):
                    _ = self._run_core_eager(s_in, None, s_pos)
            torch.cuda.current_stream(self.dev_core).wait_stream(side_stream)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=side_stream):
                s_blocks_out, s_taps_out = self._run_core_eager(s_in, None, s_pos)
            torch.cuda.current_stream(self.dev_core).wait_stream(side_stream)

            # Restore live conv states and KV slices
            for idx, vid in enumerate(range(self.core_visit_start, self.core_visit_end)):
                st = self.states[vid]
                st.restore_conv()
                st.k_cache[:M].copy_(saved_kv[idx][0])
                st.v_cache[:M].copy_(saved_kv[idx][1])

            self._core_graphs[M] = (graph, s_in, s_pos, s_blocks_out, s_taps_out)

    def _run_core(self, packed_from_io: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Event-ordered Hop 1 -> Core (Graphed for M<=16, Eager for prefill)."""
        M = packed_from_io.size(1)

        # Order dev_core behind dev_io without stalling the CPU
        with torch.cuda.device(self.dev_io):
            self._ev_io_to_core.record(torch.cuda.current_stream(self.dev_io))

        with torch.cuda.device(self.dev_core):
            torch.cuda.current_stream(self.dev_core).wait_event(self._ev_io_to_core)

            if self.use_cuda_graph and M <= 16:
                if M not in self._core_graphs:
                    self._capture_core_graph(M)
                graph, s_in, s_pos, s_blocks_out, s_taps_out = self._core_graphs[M]
                s_in.copy_(packed_from_io, non_blocking=True)
                s_pos.fill_(start_pos)
                graph.replay()
                blocks_out, taps_out = s_blocks_out, s_taps_out
            else:
                packed_core = packed_from_io.to(self.dev_core, non_blocking=True)
                blocks_out, taps_out = self._run_core_eager(packed_core, start_pos, None)

            self._ev_core_to_io.record(torch.cuda.current_stream(self.dev_core))

        return blocks_out, taps_out

    @torch.inference_mode()
    def forward_tokens(self, token_ids: torch.Tensor, start_pos: int) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Prelude (GPU 1) -> Hop 1 -> 3x Looped Core (GPU 0) -> Hop 2 -> Coda + Logits (GPU 1).
        Chunked to M <= 256 to guarantee shared-memory safety in grouped MoE kernels.
        """
        M = int(token_ids.numel())
        assert M <= 256, f"Chunk forward_tokens to M <= 256 (got M={M})"

        cfg = self.cfg
        dspark_taps: Dict[int, torch.Tensor] = {}

        # ---- Stage 1: Prelude (Layers 0..5) on GPU 1 ----
        with torch.cuda.device(self.dev_io):
            toks_io = token_ids.to(self.dev_io, non_blocking=True)
            block_buf_io = self._get_io_block_buf(M)
            block_buf_io[0].copy_(self.weights["embed_tokens"][toks_io])
            num_io = 1
            block_buf_io[num_io].zero_()

            for visit_id in range(0, cfg.core_start):
                _, l, p = self.schedule[visit_id]
                self._run_layer(
                    self.weights["layers"][l], self.states[visit_id],
                    block_buf_io, num_io, None, p, start_pos, None
                )
                if visit_id in cfg.mtp_target_visits:
                    dspark_taps[visit_id] = block_buf_io[num_io][-1:].clone()
            num_io = 2  # Completed [h0, B1]
            packed_to_core = block_buf_io[:2]

        # ---- Stage 2: Hop 1 + 36-Visit Looped Core on GPU 0 ----
        core_blocks_out, core_taps_out = self._run_core(packed_to_core, start_pos)

        # ---- Stage 3: Hop 2 + Coda (Layers 18..23) + Logits on GPU 1 ----
        with torch.cuda.device(self.dev_io):
            torch.cuda.current_stream(self.dev_io).wait_event(self._ev_core_to_io)

            n_core_blocks = core_blocks_out.size(0)  # 6
            block_buf_io[num_io:num_io + n_core_blocks].copy_(core_blocks_out, non_blocking=True)
            num_io += n_core_blocks  # 8

            if self.core_tap_visits:
                core_taps_io = core_taps_out.to(self.dev_io, non_blocking=True)
                for idx, vid in enumerate(self.core_tap_visits):
                    dspark_taps[vid] = core_taps_io[idx:idx + 1]

            block_buf_io[num_io].zero_()
            for visit_id in range(self.core_visit_end, cfg.total_visits):
                _, l, p = self.schedule[visit_id]
                self._run_layer(
                    self.weights["layers"][l], self.states[visit_id],
                    block_buf_io, num_io, None, p, start_pos, None
                )
                if visit_id in cfg.mtp_target_visits:
                    dspark_taps[visit_id] = block_buf_io[num_io][-1:].clone()

            # Final FP32-safe RMSNorm + LM Head + Logit Softcap (30.0)
            final_h = block_buf_io[num_io - 1] + block_buf_io[num_io]
            xf = final_h.float()
            x_norm = (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + cfg.rms_norm_eps)).to(torch.float16)
            x_norm.mul_(self.weights["final_norm"])
            logits = F.linear(x_norm, self.weights["lm_head"]).float()
            logits = cfg.logit_softcap * torch.tanh(logits / cfg.logit_softcap)

            ordered_taps = [dspark_taps.get(v, final_h[-1:]) for v in cfg.mtp_target_visits]
            return logits, ordered_taps

    @torch.inference_mode()
    def generate(self, prompt_ids: List[int], max_new_tokens: int = 64, use_dspark: bool = True) -> List[int]:
        """End-to-end chunked prefill + greedy decode (with optional DSpark speculation)."""
        self.reset_cache()
        prompt_t = torch.tensor(prompt_ids, dtype=torch.long)

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
                for st in self.states.values():
                    st.snapshot_conv()

                k = min(self.cfg.mtp_draft_steps, max_new_tokens - len(generated))
                draft = self.dspark.draft_k_tokens(next_tok, taps, self.pos, k)
                verify_ids = torch.tensor([next_tok] + draft, dtype=torch.long)

                v_logits, taps = self.forward_tokens(verify_ids, self.pos)
                preds = torch.argmax(v_logits, dim=-1).tolist()

                accepted = 0
                for j, d_tok in enumerate(draft):
                    if preds[j] == d_tok:
                        generated.append(d_tok)
                        accepted += 1
                    else:
                        generated.append(preds[j])
                        break
                else:
                    generated.append(preds[k])

                if accepted < k:
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
