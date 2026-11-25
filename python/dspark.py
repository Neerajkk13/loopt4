import torch
import torch.nn.functional as F
import loopt4_ext
from typing import List
from .config import Inkling2MiniConfig

class DSparkDrafter:
    """
    3-layer dense DSpark speculator (mtp.safetensors) co-located on GPU 1
    with Embeddings and LM Head -> zero PCIe hops during K-step drafting.
    """
    def __init__(self, cfg: Inkling2MiniConfig, device: torch.device, embed_w: torch.Tensor, lm_head_w: torch.Tensor):
        self.cfg = cfg
        self.device = device
        self.embed_w = embed_w
        self.lm_head_w = lm_head_w
        D = cfg.hidden_size
        I = cfg.mtp_intermediate_size
        R = cfg.mtp_markov_rank

        # Tap fusion projection: 4 target taps -> hidden_size
        self.tap_proj = torch.randn(D, 4 * D, device=device, dtype=torch.float16) * 0.01
        # RNN Markov state projector (rank 256)
        self.rnn_in  = torch.randn(R, 2 * D, device=device, dtype=torch.float16) * 0.02
        self.rnn_out = torch.randn(D, R, device=device, dtype=torch.float16) * 0.02

        self.layers = []
        self.k_cache = []
        self.v_cache = []
        for _ in range(cfg.mtp_num_layers):
            self.layers.append({
                "norm1": torch.ones(D, device=device, dtype=torch.float16),
                "w_q":   torch.randn(24 * 128, D, device=device, dtype=torch.float16) * 0.02,
                "w_k":   torch.randn(8 * 128, D, device=device, dtype=torch.float16) * 0.02,
                "w_v":   torch.randn(8 * 128, D, device=device, dtype=torch.float16) * 0.02,
                "w_o":   torch.randn(D, 24 * 128, device=device, dtype=torch.float16) * 0.02,
                "sink":  torch.full((24,), -4.0, device=device, dtype=torch.float32),
                "norm2": torch.ones(D, device=device, dtype=torch.float16),
                "w_gate": torch.randn(I, D, device=device, dtype=torch.float16) * 0.02,
                "w_up":   torch.randn(I, D, device=device, dtype=torch.float16) * 0.02,
                "w_down": torch.randn(D, I, device=device, dtype=torch.float16) * 0.02,
            })
            self.k_cache.append(torch.zeros(cfg.window_size, 8, 128, device=device, dtype=torch.float16))
            self.v_cache.append(torch.zeros(cfg.window_size, 8, 128, device=device, dtype=torch.float16))

    def _rms_norm(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        inv = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.cfg.rms_norm_eps)
        return (xf * inv).to(torch.float16) * w

    @torch.inference_mode()
    def draft_k_tokens(
        self,
        last_token: int,
        target_taps: List[torch.Tensor], # 4 tensors of [1, D] on GPU 1
        start_pos: int,
        k_steps: int,
    ) -> List[int]:
        """Runs K autoregressive draft steps locally on GPU 1."""
        fused_ctx = F.linear(torch.cat(target_taps, dim=-1), self.tap_proj) # [1, D]
        rnn_h = fused_ctx
        cur_tok = last_token
        draft_ids: List[int] = []

        for step in range(k_steps):
            emb = self.embed_w[cur_tok].unsqueeze(0) # [1, D]
            # RNN Markov hidden update
            z = F.silu(F.linear(torch.cat([emb, rnn_h], dim=-1), self.rnn_in))
            x = emb + F.linear(z, self.rnn_out)

            pos = start_pos + step
            for idx, ly in enumerate(self.layers):
                xn = self._rms_norm(x, ly["norm1"])
                q = F.linear(xn, ly["w_q"]).view(1, 24, 128)
                k = F.linear(xn, ly["w_k"]).view(1, 8, 128)
                v = F.linear(xn, ly["w_v"]).view(1, 8, 128)
                attn_out = loopt4_ext.hybrid_gqa_attn(
                    q, k, v, None, self.k_cache[idx], self.v_cache[idx],
                    None, ly["sink"], pos, True, self.cfg.window_size,
                    self.cfg.qk_l2_clip_threshold, self.cfg.rms_norm_eps,
                    self.cfg.log_scaling_n_floor, self.cfg.log_scaling_alpha,
                )
                x = x + F.linear(attn_out.view(1, -1), ly["w_o"])

                xn2 = self._rms_norm(x, ly["norm2"])
                mlp = F.silu(F.linear(xn2, ly["w_gate"])) * F.linear(xn2, ly["w_up"])
                x = x + F.linear(mlp, ly["w_down"])

            rnn_h = x
            logits = F.linear(x, self.lm_head_w)
            next_tok = int(torch.argmax(logits, dim=-1).item())
            draft_ids.append(next_tok)
            cur_tok = next_tok

        return draft_ids
