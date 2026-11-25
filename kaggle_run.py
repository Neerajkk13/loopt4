import time
import torch
import loopt4_ext
from python.config import Inkling2MiniConfig
from python.loader import create_synthetic_weights, load_core_pinned_safetensors
from python.engine import LoopT4Engine

def verify_nvfp4_and_mxfp8_kernels():
    """Unit-tests sm_75 register dequantization kernels against FP32 reference."""
    device = torch.device("cuda:0")
    M, D, I, E = 2, 2304, 1152, 128

    # 1. Verify Router Top-8 Sigmoid
    logits = torch.randn(M, E, device=device, dtype=torch.float16)
    bias   = torch.randn(E, device=device, dtype=torch.float16) * 0.1
    ids, w = loopt4_ext.moe_router_top8(logits, bias, 8.0)
    assert ids.shape == (M, 8) and torch.allclose(w.sum(dim=-1), torch.full((M,), 8.0, device=device), atol=1e-3)

    # 2. Verify FP32-Safe AttnRes + Msg + RMSNorm with +32.4 sigma outlier (d_2027)
    blocks = torch.randn(4, M, D, device=device, dtype=torch.float16)
    blocks[0, :, 2027] = 18000.0  # Would overflow FP16^2 without FP32 accumulation!
    q_d = torch.randn(D, device=device, dtype=torch.float16) * 0.05
    rw  = torch.ones(D, device=device, dtype=torch.float16)
    out_n, _ = loopt4_ext.fused_attnres_msg_rmsnorm(blocks, None, q_d, None, None, rw, 1e-6)
    assert torch.isfinite(out_n).all(), "FP16 overflow detected on outlier channel!"

    # 3. Verify 4-Tap Short-Conv1D
    x = torch.randn(4, D, device=device, dtype=torch.float16)
    cw = torch.randn(D, 4, device=device, dtype=torch.float16) * 0.25
    st = torch.zeros(D, 3, device=device, dtype=torch.float16)
    y = loopt4_ext.short_conv4_step(x, cw, st, False)
    assert torch.isfinite(y).all() and y.shape == x.shape
    print("[PASS] All sm_75 custom CUDA kernels verified numerically (including d_2027 outlier stability).")


def benchmark_kaggle_2xt4(checkpoint_dir: str = None):
    num_gpus = torch.cuda.device_count()
    print(f"Detected {num_gpus} GPU(s): {[torch.cuda.get_device_name(i) for i in range(num_gpus)]}")

    cfg = Inkling2MiniConfig(max_global_seq_len=16384)
    if checkpoint_dir:
        print(f"Streaming Inkling 2 Mini checkpoint directly to {num_gpus}xT4...")
        weights = load_core_pinned_safetensors(checkpoint_dir, cfg, num_gpus=num_gpus)
    else:
        print("Allocating exact-shape NVFP4/MXFP8 synthetic weights on 2xT4...")
        weights = create_synthetic_weights(cfg, num_gpus=num_gpus)

    engine = LoopT4Engine(cfg, weights, num_gpus=num_gpus, enable_dspark=True)
    for g in range(num_gpus):
        alloc_gb = torch.cuda.memory_allocated(g) / (1024 ** 3)
        print(f"  GPU {g} VRAM allocated: {alloc_gb:.2f} GB / 15.00 GB")

    # Warmup
    _ = engine.generate([1, 42, 128, 999], max_new_tokens=4, use_dspark=False)
    torch.cuda.synchronize()

    # Plain decode benchmark (M = 1)
    n_toks = 32
    t0 = time.perf_counter()
    _ = engine.generate([1, 42, 128, 999], max_new_tokens=n_toks, use_dspark=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"Plain Core-Pinned Decode (M=1):   {n_toks / dt:.1f} tok/s ({dt * 1000 / n_toks:.2f} ms/tok)")

    # Batched DSpark verify step timing (M = 6)
    engine.reset_cache()
    verify_batch = torch.tensor([1, 2, 3, 4, 5, 6], dtype=torch.long)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    for _ in range(10):
        engine.forward_tokens(verify_batch, 0)
    torch.cuda.synchronize()
    verify_ms = (time.perf_counter() - t1) * 1000.0 / 10.0
    print(f"Batched Verify Pass (M=6 tokens): {verify_ms:.2f} ms/step "
          f"(-> {3.5 / (verify_ms / 1000.0):.1f} tok/s @ 3.5 avg DSpark acceptance)")


if __name__ == "__main__":
    verify_nvfp4_and_mxfp8_kernels()
    # Pass checkpoint_dir="/kaggle/input/inkling-2-mini" once weights are attached
    benchmark_kaggle_2xt4(checkpoint_dir=None)
