"""
Memory diagnostic: replicate baseline eval and report VRAM at each stage.

Run from llm_training/ root:
    /home/bmartins/anaconda3/envs/llm-scratch/bin/python src/mem_diag.py
"""
import math
import sys
import torch
import torch.nn as nn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from gpt_model import GPTModel, GPT_CONFIG_774M as CONFIG


def gb(b: int) -> float:
    return b / 1024**3


def report(stage: str):
    torch.cuda.synchronize()
    alloc = torch.cuda.memory_allocated()
    reserv = torch.cuda.memory_reserved()
    peak = torch.cuda.max_memory_allocated()
    print(f"[{stage:30s}] alloc={gb(alloc):5.2f} GB | reserved={gb(reserv):5.2f} GB | peak={gb(peak):5.2f} GB")


def main():
    device = torch.device("cuda")
    dtype = torch.bfloat16
    BATCH = 2
    SEQ = CONFIG["context_length"]
    VOCAB = CONFIG["vocab_size"]

    print(f"torch: {torch.__version__}")
    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"total VRAM: {gb(torch.cuda.get_device_properties(0).total_memory):.2f} GB")
    print(f"model: GPT-2 Large (CONFIG={CONFIG})\n")

    torch.cuda.reset_peak_memory_stats()
    report("start")

    # --- 1. Build model on CPU, move to GPU ---
    torch.manual_seed(123)
    model = GPTModel(CONFIG)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nparam count: {n_params:,}  ({n_params * 4 / 1024**3:.2f} GB fp32)\n")

    model = model.to(device)
    report("after model.to(device)")

    # --- 2. Single forward pass under no_grad + autocast (mimics baseline eval) ---
    model.eval()
    x = torch.randint(0, VOCAB, (BATCH, SEQ), device=device, dtype=torch.long)
    y = torch.randint(0, VOCAB, (BATCH, SEQ), device=device, dtype=torch.long)
    report("after dummy x,y on GPU")

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=dtype):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    print(f"\nloss = {loss.item():.4f}  (uniform ref: {math.log(VOCAB):.3f})")
    report("after 1 forward (no_grad)")
    del logits, loss

    # Repeat 5 forwards to see if cache grows
    for i in range(5):
        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
        del logits, loss
    report("after 5 more forwards")

    # --- 3. Now switch to TRAINING mode (grads on) — this is what blows up ---
    print("\n--- switching to training mode (grads on) ---")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    report("after empty_cache")

    model.train()
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    report("after 1 forward (with grad)")

    loss.backward()
    report("after backward")

    # --- 4. Add optimizer (full training state) ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-4, fused=True)
    optimizer.step()  # this materializes m, v in optimizer state
    optimizer.zero_grad(set_to_none=True)
    report("after optimizer.step (m,v init)")

    # second forward+backward+step under full training conditions
    with torch.autocast(device_type="cuda", dtype=dtype):
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), y.flatten())
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    report("after 2nd train step")


if __name__ == "__main__":
    main()
