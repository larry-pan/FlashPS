"""FLUX inpaint partial-sampling latency by mask COVERAGE BUCKET x batch x resolution (1 GPU).

Clusters masks into 0.05-wide coverage buckets (by white_px_frac) and evaluates latency across
batch sizes and resolutions. For each (resolution, batch, coverage bucket):

  full_ms   -- MEASURED full/normal forward at that resolution+batch (use_cached_kv=False).
               Carries the resolution axis (bigger image = more tokens = slower).
  partial_ms -- FlashPS partial-sampling latency for a mask of that coverage. Its compute
               scales with active tokens = coverage x full_tokens(resolution).
  speedup_vs_full = full_ms / partial_ms.

IMPORTANT (vendored-code constraint): the cached path's KV buffer is hardcoded to 1024^2 (4096
tokens) in attention_processor_short.py:1292, so partial latency is only MEASURED at 1024^2. At
512^2 / 768^2 partial_ms and speedup are left n/a (only full_ms is reported there). Also the
partial timing path is stubbed-KV, so its *image* is meaningless -- we only take the *latency*
(which is real).

token math: 1 token = 16x16 px patch. full_tokens = (size//16)^2  (512->1024, 768->2304,
1024->4096). Partial buffer constraint: 512 (text) + active_tokens <= 4096.

Batch>1 needs per-(resolution,batch) reference latents; we precompute them (a full run at that
resolution and batch). If the cached+batch forward still fails it is recorded FAILED, not fatal.

Run on the GPU box, from scheduler/ (auto-picks a free GPU):
  python flux_bucket_sweep.py --model <snapshot> --out-dir out_buckets \
    --sizes 512,768,1024 --batch-sizes 1,2,4 --repeats 3 --warmup 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import types

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flux_timing_test as ft   # noqa: E402  (_make_edit_config, TEXT_SEQLEN)
import flux_tryon_demo as demo  # noqa: E402  (_torso_mask, _full_mask, _neutral_image)
import flux_mask_sweep as fms   # noqa: E402  (_time_forward, _white_frac, _setup_visible_devices, _dir_masks)

TEXT_SEQLEN = ft.TEXT_SEQLEN   # 512
CACHE_BUF = 4096               # hardcoded KV buffer (1024^2 tokens)
BUFFER_RES = 1024              # resolution the cache buffer matches -> partial measurable here only


def _full_tokens(size):
    return (size // 16) ** 2


def _active_tokens(coverage, size):
    return max(1, round(coverage * _full_tokens(size)))


def _coverages(min_c, max_c, step):
    out, c = [], min_c
    while c <= max_c + 1e-9:
        out.append(round(c, 4))
        c += step
    return out


# --- timing (reuses fms._time_forward for the batched forward) --------------------------------

def _precompute(pipe, image, mask, EditConfig, *, size, steps, batch, prompt, folder):
    """Full reference run at (size, batch) with save_latents -> {latents_i} (batch-B tensors)."""
    import torch
    os.makedirs(folder, exist_ok=True)
    cfg = ft._make_edit_config(EditConfig, steps=steps, prompt=prompt, use_cached_kv=False,
                               generated_seqlen=_full_tokens(size), save_latents=True,
                               cached_latents_folder=folder)
    cfg.batch_size = batch
    fms._time_forward(pipe, image, mask, cfg, size=size, steps=steps, strength=1.0,
                      batch=batch, repeats=1, warmup=0)  # one call; saves latents_i.pt
    return {f"latents_{i}": torch.load(os.path.join(folder, f"latents_{i}.pt"))
            for i in range(steps)}


def _measure_full(pipe, image, mask, EditConfig, args, *, size, batch):
    cfg = ft._make_edit_config(EditConfig, steps=args.steps, prompt=args.edit_prompt,
                               use_cached_kv=False, generated_seqlen=_full_tokens(size))
    cfg.batch_size = batch
    try:
        return fms._time_forward(pipe, image, mask, cfg, size=size, steps=args.steps,
                                 strength=1.0, batch=batch, repeats=args.repeats,
                                 warmup=args.warmup), "ok"
    except Exception as e:  # noqa: BLE001
        return None, f"FAILED: {type(e).__name__}"


def _measure_partial(pipe, image, mask, EditConfig, args, *, size, batch, active_tokens,
                     cached_latents, folder):
    cfg = ft._make_edit_config(EditConfig, steps=args.steps, prompt=args.edit_prompt,
                               use_cached_kv=True, generated_seqlen=active_tokens,
                               cached_latents_folder=folder)
    cfg.cached_latents = cached_latents
    cfg.batch_size = batch
    try:
        return fms._time_forward(pipe, image, mask, cfg, size=size, steps=args.steps,
                                 strength=1.0, batch=batch, repeats=args.repeats,
                                 warmup=args.warmup), "ok"
    except Exception as e:  # noqa: BLE001  (cached+batch>1 untested -> record & move on)
        return None, f"FAILED: {type(e).__name__}"


def _mkrow(size, batch, cov, full_ms, partial_ms, speedup, status):
    return {"resolution": size, "batch_size": batch, "coverage": cov, "full_ms": full_ms,
            "partial_ms": partial_ms, "speedup_vs_full": speedup, "status": status}


def _write_report(rows, csv_path):
    rows.sort(key=lambda r: (r["resolution"], r["batch_size"], r["coverage"]))
    print("\n===== FLUX partial-sampling latency by coverage bucket x batch x resolution =====")
    hdr = f"{'res':>5}{'B':>3}{'cov':>6}{'full_ms':>9}{'partial_ms':>11}{'speedup':>9}  status"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        fm = f"{r['full_ms']:9.1f}" if r["full_ms"] else f"{'n/a':>9}"
        pm = f"{r['partial_ms']:11.1f}" if r["partial_ms"] else f"{'n/a':>11}"
        sp = f"{r['speedup_vs_full']:8.2f}x" if r["speedup_vs_full"] else f"{'n/a':>9}"
        print(f"{r['resolution']:>5}{r['batch_size']:>3}{r['coverage']:>6.2f}{fm}{pm}{sp}  {r['status']}")

    cols = ["resolution", "batch_size", "coverage", "full_ms", "partial_ms", "speedup_vs_full", "status"]
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            def _fmt(v):
                if v is None:
                    return ""
                return f"{v:.4f}" if isinstance(v, float) else str(v)
            f.write(",".join(_fmt(r[c]) for c in cols) + "\n")
    print(f"\n[bucket] wrote {csv_path}")
    print("[bucket] full_ms measured at all resolutions; partial_ms/speedup MEASURED only at "
          "1024^2 (cache buffer hardcoded to 1024^2) -> n/a at 512/768. speedup=full_ms/partial_ms.")


def main():
    p = argparse.ArgumentParser(description="FLUX partial-sampling latency by coverage bucket x "
                                            "batch x resolution (single GPU).")
    p.add_argument("--model", default="black-forest-labs/FLUX.1-schnell")
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--edit-prompt", default=demo.EDIT_PROMPT)
    p.add_argument("--out-dir", default="/tmp/flux_bucket_sweep")
    p.add_argument("--sizes", default="512,768,1024", help="comma-sep resolutions")
    p.add_argument("--batch-sizes", default="1", help="comma-sep batch sizes, e.g. 1,2,4")
    p.add_argument("--min-cov", type=float, default=0.05)
    p.add_argument("--max-cov", type=float, default=0.85)
    p.add_argument("--bucket-step", type=float, default=0.05)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--devices", default=None, help="explicit physical GPU indices (CUDA_VISIBLE_DEVICES)")
    p.add_argument("--min-free-gb", type=float, default=38.0)
    p.add_argument("--csv", default=None, help="default <out-dir>/flux_bucket_grid.csv")
    args = p.parse_args()

    args.gpu_counts = "1"  # single-GPU sweep; reuse fms free-GPU auto-selection
    fms._setup_visible_devices(args)  # MUST run before torch initializes CUDA

    import torch
    from diffusers import FluxInpaintPipeline
    from configs.edit_config import EditConfig

    sizes = [int(s) for s in args.sizes.split(",") if s.strip()]
    batches = [int(b) for b in args.batch_sizes.split(",") if b.strip()]
    coverages = _coverages(args.min_cov, args.max_cov, args.bucket_step)
    os.makedirs(args.out_dir, exist_ok=True)
    csv_path = args.csv or os.path.join(args.out_dir, "flux_bucket_grid.csv")

    print(f"[bucket] sizes={sizes} batches={batches} coverages={coverages}\n")

    print(f"[bucket] loading {args.model} on cuda:0 ...")
    t0 = time.time()
    kw = {"torch_dtype": torch.bfloat16}
    if args.cache_dir:
        kw["cache_dir"] = args.cache_dir
    if os.path.isdir(args.model):
        kw["local_files_only"] = True
    pipe = FluxInpaintPipeline.from_pretrained(args.model, **kw).to("cuda:0")
    print(f"[bucket] model ready in {time.time() - t0:.1f}s\n")

    rows = []
    for size in sizes:
        image = demo._neutral_image(size)
        mask = demo._full_mask(size)
        ftok = _full_tokens(size)
        measurable = (size == BUFFER_RES)  # partial only trustworthy where buffer matches
        for batch in batches:
            cached = None
            if measurable:
                folder = os.path.join(args.out_dir, "cache", f"{size}_{batch}")
                cached = _precompute(pipe, image, mask, EditConfig, size=size, steps=args.steps,
                                     batch=batch, prompt=args.edit_prompt, folder=folder)
            full_ms, fstatus = _measure_full(pipe, image, mask, EditConfig, args, size=size, batch=batch)
            for cov in coverages:
                active = _active_tokens(cov, size)
                if measurable and cached is not None and full_ms and TEXT_SEQLEN + active <= CACHE_BUF:
                    partial_ms, pstatus = _measure_partial(
                        pipe, image, mask, EditConfig, args, size=size, batch=batch,
                        active_tokens=active, cached_latents=cached, folder=folder)
                    if partial_ms:
                        rows.append(_mkrow(size, batch, cov, full_ms, partial_ms,
                                           full_ms / partial_ms, "ok"))
                    else:
                        rows.append(_mkrow(size, batch, cov, full_ms, None, None, pstatus))
                else:
                    # partial not measurable here -> n/a (only 1024^2 matches the cache buffer)
                    if not full_ms:
                        status = fstatus
                    elif TEXT_SEQLEN + active > CACHE_BUF:
                        status = "n/a (buffer overflow)"
                    else:
                        status = "n/a (partial 1024-only)"
                    rows.append(_mkrow(size, batch, cov, full_ms, None, None, status))
            tag = "measured" if measurable else "n/a"
            fm = f"{full_ms:.1f}ms" if full_ms else fstatus
            print(f"[bucket] res={size} B={batch} full={fm} partial={tag} "
                  f"({len(coverages)} buckets)")

    _write_report(rows, csv_path)


if __name__ == "__main__":
    main()
