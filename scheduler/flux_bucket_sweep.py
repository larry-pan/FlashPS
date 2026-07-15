"""FLUX inpaint partial-sampling latency by mask COVERAGE BUCKET x batch x resolution (1 GPU).

Clusters masks into 0.05-wide coverage buckets (by white_px_frac) and evaluates latency across
batch sizes and resolutions. For each (resolution, batch, coverage bucket):

  full_ms   -- MEASURED full/normal forward at that resolution+batch (use_cached_kv=False).
               Carries the resolution axis (bigger image = more tokens = slower).
  partial_ms -- FlashPS partial-sampling latency for a mask of that coverage. Its compute
               scales with active tokens = coverage x full_tokens(resolution).
  speedup_vs_full = full_ms / partial_ms.

The cached KV buffer is now RESOLUTION-AWARE (edit_config.image_seqlen = full image tokens,
set from the real mask in the pipeline; attention_processor_short.py:1292), so partial latency is
MEASURED at every resolution. Per-resolution cap: text(512) + active_tokens <= full_tokens(size)
(so at 512^2 coverage <= ~0.5, 768^2 <= ~0.78, 1024^2 <= ~0.875); buckets above the cap show
n/a (buffer overflow). The partial timing path is stubbed-KV, so its *image* is meaningless -- we
only take the *latency* (which is real).

token math: 1 token = 16x16 px patch. full_tokens = (size//16)^2  (512->1024, 768->2304,
1024->4096). Partial buffer constraint: 512 (text) + active_tokens <= 4096.

Batch axis: full_ms is measured at every batch (shows batch scaling). partial_ms is measured at
every batch too, via TWO cache paths:
  * batch=1 -> fixed-length cache path (_setup_fixed_length_caching; validated on GPU).
  * batch>1 -> variable-length batched path (test_varlen=True, _setup_variable_length_caching;
    builds per-item offset indices + flash_attn_varlen_func). Cached latents are precomputed once
    per resolution at batch=1 and replicated per item by the pipeline (pipeline:1537). The batched
    query buffer was made resolution-aware (max_batch*(512+image_seqlen), grows across cells,
    attention_processor_short.py:~1474), so batched partial works at ANY resolution -- bounded only
    by GPU memory (large res*batch cells that OOM are caught as FAILED).
NOTE: the standalone varlen path is untested end-to-end (the repo exercises it via the server/async
path), so batch>1 cells may come back FAILED; if so, full_ms is still recorded and the cell is
marked rather than crashing the sweep.

Run on the GPU box, from scheduler/ (auto-picks a free GPU):
  python flux_bucket_sweep.py --model <snapshot> --out-dir out_buckets \
    --sizes 512,768,1024 --batch-sizes 1,2,4 --repeats 3 --warmup 1
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import traceback
import types

# FLUX_DEBUG=1 -> print the full traceback when a cell fails (and flux_mask_sweep stops swallowing
# the vendored per-block shape prints), so shape/index errors are visible instead of "FAILED".
_DEBUG = bool(os.environ.get("FLUX_DEBUG"))

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flux_timing_test as ft   # noqa: E402  (_make_edit_config, TEXT_SEQLEN)
import flux_tryon_demo as demo  # noqa: E402  (_torso_mask, _full_mask, _neutral_image)
import flux_mask_sweep as fms   # noqa: E402  (_time_forward, _white_frac, _setup_visible_devices, _dir_masks)

TEXT_SEQLEN = ft.TEXT_SEQLEN   # 512
# Cache buffer is now resolution-aware (edit_config.image_seqlen = full image tokens), so partial
# is measurable at every resolution. Per-resolution cap: text(512) + active <= full_tokens(size).


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

def _precompute(pipe, image, mask, EditConfig, *, size, steps, prompt, folder):
    """Full reference run at (size, batch=1) with save_latents -> {latents_i} (single-item tensors).

    ALWAYS batch=1: the cache path repeats these per batch item
    (pipeline_flux_inpaint.py:1537 `cached_noise_pred.repeat(batch_size, 1, 1)`), so the stored
    latents must be a single item or the repeat/view math is wrong at batch>1.
    """
    import torch
    os.makedirs(folder, exist_ok=True)
    cfg = ft._make_edit_config(EditConfig, steps=steps, prompt=prompt, use_cached_kv=False,
                               generated_seqlen=_full_tokens(size), save_latents=True,
                               cached_latents_folder=folder)
    cfg.batch_size = 1
    fms._time_forward(pipe, image, mask, cfg, size=size, steps=steps, strength=1.0,
                      batch=1, repeats=1, warmup=0)  # one call; saves latents_i.pt
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
        if _DEBUG:
            traceback.print_exc()
        return None, f"FAILED: {type(e).__name__}"


def _measure_partial(pipe, image, mask, EditConfig, args, *, size, batch, active_tokens,
                     cached_latents, folder, varlen, max_batch):
    """Partial-sampling latency. batch=1 uses the fixed-length cache path (validated); batch>1
    uses the variable-length (batched) path via test_varlen=True. max_batch pre-sizes the
    persistent _query buffer to the largest batch in the sweep."""
    cfg = ft._make_edit_config(EditConfig, steps=args.steps, prompt=args.edit_prompt,
                               use_cached_kv=True, generated_seqlen=active_tokens,
                               cached_latents_folder=folder, test_varlen=varlen,
                               max_batch_size=max_batch)
    cfg.cached_latents = cached_latents
    cfg.batch_size = batch
    try:
        return fms._time_forward(pipe, image, mask, cfg, size=size, steps=args.steps,
                                 strength=1.0, batch=batch, repeats=args.repeats,
                                 warmup=args.warmup), "ok"
    except Exception as e:  # noqa: BLE001  (varlen batched is untested standalone -> record & move on)
        if _DEBUG:
            traceback.print_exc()
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
    print("[bucket] full_ms measured at all resolutions & batches. partial_ms: batch=1 via the "
          "fixed-length cache path (validated); batch>1 via the variable-length batched path "
          "(test_varlen=True) with a resolution-aware query buffer -> works at any resolution "
          "(large res*batch may OOM -> FAILED). speedup=full_ms/partial_ms. Buckets past "
          "512+active>full_tokens(size) are n/a (buffer overflow): ~cov>0.5 @512, >0.75 @768, "
          ">0.85 @1024.")


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

    max_batch = max(batches)  # pre-size the persistent varlen _query buffer for the whole sweep
    # The batched varlen _query buffer is now resolution-aware (max_batch*(512+image_seqlen),
    # attention_processor_short.py:~1474) and grows across cells, so batched partial works at ANY
    # resolution -- no 1024^2 cap. Sizes are still bounded only by GPU memory (caught as FAILED).

    rows = []
    for size in sizes:
        image = demo._neutral_image(size)
        mask = demo._full_mask(size)
        ftok = _full_tokens(size)
        # Precompute cached latents ONCE per resolution at batch=1 (the cache path repeats them
        # per batch item; storing batch-B latents would break the repeat/view math at batch>1).
        folder = os.path.join(args.out_dir, "cache", f"{size}")
        cached = _precompute(pipe, image, mask, EditConfig, size=size, steps=args.steps,
                             prompt=args.edit_prompt, folder=folder)
        for batch in batches:
            full_ms, fstatus = _measure_full(pipe, image, mask, EditConfig, args, size=size, batch=batch)
            varlen = batch > 1  # batch=1 -> fixed-length (validated); batch>1 -> variable-length
            for cov in coverages:
                active = _active_tokens(cov, size)
                if not full_ms:
                    rows.append(_mkrow(size, batch, cov, None, None, None, fstatus))
                elif TEXT_SEQLEN + active > ftok:
                    # buffer holds full image tokens (edit_config.image_seqlen); text(512)+active must fit
                    rows.append(_mkrow(size, batch, cov, full_ms, None, None, "n/a (buffer overflow)"))
                else:
                    partial_ms, pstatus = _measure_partial(
                        pipe, image, mask, EditConfig, args, size=size, batch=batch,
                        active_tokens=active, cached_latents=cached, folder=folder,
                        varlen=varlen, max_batch=max_batch)
                    if partial_ms:
                        rows.append(_mkrow(size, batch, cov, full_ms, partial_ms,
                                           full_ms / partial_ms, "ok"))
                    else:
                        rows.append(_mkrow(size, batch, cov, full_ms, None, None, pstatus))
            fm = f"{full_ms:.1f}ms" if full_ms else fstatus
            ptag = "fixed-len" if not varlen else "varlen"
            print(f"[bucket] res={size} B={batch} full={fm} partial={ptag} ({len(coverages)} buckets)")

    _write_report(rows, csv_path)


if __name__ == "__main__":
    main()
