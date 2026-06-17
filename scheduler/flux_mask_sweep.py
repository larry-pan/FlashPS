"""Batch FLUX.1-schnell inpaint timing sweep over a GRID of mask size/shape x batch x GPU count.

Single-node, standalone pipeline (no server). Measures the NORMAL (full) inpaint forward only
-- one number per cell, `timing_median` (ms). No partial-sampling / cached-KV pass (that path is
zero-stubbed standalone and was fragile at batch>1); this is just clean, real inference timing.

Per (gpu_count, batch_size, mask) cell:
  * timing_median = median wall-time (ms) of a full `use_cached_kv=False` forward at that batch
    (num_images_per_prompt=B, strength 1.0).
  * throughput_reqs = requests/s. For G>1 it is the sum of the concurrent per-GPU throughputs.

Axes:
  * mask size/shape — an auto family (box + person, small/med/large) + an equal-area /
    different-shape control + any PNGs in --mask-dir. NOTE: full-compute timing is ~independent
    of mask size (the whole image is recomputed), so `timing_median` will be roughly flat across
    masks; the mask column mainly identifies the usable image written for it.
  * batch size — a real batched forward (num_images_per_prompt=B).
  * gpu count — REAL data-parallel replication: G worker processes, each pinned to a physical
    GPU (torch.multiprocessing + set_device), run the same cell concurrently (barrier-synced).
    The standalone pipeline is single-device, so more GPUs = more replicas = more throughput.
    Cells with gpu_count > available GPUs are SKIPPED.

Usable, proper edited images (edited_<mask>.png) are written once during bootstrap (batch 1,
plain inpaint at --edit-strength). GPU selection is automatic (freest cards via nvidia-smi;
override with --devices or CUDA_VISIBLE_DEVICES).

Run on the GPU box, from scheduler/:
  python flux_mask_sweep.py --model <snapshot> --template-image template.png --out-dir out \
    --gpu-counts 1,2,4 --batch-sizes 1,2,4 --repeats 5 --warmup 2
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time
import types

# Reduce allocator fragmentation (the OOM message itself recommends this). Must be set before
# torch initializes CUDA; torch is only imported later, inside main()/workers.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# import sibling modules (this file lives in scheduler/) + the repo configs package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import flux_timing_test as ft   # noqa: E402  (TEXT_SEQLEN, _make_edit_config)
import flux_tryon_demo as demo  # noqa: E402  (_torso_mask, _person_mask, _full_mask, _edit_config, _run)

FULL_SEQLEN = 4096        # 1024^2 -> 64x64 packed tokens


# --- mask -> token / stats helpers (dependency-free: numpy + PIL) ---------------------------

def _mask_tokens(mask, size, patch_thresh=0.5):
    """Active-token count: average-pool the mask to a 16px patch grid, count patches whose white
    fraction >= patch_thresh (the region that would be regenerated). Range [0, 4096]."""
    import numpy as np
    grid = size // 16  # 64 for size=1024
    arr = (np.asarray(mask.convert("L").resize((size, size))).astype(np.float32) / 255.0)
    arr = arr[: grid * 16, : grid * 16].reshape(grid, 16, grid, 16).mean(axis=(1, 3))
    return int((arr >= patch_thresh).sum())


def _white_frac(mask, size):
    import numpy as np
    arr = np.asarray(mask.convert("L").resize((size, size)))
    return float((arr > 127).mean())


# --- mask family -----------------------------------------------------------------------------

def _auto_masks(template, size, bg_thresh):
    """A size family (box + person modes) plus an equal-area / different-shape control pair."""
    boxes = {
        "small":  (0.35, 0.30, 0.65, 0.55),
        "medium": (0.30, 0.22, 0.70, 0.66),
        "large":  (0.24, 0.16, 0.76, 0.78),
    }
    masks = []  # (name, source, crisp_mask)
    for name, box in boxes.items():
        masks.append((f"box_{name}", "auto", demo._torso_mask(size, box)))
        masks.append((f"person_{name}", "auto", demo._person_mask(template, size, box, bg_thresh)))
    masks.append(("shape_ctrl_wide", "auto",
                  demo._torso_mask(size, (0.1875, 0.40625, 0.8125, 0.59375))))  # 40 x 12 patches
    masks.append(("shape_ctrl_tall", "auto",
                  demo._torso_mask(size, (0.3125, 0.34375, 0.6875, 0.65625))))  # 24 x 20 patches
    return masks


def _dir_masks(mask_dir, size):
    from PIL import Image
    out = []
    for path in sorted(glob.glob(os.path.join(mask_dir, "*.png")) +
                       glob.glob(os.path.join(mask_dir, "*.jpg")) +
                       glob.glob(os.path.join(mask_dir, "*.jpeg"))):
        name = os.path.splitext(os.path.basename(path))[0]
        out.append((name, "photoshop", Image.open(path).convert("RGB").resize((size, size))))
    return out


# --- timing primitives -----------------------------------------------------------------------

def _time_forward(pipe, image, mask, edit_config, *, size, steps, strength, batch, repeats, warmup):
    """Median wall-time (ms) of one batched pipeline forward on the CURRENT cuda device."""
    import contextlib
    import io
    import torch
    gen = torch.Generator(device="cuda").manual_seed(0)  # current device (set via set_device)

    def one():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):  # swallow vendored debug prints
            pipe(prompt=edit_config.prompt, image=image, mask_image=mask, height=size, width=size,
                 strength=strength, num_inference_steps=steps, guidance_scale=0.0,
                 max_sequence_length=ft.TEXT_SEQLEN, num_images_per_prompt=batch,
                 generator=gen, edit_config=edit_config, output_type="latent")
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        one()
    return statistics.median(one() for _ in range(repeats)) * 1000.0


def _measure_timing(pipe, template, mask_img, EditConfig, args, batch):
    """One grid cell on the current device: (timing_median_ms, status). Full/normal forward only
    (use_cached_kv=False, strength=1.0)."""
    cfg = ft._make_edit_config(EditConfig, steps=args.steps, prompt=args.edit_prompt,
                               use_cached_kv=False, generated_seqlen=FULL_SEQLEN)
    cfg.batch_size = batch
    try:
        ms = _time_forward(pipe, template, mask_img, cfg, size=args.size, steps=args.steps,
                           strength=1.0, batch=batch, repeats=args.repeats, warmup=args.warmup)
        return ms, "ok"
    except Exception as e:  # noqa: BLE001  (e.g. OOM at large batch -> record & move on)
        return None, f"FAILED: {type(e).__name__}"


def _mkrow(G, B, name, source, tokens, white_frac, timing_ms, tp, status):
    # speedup_vs_full = PROJECTED partial-sampling speedup from the mask area (analytical; the
    # cached pass isn't run). A full forward processes 512 text + 4096 image tokens; partial
    # sampling would process 512 + active_tokens -> ~ (512+4096)/(512+active_tokens).
    speedup = (ft.TEXT_SEQLEN + FULL_SEQLEN) / (ft.TEXT_SEQLEN + tokens) if tokens is not None else None
    return {
        "gpu_count": G, "batch_size": B, "mask_name": name, "source": source,
        "active_tokens": tokens, "token_ratio": (tokens / FULL_SEQLEN) if tokens is not None else None,
        "white_px_frac": white_frac, "timing_median": timing_ms, "throughput_reqs": tp,
        "speedup_vs_full": speedup, "status": status,
    }


def _tp(ms, reqs):
    return (reqs / (ms / 1000.0)) if ms else None


# --- multi-GPU worker (module-level so it is picklable for spawn) -----------------------------

def _grid_worker(rank, G, batch_sizes, mask_manifest, args_dict, template_path, result_q, barrier):
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from diffusers import FluxInpaintPipeline
    from PIL import Image, ImageFilter
    from configs.edit_config import EditConfig

    args = types.SimpleNamespace(**args_dict)
    # Init inside try so a load failure still keeps this worker in barrier lockstep (below),
    # rather than dying and hanging the other ranks at the next barrier.
    pipe = template = masks = None
    init_err = None
    try:
        torch.cuda.set_device(rank)
        kw = {"torch_dtype": torch.bfloat16}
        if os.path.isdir(args.model):
            kw["local_files_only"] = True
        pipe = FluxInpaintPipeline.from_pretrained(args.model, **kw).to(f"cuda:{rank}")
        template = Image.open(template_path).convert("RGB").resize((args.size, args.size))
        masks = [(m["name"], m["source"],
                  Image.open(m["path"]).convert("RGB").resize((args.size, args.size)), m["tokens"])
                 for m in mask_manifest]
    except Exception as e:  # noqa: BLE001
        init_err = f"FAILED (worker init): {type(e).__name__}"

    for B in batch_sizes:
        for m in mask_manifest:
            name, source, tokens = m["name"], m["source"], m["tokens"]
            barrier.wait()  # all G ranks start this cell together (real contention; no deadlock)
            if init_err is not None:
                timing_ms, status = None, init_err
            else:
                mask_crisp = next(mc for nm, _, mc, _ in masks if nm == name)
                mask_img = (mask_crisp.filter(ImageFilter.GaussianBlur(args.mask_blur))
                            if args.mask_blur > 0 else mask_crisp)
                timing_ms, status = _measure_timing(pipe, template, mask_img, EditConfig, args, B)
            result_q.put({"rank": rank, "gpu_count": G, "batch_size": B, "mask_name": name,
                          "source": source, "active_tokens": tokens,
                          "white_px_frac": m["white_px_frac"], "timing_ms": timing_ms,
                          "status": status})


def run_spawned(G, batch_sizes, mask_manifest, template_path, args, ctx):
    """Spawn G workers on G physical GPUs; aggregate per (batch, mask)."""
    from collections import defaultdict
    barrier = ctx.Barrier(G)
    q = ctx.Queue()
    args_dict = vars(args)
    procs = [ctx.Process(target=_grid_worker,
                         args=(r, G, batch_sizes, mask_manifest, args_dict, template_path, q, barrier))
             for r in range(G)]
    for p in procs:
        p.start()
    n = G * len(batch_sizes) * len(mask_manifest)
    raw = [q.get() for _ in range(n)]
    for p in procs:
        p.join()

    groups = defaultdict(list)
    for r in raw:
        groups[(r["batch_size"], r["mask_name"])].append(r)

    rows = []
    for (B, name), items in groups.items():
        times = [i["timing_ms"] for i in items if i["timing_ms"]]
        timing = statistics.median(times) if times else None
        tp = sum(_tp(i["timing_ms"], B) for i in items if i["timing_ms"]) or None
        statuses = set(i["status"] for i in items)
        status = "ok" if statuses == {"ok"} else ";".join(sorted(statuses))
        rows.append(_mkrow(G, B, name, items[0]["source"], items[0]["active_tokens"],
                           items[0]["white_px_frac"], timing, tp, status))
    return rows


def run_inprocess(pipe, template, mask_records, args, EditConfig, batch_sizes):
    """gpu_count == 1: reuse the already-loaded parent pipe (no spawn overhead)."""
    from PIL import ImageFilter
    rows = []
    for B in batch_sizes:
        for name, source, mask_crisp, tokens, wfrac in mask_records:
            mask_img = (mask_crisp.filter(ImageFilter.GaussianBlur(args.mask_blur))
                        if args.mask_blur > 0 else mask_crisp)
            timing_ms, status = _measure_timing(pipe, template, mask_img, EditConfig, args, B)
            rows.append(_mkrow(1, B, name, source, tokens, wfrac, timing_ms, _tp(timing_ms, B), status))
            tm = f"{timing_ms:8.1f}" if timing_ms else "     n/a"
            print(f"[grid] G=1 B={B} {name:<18} tok={tokens:<5} timing_median={tm}ms  {status}")
    return rows


# --- bootstrap (single GPU): template, masks, representative images ---------------------------

def _bootstrap(pipe, args, EditConfig):
    from PIL import Image, ImageFilter
    os.makedirs(args.out_dir, exist_ok=True)
    masks_dir = os.path.join(args.out_dir, "masks"); os.makedirs(masks_dir, exist_ok=True)

    template_path = os.path.join(args.out_dir, "template.png")
    if args.template_image:
        template = Image.open(args.template_image).convert("RGB").resize((args.size, args.size))
        print(f"[boot] using provided template -> {template_path}")
    else:
        print("[boot] generating template (full regen) ...")
        gcfg = demo._edit_config(EditConfig, steps=args.steps, prompt=args.template_prompt)
        template = demo._run(pipe, demo._neutral_image(args.size), demo._full_mask(args.size), gcfg,
                             size=args.size, steps=args.steps, strength=1.0, seed=args.seed)
    template.save(template_path)

    mask_set = []
    if args.auto:
        mask_set += _auto_masks(template, args.size, args.bg_thresh)
    if args.mask_dir:
        mask_set += _dir_masks(args.mask_dir, args.size)
    if not mask_set:
        raise SystemExit("no masks: pass --mask-dir and/or keep --auto on")

    records, manifest = [], []
    for name, source, mask_crisp in mask_set:
        tokens = _mask_tokens(mask_crisp, args.size, args.patch_thresh)
        wfrac = _white_frac(mask_crisp, args.size)
        p = os.path.join(masks_dir, f"{name}.png"); mask_crisp.save(p)
        Image.blend(template.convert("RGB"), mask_crisp.convert("RGB"), 0.4).save(
            os.path.join(masks_dir, f"{name}_overlay.png"))
        records.append((name, source, mask_crisp, tokens, wfrac))
        manifest.append({"name": name, "source": source, "path": p, "tokens": tokens,
                         "white_px_frac": wfrac})
        # representative usable image (batch 1, plain inpaint at edit-strength)
        mask_img = (mask_crisp.filter(ImageFilter.GaussianBlur(args.mask_blur))
                    if args.mask_blur > 0 else mask_crisp)
        img_cfg = demo._edit_config(EditConfig, steps=args.steps, prompt=args.edit_prompt)
        edited = demo._run(pipe, template, mask_img, img_cfg, size=args.size, steps=args.steps,
                           strength=args.edit_strength, seed=args.seed)
        edited.save(os.path.join(args.out_dir, f"edited_{name}.png"))
    print(f"[boot] {len(records)} masks + edited_*.png written -> {args.out_dir}\n")
    return template, template_path, records, manifest


# --- report ----------------------------------------------------------------------------------

def _write_report(rows, csv_path):
    rows.sort(key=lambda r: (r["gpu_count"], r["batch_size"], r["active_tokens"]))
    # Console: rich (keeps gpu/batch/throughput visible). CSV: just the 4 requested columns.
    print("\n===== FLUX inpaint timing grid (timing_median ms; speedup_vs_full = projected) =====")
    hdr = f"{'G':>2}{'B':>3}{'mask':>16}{'white%':>7}{'tok':>6}{'timing_ms':>11}{'speedup':>9}{'thru':>8}  status"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        tm = f"{r['timing_median']:11.1f}" if r["timing_median"] else f"{'n/a':>11}"
        sp = f"{r['speedup_vs_full']:8.2f}x" if r["speedup_vs_full"] else f"{'n/a':>9}"
        tp = f"{r['throughput_reqs']:8.2f}" if r["throughput_reqs"] else f"{'n/a':>8}"
        wf = f"{r['white_px_frac'] * 100:6.1f}" if r["white_px_frac"] is not None else f"{'n/a':>6}"
        flag = " *shape" if r["mask_name"].startswith("shape_ctrl") else ""
        print(f"{r['gpu_count']:>2}{r['batch_size']:>3}{r['mask_name']:>16}{wf}{r['active_tokens']:>6}"
              f"{tm}{sp}{tp}  {r['status']}{flag}")

    # Full grid CSV: one row per (gpu_count, batch_size, mask). G and B identify each cell so
    # all three axes -- mask size/shape, gpu count, batch size -- are captured.
    cols = ["gpu_count", "batch_size", "mask_name", "white_px_frac", "active_tokens",
            "timing_median_ms", "throughput_reqs", "speedup_vs_full", "status"]
    keymap = {"timing_median_ms": "timing_median"}  # CSV name -> row key; others match
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            def _fmt(v):
                if v is None:
                    return ""
                return f"{v:.4f}" if isinstance(v, float) else str(v)
            f.write(",".join(_fmt(r[keymap.get(c, c)]) for c in cols) + "\n")
    print(f"\n[grid] wrote {csv_path}  (cols: {', '.join(cols)})")
    print("[grid] one row per (gpu_count x batch_size x mask). timing_median_ms = measured full "
          "forward (mask-independent; batch/gpu drive it). throughput_reqs = req/s (summed across "
          "GPUs for G>1) -- the batch/gpu scaling metric. speedup_vs_full = PROJECTED partial-"
          "sampling speedup from mask area = (512+4096)/(512+active_tokens). Images: edited_<mask>.png.")


def _nvidia_smi_free_mb():
    """[(physical_index, free_MiB)] via nvidia-smi (no CUDA context -> safe on a busy GPU)."""
    import subprocess
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
        text=True)
    res = []
    for line in out.strip().splitlines():
        idx, free = line.split(",")
        res.append((int(idx), int(free)))
    return res


def _setup_visible_devices(args):
    """Pin the process to free GPUs by setting CUDA_VISIBLE_DEVICES *before* torch loads, so
    cuda:0..G-1 always map to cards with enough free memory. Respects --devices or a user-set
    CUDA_VISIBLE_DEVICES."""
    if args.devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
        print(f"[grid] CUDA_VISIBLE_DEVICES={args.devices} (from --devices)")
        return
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        print(f"[grid] CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']} (from environment)")
        return
    want = max((int(x) for x in args.gpu_counts.split(",") if x.strip()), default=1)
    try:
        free = _nvidia_smi_free_mb()
    except Exception as e:  # noqa: BLE001  (nvidia-smi missing/odd output -> let torch use all)
        print(f"[grid] could not query nvidia-smi ({type(e).__name__}); using all GPUs as-is")
        return
    free.sort(key=lambda t: t[1], reverse=True)
    need_mb = int(args.min_free_gb * 1024)
    usable = [i for i, mb in free if mb >= need_mb]
    if not usable:
        usable = [free[0][0]]
        print(f"[grid] WARNING: no GPU has >= {args.min_free_gb} GiB free; trying the freest "
              f"(GPU {usable[0]}, {free[0][1]/1024:.1f} GiB free) — expect possible OOM")
    sel = usable[:max(want, 1)]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in sel)
    freemap = dict(free)
    print(f"[grid] auto-selected GPUs {sel} by free memory "
          f"({', '.join(f'{i}:{freemap[i]/1024:.0f}G' for i in sel)}) "
          f"-> CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")


def main():
    p = argparse.ArgumentParser(description="FLUX inpaint timing grid: mask x batch x gpu "
                                            "(full/normal forward only; usable images from bootstrap).")
    p.add_argument("--model", default="black-forest-labs/FLUX.1-schnell",
                   help="HF repo id OR a local snapshot dir (…/snapshots/<hash>) to load offline")
    p.add_argument("--cache-dir", default=None, help="HF weights cache dir")
    p.add_argument("--template-image", default=None,
                   help="base image the masks apply to; if omitted, one is generated")
    p.add_argument("--template-prompt", default=demo.TEMPLATE_PROMPT)
    p.add_argument("--edit-prompt", default=demo.EDIT_PROMPT)
    p.add_argument("--out-dir", default="/tmp/flux_mask_sweep")
    p.add_argument("--mask-dir", default=None, help="folder of hand-authored (e.g. Photoshop) masks")
    p.add_argument("--auto", dest="auto", action="store_true", default=True,
                   help="include the auto-generated size family (default on)")
    p.add_argument("--no-auto", dest="auto", action="store_false")
    p.add_argument("--gpu-counts", default="1", help="comma-sep data-parallel GPU counts, e.g. 1,2,4")
    p.add_argument("--batch-sizes", default="1", help="comma-sep batch sizes, e.g. 1,2,4")
    p.add_argument("--devices", default=None,
                   help="explicit physical GPU indices to use (sets CUDA_VISIBLE_DEVICES), e.g. 1,2,3; "
                        "default = auto-pick the freest by nvidia-smi")
    p.add_argument("--min-free-gb", type=float, default=38.0,
                   help="a GPU needs at least this much free VRAM to be auto-selected (FLUX ~34GB)")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=4, help="schnell is distilled; ~4 steps")
    p.add_argument("--repeats", type=int, default=5, help="timed samples per cell (median)")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--patch-thresh", type=float, default=0.5,
                   help="a 16px patch counts as active if >= this fraction is white")
    p.add_argument("--edit-strength", type=float, default=0.75,
                   help="bootstrap proper-image pass: lower keeps the new shirt anchored")
    p.add_argument("--mask-blur", type=int, default=24,
                   help="feather (px) applied to the mask for the forward (0 = hard edge)")
    p.add_argument("--bg-thresh", type=float, default=42.0, help="person-mode backdrop color-key distance")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--csv", default=None, help="default <out-dir>/flux_mask_grid.csv")
    args = p.parse_args()

    _setup_visible_devices(args)  # MUST run before torch initializes CUDA

    import torch
    from diffusers import FluxInpaintPipeline
    from configs.edit_config import EditConfig

    gpu_counts = sorted({int(x) for x in args.gpu_counts.split(",") if x.strip()})
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x.strip()]
    device_count = torch.cuda.device_count()
    csv_path = args.csv or os.path.join(args.out_dir, "flux_mask_grid.csv")
    print(f"[grid] gpu_counts={gpu_counts} batch_sizes={batch_sizes} "
          f"(available GPUs={device_count})\n")

    print(f"[grid] loading {args.model} on cuda:0 (bootstrap) ...")
    t0 = time.time()
    kw = {"torch_dtype": torch.bfloat16}
    if args.cache_dir:
        kw["cache_dir"] = args.cache_dir
    if os.path.isdir(args.model):
        kw["local_files_only"] = True
    pipe = FluxInpaintPipeline.from_pretrained(args.model, **kw).to("cuda:0")
    print(f"[grid] model ready in {time.time() - t0:.1f}s\n")

    template, template_path, records, manifest = _bootstrap(pipe, args, EditConfig)

    rows = []
    if 1 in gpu_counts:
        rows += run_inprocess(pipe, template, records, args, EditConfig, batch_sizes)

    multi = [G for G in gpu_counts if G > 1]
    if multi:
        del pipe
        import gc
        gc.collect(); torch.cuda.empty_cache()  # free cuda:0 before workers reload the model
        import torch.multiprocessing as tmp
        ctx = tmp.get_context("spawn")
        for G in multi:
            if G > device_count:
                print(f"[grid] SKIP gpu_count={G}: only {device_count} GPUs available")
                for B in batch_sizes:
                    for m in manifest:
                        rows.append(_mkrow(G, B, m["name"], m["source"], m["tokens"],
                                           m["white_px_frac"], None, None,
                                           f"SKIPPED (need {G} GPUs, have {device_count})"))
                continue
            print(f"[grid] gpu_count={G}: spawning {G} workers on cuda:0..{G-1} ...")
            rows += run_spawned(G, batch_sizes, manifest, template_path, args, ctx)

    _write_report(rows, csv_path)


if __name__ == "__main__":
    main()
