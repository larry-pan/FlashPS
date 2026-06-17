from __future__ import annotations

import argparse
import contextlib
import io
import os
import statistics
import sys
import time
import traceback

# The cache-setup path hardcodes text_length=512 (pipeline_flux_inpaint
# _setup_caching_configuration); the pipe() call must use the same value or the
# masked-region index_copy_ shapes mismatch.
TEXT_SEQLEN = 512


def _synth_inputs(size: int):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), (128, 128, 128))
    mask = Image.new("RGB", (size, size), (0, 0, 0))
    q = size // 4
    ImageDraw.Draw(mask).rectangle([q, q, size - q, size - q], fill=(255, 255, 255))
    return image, mask


def _make_edit_config(EditConfig, *, steps, prompt, use_cached_kv, generated_seqlen,
                      save_latents=False, cached_latents_folder=""):
    return EditConfig({
        "launch_script": "run_edit.py",
        "num_inference_steps": steps,
        "use_cached_kv": use_cached_kv,
        "use_cached_o": False,
        "save_kv": False, "save_o": False, "save_latents": save_latents,
        "use_flash_attn_rope": bool(use_cached_kv),
        "test_seqlen": bool(use_cached_kv),
        "generated_seqlen": generated_seqlen,
        "cached_latents_folder": cached_latents_folder,
        "test_varlen": False,
        "real_varlen": False,
        "async_copy": False,
        "batch_size": 1,
        "device_num": 0,
        "prompt": prompt,
    })


def _run_pipe(pipe, image, mask, edit_config, *, size, steps, strength, output_type="latent"):
    import torch
    gen = torch.Generator(device="cuda").manual_seed(0)
    with contextlib.redirect_stdout(io.StringIO()):  # swallow vendored debug prints
        return pipe(
            prompt=edit_config.prompt,
            image=image,
            mask_image=mask,
            height=size,
            width=size,
            strength=strength,
            num_inference_steps=steps,
            guidance_scale=0.0,
            max_sequence_length=TEXT_SEQLEN,  # must match cache-setup text_length=512
            generator=gen,
            edit_config=edit_config,
            output_type=output_type,
        )


def _save_images(pipe, image, mask, EditConfig, *, size, steps, strength, prompt, out_dir):
    """Save input, mask, and the REAL non-edit (full regen) output. Edit-path outputs are
    meaningless (stubbed caches), so we don't save those."""
    os.makedirs(out_dir, exist_ok=True)
    image.save(os.path.join(out_dir, "input.png"))
    mask.save(os.path.join(out_dir, "mask.png"))
    cfg = _make_edit_config(EditConfig, steps=steps, prompt=prompt,
                            use_cached_kv=False, generated_seqlen=4096)
    out = _run_pipe(pipe, image, mask, cfg, size=size, steps=steps,
                    strength=strength, output_type="pil").images[0]
    out.save(os.path.join(out_dir, "output_non_edit.png"))
    return out_dir


def _precompute_latents(pipe, image, mask, EditConfig, *, size, steps, strength, prompt, folder):
    """Full reference run with save_latents=True; the edit path splices into these.

    Returns {f"latents_{i}": tensor} loaded back from disk (what edit_config.cached_latents
    must hold for use_cached_kv to work).
    """
    import torch
    os.makedirs(folder, exist_ok=True)
    cfg = _make_edit_config(EditConfig, steps=steps, prompt=prompt, use_cached_kv=False,
                            generated_seqlen=4096, save_latents=True, cached_latents_folder=folder)
    _run_pipe(pipe, image, mask, cfg, size=size, steps=steps, strength=strength)
    return {f"latents_{i}": torch.load(os.path.join(folder, f"latents_{i}.pt"))
            for i in range(steps)}


def _time_config(pipe, image, mask, edit_config, *, size, steps, strength,
                 repeats, warmup):
    import torch

    def one():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _run_pipe(pipe, image, mask, edit_config, size=size, steps=steps, strength=strength)
        torch.cuda.synchronize()
        return time.perf_counter() - t0

    for _ in range(warmup):
        one()
    samples = [one() for _ in range(repeats)]
    return {
        "median": statistics.median(samples),
        "min": min(samples),
        "max": max(samples),
        "samples": samples,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="FlashPS partial-sampling latency benchmark (1 GPU).")
    p.add_argument("--image", default=None)
    p.add_argument("--mask", default=None)
    p.add_argument("--prompt", default=(
        "a full-body studio fashion photograph of a young woman with very short cropped dark "
        "hair, standing against a warm beige seamless backdrop, wearing a bright yellow ribbed "
        "sleeveless crop top and a white high-waisted pencil skirt, soft even studio lighting, "
        "photorealistic, sharp focus, high detail"))
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=4, help="schnell denoising steps")
    p.add_argument("--strength", type=float, default=1.0, help="1.0 -> run all steps (fair timing)")
    p.add_argument("--seqlens", default="128,256,512,1024,2048,3072",
                   help="comma-sep generated_seqlen (masked tokens) to sweep for edit; "
                        "must satisfy 512+seqlen<=4096 (hardcoded cache buffer)")
    p.add_argument("--full-seqlen", type=int, default=4096, help="token count for the non-edit baseline")
    p.add_argument("--model", default="black-forest-labs/FLUX.1-schnell",
                   help="HF repo id OR a local snapshot dir (…/snapshots/<hash>) to load offline")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--csv", default="flux_timing.csv")
    p.add_argument("--latents-dir", default="./flux_timing_latents",
                   help="where the precomputed reference latents are written/read")
    p.add_argument("--images-dir", default=None,
                   help="if set, save input.png, mask.png, and the real non-edit output.png here")
    args = p.parse_args()

    import torch
    from diffusers import FluxInpaintPipeline

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from configs.edit_config import EditConfig

    if args.image and args.mask:
        from PIL import Image
        image = Image.open(args.image).convert("RGB").resize((args.size, args.size))
        mask = Image.open(args.mask).convert("RGB").resize((args.size, args.size))
    else:
        image, mask = _synth_inputs(args.size)

    print(f"[timing] loading {args.model} ...")
    t0 = time.time()
    _kw = {"torch_dtype": torch.bfloat16}
    if os.path.isdir(args.model):
        _kw["local_files_only"] = True
    pipe = FluxInpaintPipeline.from_pretrained(args.model, **_kw).to("cuda")
    print(f"[timing] model ready in {time.time() - t0:.1f}s "
          f"({args.steps} steps, {args.repeats} repeats, {args.warmup} warmup)\n")

    seqlens = [int(s) for s in args.seqlens.split(",") if s.strip()]

    if args.images_dir:
        print(f"[timing] saving input/mask/non-edit output -> {args.images_dir}")
        _save_images(pipe, image, mask, EditConfig, size=args.size, steps=args.steps,
                     strength=args.strength, prompt=args.prompt, out_dir=args.images_dir)

    print("[timing] precomputing reference latents (full run, save_latents=True) ...")
    cached_latents = _precompute_latents(pipe, image, mask, EditConfig, size=args.size,
                                         steps=args.steps, strength=args.strength,
                                         prompt=args.prompt, folder=args.latents_dir)
    print(f"[timing] cached {len(cached_latents)} reference latents -> {args.latents_dir}\n")

    configs = [("non_edit_full", False, args.full_seqlen)]
    configs += [(f"edit_seqlen_{k}", True, k) for k in seqlens]

    results = {}
    CACHE_BUF = 4096  # single-block cache is hardcoded (1,4096,3072)
    for label, use_cached_kv, gseq in configs:
        if use_cached_kv and TEXT_SEQLEN + gseq > CACHE_BUF:
            print(f"[timing] {label:<18} seqlen={gseq:<5} SKIPPED "
                  f"({TEXT_SEQLEN} text + {gseq} > {CACHE_BUF} cache buffer would overflow)")
            continue
        cfg = _make_edit_config(EditConfig, steps=args.steps, prompt=args.prompt,
                                use_cached_kv=use_cached_kv, generated_seqlen=gseq,
                                cached_latents_folder=args.latents_dir)
        if use_cached_kv:
            cfg.cached_latents = cached_latents  # spliced into per-step (line ~1534)
        try:
            r = _time_config(pipe, image, mask, cfg, size=args.size, steps=args.steps,
                             strength=args.strength, repeats=args.repeats, warmup=args.warmup)
            results[label] = {"seqlen": gseq, **r}
            print(f"[timing] {label:<18} seqlen={gseq:<5} median={r['median']*1000:8.1f} ms "
                  f"(min {r['min']*1000:.1f}, max {r['max']*1000:.1f})")
        except Exception as e:  # noqa: BLE001
            print(f"[timing] {label:<18} seqlen={gseq:<5} FAILED: {type(e).__name__}: {e}")
            traceback.print_exc()

    base = results.get("non_edit_full", {}).get("median")
    print("\n===== FlashPS partial-sampling latency (1 GPU) =====")
    header = f"{'config':<18}{'mask tokens':>12}{'median ms':>12}{'ms/step':>10}{'speedup':>10}"
    print(header)
    print("-" * len(header))
    rows = []
    for label, _, gseq in configs:
        if label not in results:
            continue
        med = results[label]["median"]
        speedup = (base / med) if base else float("nan")
        print(f"{label:<18}{gseq:>12}{med*1000:>12.1f}{med*1000/args.steps:>10.1f}{speedup:>9.2f}x")
        rows.append((label, gseq, med, speedup))

    with open(args.csv, "w") as f:
        f.write("config,mask_tokens,median_s,ms_per_step,speedup_vs_full\n")
        for label, gseq, med, speedup in rows:
            f.write(f"{label},{gseq},{med:.6f},{med*1000/args.steps:.3f},{speedup:.4f}\n")
    print(f"\n[timing] wrote {args.csv}")
    print("[timing] NOTE: edit-path images are meaningless (stubbed caches); this measures "
          "denoising latency vs masked-region size only.")


if __name__ == "__main__":
    main()
