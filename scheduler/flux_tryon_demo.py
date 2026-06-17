"""Virtual try-on demo for FlashPS Flux_inpaint (FLUX.1-schnell), single GPU.

Two stages, mirroring the paper's figure (template -> mask -> shirt-swapped output):

  Stage A  "generate image to get kvcache":
           text-to-image-style base generation (full white mask, strength 1.0) with the
           TEMPLATE prompt, run with save_kv=True + save_latents=True. This produces a
           usable initial image AND writes the REAL attention K/V cache (k_*.pt / v_*.pt)
           plus per-step latents into the cache folder. This is the only wired path that
           actually populates cached_kv_folder.

  Stage B  "shirt swap edit":
           inpaint the torso region (white=regenerate) with the EDIT prompt to put a
           different shirt on the same person. Produces a usable edited image.

Run on the GPU box, from scheduler/:
    python flux_tryon_demo.py                       # generate template + swap shirt
    python flux_tryon_demo.py --steps 4 --seed 1
    python flux_tryon_demo.py --template-image me.png --mask torso.png   # use a real photo

End-to-end KV caching status (verified against the vendored diffusers):
  - Stage A writes REAL KV: the save happens in the shared attn processor
    (attention_processor_short.py:1378/1383) and the standalone pipeline sets
    block_name/denoising_step per block+step, so distinct files are written.
  - REUSING that KV for a partial-sampling speedup is NOT wired in this standalone
    pipeline: the non-async load is commented out (attention_processor_short.py:1301-1313)
    and there is no loader, so an edit with use_cached_kv=True here would read zeros
    (garbage). Real KV reuse only works through the SERVER path
    (server.py:192-208 load_cache_kv -> transformer_flux._copy_cache_buffers ->
    current_key_cache), which requires async_copy=True. So this demo does the edit as a
    plain inpaint (use_cached_kv=False) to keep the output usable, and just reports that
    the cache was populated for the server to consume.
"""

import argparse
import glob
import os
import sys
import time

TEMPLATE_PROMPT = (
    "a full-body studio fashion photograph of a young woman with very short cropped dark "
    "hair, standing against a warm beige seamless backdrop, wearing a bright yellow ribbed "
    "sleeveless crop top and a white high-waisted pencil skirt, soft even studio lighting, "
    "photorealistic, sharp focus, high detail"
)
EDIT_PROMPT = (
    "a full-body studio fashion photograph of a young woman with very short cropped dark "
    "hair, standing against a warm beige seamless backdrop, wearing a black graphic t-shirt "
    "printed with a grayscale portrait and bold white slogan text across the chest, tucked "
    "into a white high-waisted pencil skirt, soft even studio lighting, photorealistic, "
    "sharp focus, high detail"
)

# The server's cache setup hardcodes text_length=512; generate KV at 512 so the written
# k_*.pt/v_*.pt are reusable by the server's async path.
TEXT_SEQLEN = 512


def _neutral_image(size):
    from PIL import Image
    return Image.new("RGB", (size, size), (128, 128, 128))


def _full_mask(size):
    from PIL import Image
    return Image.new("RGB", (size, size), (255, 255, 255))  # white = regenerate everything


def _torso_mask(size, box):
    # box = (left, top, right, bottom) as fractions of the image. White = regenerate, black = keep.
    from PIL import Image, ImageDraw
    left, top, right, bottom = box
    mask = Image.new("RGB", (size, size), (0, 0, 0))
    ImageDraw.Draw(mask).rectangle(
        [size * left, size * top, size * right, size * bottom], fill=(255, 255, 255))
    return mask


def _person_mask(template, size, box, bg_thresh):
    """Garment-shaped mask (shirt + arms, no background) via background color-key.

    Works because the template has a flat, seamless backdrop: sample its colour from the
    corners, mark pixels far from it as foreground (the person), and keep only the part
    inside the torso `box`. No model / weights needed — just numpy + PIL.
    """
    import numpy as np
    from PIL import Image, ImageFilter
    arr = np.asarray(template.convert("RGB").resize((size, size))).astype(np.float32)
    c = max(4, size // 32)
    corners = np.concatenate([arr[:c, :c].reshape(-1, 3), arr[:c, -c:].reshape(-1, 3),
                              arr[-c:, :c].reshape(-1, 3), arr[-c:, -c:].reshape(-1, 3)])
    bg = np.median(corners, axis=0)
    fg = np.sqrt(((arr - bg) ** 2).sum(axis=-1)) > bg_thresh  # True = person/foreground
    left, top, right, bottom = box
    boxm = np.zeros((size, size), dtype=bool)
    boxm[int(top * size):int(bottom * size), int(left * size):int(right * size)] = True
    m = np.repeat(((fg & boxm).astype(np.uint8) * 255)[:, :, None], 3, axis=2)
    mask = Image.fromarray(m)
    # dependency-free morphological close to fill speckle holes (e.g. print on the shirt)
    return mask.filter(ImageFilter.MaxFilter(7)).filter(ImageFilter.MinFilter(7))


def _edit_config(EditConfig, *, steps, prompt, save_kv=False, save_latents=False,
                 use_cached_kv=False, cached_kv_folder="", cached_latents_folder=""):
    return EditConfig({
        "launch_script": "run_edit.py",
        "num_inference_steps": steps,
        "prompt": prompt,
        "use_cached_o": False,
        "use_cached_kv": use_cached_kv,
        "save_o": False,
        "save_kv": save_kv,
        "save_latents": save_latents,
        "cached_kv_folder": cached_kv_folder,
        "cached_latents_folder": cached_latents_folder,
        "generated_seqlen": 4096,
        "use_flash_attn_rope": False,
        "test_seqlen": False,
        "test_varlen": False,
        "real_varlen": False,
        "async_copy": False,
        "batch_size": 1,
        "device_num": 0,
    })


def _run(pipe, image, mask, edit_config, *, size, steps, strength, seed):
    import torch
    gen = torch.Generator(device="cuda").manual_seed(seed)
    return pipe(
        prompt=edit_config.prompt,
        image=image,
        mask_image=mask,
        height=size,
        width=size,
        strength=strength,
        num_inference_steps=steps,
        guidance_scale=0.0,
        max_sequence_length=TEXT_SEQLEN,
        generator=gen,
        edit_config=edit_config,
    ).images[0]


def _count_cache(folder):
    ks = glob.glob(os.path.join(folder, "k_*.pt"))
    vs = glob.glob(os.path.join(folder, "v_*.pt"))
    nbytes = sum(os.path.getsize(p) for p in ks + vs)
    return len(ks), len(vs), nbytes


def main():
    p = argparse.ArgumentParser(description="FLUX.1-schnell virtual try-on demo (single GPU).")
    p.add_argument("--out-dir", default="/tmp/flux_tryon")
    p.add_argument("--template-prompt", default=TEMPLATE_PROMPT)
    p.add_argument("--edit-prompt", default=EDIT_PROMPT)
    p.add_argument("--template-image", default=None,
                   help="use a real photo as the template (skips generation; no KV is written)")
    p.add_argument("--mask", default=None, help="custom mask PNG (white=swap). Overrides --mask-mode")
    p.add_argument("--mask-mode", choices=["person", "box"], default="person",
                   help="'person' = garment-shaped mask (shirt+arms, no background) via backdrop "
                        "color-key inside the box; 'box' = plain rectangle")
    p.add_argument("--mask-box", default="0.30,0.16,0.70,0.62",
                   help="left,top,right,bottom fractions bounding the region (both modes use it). "
                        "Move it down / make it taller if the shirt sits lower than the box.")
    p.add_argument("--bg-thresh", type=float, default=42.0,
                   help="person-mode: RGB distance from backdrop to count as foreground. "
                        "Lower = grabs more (incl. near-backdrop tones); higher = tighter to the person.")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=4, help="schnell is distilled; ~4 steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--base-strength", type=float, default=1.0, help="1.0 -> full regen for the template")
    p.add_argument("--edit-strength", type=float, default=0.75,
                   help="lower (0.5-0.8) keeps the new shirt anchored to the body under the mask "
                        "(less distortion/misalignment); higher changes more but drifts")
    p.add_argument("--mask-blur", type=int, default=24,
                   help="feather the mask edges by this many px (0 = hard edge); softens seams")
    p.add_argument("--model", default="black-forest-labs/FLUX.1-schnell",
                   help="HF repo id OR a local snapshot dir (…/snapshots/<hash>) to load offline")
    p.add_argument("--cache-dir", default=None, help="HF weights cache dir")
    p.add_argument("--kv-dir", default=None,
                   help="where Stage A writes k_*.pt/v_*.pt (default <out>/cache/cached_kv). "
                        "Point at the server's flux_cache/cached_kv to precompute for a cached run.")
    p.add_argument("--latents-dir", default=None,
                   help="where Stage A writes latents_*.pt (default <out>/cache/cached_latents)")
    args = p.parse_args()

    import torch
    from diffusers import FluxInpaintPipeline

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from configs.edit_config import EditConfig

    os.makedirs(args.out_dir, exist_ok=True)
    cache_kv = args.kv_dir or os.path.join(args.out_dir, "cache", "cached_kv")
    cache_lat = args.latents_dir or os.path.join(args.out_dir, "cache", "cached_latents")
    os.makedirs(cache_kv, exist_ok=True)
    os.makedirs(cache_lat, exist_ok=True)

    print(f"[tryon] loading {args.model} ...")
    t0 = time.time()
    kw = {"torch_dtype": torch.bfloat16}
    if args.cache_dir:
        kw["cache_dir"] = args.cache_dir
    if os.path.isdir(args.model):
        kw["local_files_only"] = True  # loading straight from a local snapshot dir
    pipe = FluxInpaintPipeline.from_pretrained(args.model, **kw).to("cuda")
    print(f"[tryon] model ready in {time.time() - t0:.1f}s\n")

    from PIL import Image

    # ---- Stage A: template (generate image to get kvcache) ----
    template_path = os.path.join(args.out_dir, "template.png")
    if args.template_image:
        template = Image.open(args.template_image).convert("RGB").resize((args.size, args.size))
        template.save(template_path)
        print(f"[tryon] using provided template -> {template_path} (no KV written in this mode)\n")
    else:
        print("[tryon] Stage A: generating template + writing REAL kvcache (save_kv=True) ...")
        cfg = _edit_config(EditConfig, steps=args.steps, prompt=args.template_prompt,
                           save_kv=True, save_latents=True,
                           cached_kv_folder=cache_kv, cached_latents_folder=cache_lat)
        t0 = time.time()
        template = _run(pipe, _neutral_image(args.size), _full_mask(args.size), cfg,
                        size=args.size, steps=args.steps, strength=args.base_strength, seed=args.seed)
        template.save(template_path)
        nk, nv, nbytes = _count_cache(cache_kv)
        nlat = len(glob.glob(os.path.join(cache_lat, "latents_*.pt")))
        print(f"[tryon] template generated in {time.time() - t0:.1f}s -> {template_path}")
        print(f"[tryon] kvcache written: {nk} k_*.pt + {nv} v_*.pt ({nbytes/1e9:.2f} GB), "
              f"{nlat} latent steps -> {os.path.dirname(cache_kv)}\n")

    # ---- Stage B: shirt-swap edit (usable, plain inpaint) ----
    print("[tryon] Stage B: shirt-swap edit (torso mask) ...")
    if args.mask:
        mask = Image.open(args.mask).convert("RGB").resize((args.size, args.size))
    else:
        box = tuple(float(x) for x in args.mask_box.split(","))
        if args.mask_mode == "person":
            mask = _person_mask(template, args.size, box, args.bg_thresh)
        else:
            mask = _torso_mask(args.size, box)
    if args.mask_blur > 0:
        from PIL import ImageFilter
        mask = mask.filter(ImageFilter.GaussianBlur(args.mask_blur))  # feather -> smoother seam
    mask_path = os.path.join(args.out_dir, "mask.png")
    mask.save(mask_path)
    # Overlay the mask on the template so you can see what region gets regenerated (tuning aid).
    overlay_path = os.path.join(args.out_dir, "mask_overlay.png")
    Image.blend(template.convert("RGB"), mask.convert("RGB"), 0.4).save(overlay_path)

    cfg = _edit_config(EditConfig, steps=args.steps, prompt=args.edit_prompt)
    t0 = time.time()
    edited = _run(pipe, template, mask, cfg,
                  size=args.size, steps=args.steps, strength=args.edit_strength, seed=args.seed)
    edited_path = os.path.join(args.out_dir, "edited.png")
    edited.save(edited_path)
    print(f"[tryon] edit done in {time.time() - t0:.1f}s -> {edited_path}\n")

    print("===== try-on demo complete =====")
    print(f"  template : {template_path}")
    print(f"  mask     : {mask_path}")
    print(f"  overlay  : {overlay_path}  (check the box lines up with the shirt)")
    print(f"  edited   : {edited_path}")
    if not args.template_image:
        print(f"  kvcache  : {os.path.dirname(cache_kv)}  (real K/V + latents, server-reusable)")
    print("\nNOTE: the edit above is a plain inpaint (use_cached_kv=False) so the output is")
    print("real/usable. To REUSE the kvcache for a partial-sampling speedup you must run the")
    print("SERVER path (server.py, async_copy=True); standalone KV reuse is stubbed to zeros")
    print("at attention_processor_short.py:1301-1313.")


if __name__ == "__main__":
    main()
