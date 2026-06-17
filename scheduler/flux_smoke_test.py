import argparse
import os
import sys
import time


def _synth_inputs(size):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (size, size), (128, 128, 128))
    mask = Image.new("RGB", (size, size), (0, 0, 0))
    q = size // 4
    ImageDraw.Draw(mask).rectangle([q, q, size - q, size - q], fill=(255, 255, 255))
    return image, mask


def main():
    p = argparse.ArgumentParser(description="Standalone FLUX.1-schnell inpaint smoke test (no FlashPS).")
    p.add_argument("--image", default=None, help="RGB image path (synthesized if omitted)")
    p.add_argument("--mask", default=None, help="mask path: white=regenerate, black=keep (synthesized if omitted)")
    p.add_argument("--prompt", default=(
        "a full-body studio fashion photograph of a young woman with very short cropped dark "
        "hair, standing against a warm beige seamless backdrop, wearing a black graphic t-shirt "
        "printed with a grayscale portrait and bold white slogan text across the chest, tucked "
        "into a white high-waisted pencil skirt, soft even studio lighting, photorealistic, "
        "sharp focus, high detail"))
    p.add_argument("--out", default="/tmp/flux_smoke_out.png")
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--steps", type=int, default=4, help="schnell is distilled; ~4 steps")
    p.add_argument("--strength", type=float, default=0.85)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--model", default="black-forest-labs/FLUX.1-schnell",
                   help="HF repo id OR a local snapshot dir (…/snapshots/<hash>) to load offline")
    p.add_argument("--cache-dir", default=None, help="HF weights cache dir (default: HF_HOME/~/.cache)")
    args = p.parse_args()

    import torch
    from diffusers import FluxInpaintPipeline

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from configs.edit_config import EditConfig

    edit_config = EditConfig({
        "launch_script": "run_edit.py",
        "num_inference_steps": args.steps,
        "use_cached_kv": False,
        "use_cached_o": False,
        "save_kv": False,
        "save_o": False,
        "save_latents": False,
        "use_flash_attn_rope": False,
        "test_seqlen": False,
        "test_varlen": False,
        "real_varlen": False,
        "async_copy": False,
        "batch_size": 1,
        "device_num": 0,
        "prompt": args.prompt,
    })

    if args.image and args.mask:
        from PIL import Image

        image = Image.open(args.image).convert("RGB").resize((args.size, args.size))
        mask = Image.open(args.mask).convert("RGB").resize((args.size, args.size))
    else:
        image, mask = _synth_inputs(args.size)

    print(f"[smoke] loading {args.model} ...")
    t0 = time.time()
    kw = {"torch_dtype": torch.bfloat16}
    if args.cache_dir:
        kw["cache_dir"] = args.cache_dir
    if os.path.isdir(args.model):
        kw["local_files_only"] = True
    pipe = FluxInpaintPipeline.from_pretrained(args.model, **kw).to("cuda")
    print(f"[smoke] model ready in {time.time() - t0:.1f}s")

    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.time()
    out = pipe(
        prompt=args.prompt,
        image=image,
        mask_image=mask,
        height=args.size,
        width=args.size,
        strength=args.strength,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        max_sequence_length=256,
        generator=gen,
        edit_config=edit_config,
    ).images[0]
    print(f"[smoke] generated in {time.time() - t0:.1f}s")

    out.save(args.out)
    print(f"[smoke] saved -> {args.out}")


if __name__ == "__main__":
    main()
