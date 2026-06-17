"""Write a sample 1024x1024 image.png + mask.png for driving the FlashPS Flux server.

The server's prepare_for_inference reads image_path / mask_image_path off disk, so the
benchmark needs real files. This makes a simple pair (mask: white=regenerate, black=keep).

    python scheduler/make_sample_inputs.py            # -> scheduler/sample_image.png, sample_mask.png
    python scheduler/make_sample_inputs.py --mask-frac 0.4
"""

import argparse
import os

from PIL import Image, ImageDraw


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=1024)
    p.add_argument("--mask-frac", type=float, default=0.25,
                   help="side length of the centered edit box, as a fraction of the image")
    p.add_argument("--image", default=os.path.join(here, "sample_image.png"))
    p.add_argument("--mask", default=os.path.join(here, "sample_mask.png"))
    args = p.parse_args()

    img = Image.new("RGB", (args.size, args.size), (130, 150, 170))
    ImageDraw.Draw(img).ellipse(
        [args.size * 0.3, args.size * 0.3, args.size * 0.7, args.size * 0.7], fill=(200, 120, 90)
    )
    img.save(args.image)

    mask = Image.new("RGB", (args.size, args.size), (0, 0, 0))  # black = keep
    half = args.size * args.mask_frac / 2
    c = args.size / 2
    ImageDraw.Draw(mask).rectangle([c - half, c - half, c + half, c + half], fill=(255, 255, 255))
    mask.save(args.mask)

    print(f"wrote:\n  {args.image}\n  {args.mask}\n  ({args.size}x{args.size}, mask ~{args.mask_frac:.0%} side)")


if __name__ == "__main__":
    main()
