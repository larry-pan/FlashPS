"""Send ONE Flux_inpaint cached-edit request to a running FlashPS server and save the image.

The checked-in client (fisedit_client_async.py) is hardcoded to SD2 and sends no
image/mask/edit_config, so it can't drive a real Flux edit. This does, with stdlib only.

Use ABSOLUTE paths (the server reads image_path/mask_image_path/edit_config_path off disk
in a worker process). strength defaults to 1.0 so the edit runs the full step trajectory
that the precomputed per-step latents_*.pt were saved for (a smaller strength shifts the
timesteps and misaligns the cache).

    python flux_edit_request.py \
      --image   /abs/path/template.png \
      --mask    /abs/path/mask.png \
      --edit-config /abs/path/configs/flux_inpaint_cached_edit.yml \
      --out     /abs/path/edited_cached.png
"""

import argparse
import base64
import json
import os
import time
import urllib.request

EDIT_PROMPT = (
    "a full-body studio fashion photograph of a young woman with very short cropped dark "
    "hair, standing against a warm beige seamless backdrop, wearing a black graphic t-shirt "
    "printed with a grayscale portrait and bold white slogan text across the chest, tucked "
    "into a white high-waisted pencil skirt, soft even studio lighting, photorealistic, "
    "sharp focus, high detail"
)


def main():
    ap = argparse.ArgumentParser(description="One Flux_inpaint cached-edit request.")
    ap.add_argument("--server-url", default="http://localhost:8005")
    ap.add_argument("--image", required=True, help="ABSOLUTE path to the template (base) image")
    ap.add_argument("--mask", required=True, help="ABSOLUTE path to the torso mask (white=swap)")
    ap.add_argument("--edit-config", required=True,
                    help="ABSOLUTE path to configs/flux_inpaint_cached_edit.yml")
    ap.add_argument("--prompt", default=EDIT_PROMPT)
    ap.add_argument("--strength", type=float, default=1.0,
                    help="keep 1.0 so steps align with the precomputed latents")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="edited_cached.png")
    args = ap.parse_args()

    for label, path in (("image", args.image), ("mask", args.mask), ("edit-config", args.edit_config)):
        if not os.path.isabs(path):
            raise SystemExit(f"--{label} must be an absolute path (server reads it off disk): {path}")

    body = json.dumps({"inputs": {
        "prompt": args.prompt,
        "image_path": args.image,
        "mask_image_path": args.mask,
        "strength": args.strength,
        "seed": args.seed,
        "edit_config_path": args.edit_config,
    }}).encode()

    url = f"{args.server_url}/api/workflow/Flux_inpaint/inference"
    print(f"[edit] POST {url}")
    t0 = time.time()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        resp = json.loads(r.read().decode())
    dt = time.time() - t0

    results = resp["results"]
    imgs = results["img_str_list"]
    with open(args.out, "wb") as f:
        f.write(base64.b64decode(imgs[0]))
    print(f"[edit] status={resp['status']} imgs={len(imgs)} "
          f"post_proc={results.get('post_processing_latency', 0)*1000:.0f}ms "
          f"round_trip={dt*1000:.0f}ms -> {args.out}")


if __name__ == "__main__":
    main()
