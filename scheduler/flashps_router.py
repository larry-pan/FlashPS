from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import yaml

SERVICE_ID = "Flux_inpaint"
DEFAULT_FULL_SEQLEN = 4096
MIN_EDIT_SEQLEN = 256


@dataclass
class RequestSpec:
    prompt: str
    image_path: str
    mask_image_path: Optional[str] = None
    strength: float = 0.6
    seed: int = 0
    edit: Optional[bool] = None
    mask_seq_length: Optional[int] = None


def calculate_mask_seq_length(mask_path, full_seqlen=DEFAULT_FULL_SEQLEN):
    if not mask_path or not os.path.exists(mask_path):
        return full_seqlen
    from PIL import Image
    import numpy as np
    mask_array = np.array(Image.open(mask_path).convert("L"))
    black_proportion = 1.0 - (np.count_nonzero(mask_array) / mask_array.size)
    seqlen = int(black_proportion * full_seqlen)
    return max(MIN_EDIT_SEQLEN, min(seqlen, full_seqlen))


def classify(req, full_seqlen=DEFAULT_FULL_SEQLEN, edit_ratio_threshold=0.6):
    if req.edit is not None:
        return "edit" if req.edit else "non_edit"
    if not req.mask_image_path:
        return "non_edit"
    seqlen = req.mask_seq_length
    if seqlen is None:
        seqlen = calculate_mask_seq_length(req.mask_image_path, full_seqlen)
    return "non_edit" if seqlen >= edit_ratio_threshold * full_seqlen else "edit"


def resolve_seqlen(req, route, full_seqlen=DEFAULT_FULL_SEQLEN):
    if route == "non_edit":
        return full_seqlen
    if req.mask_seq_length is not None:
        return max(MIN_EDIT_SEQLEN, min(req.mask_seq_length, full_seqlen))
    if req.mask_image_path:
        return calculate_mask_seq_length(req.mask_image_path, full_seqlen)
    return max(MIN_EDIT_SEQLEN, full_seqlen // 4)


def render_config(base_yaml, route, seqlen, out_dir, idx):
    with open(base_yaml) as f:
        config = yaml.safe_load(f)
    config["use_cached_kv"] = route == "edit"
    config["generated_seqlen"] = int(seqlen)
    config["test_seqlen"] = True
    # Flash-attn varlen rope needs cache-only cos/sin; only valid on the edit (cache-on) path.
    config["use_flash_attn_rope"] = route == "edit"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"temp_config_{idx}.yml")
    with open(out_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    return out_path


def build_request(req, base_yaml, out_dir, idx, full_seqlen=DEFAULT_FULL_SEQLEN,
                  edit_ratio_threshold=0.6, num_inference_steps=4):
    route = classify(req, full_seqlen, edit_ratio_threshold)
    seqlen = resolve_seqlen(req, route, full_seqlen)
    config_path = render_config(base_yaml, route, seqlen, out_dir, idx)
    inputs = {
        "prompt": req.prompt,
        "image_path": req.image_path,
        "mask_image_path": req.mask_image_path,
        "strength": req.strength,
        "seed": req.seed,
        "edit_config_path": config_path,
        "num_inference_steps": num_inference_steps,
        "mask_seq_length": seqlen,
    }
    return SERVICE_ID, inputs, route
