import argparse
import asyncio
import base64
import io
import os
import sys

import uvicorn
import yaml
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from schedule_methods import cal_compute_time
from flashps_router import DEFAULT_FULL_SEQLEN

LATENCY_SCALE = float(os.environ.get("FLASHPS_MOCK_SCALE", "0.02"))
app = FastAPI()


def _tiny_png_b64():
    from PIL import Image
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (127, 127, 127)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


_PNG = _tiny_png_b64()


class InferenceRequest(BaseModel):
    inputs: dict


def _model_latency(inputs):
    with open(inputs["edit_config_path"]) as f:
        cfg = yaml.safe_load(f)
    seqlen = cfg["generated_seqlen"] if cfg["use_cached_kv"] else DEFAULT_FULL_SEQLEN
    return cfg["num_inference_steps"] * cal_compute_time(seqlen)


@app.post("/api/workflow/{service_id}/inference")
async def run_inference(service_id: str, request: InferenceRequest):
    latency = _model_latency(request.inputs) * LATENCY_SCALE
    await asyncio.sleep(latency)
    return {
        "status": "success",
        "results": {
            "status": "completed",
            "img_str_list": [_PNG],
            "inference_latency": latency,
            "post_processing_latency": 0.0,
        },
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8005)
    p.add_argument("--latency-scale", type=float, default=None,
                   help="Multiply analytic latency by this factor (default env FLASHPS_MOCK_SCALE or 0.02)")
    args = p.parse_args()
    if args.latency_scale is not None:
        LATENCY_SCALE = args.latency_scale
    print(f"[mock] FlashPS mock server on {args.host}:{args.port}, latency_scale={LATENCY_SCALE}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")
