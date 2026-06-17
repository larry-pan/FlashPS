import argparse
import asyncio
import glob
import json
import os
import sys
import time
from datetime import datetime

import aiohttp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from flashps_router import RequestSpec, build_request, DEFAULT_FULL_SEQLEN

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE_CONFIG = os.path.join(REPO_ROOT, "configs", "flux_inpaint_base.yml")
RUNS_DIR = os.path.join(REPO_ROOT, "scheduler", "flashps_runs")
MASK_RATIO_DIR = os.path.join(REPO_ROOT, "mask_ratio_distribution")
FULL_SEQLEN = DEFAULT_FULL_SEQLEN
PROMPT = "a photo"
STRENGTH = 0.6
SEED = 0
TIMEOUT_S = 1800


def _load_edit_ratios():
    arrays = []
    for f in sorted(glob.glob(os.path.join(MASK_RATIO_DIR, "*.npy"))):
        a = np.load(f).astype(float).ravel()
        arrays.append(a[(a > 0) & (a < 1)])
    return np.concatenate(arrays) if arrays else None


def build_workload(n, edit_frac, image_path, mask_path, rng):
    ratios = _load_edit_ratios()
    n_edit = int(round(n * edit_frac))
    specs = []
    for i in range(n):
        is_edit = i < n_edit
        if is_edit:
            ratio = float(rng.choice(ratios)) if ratios is not None else float(rng.uniform(0.1, 0.4))
            seqlen = max(1, int(ratio * FULL_SEQLEN))
        else:
            seqlen = FULL_SEQLEN
        specs.append(RequestSpec(prompt=PROMPT, image_path=image_path, mask_image_path=mask_path,
                                 strength=STRENGTH, seed=SEED + i, edit=is_edit, mask_seq_length=seqlen))
    rng.shuffle(specs)
    return specs


def poisson_offsets(n, rps, rng):
    if rps <= 0:
        return np.zeros(n)
    return np.cumsum(rng.exponential(1.0 / rps, size=n))


async def _send_one(session, server_url, service_id, inputs, route, seq_id):
    inputs = dict(inputs)
    inputs.setdefault("metadata", {})["sequence_id"] = seq_id
    start = time.time()
    try:
        async with session.post(
            f"{server_url}/api/workflow/{service_id}/inference", json={"inputs": inputs}
        ) as resp:
            body = await resp.json()
        results = (body or {}).get("results", {}) if isinstance(body, dict) else {}
        return {"route": route, "ok": isinstance(body, dict) and body.get("status") == "success",
                "latency": time.time() - start, "inference_latency": results.get("inference_latency")}
    except Exception as e:  # noqa: BLE001
        return {"route": route, "ok": False, "latency": time.time() - start,
                "inference_latency": None, "error": str(e)}


async def run_benchmark(server_url, n, edit_frac, rps, image_path, mask_path, out_dir):
    rng = np.random.default_rng(SEED)
    specs = build_workload(n, edit_frac, image_path, mask_path, rng)

    cfg_dir = os.path.join(out_dir, "configs")
    routed = [build_request(s, BASE_CONFIG, cfg_dir, i) for i, s in enumerate(specs)]
    n_edit = sum(1 for _, _, r in routed if r == "edit")
    print(f"Built {n} requests: {n_edit} edit, {n - n_edit} non-edit -> {server_url}")

    offsets = poisson_offsets(n, rps, rng)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=TIMEOUT_S)) as session:
        tasks = []
        wall_start = time.time()
        for i, ((service_id, inputs, route), offset) in enumerate(zip(routed, offsets)):
            wait = offset - (time.time() - wall_start)
            if wait > 0:
                await asyncio.sleep(wait)
            tasks.append(asyncio.create_task(_send_one(session, server_url, service_id, inputs, route, i)))
        results = await asyncio.gather(*tasks)
        wall = time.time() - wall_start

    report = summarize(results, wall)
    _print_report(report)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump({"report": report, "raw": results}, f, indent=2)
    print(f"\nRun dir -> {out_dir}")
    return report


def _pcts(values):
    if not values:
        return {"count": 0, "p50": None, "p90": None, "p99": None}
    arr = np.array(values, dtype=float)
    return {"count": len(values), "p50": float(np.percentile(arr, 50)),
            "p90": float(np.percentile(arr, 90)), "p99": float(np.percentile(arr, 99))}


def _class_stats(rows, wall):
    ok = [r for r in rows if r["ok"]]
    return {
        "requests": len(rows),
        "succeeded": len(ok),
        "throughput_rps": (len(ok) / wall) if wall > 0 else None,
        "e2e_latency": _pcts([r["latency"] for r in ok]),
        "server_inference_latency": _pcts([r["inference_latency"] for r in ok
                                           if r.get("inference_latency") is not None]),
    }


def summarize(results, wall):
    return {
        "wallclock_s": wall,
        "overall": _class_stats(results, wall),
        "edit": _class_stats([r for r in results if r["route"] == "edit"], wall),
        "non_edit": _class_stats([r for r in results if r["route"] == "non_edit"], wall),
    }


def _fmt(x):
    return f"{x:.3f}" if isinstance(x, (int, float)) else "  -  "


def _print_report(rep):
    print(f"\n===== FlashPS bench ({rep['wallclock_s']:.1f}s wall) =====")
    header = f"{'class':<10}{'ok/tot':>9}{'thrpt':>9}{'e2e p50':>10}{'e2e p90':>10}{'e2e p99':>10}{'inf p50':>10}"
    print(header)
    print("-" * len(header))
    for name in ("overall", "edit", "non_edit"):
        s = rep[name]
        e, inf = s["e2e_latency"], s["server_inference_latency"]
        print(f"{name:<10}{str(s['succeeded']) + '/' + str(s['requests']):>9}"
              f"{_fmt(s['throughput_rps']):>9}{_fmt(e['p50']):>10}{_fmt(e['p90']):>10}"
              f"{_fmt(e['p99']):>10}{_fmt(inf['p50']):>10}")


def main():
    p = argparse.ArgumentParser(description="FlashPS edit/non-edit batching + speed benchmark")
    p.add_argument("--server-url", default="http://localhost:8005")
    p.add_argument("--n", type=int, default=40, help="total requests")
    p.add_argument("--edit-frac", type=float, default=0.5, help="fraction routed as edit")
    p.add_argument("--rps", type=float, default=4.0, help="target arrival rate")
    p.add_argument("--image-path", default="/tmp/flashps_dummy_image.png",
                   help="real image (required by a real server; ignored by the mock)")
    p.add_argument("--mask-path", default="/tmp/flashps_dummy_mask.png")
    args = p.parse_args()

    out_dir = os.path.join(RUNS_DIR, f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"{datetime.now()} bench: n={args.n} edit_frac={args.edit_frac} rps={args.rps}")
    asyncio.run(run_benchmark(args.server_url, args.n, args.edit_frac, args.rps,
                              args.image_path, args.mask_path, out_dir))


if __name__ == "__main__":
    main()
