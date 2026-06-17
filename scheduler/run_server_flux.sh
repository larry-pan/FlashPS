#!/usr/bin/env bash
# Launch the FlashPS server for Flux_inpaint (FLUX.1-schnell) with continuous batching.
#
# Run from the scheduler/ directory:  bash run_server_flux.sh
#
# Needs >=2 visible GPUs: GPU0 = coordinator (post-processing), GPU1..N-1 = workers.
# Hostname + GPU count are auto-detected by server.py (get_node_config / nvidia-smi).
# Each worker loads a full FLUX copy (~33GB), so each GPU wants >=40GB.
set -e

timestamp=$(date +"%Y%m%d_%H%M%S")

# Cache folders only need to exist; empty is fine for non-edit / cache-off serving.
mkdir -p flux_cache/cached_kv flux_cache/cached_latents
mkdir -p 8gpu_server_log

LOG="8gpu_server_log/log_flux_${timestamp}.log"
echo "Server log -> $(pwd)/$LOG"

python -u server.py \
  --config dist_config.yml \
  --pipeline-name Flux_inpaint \
  --worker-max-batch-size 8 \
  --scheduling-baseline flops_balance \
  --cache-config cache_configs/flux_cache_config.yml \
  > "$LOG" 2>&1 &

echo "Booting (each worker loads FLUX weights; ~1-3 min)... tailing log:"
sleep 120
tail -n 40 "$LOG"
echo
echo "Ready when the log shows uvicorn 'Application startup complete' + 'Uvicorn running on http://0.0.0.0:8005'."
echo "(There is no /health endpoint; readiness = that log line, or a successful inference POST.)"
