#!/usr/bin/env bash
# Production launcher: Qwen3.6-27B Q4_K_M + DFlash DDTree decode + PFlash
# speculative prefill on a single RTX 3090 (24 GB).
#
# Accepts source prompts up to ~136K tokens (139264): PFlash compresses them
# with the Qwen3-0.6B drafter so the target only prefills the kept spans.
set -euo pipefail

BIN="${BIN:-/home/rhonstin/lucebox-hub/server/build-sm86/dflash_server}"
MODELS="${MODELS:-/mnt/models}"
PORT="${PORT:-8082}"

# Pin to the RTX 3090 by UUID: the CMP 90HX (GPU 1) is reserved for sidecar
# services (embeddings/vision) and must not get a stray CUDA context.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-GPU-fd98efd2-8559-aa03-d51e-99d494ae8061}"

# Shrink drafter prefill chunk transients (default 4096) so the 136K drafter
# pass fits alongside the parked server allocations on 24 GB.
export DFLASH_FP_CHUNK_S="${DFLASH_FP_CHUNK_S:-1024}"
# CUDA graphs cost ~1 GB of resident VRAM here for no measurable decode gain
# (62.9 vs 63.2 tok/s); that headroom is needed by the 136K drafter pass.
export GGML_CUDA_DISABLE_GRAPHS="${GGML_CUDA_DISABLE_GRAPHS:-1}"
# Sampled-verify: spec decode at any temperature. Requires the full-attention
# verify path (--fa-window 0 below) — a finite window starves the verify
# logit tail at long context and breaks sampling (see commit 9850dd3).
export DFLASH_SAMPLED_VERIFY="${DFLASH_SAMPLED_VERIFY:-1}"
# Disable pre-RoPE tail scoring: the NoPE K copy costs another ~7.6 GB at
# 136K source, which cannot fit on 24 GB together with the 15.2 GB drafter
# KV. Costs some far-position score quality (NIAH-validated below).
export DFLASH_FP_NOPE_TAIL="${DFLASH_FP_NOPE_TAIL:-0}"
# Use quantized matmul kernels instead of dequant+cuBLAS. Without this the
# target backend's CUDA pool retains the LM-head dequant high-water (~2.4 GB)
# after the first prefill, and every subsequent 136K drafter pass OOMs.
export GGML_CUDA_FORCE_MMQ="${GGML_CUDA_FORCE_MMQ:-1}"
# DDTree tree verify: verify a 22-node draft token tree per step instead of
# the single top-1 chain, recovering from single-token draft divergences
# (sampled or greedy). Measured: +29% decode at 57K ctx temp 1.0
# (23.7 -> 30.6 tok/s), +6-10% on short prompts, 12/12 tool calls.
export DFLASH_TREE_VERIFY="${DFLASH_TREE_VERIFY:-1}"

exec "$BIN" "$MODELS/Qwen3.6-27B-Q4_K_M.gguf" \
  --draft "$MODELS/dflash-draft-3.6-q4_k_m.gguf" \
  --draft-swa 2048 \
  --host 0.0.0.0 --port "$PORT" \
  --max-ctx 114688 \
  --chunk 256 \
  --fa-window 0 \
  --cache-type-k q4_0 --cache-type-v q4_0 \
  --ddtree --ddtree-budget 22 \
  --prefill-compression auto \
  --prefill-threshold 16000 \
  --prefill-curve 16000:0.5 40000:0.2 100000:0.1 139264:0.05 \
  --prefill-drafter "$MODELS/Qwen3-0.6B-BF16.gguf" \
  --draft-residency request-scoped \
  --kv-cache-dir /mnt/models/.cache/dflash-kv \
  --kv-cache-budget 16384 \
  --model-name qwen3.6-27b
