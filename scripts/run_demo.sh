#!/usr/bin/env bash
# 最小動作確認: 学習 → 可視化
set -euo pipefail
cd "$(dirname "$0")/.."

uv sync

# 1) 学習（head のみ）
uv run python -m clipmatch.train --csv data/sample.csv --out checkpoints/demo --epochs 3 --batch-size 8

# 2) 可視化（学習済み）
uv run python -m clipmatch.visualize --ckpt checkpoints/demo/best.pt --csv data/sample.csv --out outputs/demo

# 3) 可視化（zero-shot，学習不要）
uv run python -m clipmatch.visualize --zeroshot --threshold 0.25 --csv data/sample.csv --out outputs/zeroshot
