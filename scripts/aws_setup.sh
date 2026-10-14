#!/usr/bin/env bash
# EC2（Deep Learning AMI, Ubuntu）で GRPO を回す準備。使い方: bash aws_setup.sh s3://<bucket>
# 1) uv と リポジトリ 2) 依存パッケージ 3) S3 からフレームを展開し、動画のダミーファイルを置く 4) チェックポイントがあれば戻す
set -euo pipefail
BUCKET="${1:?S3 バケット（例: s3://movie-llm-xxx）を指定}"
cd ~
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d movie_llm ] || git clone -q https://github.com/kanaharu20/movie_llm.git
cd movie_llm && git pull -q
uv sync -q
mkdir -p data/ja/frames data/ja/videos data/jvqa/frames data/jvqa/videos results
# フレーム（zip は手元で scripts/jvqa_cache_frames.py --zip で作ったもの）
aws s3 cp "$BUCKET/frames/ja_frames_f32_s640.zip" /tmp/ja.zip && uv run python -c "import zipfile;zipfile.ZipFile('/tmp/ja.zip').extractall('data/ja/frames')"
aws s3 cp "$BUCKET/frames/jvqa_frames_f32_s640.zip" /tmp/jvqa.zip && uv run python -c "import zipfile;zipfile.ZipFile('/tmp/jvqa.zip').extractall('data/jvqa/frames')"
for d in data/ja data/jvqa; do for v in "$d"/frames/*/; do touch "$d/videos/$(basename "$v").mp4"; done; done
# 学習データと dev（S3 に置いてあれば）
aws s3 cp "$BUCKET/data/train_time.jsonl" data/ja/train_time.jsonl || true
aws s3 cp "$BUCKET/data/dev_time.jsonl" data/ja/dev_time.jsonl || true
# 途中のチェックポイントがあれば戻す（スポットが止まったあとの再開）
aws s3 sync "$BUCKET/results/grpo3_time" results/grpo3_time || true
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "準備完了: cd ~/movie_llm で docs/aws.md の手順 4 を実行"
