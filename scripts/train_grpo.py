"""GRPO（LoRA）で動画 QA を学習する。

    # 手元の 2B で動作確認（ベンチマークの 4 問、小さい設定）
    uv run python scripts/train_grpo.py --items data/perception_test/smoke_items.jsonl --out results/grpo_smoke \\
        --model Qwen/Qwen3-VL-2B-Instruct --frames 8 --group-size 4 --prompts-per-step 2 --micro-batch 2 --max-steps 2

    # Colab A100: Perception Test の学習セット（2 回目: count は数を答える形式、動的サンプリング）
    uv run python scripts/train_grpo.py --items data/perception_test/train_grpo2.jsonl --out /content/drive/MyDrive/movie_llm/grpo2_4b \\
        --model Qwen/Qwen3-VL-4B-Instruct --frames 32 --video-dir /content/pt_videos --frames-dir /content/pt_frames \\
        --group-size 8 --prompts-per-step 4 --micro-batch 4 --max-steps 300 --save-every 20
"""

from __future__ import annotations

import argparse
import json

from movie_llm.train.grpo import GRPOConfig, GRPOTrainer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen3-VL-2B-Instruct")
    ap.add_argument("--video-dir", default="data/jvqa/videos")
    ap.add_argument("--frames-dir", default="data/jvqa/frames")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=4)
    ap.add_argument("--micro-batch", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--beta", type=float, default=0.02)
    ap.add_argument("--max-steps", type=int, default=100)
    ap.add_argument("--save-every", type=int, default=20)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--numeric-partial", type=float, default=0.5, help="数を答える問題で差 1 のときの報酬")
    ap.add_argument("--max-draws", type=int, default=12, help="1 ステップで引く問題数の上限（動的サンプリング）")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    cfg = GRPOConfig(
        model_id=args.model, out_dir=args.out, video_dir=args.video_dir, frames_dir=args.frames_dir, num_frames=args.frames,
        group_size=args.group_size, prompts_per_step=args.prompts_per_step, micro_batch=args.micro_batch,
        max_new_tokens=args.max_new_tokens, lr=args.lr, beta=args.beta, max_steps=args.max_steps, save_every=args.save_every,
        lora_r=args.lora_r, seed=args.seed, gradient_checkpointing=not args.no_grad_ckpt,
        numeric_partial=args.numeric_partial, max_draws=args.max_draws,
    )
    print(f"学習データ {len(items)} 問")
    GRPOTrainer(cfg, items).train()


if __name__ == "__main__":
    main()
