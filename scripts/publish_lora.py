"""学習済み LoRA（時刻特定、Qwen3-VL-4B 用）を Hugging Face Hub に公開する。

先に `uv run huggingface-cli login`（書き込み権限のトークン）。

    uv run python scripts/publish_lora.py --src results/grpo3_time/step_00400 --repo kanaharu20/movie-llm-timeloc-lora
"""

from __future__ import annotations

import argparse
from pathlib import Path

CARD = """---
license: apache-2.0
base_model: Qwen/Qwen3-VL-4B-Instruct
library_name: peft
language: [ja]
tags: [video, temporal-grounding, lora, grpo, japanese]
---

# movie_llm 時刻特定 LoRA（Qwen3-VL-4B-Instruct 用）

日本語の問いが指す場面が「動画の何分何秒か」を答えるように、Qwen3-VL-4B-Instruct を GRPO（強化学習）で学習した LoRA（r=16、33M パラメータ）。
出力は「映像の根拠 1〜2 行 + 答え: 分:秒」。

- 学習データ: 日本の CC BY 動画 1,000 本から作った場面データ 3,145 問（問い + 時刻区間。映像は含まない）
- 400 ステップ、A100 12 時間。詳細は https://github.com/kanaharu20/movie_llm の docs/grpo3_time.md
- Japanese Video-QA 722 問の時刻特定: 学習前 55.4% → 62.2%（32 フレーム）、69.9%（64 フレーム）。4 択の精度は変わらない

## 使い方

```python
from movie_llm.infer.qwen3vl import Qwen3VL
m = Qwen3VL("Qwen/Qwen3-VL-4B-Instruct", num_frames=32, adapter="{repo}")
```

デモ（VideoJev-4B と組み合わせた 2 段構成）は上のリポジトリの README。
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="results/grpo3_time/step_00400")
    ap.add_argument("--repo", default="kanaharu20/movie-llm-timeloc-lora")
    ap.add_argument("--private", action="store_true")
    args = ap.parse_args()
    from huggingface_hub import HfApi

    src = Path(args.src)
    assert (src / "adapter_model.safetensors").exists(), src
    (src / "README.md").write_text(CARD.replace("{repo}", args.repo), encoding="utf-8")
    api = HfApi()
    api.create_repo(args.repo, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(folder_path=str(src), repo_id=args.repo, repo_type="model", allow_patterns=["adapter_config.json", "adapter_model.safetensors", "README.md"])
    print(f"公開: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
