"""中身を見ずに答える自明なベースラインの予測ファイルを作る。

- always_B: 選択式は常に B、はい/いいえは常に はい、open は空
- 比較表の下限として使う。規則採点だけで採点できる

使い方:
    uv run python scripts/jvqa_trivial_baselines.py
    uv run python scripts/jvqa_eval.py --predictions results/always_B/predictions.jsonl --name always_B
"""

from __future__ import annotations

import json
from pathlib import Path

from movie_llm.eval.dataset import load_items


def main() -> None:
    items = load_items()
    for name, mc, yn in (("always_B", "B", "はい"), ("always_A", "A", "いいえ")):
        out = Path("results") / name
        out.mkdir(parents=True, exist_ok=True)
        with (out / "predictions.jsonl").open("w", encoding="utf-8") as f:
            for it in items:
                p = mc if it["type"] == "multi_choice" else yn if it["type"] == "yes_no" else ""
                f.write(json.dumps({"idx": it["idx"], "prediction": p}, ensure_ascii=False) + "\n")
        print(f"wrote {out / 'predictions.jsonl'}")


if __name__ == "__main__":
    main()
