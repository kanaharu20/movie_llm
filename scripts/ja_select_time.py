"""検査を通った場面（data/ja/time_verified.jsonl の pass=true）を、動画単位で学習用と dev に分ける。

- 1 場面 = 1 問。問いの言い方は「〜している場面」（query）と「〜のは動画のどの時点ですか」（question）を半々にする
- dev は動画単位で取り分ける（既定 15% の動画）。train と dev の動画は重ならない
- 出力の項目は GRPO（type=time）と評価がそのまま読める: id, video_id, question, start, end, video_duration, type

    uv run python scripts/ja_select_time.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

DATA = Path("data/ja")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verified", default=str(DATA / "time_verified.jsonl"))
    ap.add_argument("--dev-ratio", type=float, default=0.15)
    ap.add_argument("--train-out", default=str(DATA / "train_time.jsonl"))
    ap.add_argument("--dev-out", default=str(DATA / "dev_time.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    rows = [json.loads(l) for l in open(args.verified, encoding="utf-8") if l.strip()]
    ok = [r for r in rows if r.get("pass")]
    videos = sorted({r["video_id"] for r in ok})
    rng.shuffle(videos)
    n_dev = max(1, int(len(videos) * args.dev_ratio))
    dev_videos = set(videos[:n_dev])

    def to_item(r: dict, k: int) -> dict:
        q = r["question"] if k % 2 == 0 else r["query"]
        return {"id": r["id"], "video_id": r["video_id"], "question": q, "phrasing": "question" if k % 2 == 0 else "query",
                "start": r["start"], "end": r["end"], "video_duration": r["video_duration"], "type": "time", "domain": r.get("domain")}

    train, dev = [], []
    for k, r in enumerate(ok):
        (dev if r["video_id"] in dev_videos else train).append(to_item(r, k))
    rng.shuffle(train)
    for path, items in ((args.train_out, train), (args.dev_out, dev)):
        with open(path, "w", encoding="utf-8") as f:
            for it in items:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        doms = dict(sorted(collections.Counter(it["domain"] for it in items).items()))
        print(f"{Path(path).name}: {len(items)} 問、動画 {len({it['video_id'] for it in items})} 本、domain {doms}")


if __name__ == "__main__":
    main()
