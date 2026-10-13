"""難易度の測定結果から GRPO の学習セットを作る。

- 全正解（8/8）と全不正解（0/8）を除く（グループ内にばらつきがなく学習信号が出ない）
- 問題に学習前の正解率 base_acc を付ける（学習後の比較、カリキュラムに使う）
- 出力: data/perception_test/train_grpo.jsonl、学習に使わない問題のうち評価用に 300 問を held-out として dev_grpo.jsonl

    uv run python scripts/pt_select.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

DATA = Path("data/perception_test")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(DATA / "items_train.jsonl"))
    ap.add_argument("--difficulty", default=str(DATA / "difficulty_4b_f32.jsonl"))
    ap.add_argument("--lo", type=float, default=0.125, help="この正解率未満は除く（既定: 0/8 だけ除く）")
    ap.add_argument("--hi", type=float, default=0.875, help="この正解率超は除く（既定: 8/8 だけ除く）")
    ap.add_argument("--dev", type=int, default=300, help="評価用に取り分ける問題数（学習セットと動画が重ならないよう選ぶ）")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()}
    diff = {json.loads(l)["id"]: json.loads(l) for l in open(args.difficulty, encoding="utf-8") if l.strip()}
    rng = random.Random(args.seed)

    train = []
    for iid, d in diff.items():
        if iid not in items or not (args.lo <= d["acc"] <= args.hi):
            continue
        it = dict(items[iid])
        it["base_acc"] = d["acc"]
        train.append(it)
    train_videos = {it["video_id"] for it in train}

    # dev: 学習セットに含まれない動画の問題から、全正解・全不正解を含めて無作為に選ぶ（偏りのない評価用）
    pool = [dict(items[i], base_acc=d["acc"]) for i, d in diff.items() if i in items and items[i]["video_id"] not in train_videos]
    rng.shuffle(pool)
    dev = pool[: args.dev]

    for name, rows in (("train_grpo.jsonl", train), ("dev_grpo.jsonl", dev)):
        with (DATA / name).open("w", encoding="utf-8") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        cats = dict(collections.Counter(it["category"] for it in rows))
        accs = [it["base_acc"] for it in rows]
        print(f"{name}: {len(rows)} 問、動画 {len({it['video_id'] for it in rows})} 本、学習前の正解率 {sum(accs)/len(accs):.3f}、category {cats}")
    print("数を答える形式にできる学習問題:", sum("numeric_answer" in it for it in train))


if __name__ == "__main__":
    main()
