"""難易度の測定結果から GRPO の学習セットと dev を作る。

- 全正解（8/8）と全不正解（0/8）を除く（グループ内にばらつきがなく学習信号が出ない）
- 残った問題（1〜7/8）を動画単位で train と dev に分ける。dev は category ごとの下限（既定: count 100、temporal 100、
  他 50）を満たすまで動画を取り分ける。train と dev は同じ分布から取り、動画は重ならない
- count で選択肢が全部数字の問題は「数を答える」形式（type=numeric、answer は数字）に変える。元の選択肢と記号は
  choices / mc_answer に残す。--keep-mc を付けると変えない
- 問題に学習前（3択）の正解率 base_acc を付ける

1 回目（train_grpo.jsonl / dev_grpo.jsonl）は旧版で作った。dev が学習に使わない 0/8・8/8 の問題から取られ、
count が 27 問しか無かったので、2 回目から本スクリプトの方式にした。

    uv run python scripts/pt_select.py            # -> train_grpo2.jsonl, dev_grpo2.jsonl
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path

DATA = Path("data/perception_test")
DEV_MIN = {"count": 100, "temporal": 100, "causal": 50, "spatial": 50, "action": 50}


def to_numeric(it: dict) -> dict:
    """3択の count 問題を、数字だけで答える形式にする。"""
    it = dict(it)
    it["mc_answer"] = it["answer"]
    it["answer"] = it["numeric_answer"]
    it["type"] = "numeric"
    return it


def describe(name: str, rows: list[dict]) -> None:
    cats = dict(sorted(collections.Counter(it["category"] for it in rows).items()))
    types = dict(collections.Counter(it["type"] for it in rows))
    accs = [it["base_acc"] for it in rows]
    print(f"{name}: {len(rows)} 問、動画 {len({it['video_id'] for it in rows})} 本、学習前(3択)の正解率 {sum(accs)/len(accs):.3f}")
    print(f"  category {cats}\n  type {types}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(DATA / "items_train.jsonl"))
    ap.add_argument("--difficulty", default=str(DATA / "difficulty_4b_f32.jsonl"))
    ap.add_argument("--lo", type=float, default=0.125, help="この正解率未満は除く（既定: 0/8 だけ除く）")
    ap.add_argument("--hi", type=float, default=0.875, help="この正解率超は除く（既定: 8/8 だけ除く）")
    ap.add_argument("--dev-min", default=",".join(f"{k}={v}" for k, v in DEV_MIN.items()), help="dev の category ごとの最低問数")
    ap.add_argument("--keep-mc", action="store_true", help="count を 3択のまま残す")
    ap.add_argument("--train-out", default=str(DATA / "train_grpo2.jsonl"))
    ap.add_argument("--dev-out", default=str(DATA / "dev_grpo2.jsonl"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev_min = {k: int(v) for k, v in (kv.split("=") for kv in args.dev_min.split(","))}

    items = {json.loads(l)["id"]: json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()}
    diff = {json.loads(l)["id"]: json.loads(l) for l in open(args.difficulty, encoding="utf-8") if l.strip()}
    rng = random.Random(args.seed)

    pool = []
    for iid, d in diff.items():
        if iid not in items or not (args.lo <= d["acc"] <= args.hi):
            continue
        it = dict(items[iid], base_acc=d["acc"])
        if not args.keep_mc and it["category"] == "count" and "numeric_answer" in it:
            it = to_numeric(it)
        pool.append(it)

    # 動画単位で dev を取り分ける。足りない category の問題を含む動画を優先して加える
    by_video: dict[str, list[dict]] = collections.defaultdict(list)
    for it in pool:
        by_video[it["video_id"]].append(it)
    videos = sorted(by_video)
    rng.shuffle(videos)
    dev_count: collections.Counter = collections.Counter()
    dev_videos: set[str] = set()
    for v in videos:
        if all(dev_count[c] >= n for c, n in dev_min.items()):
            break
        if any(dev_count[it["category"]] < dev_min.get(it["category"], 0) for it in by_video[v]):
            dev_videos.add(v)
            dev_count.update(it["category"] for it in by_video[v])

    dev = [it for it in pool if it["video_id"] in dev_videos]
    train = [it for it in pool if it["video_id"] not in dev_videos]
    rng.shuffle(dev)

    for path, rows in ((args.train_out, train), (args.dev_out, dev)):
        with open(path, "w", encoding="utf-8") as f:
            for it in rows:
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        describe(Path(path).name, rows)
    assert not ({it["video_id"] for it in train} & {it["video_id"] for it in dev})


if __name__ == "__main__":
    main()
