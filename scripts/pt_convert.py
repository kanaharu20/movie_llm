"""Perception Test の 3択 QA を、本リポジトリの学習用の形式（日本語）に変換する。

出力: data/perception_test/train_items.jsonl（1行1問）
    {"id": "pt_train_video_9431_0", "video_id": "video_9431", "split": "train",
     "question": "...", "choices": ["A) ...", "B) ...", "C) ..."], "answer": "B",
     "type": "multi_choice", "category": "count", "tags": [...], "area": "...", "reasoning": "...",
     "numeric_answer": "3"  # 選択肢が全部数字のときだけ。数を答える open 形式に使う
    }

- 問いは人手の対訳表（translations_questions_manual.json）、選択肢はモデル訳（translations.json）
- 文字並べの問いの選択肢と数字だけの選択肢は訳さない
- 正解の記号は均等になるよう、問題ごとに選択肢をシャッフルする（乱数の種を固定）
- 訳が無い選択肢を含む問題は除外して件数を報告する

    uv run python scripts/pt_convert.py
    uv run python scripts/pt_convert.py --splits train valid
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import re
from pathlib import Path

DATA = Path("data/perception_test")
LETTER_Q = re.compile(r"\bletters?\b|\bword\b", re.I)
LETTERS = "ABC"

def is_letter_sequence(option: str) -> bool:
    """文字並べ問題の選択肢（'bad', 'dab', 'b, a, d', 'gt'）かどうか。英語の語句は False。"""
    toks = [t.strip() for t in option.split(",")]
    return all(re.fullmatch(r"[a-z]{1,5}", t) for t in toks) and option.lower() not in {"yes", "no", "none"}

# Perception Test の tag/area/reasoning → ベンチマークの category
def to_category(q: dict) -> str:
    tags = set(q["tag"])
    if tags & {"object counting", "action counting", "event counting"}:
        return "count"
    if q["reasoning"] in ("explanatory", "counterfactual", "predictive"):
        return "causal"
    if tags & {"sequencing", "event recall", "change detection", "object permanence", "task completion"}:
        return "temporal"
    if tags & {"action recognition", "adversarial action", "distractor action", "motion"}:
        return "action"
    return "spatial"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", nargs="+", default=["train", "valid"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(DATA / "items.jsonl"))
    args = ap.parse_args()

    qs = [q for q in json.load(open(DATA / "mc_all.json")) if q["split"] in args.splits]
    q_ja = json.load(open(DATA / "translations_questions_manual.json"))
    o_ja = json.load(open(DATA / "translations.json")) if (DATA / "translations.json").exists() else {}
    rng = random.Random(args.seed)

    out, missing_opts, skipped = [], collections.Counter(), 0
    for q in qs:
        if q["question"] not in q_ja:
            skipped += 1
            continue
        is_letter = bool(LETTER_Q.search(q["question"]))
        opts_ja = []
        ok = True
        for o in q["options"]:
            if (is_letter and is_letter_sequence(o)) or re.fullmatch(r"\d+", o):
                opts_ja.append(o)
            elif o in o_ja:
                opts_ja.append(o_ja[o])
            else:
                missing_opts[o] += 1
                ok = False
        if not ok:
            skipped += 1
            continue
        order = list(range(3))
        rng.shuffle(order)
        choices = [f"{LETTERS[i]}) {opts_ja[j]}" for i, j in enumerate(order)]
        answer = LETTERS[order.index(q["answer_id"])]
        item = {
            "id": f"pt_{q['split']}_{q['video_id']}_{q['id']}",
            "video_id": q["video_id"],
            "split": q["split"],
            "question": q_ja[q["question"]],
            "question_en": q["question"],
            "choices": choices,
            "answer": answer,
            "type": "multi_choice",
            "category": to_category(q),
            "tags": q["tag"],
            "area": q["area"],
            "reasoning": q["reasoning"],
        }
        if all(re.fullmatch(r"\d+", o) for o in q["options"]):
            item["numeric_answer"] = q["options"][q["answer_id"]]
        out.append(item)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in out:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"出力 {len(out)} 問 → {args.out}（除外 {skipped}、訳が無い選択肢 {len(missing_opts)} 種）")
    print("category:", dict(collections.Counter(it["category"] for it in out)))
    print("answer:", dict(collections.Counter(it["answer"] for it in out)))
    print("数を答える形式にできる問題:", sum("numeric_answer" in it for it in out))
    for it in out[:3]:
        print(json.dumps({k: it[k] for k in ("question", "choices", "answer", "category")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
