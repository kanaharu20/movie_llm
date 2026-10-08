"""Japanese Video-QA の予測ファイルを採点する。

予測ファイルは JSONL。1行1問で、少なくとも "idx" と "prediction" を持つ。
    {"idx": 0, "prediction": "A) 江戸の吉原"}

使い方:
    uv run python scripts/jvqa_eval.py --predictions results/foo/predictions.jsonl --name foo
    uv run python scripts/jvqa_eval.py --predictions ... --name foo --judge          # GPT-4o でも採点（要 OPENAI_API_KEY）
    uv run python scripts/jvqa_eval.py --predictions ... --name foo --judge --limit 20

出力: results/<name>/items.jsonl（問題ごとの点）と results/<name>/report.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from movie_llm.eval.dataset import load_items
from movie_llm.eval.judge import Judge
from movie_llm.eval.report import build_report
from movie_llm.eval.rules import extract_choice, extract_yes_no, score_rule


def read_predictions(path: Path) -> dict[int, str]:
    preds: dict[int, str] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                preds[int(rec["idx"])] = str(rec.get("prediction", ""))
    return preds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--name", required=True, help="results/<name>/ に出力")
    ap.add_argument("--judge", action="store_true", help="GPT-4o でも採点する")
    ap.add_argument("--judge-model", default="gpt-4o")
    ap.add_argument("--judge-only-open", action="store_true", help="審判は open と規則採点できない問題だけに使う（API 節約）")
    ap.add_argument("--limit", type=int, default=None, help="先頭 N 問だけ採点")
    ap.add_argument("--all", action="store_true", help="予測が無い問題も空文字として採点する（既定は予測がある問題だけ）")
    ap.add_argument("--data-dir", default="data/jvqa")
    args = ap.parse_args()

    items = load_items(args.data_dir)
    if args.limit:
        items = items[: args.limit]
    preds = read_predictions(Path(args.predictions))
    missing = [it["idx"] for it in items if it["idx"] not in preds]
    if missing and args.all:
        print(f"警告: 予測が無い問題が {len(missing)} 件。空文字として採点する（例: {missing[:5]}）")
    elif missing:
        print(f"予測がある {len(items) - len(missing)} 問だけを採点する（全体 {len(items)} 問）")
        items = [it for it in items if it["idx"] in preds]

    rows = []
    for it in items:
        p = preds.get(it["idx"], "")
        extracted = None
        if it["type"] == "multi_choice":
            extracted = extract_choice(p, it["choices"])
        elif it["type"] == "yes_no":
            extracted = extract_yes_no(p)
        rows.append(
            {
                "idx": it["idx"],
                "video_id": it["video_id"],
                "type": it["type"],
                "category": it["category"],
                "domain": it["domain"],
                "duration": it["duration"],
                "answer": it["answer"],
                "prediction": p,
                "rule_extracted": extracted,
                "rule_score": score_rule(it, p),
                "judge_score": None,
            }
        )

    if args.judge:
        judge = Judge(model=args.judge_model)
        targets = [
            (i, it)
            for i, it in enumerate(items)
            if not (args.judge_only_open and rows[i]["rule_score"] is not None)
        ]
        jobs = [(it["question"], it["answer"], rows[i]["prediction"], it["choices"]) for i, it in targets]
        print(f"審判採点: {len(jobs)} 問 (model={args.judge_model})")
        scores = judge.score_many(jobs)
        for (i, _), s in zip(targets, scores):
            rows[i]["judge_score"] = s
        print(f"API 呼び出し: {judge.api_calls} 回（残りはキャッシュ）")

    for r in rows:
        r["final_score"] = r["judge_score"] if r["judge_score"] is not None else r["rule_score"]

    out_dir = Path("results") / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = build_report(rows, f"評価結果: {args.name}")
    (out_dir / "report.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
