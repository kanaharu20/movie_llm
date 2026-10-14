"""場面の時刻を当てる課題の採点（scripts/jvqa_predict.py --task time の出力を読む）。

- 正解: 問いの start_time / end_time（動画の長さで表記を判別）
- 許容幅: max(5 秒, フレーム間隔 / 2)。フレーム数は予測ファイルの frames 列
- 正解率 = 区間 ± 許容幅に入った割合。得点 = 部分点込みの平均（学習の報酬と同じ式）
- 出力: results/<name>/items_time.jsonl と report_time.md

    uv run python scripts/jvqa_eval_time.py --predictions results/time-4b_f32/predictions.jsonl --name time-4b_f32
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import pandas as pd

from movie_llm.eval.dataset import load_items
from movie_llm.eval.report import md_table
from movie_llm.eval.timeloc import extract_time, gold_segment, score_time, tolerance


def summarize_time(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    def agg(g: pd.DataFrame) -> pd.Series:
        d = g["dist"].dropna()
        return pd.Series({
            "n": len(g),
            "正解率(%)": round(g["hit"].mean() * 100, 1),
            "得点": round(g["score"].mean(), 3),
            "時刻を読めず": int(g["pred_sec"].isna().sum()),
            "距離の中央値(秒)": round(statistics.median(d), 1) if len(d) else None,
            "許容幅の中央値(秒)": round(g["tol"].median(), 1),
        })

    if by is None:
        res = agg(df).to_frame("全体").T
        res.index.name = "区分"
    else:
        res = df.groupby(by).apply(agg, include_groups=False)
    res["n"] = res["n"].astype(int)
    res["時刻を読めず"] = res["時刻を読めず"].astype(int)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out-root", default="results")
    args = ap.parse_args()

    preds = {}
    for line in open(args.predictions, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            preds[r["idx"]] = r
    items = [it for it in load_items() if it["idx"] in preds]
    print(f"予測がある {len(items)} 問を採点する")

    rows = []
    for it in items:
        p = preds[it["idx"]]
        dur = p.get("video_duration")
        seg = gold_segment(it, dur)
        if seg is None or dur is None:
            continue
        tol = tolerance(dur, p.get("frames", 32))
        pred_sec = extract_time(p["prediction"])
        score, hit, dist = score_time(pred_sec, seg[0], seg[1], tol)
        rows.append({
            "idx": it["idx"], "video_id": it["video_id"], "type": it["type"], "category": it["category"], "domain": it["domain"],
            "duration": it["duration"], "video_duration": round(dur, 1), "start": round(seg[0], 1), "end": round(seg[1], 1), "tol": round(tol, 1),
            "prediction": p["prediction"], "pred_sec": pred_sec, "dist": dist, "hit": hit, "score": round(score, 3),
        })
    df = pd.DataFrame(rows)
    out_dir = Path(args.out_root) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "items_time.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 当て推量の基準: いつも動画の中央と答えた場合
    base_hits = []
    for r in rows:
        mid = r["video_duration"] / 2
        base_hits.append(score_time(mid, r["start"], r["end"], r["tol"])[1])
    out = [f"# 場面の時刻を当てる課題: {args.name}\n",
           f"- 問題数 {len(df)}（区間が読めなかった問題は除く）。正解 = 答えの時刻が正解区間 ± 許容幅に入る。許容幅 = max(5 秒, フレーム間隔 / 2)",
           f"- 得点 = 部分点込み（許容幅から外れた秒数 d で max(0, 1 - d / 30)）。学習の報酬と同じ式",
           f"- 当て推量の基準（常に動画の中央と答える）: 正解率 {sum(base_hits) / len(base_hits) * 100:.1f}%\n",
           "## 全体\n", md_table(summarize_time(df)) + "\n"]
    for col in ("duration", "category", "type", "domain"):
        out.append(f"## {col} 別\n")
        out.append(md_table(summarize_time(df, col)) + "\n")
    report = "\n".join(out)
    (out_dir / "report_time.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
