"""Japanese Video-QA (nablasinc/Japanese_Video-QA) の内訳を集計して Markdown に書き出す。

使い方:
    uv run python scripts/jvqa_stats.py            # data/jvqa に自動ダウンロードして集計
    uv run python scripts/jvqa_stats.py --out docs/jvqa_stats.md
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from movie_llm.eval.dataset import REPO, load_dataframe as load


def md_table(df: pd.DataFrame, index_name: str | None = None) -> str:
    """DataFrame を Markdown の表にする（pandas の to_markdown は tabulate 依存なので自前で）。"""
    df = df.copy()
    if index_name is not None:
        df.index.name = index_name
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def count_with_pct(s: pd.Series, name: str) -> pd.DataFrame:
    c = s.value_counts()
    return pd.DataFrame({"件数": c, "割合(%)": (c / len(s) * 100).round(1)}).rename_axis(name)


def parse_time(t: str) -> float | None:
    """'00:00:59.000' / '06:20:000' / '1:23' などを秒に変換。解釈できなければ None。"""
    if not isinstance(t, str) or not t.strip():
        return None
    parts = t.strip().split(":")
    try:
        nums = [float(p) for p in parts]
    except ValueError:
        return None
    if len(nums) == 3:
        h, m, s = nums
        # '06:20:000' は mm:ss:ms の表記と思われる（末尾が3桁で、先頭が小さい）
        if re.fullmatch(r"\d{1,2}:\d{2}:\d{3}", t.strip()):
            return h * 60 + m + s / 1000
        return h * 3600 + m * 60 + s
    if len(nums) == 2:
        m, s = nums
        return m * 60 + s
    if len(nums) == 1:
        return nums[0]
    return None


def time_format_label(t: str) -> str:
    if not isinstance(t, str) or not t.strip():
        return "(空)"
    t = t.strip()
    if re.fullmatch(r"\d{2}:\d{2}:\d{2}\.\d{3}", t):
        return "HH:MM:SS.mmm"
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{3}", t):
        return "MM:SS:mmm"
    if re.fullmatch(r"\d{1,2}:\d{2}:\d{2}", t):
        return "HH:MM:SS"
    if re.fullmatch(r"\d{1,2}:\d{2}\.\d{3}", t):
        return "MM:SS.mmm"
    if re.fullmatch(r"\d{1,2}:\d{2}", t):
        return "MM:SS"
    return "その他"


def build_report(df: pd.DataFrame) -> str:
    n = len(df)
    out: list[str] = []
    out.append("# Japanese Video-QA 内訳集計\n")
    out.append(f"データ: `{REPO}` test 分割、{n} 問 / 動画 {df['video_id'].nunique()} 本\n")
    out.append("集計スクリプト: `scripts/jvqa_stats.py`\n")

    out.append("## 回答形式 (type)\n")
    out.append(md_table(count_with_pct(df["type"], "type")) + "\n")

    out.append("## 質問カテゴリ (category)\n")
    out.append(md_table(count_with_pct(df["category"], "category")) + "\n")

    out.append("## 領域 (domain)\n")
    out.append(md_table(count_with_pct(df["domain"], "domain")) + "\n")

    out.append("## 動画の長さ (duration)\n")
    out.append(md_table(count_with_pct(df["duration"], "duration")) + "\n")

    out.append("## 回答形式 × 質問カテゴリ\n")
    ct = pd.crosstab(df["category"], df["type"], margins=True, margins_name="合計")
    out.append(md_table(ct, "category \\ type") + "\n")

    out.append("## 回答形式 × 領域\n")
    ct = pd.crosstab(df["domain"], df["type"], margins=True, margins_name="合計")
    out.append(md_table(ct, "domain \\ type") + "\n")

    out.append("## 質問カテゴリ × 領域\n")
    ct = pd.crosstab(df["domain"], df["category"], margins=True, margins_name="合計")
    out.append(md_table(ct, "domain \\ category") + "\n")

    out.append("## 動画の長さ × 回答形式\n")
    ct = pd.crosstab(df["duration"], df["type"], margins=True, margins_name="合計")
    out.append(md_table(ct, "duration \\ type") + "\n")

    out.append("## 自動採点できる問題の数\n")
    auto = df[df["type"].isin(["multi_choice", "yes_no"])]
    out.append(f"- multi_choice + yes_no: {len(auto)} 問 ({len(auto) / n * 100:.1f}%)。報酬に使えるのはこの範囲\n")
    out.append(f"- open: {int((df['type'] == 'open').sum())} 問。審判モデル(GPT-4o)での採点が必要\n")

    out.append("## 選択式の正解ラベル分布\n")
    mc = df[df["type"] == "multi_choice"]
    out.append(md_table(count_with_pct(mc["answer"], "answer")) + "\n")
    nchoice = mc["choices"].apply(len)
    out.append("選択肢の数:\n")
    out.append(md_table(count_with_pct(nchoice, "選択肢数")) + "\n")

    out.append("## はい/いいえ の正解分布\n")
    yn = df[df["type"] == "yes_no"]
    out.append(md_table(count_with_pct(yn["answer"], "answer")) + "\n")

    out.append("## 自由記述 (open) の答えの長さ\n")
    op = df[df["type"] == "open"]
    ln = op["answer"].str.len()
    out.append(f"- 文字数: 平均 {ln.mean():.1f}、中央値 {ln.median():.0f}、最小 {ln.min()}、最大 {ln.max()}\n")
    out.append("- 例:\n")
    for _, r in op.sample(5, random_state=0).iterrows():
        out.append(f"  - Q: {r['question']} / A: {r['answer']} ({r['category']})")
    out.append("")

    out.append("## 1本の動画あたりの問題数\n")
    per_video = df.groupby("video_id").size()
    out.append(md_table(count_with_pct(per_video, "問/動画")) + "\n")

    out.append("## 小領域 (subdomain)\n")
    out.append(f"- 種類: {df['subdomain'].nunique()}\n")
    sub = df.groupby(["domain", "subdomain"]).size().rename("件数").reset_index()
    sub = sub.sort_values(["domain", "件数"], ascending=[True, False]).set_index(["domain", "subdomain"])
    out.append(md_table(sub) + "\n")

    out.append("## 時刻 (start_time / end_time) の表記ゆれ\n")
    fmt = Counter(time_format_label(t) for t in pd.concat([df["start_time"], df["end_time"]]))
    out.append(md_table(pd.DataFrame({"件数": pd.Series(fmt)}).rename_axis("表記")) + "\n")
    out.append("参考値なので評価には使わない前提だが、使う場合は `parse_time` のように表記を揃える必要がある。\n")
    span = (df["end_time"].apply(parse_time) - df["start_time"].apply(parse_time)).dropna()
    span = span[span >= 0]
    out.append(f"- 根拠区間の長さ(秒、解釈できた {len(span)} 件): 中央値 {span.median():.1f}、平均 {span.mean():.1f}、最大 {span.max():.1f}\n")

    out.append("## データ品質のメモ\n")
    bad_mc = df[(df["type"] == "multi_choice") & ~df["answer"].isin(list("ABCD"))]
    bad_yn = df[(df["type"] == "yes_no") & ~df["answer"].isin(["はい", "いいえ"])]
    stray = df[(df["type"] != "multi_choice") & (df["choices"].apply(len) > 0)]
    out.append(f"- multi_choice で answer が A〜D でない: {len(bad_mc)} 件。自動採点では選択肢本文との一致も見る必要がある")
    for _, r in bad_mc.iterrows():
        out.append(f"  - {r['video_id']}: answer=`{r['answer']}` choices={list(r['choices'])}")
    out.append(f"- yes_no で answer が はい/いいえ でない: {len(bad_yn)} 件。実質 open として扱う")
    for _, r in bad_yn.iterrows():
        out.append(f"  - {r['video_id']}: Q: {r['question']} / A: {r['answer']}")
    out.append(f"- multi_choice 以外で choices が空でない: {len(stray)} 件")
    out.append("- 選択式の正解は B に偏っている。常に B と答えるだけで約4割取れるので、ベースラインの比較対象として記録しておく")
    out.append("- 選択肢の書式は `A) 本文` で統一されている\n")

    out.append("## 質問文の長さ\n")
    ql = df["question"].str.len()
    out.append(f"- 文字数: 平均 {ql.mean():.1f}、中央値 {ql.median():.0f}、最小 {ql.min()}、最大 {ql.max()}\n")

    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/jvqa")
    ap.add_argument("--out", default="docs/jvqa_stats.md")
    args = ap.parse_args()

    df = load(Path(args.data_dir))
    report = build_report(df)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
