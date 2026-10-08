"""採点結果の集計と Markdown レポート。"""

from __future__ import annotations

from typing import Any

import pandas as pd


def md_table(df: pd.DataFrame) -> str:
    df = df.reset_index()
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def summarize(df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """final_score から 公式表と同じ列（Avg, Score 1/2/3）と正解率を出す。"""

    def agg(g: pd.DataFrame) -> pd.Series:
        s = g["final_score"].dropna()
        return pd.Series(
            {
                "n": len(g),
                "採点済": len(s),
                "Avg": round(s.mean(), 3) if len(s) else None,
                "Score 1": int((s == 1).sum()),
                "Score 2": int((s == 2).sum()),
                "Score 3": int((s == 3).sum()),
                "正解率(%)": round((s == 3).mean() * 100, 1) if len(s) else None,
            }
        )

    if by is None:
        res = agg(df).to_frame("全体").T
        res.index.name = "区分"
    else:
        res = df.groupby(by).apply(agg, include_groups=False)
    for c in ("n", "採点済", "Score 1", "Score 2", "Score 3"):
        res[c] = res[c].astype(int)
    return res


def build_report(rows: list[dict[str, Any]], title: str) -> str:
    df = pd.DataFrame(rows)
    out = [f"# {title}\n"]
    n_rule = df["rule_score"].notna().sum()
    n_judge = df["judge_score"].notna().sum()
    out.append(f"- 問題数: {len(df)}、規則採点: {n_rule}、審判採点: {n_judge}")
    out.append("- final_score は審判の点があればそれを、なければ規則採点の点を使う")
    out.append("- 正解率は Score 3 の割合。公式表の Avg は (1×Score1 + 2×Score2 + 3×Score3) / 800\n")

    out.append("## 全体\n")
    out.append(md_table(summarize(df)) + "\n")
    for col in ("type", "category", "domain", "duration"):
        out.append(f"## {col} 別\n")
        out.append(md_table(summarize(df, col)) + "\n")

    both = df.dropna(subset=["rule_score", "judge_score"])
    if len(both):
        agree = (both["rule_score"] == both["judge_score"]).mean() * 100
        out.append("## 規則採点と審判採点の一致\n")
        out.append(f"- 両方の点がある {len(both)} 問のうち一致 {agree:.1f}%")
        ct = pd.crosstab(both["rule_score"], both["judge_score"])
        ct.index.name = "rule \\ judge"
        out.append(md_table(ct) + "\n")

    unparsed = df[(df["type"].isin(["multi_choice", "yes_no"])) & (df["rule_extracted"].isna())]
    if len(unparsed):
        out.append("## 規則採点で選択を読み取れなかった出力\n")
        out.append(f"- {len(unparsed)} 件（1点扱い）。例:")
        for _, r in unparsed.head(10).iterrows():
            out.append(f"  - idx={r['idx']} ({r['type']}): {str(r['prediction'])[:80]!r}")
        out.append("")
    return "\n".join(out)
