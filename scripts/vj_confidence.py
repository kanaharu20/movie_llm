"""VideoJev の確率と 2 段目（Qwen）の一致から、自動 / 候補提示 / 該当なし の 3 段に振り分ける境目を決める。

入力:
  --vj   VideoJev の予測（probabilities つき、scripts/videojev_predict.py）と、その採点 items_time.jsonl（同じフォルダ）
  --ts   2 段構成の予測（scripts/jvqa_two_stage.py）。無ければ一致の信号は使わない
信頼度:
  pmax  = 候補確率の最大値
  pmean = 選ばれた区間の中の候補確率の平均
  agree = 2 段目の時刻が VideoJev の区間 ± 5 秒に入ったか
答え: 一致なら 2 段目の時刻、そうでなければ VideoJev の区間の中点（docs/videojev_eval.md の「一致で切り替え」）
出力: 帯ごとの件数と正解率、候補 3 つまで見せたときに正解を含む割合（recall@3）。results/<vj>/confidence.md に保存。

使い方:
    uv run python scripts/vj_confidence.py --vj results/videojev-4b-p --ts results/two_stage-4b
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from movie_llm.demo.confidence import interval_stats
from movie_llm.eval.timeloc import score_time


def top_peaks(probs: list[float], step: float, k: int = 3, suppress_sec: float = 30.0) -> list[tuple[float, float, float]]:
    """確率の列から、互いに suppress_sec 以上離れた山を k 個まで取る。(開始秒, 終了秒, 山の確率)。"""
    taken: list[int] = []
    out = []
    order = sorted(range(len(probs)), key=lambda i: -probs[i])
    w = max(1, int(suppress_sec / step))
    for i in order:
        if any(abs(i - j) < w for j in taken):
            continue
        taken.append(i)
        # 山の裾（確率が山の半分を下回るまで）を区間にする
        lo = i
        while lo > 0 and probs[lo - 1] >= probs[i] / 2:
            lo -= 1
        hi = i
        while hi < len(probs) - 1 and probs[hi + 1] >= probs[i] / 2:
            hi += 1
        out.append((lo * step, (hi + 1) * step, probs[i]))
        if len(out) >= k:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vj", default="results/videojev-4b-p")
    ap.add_argument("--ts", default="results/two_stage-4b")
    ap.add_argument("--hi-pmax", type=float, default=0.985, help="自動で返す帯の pmax の下限")
    ap.add_argument("--lo-pmean", type=float, default=0.5, help="該当なしにする帯の pmean の上限")
    ap.add_argument("--agree-sec", type=float, default=5.0)
    args = ap.parse_args()

    vj = Path(args.vj)
    P = {r["idx"]: r for r in map(json.loads, open(vj / "predictions.jsonl", encoding="utf-8"))}
    I = {r["idx"]: r for r in map(json.loads, open(vj / "items_time.jsonl", encoding="utf-8"))}
    ts_path = Path(args.ts) / "predictions.jsonl"
    TS = {r["idx"]: r for r in map(json.loads, open(ts_path, encoding="utf-8"))} if ts_path.exists() else {}
    ids = sorted(i for i in P if i in I and P[i].get("probabilities"))

    rows = []
    for i in ids:
        p = P[i]["probabilities"]
        dur = I[i]["video_duration"]
        K = len(p)
        step = dur / K
        s, e = P[i]["interval"]
        pmax, pmean = interval_stats(p, (s, e), dur)
        t2 = TS.get(i, {}).get("stage2_abs_sec")
        agree = t2 is not None and (s - args.agree_sec) <= t2 <= (e + args.agree_sec)
        ans = t2 if agree else (s + e) / 2
        _, hit, _ = score_time(ans, I[i]["start"], I[i]["end"], I[i]["tol"])
        peaks = top_peaks(p, step)
        hit3 = any(score_time((a + b) / 2, I[i]["start"], I[i]["end"], I[i]["tol"])[1] for a, b, _ in peaks)
        if pmean < args.lo_pmean:
            band = "low"
        elif pmax >= args.hi_pmax and (agree or not TS):
            band = "high"
        else:
            band = "mid"
        rows.append(dict(idx=i, band=band, hit=bool(hit), hit3=bool(hit3), vj_hit=bool(I[i]["hit"]), pmax=pmax, pmean=pmean, agree=agree, duration=I[i]["duration"]))

    lines = [f"# 信頼度による振り分け: {vj.name}", ""]
    lines.append(f"- 高（自動で返す）: pmax ≥ {args.hi_pmax}" + ("、かつ 2 段目と一致" if TS else ""))
    lines.append(f"- 低（該当なしとして人に回す）: 区間内の確率の平均 pmean < {args.lo_pmean}")
    lines.append("- 中（候補 3 つを人に見せる）: それ以外")
    lines.append("- 答え: 一致なら 2 段目の時刻、そうでなければ VideoJev の区間の中点。recall@3 は確率の山 3 つのどれかが当たる割合")
    lines.append("")
    lines.append("| 帯 | n | 割合 | 正解率 | recall@3 | VideoJev 単体の正解率 |")
    lines.append("|---|---|---|---|---|---|")
    n_all = len(rows)
    for band in ("high", "mid", "low"):
        sub = [r for r in rows if r["band"] == band]
        if not sub:
            continue
        lines.append(f"| {band} | {len(sub)} | {len(sub)/n_all:.0%} | {sum(r['hit'] for r in sub)/len(sub):.1%} | {sum(r['hit3'] for r in sub)/len(sub):.1%} | {sum(r['vj_hit'] for r in sub)/len(sub):.1%} |")
    lines.append(f"| 全体 | {n_all} | 100% | {sum(r['hit'] for r in rows)/n_all:.1%} | {sum(r['hit3'] for r in rows)/n_all:.1%} | {sum(r['vj_hit'] for r in rows)/n_all:.1%} |")
    lines.append("")
    lines.append("## 長さ別")
    lines.append("")
    lines.append("| 長さ | 帯 | n | 正解率 |")
    lines.append("|---|---|---|---|")
    for dur in ("short", "medium"):
        for band in ("high", "mid", "low"):
            sub = [r for r in rows if r["band"] == band and r["duration"] == dur]
            if sub:
                lines.append(f"| {dur} | {band} | {len(sub)} | {sum(r['hit'] for r in sub)/len(sub):.1%} |")
    lines.append("")
    lines.append("## pmax を下げたときの自動の帯（一致あり）")
    lines.append("")
    lines.append("| pmax の下限 | n | 割合 | 正解率 |")
    lines.append("|---|---|---|---|")
    for thr in (0.995, 0.99, 0.985, 0.98, 0.97, 0.95, 0.9):
        sub = [r for r in rows if r["pmax"] >= thr and (r["agree"] or not TS) and r["pmean"] >= args.lo_pmean]
        if sub:
            lines.append(f"| {thr} | {len(sub)} | {len(sub)/n_all:.0%} | {sum(r['hit'] for r in sub)/len(sub):.1%} |")
    report = "\n".join(lines) + "\n"
    (vj / "confidence.md").write_text(report, encoding="utf-8")
    with (vj / "confidence_items.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(report)


if __name__ == "__main__":
    main()
