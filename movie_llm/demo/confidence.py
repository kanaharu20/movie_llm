"""信頼度の計算と 3 段の振り分け（docs/videojev_eval.md「信頼度による 3 段の振り分け」）。

- pmax: 候補確率の最大値
- pmean: 選ばれた区間の中の候補確率の平均
- agree: 2 段目（Qwen3-VL）の時刻が 1 段目（VideoJev）の区間 ± agree_sec に入ったか
帯:
- high: pmax >= hi_pmax かつ agree  → 自動で返す（722 問で 95.3%、56%）
- low : pmean < lo_pmean            → 見つからない可能性が高い（54.5%、8%）
- mid : それ以外                     → 候補 3 つを人に見せる（82.6%、37%）
"""

from __future__ import annotations

HI_PMAX = 0.98
LO_PMEAN = 0.5
AGREE_SEC = 5.0

BAND_LABEL = {
    "high": "高: 自動で返せる水準（この帯の正解率 95%）",
    "mid": "中: 候補を確認してください（この帯の正解率 83%）",
    "low": "低: この素材には無い可能性があります（この帯の正解率 55%）",
}


def interval_stats(probs: list[float], interval: tuple[float, float], duration: float) -> tuple[float, float]:
    """(pmax, pmean)。候補 i は [i*step, (i+1)*step] 秒、step = duration / len(probs)。"""
    if not probs:
        return 0.0, 0.0
    step = duration / len(probs)
    s, e = interval
    lo = max(0, int(s / step))
    hi = min(len(probs), max(lo + 1, int(e / step) + 1))
    seg = probs[lo:hi]
    return float(max(probs)), float(sum(seg) / len(seg))


def top_peaks(probs: list[float], step: float, k: int = 3, suppress_sec: float = 30.0, min_prob: float = 0.1) -> list[dict]:
    """確率の列から、互いに suppress_sec 以上離れた山を k 個まで取る。山の裾（確率が山の半分以上、かつ 0.05 以上）を区間にする。

    min_prob 未満の山は候補にしない（確率がほぼ 0 の平地を候補にしないため）。
    """
    taken: list[int] = []
    out: list[dict] = []
    w = max(1, int(suppress_sec / step))
    for i in sorted(range(len(probs)), key=lambda j: -probs[j]):
        if probs[i] < min_prob:
            break
        if any(abs(i - j) < w for j in taken):
            continue
        taken.append(i)
        floor = max(probs[i] / 2, 0.05)
        lo = i
        while lo > 0 and probs[lo - 1] >= floor:
            lo -= 1
        hi = i
        while hi < len(probs) - 1 and probs[hi + 1] >= floor:
            hi += 1
        out.append({"start": lo * step, "end": (hi + 1) * step, "prob": float(probs[i])})
        if len(out) >= k:
            break
    return out


def band(pmax: float, pmean: float, agree: bool | None, hi_pmax: float = HI_PMAX, lo_pmean: float = LO_PMEAN) -> str:
    if pmean < lo_pmean:
        return "low"
    if pmax >= hi_pmax and (agree is None or agree):
        return "high"
    return "mid"


def mmss(sec: float | None) -> str:
    if sec is None:
        return "—"
    m, s = divmod(int(round(max(0.0, float(sec)))), 60)
    return f"{m}:{s:02d}"
