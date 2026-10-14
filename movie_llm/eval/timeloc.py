"""場面の時刻を当てる課題（シーン検索）の採点。

Japanese Video-QA の全 800 問には、問いが指す場面の区間 start_time / end_time が付いている。
「この問いの場面は動画の何分何秒か」と聞き、答えの時刻が区間（± 許容幅）に入れば正解とする。
4 択や自由記述と違い、800 問すべてを機械で採点できる。学習（GRPO）の報酬にもそのまま使う。

- 時刻の表記は 6 種類が混在する。'00:01:00.000' は「時:分:秒」ではなく「分:秒:フレーム」で書かれていることがあり、
  動画の長さと照らして判別する
- 許容幅 = max(5 秒, フレーム間隔 / 2)。32 フレームなら 2 分の動画で 5 秒、20 分の動画で 19 秒
- 部分点: 区間 ± 許容幅から外れた秒数 d に対して max(0, 1 - d / 30)。30 秒以上外れたら 0
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from movie_llm.eval.rules import normalize

PARTIAL_WINDOW = 30.0  # 許容幅から何秒外れたら 0 点になるか


def _hms(h: int, m: int, s: float) -> float:
    return h * 3600 + m * 60 + s


def parse_time(text: str | None, video_duration: float | None = None) -> float | None:
    """注釈の時刻表記を秒に直す。video_duration があれば、長さを超える読み方を捨てて別の読み方を試す。"""
    s = normalize(text)
    if not s:
        return None
    cands: list[float] = []
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2}):(\d{1,2})\.(\d{1,3})", s)  # 時:分:秒.ミリ秒 または 分:秒:フレーム.ミリ秒
    if m:
        h, mi, se, ms = m.groups()
        cands = [_hms(int(h), int(mi), int(se) + int(ms.ljust(3, "0")) / 1000), _hms(0, int(h), int(mi) + int(se) / 100)]
    m = m or re.fullmatch(r"(\d{1,2}):(\d{1,2}):(\d{3})", s)  # 分:秒:ミリ秒
    if m and not cands:
        mi, se, ms = m.groups()
        cands = [_hms(0, int(mi), int(se) + int(ms) / 1000)]
    if not cands:
        m = re.fullmatch(r"(\d{1,2}):(\d{1,2}):(\d{1,2})", s)  # 時:分:秒 または 分:秒:フレーム
        if m:
            h, mi, se = m.groups()
            cands = [_hms(int(h), int(mi), int(se)), _hms(0, int(h), int(mi) + int(se) / 100)]
    if not cands:
        m = re.fullmatch(r"(\d{1,2}):(\d{1,2})(?:\.(\d{1,3}))?", s)  # 分:秒(.ミリ秒)
        if m:
            mi, se, ms = m.groups()
            cands = [_hms(0, int(mi), int(se) + (int(ms.ljust(3, "0")) / 1000 if ms else 0))]
    if not cands:
        return None
    if video_duration is not None:
        ok = [c for c in cands if c <= video_duration + 2]
        if ok:
            return ok[0]
    return cands[0]


def gold_segment(item: dict, video_duration: float | None) -> tuple[float, float] | None:
    s = parse_time(item.get("start_time"), video_duration)
    e = parse_time(item.get("end_time"), video_duration)
    if s is None or e is None:
        return None
    if e < s:
        s, e = e, s
    return s, e


# モデル出力から時刻を読む。「答え: 01:42」「1:42」「1分42秒」「102秒」「01:42〜01:50」（最初の時刻を採る）
_ANS_LINE = re.compile(r"答え\s*[:：]\s*(.+)")
_MMSS = re.compile(r"(?<!\d)(\d{1,2})[:：](\d{2})(?:[:：](\d{2}))?(?!\d)")
_JP = re.compile(r"(?:(\d{1,2})\s*分\s*)?(\d{1,3})(?:\.\d+)?\s*秒")
_JP_MIN = re.compile(r"(\d{1,2})\s*分(?!\d)")


def extract_time(prediction: str | None) -> float | None:
    p = normalize(prediction)
    if not p:
        return None
    m = _ANS_LINE.search(p)
    target = m.group(1) if m else p.strip().splitlines()[-1]
    for text in (target, p):
        m = _MMSS.search(text)
        if m:
            a, b, c = m.groups()
            if c is not None:  # 時:分:秒
                return _hms(int(a), int(b), int(c))
            return _hms(0, int(a), int(b))
        m = _JP.search(text)
        if m:
            mi, se = m.groups()
            return _hms(0, int(mi or 0), int(se))
        m = _JP_MIN.search(text)
        if m:
            return _hms(0, int(m.group(1)), 0)
    return None


def tolerance(video_duration: float, num_frames: int) -> float:
    return max(5.0, video_duration / num_frames / 2)


def score_time(pred: float | None, start: float, end: float, tol: float) -> tuple[float, bool, float | None]:
    """(得点 0〜1, 正解か, 区間からの距離)。区間 ± tol に入れば 1.0、外れた秒数に応じて線形に減る。"""
    if pred is None:
        return 0.0, False, None
    dist = 0.0 if start <= pred <= end else min(abs(pred - start), abs(pred - end))
    over = max(0.0, dist - tol)
    score = max(0.0, 1.0 - over / PARTIAL_WINDOW)
    return score, over == 0.0, dist


def build_time_prompt(question: str) -> str:
    return (
        "次の問いが指している場面は、動画の何分何秒に映っていますか。\n"
        f"問い: {question}\n"
        "映像の根拠を 1〜2 行で書き、最後の行に「答え: 1:42」のように時刻を 分:秒 の形で 1 つだけ書いてください。"
    )


def video_duration(video_path: str | Path, frames_dir: str | Path | None, num_frames: int, max_side: int | None) -> float | None:
    """動画の長さ（秒）。フレームキャッシュの meta.json があればそれを使い、無ければ動画を調べる。"""
    if frames_dir:
        meta = Path(frames_dir) / Path(video_path).stem / f"f{num_frames}_s{max_side or 0}" / "meta.json"
        if meta.exists():
            return float(json.loads(meta.read_text())["duration"])
        for m in (Path(frames_dir) / Path(video_path).stem).glob("f*_s*/meta.json"):
            return float(json.loads(m.read_text())["duration"])
    try:
        from movie_llm.infer.video import probe

        return float(probe(str(video_path)).duration)
    except Exception:  # noqa: BLE001
        return None
