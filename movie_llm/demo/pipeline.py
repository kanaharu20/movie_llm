"""2 段構成のシーン検索（VideoJev が探し、学習済み Qwen3-VL が確かめて説明する）+ 3 段の振り分け。

使い方:
    from movie_llm.demo.pipeline import SceneSearch
    ss = SceneSearch(adapter="models/timeloc-lora")
    r = ss.search("sample.mp4", "赤い鳥居の前で参拝している場面")
    r["band"], r["time"], r["rationale"], r["candidates"]
"""

from __future__ import annotations

import os
import time
from pathlib import Path

from movie_llm.demo.confidence import AGREE_SEC, BAND_LABEL, HI_PMAX, LO_PMEAN, band, mmss

LORA_ENV = "MOVIE_LLM_LORA"


class SceneSearch:
    def __init__(self, adapter: str | None = None, stage1_kwargs: dict | None = None, stage2_kwargs: dict | None = None,
                 hi_pmax: float = HI_PMAX, lo_pmean: float = LO_PMEAN, agree_sec: float = AGREE_SEC, margin: float = 30.0) -> None:
        self.adapter = adapter or os.environ.get(LORA_ENV) or None
        self.stage1_kwargs = stage1_kwargs or {}
        self.stage2_kwargs = stage2_kwargs or {}
        self.hi_pmax, self.lo_pmean, self.agree_sec, self.margin = hi_pmax, lo_pmean, agree_sec, margin
        self._loc = None
        self._ver = None

    @property
    def locator(self):
        if self._loc is None:
            from movie_llm.demo.stage1 import VideoJevLocator

            self._loc = VideoJevLocator(**self.stage1_kwargs)
        return self._loc

    @property
    def verifier(self):
        if self._ver is None:
            from movie_llm.demo.stage2 import Verifier

            self._ver = Verifier(adapter=self.adapter, **self.stage2_kwargs)
        return self._ver

    def search(self, video_path: str | Path, query: str, top_k: int = 3, verify: bool = True) -> dict:
        t0 = time.time()
        s1 = self.locator.locate(video_path, [query], top_k=top_k)[0]
        s, e = s1["interval"]
        agree = None
        refined = None
        if verify:
            refined = self.verifier.refine(video_path, s, e, query, margin=self.margin)
            t = refined["time"]
            agree = t is not None and (s - self.agree_sec) <= t <= (e + self.agree_sec)
        b = band(s1["pmax"], s1["pmean"], agree, self.hi_pmax, self.lo_pmean)
        answer = refined["time"] if (refined and agree) else (s + e) / 2
        return {
            "query": query,
            "band": b,
            "band_label": BAND_LABEL[b],
            "time": answer,
            "time_str": mmss(answer),
            "interval": [s, e],
            "interval_str": f"{mmss(s)}〜{mmss(e)}",
            "pmax": round(s1["pmax"], 3),
            "pmean": round(s1["pmean"], 3),
            "agree": agree,
            "rationale": refined["rationale"] if refined else "",
            "stage2_time": refined["time"] if refined else None,
            "window": refined["window"] if refined else None,
            "candidates": [{**c, "label": f"{mmss(c['start'])}〜{mmss(c['end'])}（確率 {c['prob']:.2f}）"} for c in s1["candidates"]],
            "duration": s1["duration"],
            "seconds": round(time.time() - t0, 1),
        }

    def ask(self, video_path: str | Path, start: float, end: float, question: str) -> str:
        return self.verifier.ask(video_path, start, end, question, margin=self.margin)
