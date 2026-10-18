"""1 段目: VideoJev-4B（区間分類、Apache-2.0）で、問いが指す場面の候補区間と確率を出す。

VideoJev は動画ファイルを直接読む（PyAV）。1 本の動画に複数の問いをまとめて渡せる。
"""

from __future__ import annotations

import time
from pathlib import Path

from movie_llm.demo.confidence import interval_stats, top_peaks

DEFAULT_REPO = "ZhuYuhan/VideoJev-4B"


def video_duration(path: str | Path) -> float:
    import av

    with av.open(str(path)) as c:
        if c.duration:
            return c.duration / av.time_base
        vs = c.streams.video[0]
        return float(vs.duration * vs.time_base)


class VideoJevLocator:
    def __init__(self, repo: str = DEFAULT_REPO, attn: str = "sdpa", fps: float = 2.0, max_frames: int = 1024,
                 context: int = 147456, device: str = "cuda:0", threshold: float | None = None) -> None:
        import torch
        from transformers import AutoModel, AutoProcessor

        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(repo, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(repo, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": device}, attn_implementation=attn)
        self.model.prepare_for_inference()
        self.fps, self.max_frames, self.context, self.threshold = fps, max_frames, context, threshold
        self.load_seconds = time.time() - t0

    def locate(self, video_path: str | Path, queries: list[str], top_k: int = 3) -> list[dict]:
        """問いごとに {interval, probabilities, pmax, pmean, candidates, duration, seconds} を返す。"""
        t0 = time.time()
        dur = video_duration(video_path)
        questions = [{"id": str(i), "task": "grounding", "query": q} for i, q in enumerate(queries)]
        kw = dict(fps=self.fps, max_frames=self.max_frames, context=self.context)
        if self.threshold is not None:
            kw["threshold"] = self.threshold
        results = self.model.predict_video(self.processor, str(video_path), questions, **kw)
        by_id = {str(r.get("id")): r for r in results}
        out = []
        for i, q in enumerate(queries):
            r = by_id.get(str(i), {})
            interval = r.get("interval") or r.get("span") or [None, None]
            probs = [float(x) for x in (r.get("probabilities") or [])]
            s, e = (float(interval[0]), float(interval[1])) if interval[0] is not None else (0.0, dur)
            pmax, pmean = interval_stats(probs, (s, e), dur)
            step = dur / len(probs) if probs else dur
            out.append({
                "query": q, "interval": [s, e], "probabilities": probs, "pmax": pmax, "pmean": pmean,
                "candidates": top_peaks(probs, step, k=top_k) if probs else [{"start": s, "end": e, "prob": pmax}],
                "duration": dur, "seconds": round((time.time() - t0) / len(queries), 2),
            })
        return out
