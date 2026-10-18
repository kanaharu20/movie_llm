"""2 段目: 学習済み Qwen3-VL-4B（LoRA）が、候補区間の前後だけを見て、時刻を確かめ、根拠を書き、質問に答える。"""

from __future__ import annotations

import re
from pathlib import Path

from movie_llm.eval.timeloc import build_time_prompt, extract_time
from movie_llm.infer.qwen3vl import Qwen3VL
from movie_llm.infer.video import probe, sample_frames_window

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"


def window_for(start: float, end: float, duration: float, margin: float = 30.0, min_window: float = 60.0) -> tuple[float, float]:
    ws, we = start - margin, end + margin
    if we - ws < min_window:
        c = (ws + we) / 2
        ws, we = c - min_window / 2, c + min_window / 2
    return max(0.0, ws), min(duration, we)


class Verifier:
    def __init__(self, model_id: str = DEFAULT_MODEL, adapter: str | None = None, frames: int = 32, device: str | None = None) -> None:
        self.model = Qwen3VL(model_id, device=device, num_frames=frames, adapter=adapter)
        self.frames = frames

    def refine(self, video_path: str | Path, start: float, end: float, query: str, margin: float = 30.0, min_window: float = 60.0, max_new_tokens: int = 96) -> dict:
        """切り抜きの中で場面の時刻を当て直し、根拠を返す。time は元の動画の絶対秒（読めなければ None）。"""
        dur = probe(str(video_path)).duration
        ws, we = window_for(start, end, dur, margin, min_window)
        frames, meta = sample_frames_window(str(video_path), ws, we, self.frames, self.model.max_side)
        res = self.model.ask_frames(frames, meta, build_time_prompt(query), max_new_tokens=max_new_tokens)
        rel = extract_time(res.text)
        rationale = "\n".join(ln for ln in res.text.strip().split("\n") if not re.match(r"^\s*答え\s*[:：]", ln)).strip()
        return {"window": [ws, we], "time": (ws + rel) if rel is not None else None, "rationale": rationale, "raw": res.text, "seconds": res.seconds}

    def ask(self, video_path: str | Path, start: float, end: float, question: str, margin: float = 30.0, min_window: float = 60.0, max_new_tokens: int = 128) -> str:
        """切り抜きについて自由に質問する（数、動作、何が映っているか）。"""
        dur = probe(str(video_path)).duration
        ws, we = window_for(start, end, dur, margin, min_window)
        frames, meta = sample_frames_window(str(video_path), ws, we, self.frames, self.model.max_side)
        prompt = f"この映像について日本語で簡潔に答えてください。\n質問: {question}"
        return self.model.ask_frames(frames, meta, prompt, max_new_tokens=max_new_tokens).text.strip()
