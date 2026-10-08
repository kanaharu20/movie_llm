"""Qwen3-VL で動画に質問する。

手元 (M2 Pro, MPS) でも GPU (CUDA) でも同じコードで動くようにする。
動画はファイルパスを渡し、PyAV で一様に num_frames 枚を抜き出してから processor に渡す。
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from movie_llm.infer.video import sample_frames

DEFAULT_MODEL = "Qwen/Qwen3-VL-2B-Instruct"


def pick_device(device: str | None = None) -> str:
    if device:
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_prompt(question: str, qtype: str, choices: list[str]) -> str:
    """ベンチマーク用の指示文。規則採点で読み取りやすい形で答えさせる。"""
    if qtype == "multi_choice":
        opts = "\n".join(choices)
        return f"{question}\n{opts}\n正しい選択肢の記号（A〜D）だけを答えてください。"
    if qtype == "yes_no":
        return f"{question}\n「はい」か「いいえ」で答えてください。"
    return f"{question}\n簡潔に答えてください。"


@dataclass
class GenResult:
    text: str
    num_video_tokens: int
    prompt_tokens: int
    seconds: float


class Qwen3VL:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        device: str | None = None,
        dtype: torch.dtype | None = None,
        num_frames: int = 16,
        max_pixels: int | None = 256 * 28 * 28,
        max_side: int | None = 640,
    ) -> None:
        self.device = pick_device(device)
        if dtype is None:
            dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        self.num_frames = num_frames
        self.max_side = max_side
        self.processor = AutoProcessor.from_pretrained(model_id)
        if max_pixels is not None:
            # 1フレームあたりの上限画素数。小さくするほど速く、粗くなる
            self.processor.video_processor.size = {"shortest_edge": 128 * 32 * 32, "longest_edge": max_pixels * num_frames}
        t0 = time.time()
        self.model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=dtype).to(self.device).eval()
        self.load_seconds = time.time() - t0

    @torch.inference_mode()
    def ask(self, video_path: str, prompt: str, max_new_tokens: int = 64, num_frames: int | None = None) -> GenResult:
        t0 = time.time()
        frames, meta = sample_frames(video_path, num_frames or self.num_frames, max_side=self.max_side)
        messages = [{"role": "user", "content": [{"type": "video"}, {"type": "text", "text": prompt}]}]
        text = self.processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.processor(
            text=[text],
            videos=[frames],
            video_metadata=[meta],
            do_sample_frames=False,
            return_tensors="pt",
        ).to(self.device)
        out = self.model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        new_tokens = out[0, inputs["input_ids"].shape[1]:]
        text = self.processor.decode(new_tokens, skip_special_tokens=True).strip()
        video_token_id = self.model.config.video_token_id
        n_video = int((inputs["input_ids"] == video_token_id).sum())
        return GenResult(text=text, num_video_tokens=n_video, prompt_tokens=int(inputs["input_ids"].shape[1]), seconds=time.time() - t0)
