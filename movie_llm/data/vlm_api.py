"""学習データ作成に使う、OpenAI 互換 API 経由の Qwen3-VL / Qwen3 呼び出し。

- 既定は Vercel AI Gateway。問題を作る側は Qwen3-VL-235B-A22B-Instruct（Apache-2.0）、映像なしの検査は Qwen3-32B
- 入力はフレームキャッシュ（<frames_dir>/<video_id>/f<N>_s<side>/）の JPEG を base64 で渡し、各フレームの前に時刻 [mm:ss] を置く
- 同じ入力を 2 度呼ばないよう、(モデル, メッセージ, 温度, 乱数の種) の sha256 でキャッシュする
- GPT / Gemini / Claude は規約で出力の学習利用を禁じているので、ここでは使わない
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import threading
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv

VL_MODEL = "alibaba/qwen3-vl-instruct"
TEXT_MODEL = "alibaba/qwen-3-32b"
BASE_URL = "https://ai-gateway.vercel.sh/v1"
API_KEY_ENV = "AI_GATEWAY_API_KEY"


def frame_content(frames_dir: Path | str, video_id: str, num_frames: int, max_side: int = 640) -> list[dict] | None:
    """フレームキャッシュから [時刻, 画像, 時刻, 画像, ...] の content を作る。無ければ None。"""
    d = Path(frames_dir) / video_id / f"f{num_frames}_s{max_side}"
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return None
    meta = json.loads(meta_path.read_text())
    files = sorted(d.glob("*.jpg"))
    if len(files) != len(meta["frames_indices"]):
        return None
    content: list[dict] = []
    for f, fi in zip(files, meta["frames_indices"]):
        t = fi / meta["fps"]
        content.append({"type": "text", "text": f"[{int(t // 60):02d}:{int(t % 60):02d}]"})
        b = base64.b64encode(f.read_bytes()).decode()
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}})
    return content


def parse_json(text: str) -> dict | list | None:
    """モデル出力から JSON を取り出す（```json ... ``` や前置きの文があってもよい）。"""
    text = text.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if m:
        text = m.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    start = min([i for i in (text.find("{"), text.find("[")) if i >= 0], default=-1)
    if start < 0:
        return None
    for end in range(len(text), start, -1):
        if text[end - 1] in "}]":
            try:
                return json.loads(text[start:end])
            except json.JSONDecodeError:
                continue
    return None


class Client:
    def __init__(self, cache_path: Path | str = "data/ja/api_cache.jsonl", base_url: str = BASE_URL, api_key_env: str = API_KEY_ENV, max_retries: int = 5) -> None:
        load_dotenv(find_dotenv(usecwd=True))
        from openai import OpenAI

        key = os.environ.get(api_key_env)
        if not key:
            raise RuntimeError(f"環境変数 {api_key_env} が無い（.env を確認）")
        self.client = OpenAI(base_url=base_url, api_key=key)
        self.cache_path = Path(cache_path)
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            for line in self.cache_path.open(encoding="utf-8"):
                if line.strip():
                    r = json.loads(line)
                    self.cache[r["key"]] = r["text"]
        self.max_retries = max_retries
        self.cost = 0.0
        self.calls = 0

    @staticmethod
    def _key(model: str, messages: list[dict], temperature: float, seed: int) -> str:
        # 画像の base64 を含めてハッシュ（同じフレームなら同じ鍵）
        h = hashlib.sha256()
        h.update(json.dumps({"m": model, "t": temperature, "s": seed}, sort_keys=True).encode())
        h.update(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode())
        return h.hexdigest()

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 1024, seed: int = 0) -> str:
        """seed は同じ入力で別のサンプルを得たいときの区別用（API には渡さない）。"""
        key = self._key(model, messages, temperature, seed)
        with self.lock:
            if key in self.cache:
                return self.cache[key]
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                r = self.client.chat.completions.create(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
                text = r.choices[0].message.content or ""
                cost = getattr(r.usage, "cost", None) if r.usage else None
                break
            except Exception as e:  # noqa: BLE001
                if attempt == self.max_retries - 1:
                    raise
                print(f"API 失敗（{attempt + 1}回目）: {str(e)[:120]} → {delay:.0f}s 待つ", flush=True)
                time.sleep(delay)
                delay *= 2
        with self.lock:
            self.cache[key] = text
            self.calls += 1
            if cost:
                self.cost += float(cost)
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "model": model, "text": text}, ensure_ascii=False) + "\n")
        return text
