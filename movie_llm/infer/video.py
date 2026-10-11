"""PyAV で動画から一様に N フレームを抜き出す。

transformers 5.x の動画デコードは torchcodec 前提なので、ここで自前に抜き出して
(frames, metadata) の形で processor に渡す。
"""

from __future__ import annotations

from dataclasses import dataclass

import av
import numpy as np
from transformers.video_utils import VideoMetadata


@dataclass
class VideoInfo:
    fps: float
    total_frames: int
    duration: float
    width: int
    height: int


def probe(path: str) -> VideoInfo:
    with av.open(path) as c:
        s = c.streams.video[0]
        fps = float(s.average_rate or s.guessed_rate or 30)
        duration = float(s.duration * s.time_base) if s.duration is not None else float(c.duration / av.time_base)
        total = s.frames or int(round(duration * fps))
        return VideoInfo(fps=fps, total_frames=total, duration=duration, width=s.width, height=s.height)


def sample_frames(path: str, num_frames: int, max_side: int | None = None) -> tuple[np.ndarray, VideoMetadata]:
    """先頭から末尾まで等間隔に num_frames 枚を取り、(T, H, W, 3) の uint8 配列とメタデータを返す。

    max_side を指定すると長辺をその画素数以下に縮小する（デコード後のメモリ節約）。
    """
    info = probe(path)
    n = max(1, min(num_frames, info.total_frames))
    targets = np.linspace(0, info.total_frames - 1, n).round().astype(int)
    wanted = set(targets.tolist())
    frames: dict[int, np.ndarray] = {}
    with av.open(path) as c:
        s = c.streams.video[0]
        s.thread_type = "AUTO"
        for i, frame in enumerate(c.decode(s)):
            if i in wanted:
                if max_side and max(frame.width, frame.height) > max_side:
                    scale = max_side / max(frame.width, frame.height)
                    frame = frame.reformat(width=int(frame.width * scale) // 2 * 2, height=int(frame.height * scale) // 2 * 2)
                frames[i] = frame.to_ndarray(format="rgb24")
                if len(frames) == len(wanted):
                    break
    # コンテナの frames 数が実際より多い場合、末尾の目標が取れないので取れた分で埋める
    got = sorted(frames)
    if not got:
        raise RuntimeError(f"フレームを1枚も取り出せない: {path}")
    indices = [i if i in frames else got[-1] for i in targets.tolist()]
    arr = np.stack([frames[i] for i in indices])
    meta = VideoMetadata(
        total_num_frames=info.total_frames,
        fps=info.fps,
        width=arr.shape[2],
        height=arr.shape[1],
        duration=info.duration,
        video_backend="pyav",
        frames_indices=indices,
    )
    return arr, meta


# ---------------------------------------------------------------------------
# フレームキャッシュ
# 動画のデコードは CPU が重く、Colab や学習で同じフレームを何度も読む。
# 一度抜いたフレームを JPEG + meta.json で保存し、以降はそれを読む。
# 配置: <frames_dir>/<動画の stem>/f<num_frames>_s<max_side>/{0000.jpg, ..., meta.json}
# ---------------------------------------------------------------------------

import json
from pathlib import Path

from PIL import Image


def cache_dir(frames_dir: Path | str, video_path: str, num_frames: int, max_side: int | None) -> Path:
    return Path(frames_dir) / Path(video_path).stem / f"f{num_frames}_s{max_side or 0}"


def save_frames(dir_: Path, arr: np.ndarray, meta: VideoMetadata, quality: int = 90) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(arr):
        Image.fromarray(frame).save(dir_ / f"{i:04d}.jpg", quality=quality)
    (dir_ / "meta.json").write_text(
        json.dumps(
            {
                "total_num_frames": int(meta.total_num_frames),
                "fps": float(meta.fps),
                "width": int(meta.width),
                "height": int(meta.height),
                "duration": float(meta.duration),
                "frames_indices": [int(i) for i in meta.frames_indices],
            }
        )
    )


def load_frames(dir_: Path) -> tuple[np.ndarray, VideoMetadata] | None:
    meta_path = dir_ / "meta.json"
    if not meta_path.exists():
        return None
    m = json.loads(meta_path.read_text())
    files = sorted(dir_.glob("*.jpg"))
    if len(files) != len(m["frames_indices"]):
        return None
    arr = np.stack([np.asarray(Image.open(f).convert("RGB")) for f in files])
    meta = VideoMetadata(
        total_num_frames=m["total_num_frames"],
        fps=m["fps"],
        width=m["width"],
        height=m["height"],
        duration=m["duration"],
        video_backend="cache",
        frames_indices=m["frames_indices"],
    )
    return arr, meta


def sample_frames_cached(
    video_path: str, num_frames: int, max_side: int | None, frames_dir: Path | str | None
) -> tuple[np.ndarray, VideoMetadata]:
    """キャッシュがあれば読み、なければ動画から抜いてキャッシュに保存する。frames_dir が None なら毎回デコード。"""
    if frames_dir is None:
        return sample_frames(video_path, num_frames, max_side=max_side)
    d = cache_dir(frames_dir, video_path, num_frames, max_side)
    hit = load_frames(d)
    if hit is not None:
        return hit
    arr, meta = sample_frames(video_path, num_frames, max_side=max_side)
    save_frames(d, arr, meta)
    return arr, meta
