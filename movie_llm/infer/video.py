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
