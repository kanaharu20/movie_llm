"""YouTube 動画の取得（yt-dlp）。映像のみ・480p 以下の単一ファイルを取る。音声は使わない。"""

from __future__ import annotations

import subprocess
from pathlib import Path

VIDEO_DIR = Path("data/jvqa/videos")
FORMAT = "bv*[ext=mp4][vcodec^=avc1][height<=480]/bv*[ext=mp4][height<=480]/bv*[height<=480]/bv*"


def video_path(video_id: str, video_dir: Path = VIDEO_DIR) -> Path | None:
    for ext in ("mp4", "webm", "mkv"):
        p = video_dir / f"{video_id}.{ext}"
        if p.exists():
            return p
    return None


def fetch(video_id: str, video_dir: Path = VIDEO_DIR) -> Path | None:
    """取得済みならそのパス、なければ yt-dlp で取得する。失敗したら None。"""
    p = video_path(video_id, video_dir)
    if p:
        return p
    video_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp", "-q", "--no-warnings", "-f", FORMAT,
        "-o", str(video_dir / "%(id)s.%(ext)s"),
        f"https://www.youtube.com/watch?v={video_id}",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"yt-dlp 失敗 {video_id}: {r.stderr.strip()[:200]}")
        return None
    return video_path(video_id, video_dir)
