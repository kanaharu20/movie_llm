"""Japanese Video-QA の全動画を取得する（short を先に）。再実行すると未取得分だけ試す。

結果は data/jvqa/video_status.jsonl に記録する（video_id, duration, ok, error）。
YouTube から消えた動画はここで分かる。

    uv run python scripts/jvqa_fetch_videos.py
    uv run python scripts/jvqa_fetch_videos.py --duration short
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import time
from pathlib import Path

from movie_llm.eval.dataset import load_items
from movie_llm.infer.fetch import FORMAT, VIDEO_DIR, video_path

print = functools.partial(print, flush=True)
STATUS = Path("data/jvqa/video_status.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", choices=["short", "medium"])
    ap.add_argument("--sleep", type=float, default=2.0, help="取得の間隔（秒）")
    ap.add_argument("--retry-failed", action="store_true", help="以前失敗した動画も試す")
    args = ap.parse_args()

    items = load_items()
    videos: dict[str, str] = {}
    for it in items:
        videos.setdefault(it["video_id"], it["duration"])
    order = sorted(videos, key=lambda v: (videos[v] != "short", v))
    if args.duration:
        order = [v for v in order if videos[v] == args.duration]

    failed: set[str] = set()
    if STATUS.exists() and not args.retry_failed:
        for line in STATUS.open(encoding="utf-8"):
            rec = json.loads(line)
            if not rec["ok"]:
                failed.add(rec["video_id"])

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    n_ok = n_skip = n_fail = 0
    with STATUS.open("a", encoding="utf-8") as st:
        for k, vid in enumerate(order, 1):
            if video_path(vid):
                n_skip += 1
                continue
            if vid in failed:
                n_fail += 1
                continue
            cmd = ["yt-dlp", "-q", "--no-warnings", "-f", FORMAT, "-o", str(VIDEO_DIR / "%(id)s.%(ext)s"),
                   f"https://www.youtube.com/watch?v={vid}"]
            r = subprocess.run(cmd, capture_output=True, text=True)
            ok = r.returncode == 0 and video_path(vid) is not None
            err = "" if ok else r.stderr.strip().splitlines()[-1][:200] if r.stderr.strip() else "unknown"
            st.write(json.dumps({"video_id": vid, "duration": videos[vid], "ok": ok, "error": err}, ensure_ascii=False) + "\n")
            st.flush()
            n_ok += ok
            n_fail += not ok
            print(f"[{k}/{len(order)}] {vid} {videos[vid]} {'ok' if ok else 'FAIL ' + err}")
            time.sleep(args.sleep)
    print(f"取得 {n_ok}、済み {n_skip}、失敗 {n_fail}、動画数 {len(order)}")


if __name__ == "__main__":
    main()
