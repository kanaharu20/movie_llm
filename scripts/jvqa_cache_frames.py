"""取得済みの全動画からフレームを抜いてキャッシュを作る。

Colab や学習で動画のデコードを省くため。出力は data/jvqa/frames/<video_id>/f<N>_s<side>/*.jpg + meta.json。

    uv run python scripts/jvqa_cache_frames.py --frames 16 32
    uv run python scripts/jvqa_cache_frames.py --frames 32 --zip   # frames_f32_s640.zip も作る（Drive に上げる用）
"""

from __future__ import annotations

import argparse
import functools
import shutil
import time
from pathlib import Path

from movie_llm.infer.video import cache_dir, load_frames, sample_frames, save_frames

print = functools.partial(print, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, nargs="+", default=[16, 32])
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--video-dir", default="data/jvqa/videos")
    ap.add_argument("--frames-dir", default="data/jvqa/frames")
    ap.add_argument("--zip", action="store_true", help="フレーム数ごとに zip を作る")
    args = ap.parse_args()

    videos = sorted(p for p in Path(args.video_dir).iterdir() if p.suffix in (".mp4", ".webm", ".mkv"))
    print(f"動画 {len(videos)} 本、フレーム数 {args.frames}")
    t0 = time.time()
    for n in args.frames:
        done = skipped = 0
        for k, vp in enumerate(videos, 1):
            d = cache_dir(args.frames_dir, str(vp), n, args.max_side)
            if load_frames(d) is not None:
                skipped += 1
                continue
            try:
                arr, meta = sample_frames(str(vp), n, max_side=args.max_side)
            except Exception as e:
                print(f"  失敗 {vp.name}: {e}")
                continue
            save_frames(d, arr, meta)
            done += 1
            if k % 25 == 0:
                print(f"  [{k}/{len(videos)}] f{n} {time.time() - t0:.0f}s")
        print(f"f{n}: 作成 {done}、既存 {skipped}")
        if args.zip:
            # zip の中身は <video_id>/f<N>_s<side>/... になるよう、一時ディレクトリに該当サブフォルダだけ集める
            tmp = Path(args.frames_dir).parent / f"_zip_f{n}"
            if tmp.exists():
                shutil.rmtree(tmp)
            for vd in Path(args.frames_dir).iterdir():
                sub = vd / f"f{n}_s{args.max_side}"
                if sub.is_dir():
                    (tmp / vd.name).mkdir(parents=True, exist_ok=True)
                    shutil.copytree(sub, tmp / vd.name / sub.name)
            out = Path(args.frames_dir).parent / f"frames_f{n}_s{args.max_side}"
            shutil.make_archive(str(out), "zip", tmp)
            shutil.rmtree(tmp)
            print(f"  zip: {out}.zip ({Path(str(out) + '.zip').stat().st_size / 1e6:.0f} MB)")
    print(f"合計 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
