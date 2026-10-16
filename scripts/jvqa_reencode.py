"""ベンチマークの動画を小さく再エンコードする（VideoJev など、動画ファイルそのものを読むモデルの評価用）。

- 2 fps・短辺 360px・H.264（libx264, crf 28）。7.6GB → 1〜2GB の見込み
- 出力: data/jvqa/videos_small/<video_id>.mp4。--zip で data/jvqa/videos_2fps_s360.zip も作る（Drive に上げる用）
- 途中から再開できる（出力があるものは飛ばす）

使い方:
    uv run python scripts/jvqa_reencode.py --workers 4 --zip
"""

from __future__ import annotations

import argparse
import shutil
from fractions import Fraction
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def reencode(src: Path, dst: Path, fps: float, short_side: int) -> tuple[str, str | None]:
    import av

    try:
        with av.open(str(src)) as inp:
            vs = inp.streams.video[0]
            vs.thread_type = "AUTO"
            w, h = vs.codec_context.width, vs.codec_context.height
            scale = min(1.0, short_side / max(1, min(w, h)))
            ow, oh = (int(w * scale) // 2) * 2, (int(h * scale) // 2) * 2
            tmp = dst.with_suffix(".tmp.mp4")
            with av.open(str(tmp), "w") as out:
                os_ = out.add_stream("libx264", rate=int(round(fps)))
                os_.width, os_.height = ow, oh
                os_.pix_fmt = "yuv420p"
                os_.options = {"crf": "28", "preset": "veryfast"}
                step = 1.0 / fps
                next_t = 0.0
                n = 0
                for frame in inp.decode(vs):
                    t = frame.time
                    if t is None or t + 1e-6 < next_t:
                        continue
                    f2 = frame.reformat(width=ow, height=oh, format="yuv420p")
                    f2.pts = n
                    f2.time_base = Fraction(1, int(round(fps)))  # pts = 枚数（time_base 1/fps）
                    for pkt in os_.encode(f2):
                        out.mux(pkt)
                    n += 1
                    next_t += step
                for pkt in os_.encode(None):
                    out.mux(pkt)
            if n == 0:
                tmp.unlink(missing_ok=True)
                return src.stem, "フレームなし"
            tmp.rename(dst)
            return src.stem, None
    except Exception as e:  # noqa: BLE001
        return src.stem, str(e)[:120]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-dir", default="data/jvqa/videos")
    ap.add_argument("--out-dir", default="data/jvqa/videos_small")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--short-side", type=int, default=360)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()

    src_dir, out_dir = Path(args.video_dir), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    srcs = sorted(p for p in src_dir.glob("*.mp4") if p.stat().st_size > 0)
    todo = [p for p in srcs if not (out_dir / p.name).exists()]
    print(f"動画 {len(srcs)} 本、済み {len(srcs) - len(todo)}、今回 {len(todo)}", flush=True)
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(args.workers) as ex:
        futs = {ex.submit(reencode, p, out_dir / p.name, args.fps, args.short_side): p for p in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            name, err = fut.result()
            if err:
                print(f"  失敗 {name}: {err}", flush=True)
            else:
                done += 1
            if k % 20 == 0:
                print(f"  [{k}/{len(todo)}] {time.time() - t0:.0f}s", flush=True)
    total = sum(p.stat().st_size for p in out_dir.glob("*.mp4")) / 1e9
    print(f"作成 {done}、合計 {total:.2f}GB、{time.time() - t0:.0f}s", flush=True)
    if args.zip:
        out = src_dir.parent / f"videos_{int(args.fps)}fps_s{args.short_side}"
        shutil.make_archive(str(out), "zip", out_dir)
        print(f"zip: {out}.zip ({Path(str(out) + '.zip').stat().st_size / 1e9:.2f}GB)", flush=True)


if __name__ == "__main__":
    main()
