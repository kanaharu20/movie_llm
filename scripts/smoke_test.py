"""合成動画で推論が動くかを確認する。

    uv run python scripts/smoke_test.py
    uv run python scripts/smoke_test.py --model Qwen/Qwen3-VL-4B-Instruct --frames 8
"""

from __future__ import annotations

import argparse

from movie_llm.infer.qwen3vl import DEFAULT_MODEL, Qwen3VL


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--video", default="data/samples/synthetic.mp4")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    m = Qwen3VL(args.model, device=args.device, num_frames=args.frames)
    print(f"device={m.device} load={m.load_seconds:.1f}s")
    for q in [
        "この動画には何が映っていますか？一文で答えてください。",
        "赤い四角はどちらの方向に動いていますか？\nA) 左から右\nB) 右から左\nC) 上から下\nD) 下から上\n正しい選択肢の記号（A〜D）だけを答えてください。",
        "青い丸は何回現れますか？数字だけで答えてください。",
    ]:
        r = m.ask(args.video, q)
        print(f"\nQ: {q.splitlines()[0]}\nA: {r.text}\n  (video tokens={r.num_video_tokens}, prompt tokens={r.prompt_tokens}, {r.seconds:.1f}s)")


if __name__ == "__main__":
    main()
