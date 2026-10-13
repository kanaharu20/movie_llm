"""Japanese Video-QA に対して Qwen3-VL で予測を出す。

- 動画は data/jvqa/videos/ に無ければ yt-dlp で取得する
- 出力は results/<name>/predictions.jsonl（1行1問）。途中から再開できる
- 採点は scripts/jvqa_eval.py で行う

使い方:
    uv run python scripts/jvqa_predict.py --name qwen3vl-2b_f16 --idx 2 5 7
    uv run python scripts/jvqa_predict.py --name qwen3vl-2b_f16 --type multi_choice --limit 20
    uv run python scripts/jvqa_predict.py --name qwen3vl-4b_f16 --model Qwen/Qwen3-VL-4B-Instruct --duration short
"""

from __future__ import annotations

import argparse
import json
import time
import functools
from pathlib import Path

print = functools.partial(print, flush=True)  # ログにリダイレクトしても逐次出すため

from movie_llm.eval.dataset import load_items
from movie_llm.infer.fetch import fetch, video_path
from movie_llm.infer.qwen3vl import DEFAULT_MODEL, Qwen3VL, build_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="results/<name>/ に出力")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--device", default=None)
    ap.add_argument("--idx", type=int, nargs="*", help="この idx だけ")
    ap.add_argument("--type", choices=["open", "multi_choice", "yes_no"])
    ap.add_argument("--duration", choices=["short", "medium"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-fetch", action="store_true", help="動画が無い問題は飛ばす（取得しない）")
    ap.add_argument("--video-dir", default="data/jvqa/videos", help="動画の置き場（Colab では Drive のパス）")
    ap.add_argument("--frames-dir", default=None, help="フレームキャッシュの置き場。例: data/jvqa/frames。指定するとデコードを省ける")
    ap.add_argument("--out-dir", default="results", help="出力先のルート（Colab では Drive のパス）")
    ap.add_argument("--adapter", default=None, help="学習した LoRA のフォルダ（例: results/grpo_4b/step_00300）")
    args = ap.parse_args()

    items = load_items()
    if args.idx:
        keep = set(args.idx)
        items = [it for it in items if it["idx"] in keep]
    if args.type:
        items = [it for it in items if it["type"] == args.type]
    if args.duration:
        items = [it for it in items if it["duration"] == args.duration]
    if args.limit:
        items = items[: args.limit]

    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"
    done: set[int] = set()
    if out_path.exists():
        with out_path.open(encoding="utf-8") as f:
            done = {json.loads(l)["idx"] for l in f if l.strip()}
    todo = [it for it in items if it["idx"] not in done]
    print(f"対象 {len(items)} 問、済み {len(done)}、今回 {len(todo)} 問")
    if not todo:
        return

    model = Qwen3VL(args.model, device=args.device, num_frames=args.frames, frames_dir=args.frames_dir, adapter=args.adapter)
    print(f"model={args.model} device={model.device} ({model.device_name}, {model.dtype}) frames={args.frames} load={model.load_seconds:.1f}s")
    (out_dir / "config.json").write_text(
        json.dumps({"model": args.model, "frames": args.frames, "max_new_tokens": args.max_new_tokens, "device": model.device, "device_name": model.device_name, "dtype": str(model.dtype), "frames_dir": args.frames_dir, "adapter": args.adapter}, ensure_ascii=False, indent=2)
    )

    t_all = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for k, it in enumerate(todo, 1):
            vdir = Path(args.video_dir)
            vp = video_path(it["video_id"], vdir) if args.no_fetch else fetch(it["video_id"], vdir)
            if vp is None:
                print(f"[{k}/{len(todo)}] idx={it['idx']} 動画なし、飛ばす")
                continue
            prompt = build_prompt(it["question"], it["type"], it["choices"])
            try:
                r = model.ask(str(vp), prompt, max_new_tokens=args.max_new_tokens)
            except Exception as e:  # 1問の失敗で全体を止めない
                print(f"[{k}/{len(todo)}] idx={it['idx']} 失敗: {e}")
                continue
            rec = {
                "idx": it["idx"],
                "video_id": it["video_id"],
                "type": it["type"],
                "prediction": r.text,
                "answer": it["answer"],
                "video_tokens": r.num_video_tokens,
                "prompt_tokens": r.prompt_tokens,
                "seconds": round(r.seconds, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(todo)}] idx={it['idx']} {it['type']} {r.seconds:.1f}s  pred={r.text[:40]!r}  gold={it['answer']!r}")
    print(f"合計 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
