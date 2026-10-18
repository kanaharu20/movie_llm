"""2 段構成の評価: VideoJev の候補区間の前後だけを学習済み Qwen3-VL に見せ、時刻を当て直し・根拠を書かせる。

1 段目: scripts/videojev_predict.py の出力（results/<vj_name>/predictions.jsonl、interval と probabilities）
2 段目: 区間の前後 --margin 秒を切り抜き（sample_frames_window）、時刻の指示文で Qwen3-VL に答えさせる。
        答えは切り抜き内の時刻なので、切り抜きの開始秒を足し戻して絶対時刻にする。

出力は jvqa_predict.py --task time と同じ形式（prediction の最後の行が「答え: 分:秒」の絶対時刻）。
採点は scripts/jvqa_eval_time.py。frames は許容幅の計算用に 32 と記録する（1 段目・4B 単体と同じ基準で比べる）。

使い方（Colab）:
    python scripts/jvqa_two_stage.py --vj-predictions {DRIVE}/results/videojev-4b/predictions.jsonl \
        --name two_stage-4b --video-dir /content/videos_small --adapter {ADAPTER} --out-dir {DRIVE}/results
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import time
from pathlib import Path

print = functools.partial(print, flush=True)

from movie_llm.eval.dataset import load_items
from movie_llm.eval.timeloc import build_time_prompt, extract_time
from movie_llm.infer.qwen3vl import DEFAULT_MODEL, Qwen3VL
from movie_llm.infer.video import probe, sample_frames_window


def mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vj-predictions", required=True, help="VideoJev の predictions.jsonl")
    ap.add_argument("--name", default="two_stage-4b")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct", help="LoRA は 4B 用なので既定は 4B")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--video-dir", default="data/jvqa/videos_small")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--frames", type=int, default=32, help="切り抜きに使うフレーム数")
    ap.add_argument("--margin", type=float, default=30.0, help="候補区間の前後に足す秒数")
    ap.add_argument("--min-window", type=float, default=60.0, help="切り抜きの最短の長さ（秒）")
    ap.add_argument("--max-new-tokens", type=int, default=96)
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    items = {it["idx"]: it for it in load_items()}
    vj = [json.loads(l) for l in open(args.vj_predictions, encoding="utf-8") if l.strip()]
    vj = [r for r in vj if r.get("interval") and r["interval"][0] is not None]
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"
    done: set[int] = set()
    if out_path.exists():
        done = {json.loads(l)["idx"] for l in out_path.open(encoding="utf-8") if l.strip()}
    todo = [r for r in vj if r["idx"] not in done and (Path(args.video_dir) / f"{r['video_id']}.mp4").exists()]
    if args.limit:
        todo = todo[: args.limit]
    print(f"VideoJev の予測 {len(vj)} 問、済み {len(done)}、今回 {len(todo)} 問")
    if not todo:
        return

    model = Qwen3VL(args.model, num_frames=args.frames, adapter=args.adapter)
    print(f"model={args.model} adapter={args.adapter} device={model.device} ({model.device_name}) frames={args.frames} load={model.load_seconds:.1f}s")
    (out_dir / "config.json").write_text(json.dumps({"model": args.model, "adapter": args.adapter, "task": "time", "two_stage": True, "vj_predictions": args.vj_predictions, "frames": args.frames, "margin": args.margin, "min_window": args.min_window}, ensure_ascii=False, indent=2))

    t_all = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for k, r in enumerate(todo, 1):
            it = items[r["idx"]]
            vp = Path(args.video_dir) / f"{r['video_id']}.mp4"
            dur = r.get("video_duration") or probe(str(vp)).duration
            s0, e0 = float(r["interval"][0]), float(r["interval"][1])
            ws, we = s0 - args.margin, e0 + args.margin
            if we - ws < args.min_window:  # 短すぎる切り抜きは中心を保って広げる
                c = (ws + we) / 2
                ws, we = c - args.min_window / 2, c + args.min_window / 2
            ws, we = max(0.0, ws), min(dur, we)
            try:
                frames, meta = sample_frames_window(str(vp), ws, we, args.frames, model.max_side)
                res = model.ask_frames(frames, meta, build_time_prompt(it["question"]), max_new_tokens=args.max_new_tokens)
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(todo)}] idx={r['idx']} 失敗: {str(e)[:120]}")
                continue
            rel = extract_time(res.text)
            if rel is not None:
                abs_t = ws + rel
                # 最後の行の「答え: …」を絶対時刻に書き換える（採点はこの行を読む）
                lines = res.text.rstrip().split("\n")
                lines = [ln for ln in lines if not re.match(r"^\s*答え\s*[:：]", ln)]
                pred = "\n".join(lines + [f"答え: {mmss(abs_t)}"])
            else:
                abs_t = None
                pred = res.text + "\n（時刻を読めず）"
            rec = {
                "idx": r["idx"], "video_id": r["video_id"], "type": it["type"], "prediction": pred, "answer": it["answer"],
                "task": "time", "start_time": it["start_time"], "end_time": it["end_time"], "frames": 32, "video_duration": dur,
                "stage1_interval": [s0, e0], "stage1_prob_max": r.get("prob_max"), "window": [round(ws, 1), round(we, 1)],
                "stage2_rel_sec": rel, "stage2_abs_sec": abs_t, "stage2_raw": res.text, "seconds": round(res.seconds, 2),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            print(f"[{k}/{len(todo)}] idx={r['idx']} vj={mmss(s0)}-{mmss(e0)} win={mmss(ws)}-{mmss(we)} → {mmss(abs_t) if abs_t is not None else 'なし'}  gold={it['start_time']}-{it['end_time']}  {res.seconds:.1f}s")
    print(f"合計 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
