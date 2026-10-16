"""VideoJev-4B（区間分類 + 教師あり学習の公開モデル、Apache-2.0）で、ベンチマーク 800 問の「場面の時刻」を当てる。

- 入力は動画ファイル（フレームキャッシュは使えない）。手元で scripts/jvqa_reencode.py で小さくした動画を使う
- 1 本の動画に属する問いをまとめて 1 回の predict_video に渡す（動画の符号化は 1 回）
- 出力は scripts/jvqa_predict.py --task time と同じ形式の predictions.jsonl。prediction は区間の中点を「答え: 分:秒」にしたもの。
  区間そのものも interval に残す（重なりで採点するときに使う）
- 許容幅をそろえるため frames は 32 と記録する（jvqa_eval_time.py は frames から許容幅を計算する）

使い方（Colab）:
    python scripts/videojev_predict.py --name videojev-4b --video-dir /content/videos_small --out-dir {DRIVE}/results --fps 2 --attn sdpa
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections import defaultdict
from pathlib import Path

print = functools.partial(print, flush=True)

from movie_llm.eval.dataset import load_items


def mmss(sec: float) -> str:
    sec = max(0.0, float(sec))
    m, s = divmod(int(round(sec)), 60)
    return f"{m}:{s:02d}"


def duration_of(path: Path) -> float | None:
    import av

    try:
        with av.open(str(path)) as c:
            if c.duration:
                return c.duration / av.time_base
            vs = c.streams.video[0]
            if vs.duration and vs.time_base:
                return float(vs.duration * vs.time_base)
    except Exception:  # noqa: BLE001
        return None
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="videojev-4b")
    ap.add_argument("--repo", default="ZhuYuhan/VideoJev-4B")
    ap.add_argument("--video-dir", default="data/jvqa/videos_small")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=2048)
    ap.add_argument("--context", type=int, default=147456)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--attn", default="sdpa", help="flash_attention_2 が入っていれば flash_attention_2")
    ap.add_argument("--frames-label", type=int, default=32, help="許容幅の計算に使うフレーム数（4B の評価と同じ 32）")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    items = load_items()
    vdir = Path(args.video_dir)
    out_dir = Path(args.out_dir) / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"
    done: set[int] = set()
    if out_path.exists():
        done = {json.loads(l)["idx"] for l in out_path.open(encoding="utf-8") if l.strip()}
    by_video: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        if it["idx"] in done:
            continue
        if (vdir / f"{it['video_id']}.mp4").exists():
            by_video[it["video_id"]].append(it)
    videos = sorted(by_video)
    if args.limit:
        videos = videos[: args.limit]
    n_q = sum(len(by_video[v]) for v in videos)
    print(f"対象 {len(items)} 問、済み {len(done)}、今回 {n_q} 問 / {len(videos)} 本")
    if not videos:
        return

    import torch
    from transformers import AutoModel, AutoProcessor

    t0 = time.time()
    processor = AutoProcessor.from_pretrained(args.repo, trust_remote_code=True)
    model = AutoModel.from_pretrained(args.repo, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": "cuda:0"}, attn_implementation=args.attn)
    model.prepare_for_inference()
    print(f"model={args.repo} attn={args.attn} load={time.time() - t0:.0f}s")
    (out_dir / "config.json").write_text(json.dumps({"model": args.repo, "task": "time", "fps": args.fps, "max_frames": args.max_frames, "context": args.context, "threshold": args.threshold, "attn": args.attn, "frames_label": args.frames_label}, ensure_ascii=False, indent=2))

    t_all = time.time()
    k = 0
    with out_path.open("a", encoding="utf-8") as f:
        for vi, vid in enumerate(videos, 1):
            vp = vdir / f"{vid}.mp4"
            qs = by_video[vid]
            questions = [{"id": str(it["idx"]), "task": "grounding", "query": it["question"]} for it in qs]
            t1 = time.time()
            try:
                kw = dict(fps=args.fps, max_frames=args.max_frames, context=args.context)
                if args.threshold is not None:
                    kw["threshold"] = args.threshold
                results = model.predict_video(processor, str(vp), questions, **kw)
            except Exception as e:  # noqa: BLE001
                print(f"[{vi}/{len(videos)}] {vid} 失敗: {str(e)[:160]}")
                continue
            dur = duration_of(vp)
            by_id = {str(r.get("id")): r for r in results} if isinstance(results, list) else {}
            for it in qs:
                r = by_id.get(str(it["idx"]), {})
                interval = r.get("interval") or r.get("span")
                if interval and len(interval) == 2 and interval[0] is not None:
                    s, e = float(interval[0]), float(interval[1])
                    pred = f"区間 {mmss(s)}〜{mmss(e)}\n答え: {mmss((s + e) / 2)}"
                else:
                    s = e = None
                    pred = "区間なし"
                rec = {
                    "idx": it["idx"], "video_id": vid, "type": it["type"], "prediction": pred, "answer": it["answer"],
                    "interval": [s, e], "seconds": round((time.time() - t1) / len(qs), 2),
                    "task": "time", "start_time": it["start_time"], "end_time": it["end_time"], "frames": args.frames_label, "video_duration": dur,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                k += 1
            f.flush()
            print(f"[{vi}/{len(videos)}] {vid} {len(qs)} 問 {time.time() - t1:.1f}s  例: {questions[0]['query'][:30]!r} → {by_id.get(questions[0]['id'], {}).get('interval')}")
    print(f"合計 {k} 問 {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    main()
