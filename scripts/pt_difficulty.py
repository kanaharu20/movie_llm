"""学習前のモデルに 1 問 n 回答えさせ、問題ごとの正解率を測る（GRPO 用の問題の選別）。

GRPO はグループ内の正誤にばらつきがないと学習信号が出ない。全問正解・全問不正解の問題を落とし、
正解率が lo〜hi の範囲の問題だけを学習に使う。

入力: scripts/pt_convert.py が出す items.jsonl（choices / answer / video_id を持つ）
出力: <out>/difficulty.jsonl（1行1問、n 本の回答と正解率）。途中から再開できる

    uv run python scripts/pt_difficulty.py --items data/perception_test/items.jsonl \\
        --video-dir /content/pt/videos --frames-dir /content/pt_frames --model Qwen/Qwen3-VL-4B-Instruct --frames 32
    uv run python scripts/pt_difficulty.py ... --summarize   # 集計だけ
"""

from __future__ import annotations

import argparse
import functools
import json
import time
from collections import Counter
from pathlib import Path

from movie_llm.eval.rules import extract_choice, gold_letter
from movie_llm.infer.fetch import video_path
from movie_llm.infer.qwen3vl import DEFAULT_MODEL, Qwen3VL, build_prompt

print = functools.partial(print, flush=True)


def summarize(path: Path, lo: float, hi: float) -> None:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if not rows:
        print("結果なし")
        return
    acc = [r["acc"] for r in rows]
    n = len(rows[0]["letters"])
    hist = Counter(round(a * n) for a in acc)
    print(f"問題数 {len(rows)}、平均正解率 {sum(acc) / len(acc):.3f}")
    print(f"正解数({n}本中)の分布:", dict(sorted(hist.items())))
    bycat = {}
    for r in rows:
        bycat.setdefault(r["category"], []).append(r["acc"])
    print("category 別の正解率:", {c: round(sum(a) / len(a), 3) for c, a in sorted(bycat.items())})
    keep = [r for r in rows if lo <= r["acc"] <= hi]
    print(f"学習に使える（{lo}〜{hi}）: {len(keep)} 問 ({len(keep) / len(rows) * 100:.0f}%)")
    print("  category:", dict(Counter(r["category"] for r in keep)))
    print("全問正解（易しすぎ）:", sum(a > hi for a in acc), " 全問不正解（難しすぎ）:", sum(a < lo for a in acc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True)
    ap.add_argument("--out", default=None, help="既定は items と同じフォルダの difficulty.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--video-dir", default="data/perception_test/raw/train_videos/videos")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--lo", type=float, default=0.2)
    ap.add_argument("--hi", type=float, default=0.8)
    ap.add_argument("--summarize", action="store_true")
    ap.add_argument("--adapter", default=None, help="学習した LoRA のフォルダ。学習後の評価に使う")
    ap.add_argument("--greedy", action="store_true", help="サンプリングせず貪欲法で 1 回答える（評価用。acc は 0 か 1）")
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
    out = Path(args.out) if args.out else Path(args.items).parent / "difficulty.jsonl"
    if args.summarize:
        summarize(out, args.lo, args.hi)
        return

    done = set()
    if out.exists():
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if l.strip()}
    todo = [it for it in items if it["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"対象 {len(items)}、済み {len(done)}、今回 {len(todo)}")
    if not todo:
        summarize(out, args.lo, args.hi)
        return

    model = Qwen3VL(args.model, num_frames=args.frames, frames_dir=args.frames_dir, adapter=args.adapter)
    print(f"model={args.model} device={model.device} ({model.device_name}, {model.dtype}) frames={args.frames} n={args.n}")
    vdir = Path(args.video_dir)
    t0 = time.time()
    with out.open("a", encoding="utf-8") as f:
        for k, it in enumerate(todo, 1):
            vp = video_path(it["video_id"], vdir)
            if vp is None:
                print(f"[{k}/{len(todo)}] {it['id']} 動画なし")
                continue
            prompt = build_prompt(it["question"], it["type"], it["choices"])
            try:
                if args.greedy:
                    preds = [model.ask(str(vp), prompt, max_new_tokens=32).text]
                else:
                    preds = model.sample(str(vp), prompt, n=args.n, temperature=args.temperature)
            except Exception as e:
                print(f"[{k}/{len(todo)}] {it['id']} 失敗: {e}")
                continue
            gold = gold_letter(it["answer"], it["choices"])
            letters = [extract_choice(p, it["choices"]) for p in preds]
            correct = [l == gold for l in letters]
            rec = {
                "id": it["id"],
                "video_id": it["video_id"],
                "category": it["category"],
                "answer": gold,
                "preds": preds,
                "letters": letters,
                "acc": sum(correct) / len(correct),
                "unparsed": sum(l is None for l in letters),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0 or k == 1:
                print(f"[{k}/{len(todo)}] {it['id']} acc={rec['acc']:.2f} letters={letters} {time.time() - t0:.0f}s")
    print(f"合計 {time.time() - t0:.0f}s")
    summarize(out, args.lo, args.hi)


if __name__ == "__main__":
    main()
