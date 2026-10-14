"""学習前のモデルに 1 問 n 回答えさせ、問題ごとの正解率を測る（GRPO 用の問題の選別）。

GRPO はグループ内の正誤にばらつきがないと学習信号が出ない。全問正解・全問不正解の問題を落とし、
正解率が lo〜hi の範囲の問題だけを学習に使う。

入力: scripts/pt_convert.py が出す items.jsonl（choices / answer / video_id を持つ）。
      pt_select.py が出す type=numeric（数を答える）の問題も扱える（正解は整数の一致、差 1 の部分点も集計する）
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

from movie_llm.eval.rules import extract_choice, extract_number, gold_letter, score_numeric
from movie_llm.eval.timeloc import build_time_prompt, extract_time, score_time, tolerance
from movie_llm.infer.fetch import video_path
from movie_llm.infer.qwen3vl import DEFAULT_MODEL, Qwen3VL, build_prompt

print = functools.partial(print, flush=True)


def summarize(path: Path, lo: float, hi: float) -> None:
    rows = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    if not rows:
        print("結果なし")
        return
    acc = [r["acc"] for r in rows]
    n = len(rows[0]["preds"])
    hist = Counter(round(a * n) for a in acc)
    print(f"問題数 {len(rows)}、平均正解率 {sum(acc) / len(acc):.3f}")
    print(f"正解数({n}本中)の分布:", dict(sorted(hist.items())))
    bycat = {}
    for r in rows:
        bycat.setdefault(r.get("category") or r.get("domain") or "-", []).append(r["acc"])
    print("category 別の正解率:", {c: f"{sum(a) / len(a):.3f} ({len(a)})" for c, a in sorted(bycat.items())})
    tm = [r for r in rows if r.get("type") == "time"]
    if tm:
        print(f"時刻を当てる問題 {len(tm)}: 正解率 {sum(r['acc'] for r in tm) / len(tm):.3f}、部分点込み {sum(r['partial'] for r in tm) / len(tm):.3f}、"
              f"時刻を読めなかった回答 {sum(r['unparsed'] for r in tm)}")
    num = [r for r in rows if r.get("type") == "numeric"]
    if num:
        partial = [r["partial"] for r in num]
        print(f"数を答える問題 {len(num)}: 正解率 {sum(r['acc'] for r in num) / len(num):.3f}、部分点込み {sum(partial) / len(partial):.3f}、"
              f"数字が読めなかった回答 {sum(r['unparsed'] for r in num)}")
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
            is_time = it["type"] == "time"
            prompt = build_time_prompt(it["question"]) if is_time else build_prompt(it["question"], it["type"], it["choices"])
            try:
                if args.greedy:
                    preds = [model.ask(str(vp), prompt, max_new_tokens=96 if is_time else 32).text]
                else:
                    preds = model.sample(str(vp), prompt, n=args.n, temperature=args.temperature, max_new_tokens=96 if is_time else 32)
            except Exception as e:
                print(f"[{k}/{len(todo)}] {it['id']} 失敗: {e}")
                continue
            if is_time:
                gold = f"{it['start']}-{it['end']}"
                tol = tolerance(float(it["video_duration"]), args.frames)
                parsed = [extract_time(p) for p in preds]
                scored = [score_time(t, float(it["start"]), float(it["end"]), tol) for t in parsed]
                correct = [h for _, h, _ in scored]
                partial = sum(sc for sc, _, _ in scored) / len(scored)
            elif it["type"] == "numeric":
                gold = it["answer"]
                parsed = [extract_number(p) for p in preds]
                scored = [score_numeric(gold, p) for p in preds]
                correct = [c for _, c, _ in scored]
                partial = sum(sc for sc, _, _ in scored) / len(scored)
            else:
                gold = gold_letter(it["answer"], it["choices"])
                parsed = [extract_choice(p, it["choices"]) for p in preds]
                correct = [l == gold for l in parsed]
                partial = sum(correct) / len(correct)
            rec = {
                "id": it["id"],
                "video_id": it["video_id"],
                "category": it.get("category") or it.get("domain"),
                "type": it["type"],
                "answer": gold,
                "preds": preds,
                "letters": parsed,  # numeric のときは読み取った整数
                "acc": sum(correct) / len(correct),
                "partial": partial,  # numeric: 差 1 を 0.5 とした平均。multi_choice: acc と同じ
                "unparsed": sum(l is None for l in parsed),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if k % 10 == 0 or k == 1:
                print(f"[{k}/{len(todo)}] {it['id']} acc={rec['acc']:.2f} parsed={parsed} {time.time() - t0:.0f}s")
    print(f"合計 {time.time() - t0:.0f}s")
    summarize(out, args.lo, args.hi)


if __name__ == "__main__":
    main()
