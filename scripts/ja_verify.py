"""生成した問題（scripts/ja_generate.py）を 2 つの検査で選別する。

検査 1「再回答の一致」: 答えを渡さずに同じ Qwen3-VL-235B に 32 フレームで 2 回答えさせる（温度 0.7）。
  2 回とも生成時の答えと一致した問題だけ残す。count は「映っているものを一つずつ挙げてから数を書く」別の聞き方。
検査 2「映像なしで解けない」: 文章だけの Qwen3-32B に、フレームを渡さず問題だけで 3 回答えさせる（温度 1.0）。
  3 回とも正解なら、知識や文章の癖だけで解ける問題として落とす。

出力: data/ja/verified.jsonl（検査結果つき。通った問題は pass=true）。学習前の 4B での選別（pt_difficulty.py）は別に行う。

    uv run python scripts/ja_verify.py --workers 4
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from movie_llm.data.vlm_api import TEXT_MODEL, VL_MODEL, Client, frame_content
from movie_llm.eval.rules import extract_choice, extract_number
from movie_llm.infer.qwen3vl import build_prompt

print = functools.partial(print, flush=True)
DATA = Path("data/ja")

COUNT_PROMPT = "{question}\n動画に写っているものを時刻つきで一つずつ挙げて、同じものを二重に数えないようにしてから、最後の行に「答え: 数字」の形で書いてください。"
TEXT_ONLY_NOTE = "次は動画についての問題ですが、動画は見られません。知識と推測だけで答えてください。\n"


def answer_of(item: dict, text: str) -> str | None:
    if item["type"] == "numeric":
        tail = text.rsplit("答え", 1)[-1] if "答え" in text else text.strip().splitlines()[-1] if text.strip() else ""
        n = extract_number(tail)
        return None if n is None else str(n)
    return extract_choice(text, item["choices"])


def check_item(client: Client, item: dict, frames_dir: Path, num_frames: int, vl_model: str, text_model: str) -> dict:
    content = frame_content(frames_dir, item["video_id"], num_frames)
    if content is None:
        raise FileNotFoundError(item["video_id"])
    if item["type"] == "numeric":
        prompt = COUNT_PROMPT.format(question=item["question"])
    else:
        prompt = build_prompt(item["question"], item["type"], item["choices"])
    # 検査 1
    re_answers = []
    for s in range(2):
        text = client.chat(vl_model, [{"role": "user", "content": content + [{"type": "text", "text": prompt}]}], temperature=0.7, max_tokens=600, seed=s)
        re_answers.append(answer_of(item, text))
    consistent = all(a == item["answer"] for a in re_answers)
    # 検査 2
    blind = []
    for s in range(3):
        text = client.chat(text_model, [{"role": "user", "content": TEXT_ONLY_NOTE + build_prompt(item["question"], item["type"], item["choices"])}], temperature=1.0, max_tokens=200, seed=s)
        blind.append(answer_of(item, text))
    blind_solvable = all(a == item["answer"] for a in blind)
    out = dict(item)
    out.update(re_answers=re_answers, consistent=consistent, blind_answers=blind, blind_solvable=blind_solvable, pass_=consistent and not blind_solvable)
    out["pass"] = out.pop("pass_")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(DATA / "generated.jsonl"))
    ap.add_argument("--frames-dir", default=str(DATA / "frames"))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--vl-model", default=VL_MODEL)
    ap.add_argument("--text-model", default=TEXT_MODEL)
    ap.add_argument("--out", default=str(DATA / "verified.jsonl"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--summarize", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    if not args.summarize:
        items = [json.loads(l) for l in open(args.items, encoding="utf-8") if l.strip()]
        done = {json.loads(l)["id"] for l in out.open(encoding="utf-8") if l.strip()} if out.exists() else set()
        todo = [it for it in items if it["id"] not in done]
        if args.limit:
            todo = todo[: args.limit]
        print(f"問題 {len(items)}、済み {len(done)}、今回 {len(todo)}")
        if todo:
            client = Client()
            frames_dir = Path(args.frames_dir)
            t0 = time.time()
            with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
                futs = {ex.submit(check_item, client, it, frames_dir, args.frames, args.vl_model, args.text_model): it for it in todo}
                for k, fut in enumerate(as_completed(futs), 1):
                    it = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"[{k}/{len(todo)}] {it['id']} 失敗: {str(e)[:120]}")
                        continue
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    if k % 10 == 0 or k == 1:
                        print(f"[{k}/{len(todo)}] {it['id']} 一致={r['consistent']} 映像なしで正解={r['blind_solvable']}  {time.time() - t0:.0f}s  API {client.calls} 回 ${client.cost:.3f}")

    rows = [json.loads(l) for l in out.open(encoding="utf-8") if l.strip()]
    print(f"\n検査した問題 {len(rows)}、通過 {sum(r['pass'] for r in rows)} ({sum(r['pass'] for r in rows) / max(len(rows), 1) * 100:.0f}%)")
    print("  再回答が一致しなかった:", sum(not r["consistent"] for r in rows), " 映像なしで 3 回正解:", sum(r["blind_solvable"] for r in rows))
    for cat in sorted({r["category"] for r in rows}):
        xs = [r for r in rows if r["category"] == cat]
        print(f"  {cat}: {len(xs)} 問 → 通過 {sum(r['pass'] for r in xs)}（不一致 {sum(not r['consistent'] for r in xs)}、映像なし正解 {sum(r['blind_solvable'] for r in xs)}）")


if __name__ == "__main__":
    main()
