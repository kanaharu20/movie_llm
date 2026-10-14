"""生成した場面（scripts/ja_generate_time.py）を「再特定の一致」で検査する。

区間を隠して同じ Qwen3-VL-235B に、評価と同じ指示文（movie_llm.eval.timeloc.build_time_prompt）で 2 回時刻を答えさせる（温度 0.7）。
2 回とも正解区間 ± 許容幅に入った場面だけ残す（許容幅 = max(5 秒, フレーム間隔 / 2)、評価と同じ）。
1 回目は query（「〜している場面」）、2 回目は question（「〜のは動画のどの時点ですか」）で聞き、どちらの言い方でも当たることを確かめる。

出力: data/ja/time_verified.jsonl（検査結果つき。pass=true が学習に使える）

    uv run python scripts/ja_verify_time.py --workers 4
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from movie_llm.data.vlm_api import VL_MODEL, Client, frame_content
from movie_llm.eval.timeloc import build_time_prompt, extract_time, score_time, tolerance

print = functools.partial(print, flush=True)
DATA = Path("data/ja")


def check_item(client: Client, item: dict, frames_dir: Path, num_frames: int, model: str) -> dict:
    content = frame_content(frames_dir, item["video_id"], num_frames)
    if content is None:
        raise FileNotFoundError(item["video_id"])
    tol = tolerance(item["video_duration"], num_frames)
    answers, hits, dists = [], [], []
    for s, q in enumerate((item["query"], item["question"])):
        text = client.chat(model, [{"role": "user", "content": content + [{"type": "text", "text": build_time_prompt(q)}]}], temperature=0.7, max_tokens=160, seed=s)
        t = extract_time(text)
        _, hit, dist = score_time(t, item["start"], item["end"], tol)
        answers.append(t); hits.append(hit); dists.append(dist)
    out = dict(item)
    out.update(tol=round(tol, 1), re_times=answers, re_hits=hits, re_dists=[None if d is None else round(d, 1) for d in dists])
    out["pass"] = all(hits)
    return out


def summarize(rows: list[dict]) -> None:
    n = len(rows)
    print(f"\n検査した場面 {n}、通過 {sum(r['pass'] for r in rows)} ({sum(r['pass'] for r in rows) / max(n, 1) * 100:.0f}%)")
    print("  query で当たった:", sum(r["re_hits"][0] for r in rows), " question で当たった:", sum(r["re_hits"][1] for r in rows),
          " 時刻を読めなかった回答:", sum(t is None for r in rows for t in r["re_times"]))
    for dom in sorted({r["domain"] for r in rows}):
        xs = [r for r in rows if r["domain"] == dom]
        print(f"  {dom}: {len(xs)} → 通過 {sum(r['pass'] for r in xs)}")
    short = [r for r in rows if r["video_duration"] < 240]
    med = [r for r in rows if r["video_duration"] >= 240]
    for name, xs in (("short (<4分)", short), ("medium (4〜20分)", med)):
        if xs:
            print(f"  {name}: {len(xs)} → 通過 {sum(r['pass'] for r in xs)}（許容幅の中央値 {sorted(r['tol'] for r in xs)[len(xs)//2]:.0f} 秒）")
    print("  通過した場面の動画数:", len({r["video_id"] for r in rows if r["pass"]}))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", default=str(DATA / "time_generated.jsonl"))
    ap.add_argument("--frames-dir", default=str(DATA / "frames"))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--model", default=VL_MODEL)
    ap.add_argument("--out", default=str(DATA / "time_verified.jsonl"))
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
        print(f"場面 {len(items)}、済み {len(done)}、今回 {len(todo)}")
        if todo:
            client = Client()
            frames_dir = Path(args.frames_dir)
            t0 = time.time()
            with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
                futs = {ex.submit(check_item, client, it, frames_dir, args.frames, args.model): it for it in todo}
                for k, fut in enumerate(as_completed(futs), 1):
                    it = futs[fut]
                    try:
                        r = fut.result()
                    except Exception as e:  # noqa: BLE001
                        print(f"[{k}/{len(todo)}] {it['id']} 失敗: {str(e)[:120]}")
                        continue
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
                    f.flush()
                    if k % 20 == 0 or k == 1:
                        print(f"[{k}/{len(todo)}] {it['id']} 当たり={r['re_hits']} 距離={r['re_dists']}  {time.time() - t0:.0f}s  API {client.calls} 回 ${client.cost:.3f}")
    rows = [json.loads(l) for l in out.open(encoding="utf-8") if l.strip()]
    summarize(rows)


if __name__ == "__main__":
    main()
