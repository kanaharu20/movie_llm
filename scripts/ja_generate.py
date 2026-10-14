"""日本の映像（CC BY）から、正誤を機械で判定できる問題を Qwen3-VL-235B に作らせる。

入力: data/ja/video_status.jsonl（status=ok の動画）と data/ja/frames/<id>/f32_s640/
出力: data/ja/generated.jsonl（1 行 1 問。pt_convert.py と同じ項目 + evidence_time, gen_model）

1 本につき count（数、答えは整数）2 問、temporal（順序・時刻、4 択）2 問、action / spatial（4 択）2 問を頼む。
形式が壊れた問題は捨て、1 本ぶん丸ごと壊れていたら温度を上げて 1 回だけ作り直す。途中から再開できる。

    uv run python scripts/ja_generate.py --limit 50 --workers 4
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from movie_llm.data.vlm_api import VL_MODEL, Client, frame_content, parse_json
from movie_llm.eval.rules import LETTERS, extract_number, gold_letter, split_choices

print = functools.partial(print, flush=True)
DATA = Path("data/ja")

PROMPT = """上の画像は 1 本の動画から等間隔に抜いた {n} フレームで、各フレームの前に動画内の時刻 [分:秒] を示しています。
この動画について、映像を見れば答えが一つに決まり、映像を見なければ答えられない問題を 6 問作ってください。

内訳:
- count（数を答える）2 問: 「〜は何人/何個/何回ですか」。答えは整数。フレームに写っている範囲で数えられるものだけ。答えが 0 や 1 になる問題は避ける。
  数えた対象を items に 1 つずつ「時刻 + 何か」で列挙し、answer は items の個数と一致させる。同じ対象が複数のフレームに写っていても 1 回だけ数える。列挙できないものは出題しない
- temporal（順序・時刻）2 問、4 択: 「〜の後に何が起きますか」「〜が映るのは動画のどの時点ですか（序盤/中盤/終盤 または 分:秒 の範囲）」
- action または spatial 2 問、4 択: 人や物の動作、位置関係、色や向きなど

禁止:
- 字幕・テロップ・看板などの文字を読むだけで答えられる問題
- 一般常識や有名な場所・料理・行事の知識で答えられる問題（例: 「寿司に使う魚はどれ」「金閣寺がある都市はどこ」）
- 動画の一部の場面にしか当てはまらないのに、動画全体について聞く曖昧な問題
- 人物の名前や所属など個人を特定する問題
- 映像から判断しにくい問題（温度、味、音）
- 「フレームが何枚」「何枚目の画像」「最終フレーム」など、抜き出したフレームの枚数や並びに依存する問題。動画を通して見た人が答えられる問題にする
- 「字幕が出た後」「テロップの後」など、文字の表示を手がかりにする問題

4 択は正解 1 つと、映像を見ていないともっともらしい誤答 3 つ。選択肢は "A) …" "B) …" "C) …" "D) …" の形。正解の位置はばらす。
temporal の「どの時点か」は、「序盤/中盤/終盤」ではなく、動画の長さに応じた時刻の範囲（例: 「0:00〜0:30」「1:30〜2:00」）を選択肢にする。
各問に、答えの根拠となるフレームの時刻を evidence_time に書く。問題文は自然な日本語で、「このフレーム」「画像」という言い方はせず「動画」と書く。

JSON だけを出力:
{{"questions": [{{"type": "numeric", "category": "count", "question": "...", "choices": null, "items": ["00:32 左の赤い提灯", "00:32 中央の提灯", "00:40 右奥の提灯"], "answer": "3", "evidence_time": ["00:32", "00:40"]}},
                {{"type": "multi_choice", "category": "temporal", "question": "...", "choices": ["A) ...", "B) ...", "C) ...", "D) ..."], "answer": "B", "evidence_time": ["01:10", "01:20"]}}]}}"""


def normalize_item(q: dict, video_id: str, k: int, model: str) -> dict | None:
    """生成された 1 問を学習用の形式に直す。形式が壊れていれば None。"""
    t, cat = q.get("type"), q.get("category")
    question = (q.get("question") or "").strip()
    if not question or cat not in ("count", "temporal", "action", "spatial", "causal"):
        return None
    item = {"id": f"ja_{video_id}_{k}", "video_id": video_id, "question": question, "category": cat,
            "evidence_time": q.get("evidence_time") or [], "gen_model": model}
    if t == "numeric":
        n = extract_number(str(q.get("answer", "")))
        items = q.get("items")
        # 列挙した個数と答えが合わない問題は、数えた根拠が無いので捨てる
        if n is None or n < 0 or not isinstance(items, list) or len(items) != n:
            return None
        item.update(type="numeric", choices=[], answer=str(n), items=[str(x) for x in items])
        return item
    if t == "multi_choice":
        choices = q.get("choices") or []
        if len(choices) != 4:
            return None
        fixed = []
        for i, c in enumerate(choices):
            body = re.sub(r"^[A-DＡ-Ｄ]\s*[\)\.．:：、）]\s*", "", str(c)).strip()
            if not body:
                return None
            fixed.append(f"{LETTERS[i]}) {body}")
        if len(set(split_choices(fixed).values())) != 4:
            return None
        ans = str(q.get("answer", "")).strip()
        letter = gold_letter(ans, fixed)
        if letter is None:
            # "B) 本文" の形で返ってきた場合
            m = re.match(r"^([A-D])\s*[\)\.．:：、）]", ans)
            letter = m.group(1) if m else None
        if letter is None:
            return None
        # モデルは正解を B に置きがちなので、選択肢を並べ直して正解の位置を均す（問題 ID で種を固定）
        bodies = [split_choices(fixed)[L] for L in LETTERS]
        gold_body = bodies[LETTERS.index(letter)]
        random.Random(f"{video_id}_{k}").shuffle(bodies)
        fixed = [f"{L}) {b}" for L, b in zip(LETTERS, bodies)]
        item.update(type="multi_choice", choices=fixed, answer=LETTERS[bodies.index(gold_body)])
        return item
    return None


def generate_one(client: Client, model: str, frames_dir: Path, video_id: str, num_frames: int) -> list[dict]:
    content = frame_content(frames_dir, video_id, num_frames)
    if content is None:
        raise FileNotFoundError(f"フレームなし: {video_id}")
    content = content + [{"type": "text", "text": PROMPT.format(n=num_frames)}]
    messages = [{"role": "user", "content": content}]
    items: list[dict] = []
    for attempt, temp in enumerate((0.7, 1.0)):
        text = client.chat(model, messages, temperature=temp, max_tokens=2000, seed=attempt)
        data = parse_json(text)
        qs = data.get("questions") if isinstance(data, dict) else data
        if isinstance(qs, list):
            items = [it for it in (normalize_item(q, video_id, k, model) for k, q in enumerate(qs) if isinstance(q, dict)) if it]
        if len(items) >= 3:
            break
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default=str(DATA / "video_status.jsonl"))
    ap.add_argument("--frames-dir", default=str(DATA / "frames"))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--model", default=VL_MODEL)
    ap.add_argument("--out", default=str(DATA / "generated.jsonl"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    videos = [json.loads(l) for l in open(args.status, encoding="utf-8") if l.strip()]
    videos = [v for v in videos if v["status"] == "ok"]
    out = Path(args.out)
    done = set()
    if out.exists():
        done = {json.loads(l)["video_id"] for l in out.open(encoding="utf-8") if l.strip()}
    todo = [v for v in videos if v["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"動画 {len(videos)} 本、済み {len(done)} 本、今回 {len(todo)} 本")
    if not todo:
        return
    client = Client()
    frames_dir = Path(args.frames_dir)
    t0 = time.time()
    n_items = 0
    with out.open("a", encoding="utf-8") as f, ThreadPoolExecutor(args.workers) as ex:
        futs = {ex.submit(generate_one, client, args.model, frames_dir, v["id"], args.frames): v for v in todo}
        for k, fut in enumerate(as_completed(futs), 1):
            v = futs[fut]
            try:
                items = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"[{k}/{len(todo)}] {v['id']} 失敗: {str(e)[:120]}")
                continue
            for it in items:
                it["domain"] = v["domain"]
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
            f.flush()
            n_items += len(items)
            print(f"[{k}/{len(todo)}] {v['id']} {len(items)} 問 ({dict(collections.Counter(i['category'] for i in items))})  {time.time() - t0:.0f}s  API {client.calls} 回 ${client.cost:.3f}")
    rows = [json.loads(l) for l in out.open(encoding="utf-8") if l.strip()]
    print(f"\n合計 {len(rows)} 問 / {len({r['video_id'] for r in rows})} 本。category {dict(collections.Counter(r['category'] for r in rows))}、type {dict(collections.Counter(r['type'] for r in rows))}")
    if any(r["type"] == "multi_choice" for r in rows):
        print("4 択の正解位置:", dict(collections.Counter(r["answer"] for r in rows if r["type"] == "multi_choice")))


if __name__ == "__main__":
    main()
