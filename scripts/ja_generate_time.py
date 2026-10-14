"""日本の映像（CC BY）から「場面の説明 + 時刻区間」を Qwen3-VL-235B に作らせる（時刻を当てる課題の学習データ）。

1 本につき 6〜8 場面。各場面は、動画の中で 1 回しか起きない具体的な出来事で、開始と終了の時刻を持つ。
説明は 2 つの言い方で書かせる:
  - query: シーン検索の言い方「赤い法被の男性が太鼓を叩いている場面」
  - question: ベンチマークの言い方「赤い法被の男性が太鼓を叩いているのは動画のどの時点ですか」
学習時はどちらかを問いとして使い、答えは区間の時刻。

入力: data/ja/video_status.jsonl（status=ok）と data/ja/frames/<id>/f32_s640/
出力: data/ja/time_generated.jsonl（1 行 1 場面: id, video_id, query, question, start, end（秒）, video_duration）

    uv run python scripts/ja_generate_time.py --workers 4
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from movie_llm.data.vlm_api import VL_MODEL, Client, frame_content, parse_json
from movie_llm.eval.timeloc import parse_time

print = functools.partial(print, flush=True)
DATA = Path("data/ja")

PROMPT = """上の画像は 1 本の動画（長さ {dur}）から等間隔に抜いた {n} フレームで、各フレームの前に動画内の時刻 [分:秒] を示しています。
この動画から、「場面を探す」練習問題を 6〜8 件作ってください。

各件は、動画の中で 1 回しか起きない具体的な出来事です。条件:
- 誰が（見た目の特徴）・何を・どうしているか、を具体的に書く。「人が映っている」「料理をしている」のように動画の複数の場面に当てはまる説明は禁止
- 同じ出来事が動画の別の場所でも起きるなら出題しない（答えが 1 か所に決まるものだけ）
- 字幕・テロップ・看板の文字を手がかりにしない。人物の名前や所属など個人を特定する表現を使わない
- start と end はその出来事が映っている区間（分:秒）。隣のフレームの時刻を見て、出来事が続いている範囲を書く。区間は 3 秒以上
- 場面どうしは重ならないようにし、動画の序盤・中盤・終盤から偏りなく選ぶ

2 つの言い方を両方書く:
- query: 「〜している場面」で終わる名詞句（例: 「赤い法被の男性が大太鼓を両手で叩いている場面」）
- question: 「〜のは動画のどの時点ですか」で終わる問い（例: 「赤い法被の男性が大太鼓を両手で叩いているのは動画のどの時点ですか」）

JSON だけを出力:
{{"scenes": [{{"query": "...", "question": "...", "start": "01:40", "end": "01:52"}}]}}"""


def fmt(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def normalize_scene(sc: dict, video_id: str, k: int, dur: float, model: str) -> dict | None:
    q, qq = (sc.get("query") or "").strip(), (sc.get("question") or "").strip()
    s, e = parse_time(str(sc.get("start", "")), dur), parse_time(str(sc.get("end", "")), dur)
    if not q or not qq or s is None or e is None:
        return None
    if e < s:
        s, e = e, s
    if e > dur + 2 or e - s < 2 or e - s > max(60.0, dur * 0.3):
        return None  # 動画の外、短すぎ、長すぎ（動画の 3 割超）は捨てる
    if any(w in q + qq for w in ("テロップ", "字幕", "タイトル", "文字")):
        return None
    return {"id": f"jt_{video_id}_{k}", "video_id": video_id, "query": q, "question": qq, "start": round(s, 1), "end": round(min(e, dur), 1),
            "video_duration": round(dur, 1), "type": "time", "gen_model": model}


def generate_one(client: Client, model: str, frames_dir: Path, video_id: str, num_frames: int) -> list[dict]:
    content = frame_content(frames_dir, video_id, num_frames)
    if content is None:
        raise FileNotFoundError(f"フレームなし: {video_id}")
    meta = json.loads((frames_dir / video_id / f"f{num_frames}_s640" / "meta.json").read_text())
    dur = float(meta["duration"])
    content = content + [{"type": "text", "text": PROMPT.format(n=num_frames, dur=fmt(dur))}]
    messages = [{"role": "user", "content": content}]
    items: list[dict] = []
    for attempt, temp in enumerate((0.7, 1.0)):
        text = client.chat(model, messages, temperature=temp, max_tokens=2000, seed=attempt)
        data = parse_json(text)
        scs = data.get("scenes") if isinstance(data, dict) else data
        if isinstance(scs, list):
            items = [it for it in (normalize_scene(sc, video_id, k, dur, model) for k, sc in enumerate(scs) if isinstance(sc, dict)) if it]
        if len(items) >= 3:
            break
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--status", default=str(DATA / "video_status.jsonl"))
    ap.add_argument("--frames-dir", default=str(DATA / "frames"))
    ap.add_argument("--frames", type=int, default=32)
    ap.add_argument("--model", default=VL_MODEL)
    ap.add_argument("--out", default=str(DATA / "time_generated.jsonl"))
    ap.add_argument("--limit", type=int)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    videos = [json.loads(l) for l in open(args.status, encoding="utf-8") if l.strip()]
    videos = [v for v in videos if v["status"] == "ok"]
    out = Path(args.out)
    done = {json.loads(l)["video_id"] for l in out.open(encoding="utf-8") if l.strip()} if out.exists() else set()
    todo = [v for v in videos if v["id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"動画 {len(videos)} 本、済み {len(done)} 本、今回 {len(todo)} 本")
    if not todo:
        return
    client = Client()
    frames_dir = Path(args.frames_dir)
    t0 = time.time()
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
            print(f"[{k}/{len(todo)}] {v['id']} {len(items)} 場面  {time.time() - t0:.0f}s  API {client.calls} 回 ${client.cost:.3f}")
    rows = [json.loads(l) for l in out.open(encoding="utf-8") if l.strip()]
    lens = sorted(r["end"] - r["start"] for r in rows)
    print(f"\n合計 {len(rows)} 場面 / {len({r['video_id'] for r in rows})} 本。区間の長さ 中央値 {lens[len(lens)//2]:.0f} 秒、"
          f"domain {dict(collections.Counter(r['domain'] for r in rows))}")


if __name__ == "__main__":
    main()
