"""候補動画（scripts/ja_search_cc.py の出力）を取得し、ライセンスを確認する。

- 各動画のメタデータを取り、license が "Creative Commons Attribution license (reuse allowed)" のものだけ取得する
  （検索フィルタは目安なので、ここで確定させる）
- 映像のみ・480p 以下（ベンチマークの動画と同じ形式）。音声は使わない
- 結果は data/ja/video_status.jsonl に 1 行 1 動画で記録し、途中から再開できる。出典一覧（CC BY の表示義務）にも使う
- domain ごとに本数が偏らないよう、domain を順番に回しながら取る

    uv run python scripts/ja_fetch.py --limit 50          # まず 50 本
    uv run python scripts/ja_fetch.py --limit 1000
"""

from __future__ import annotations

import argparse
import collections
import functools
import json
import subprocess
import time
from pathlib import Path

from movie_llm.infer.fetch import FORMAT, video_path

print = functools.partial(print, flush=True)
DATA = Path("data/ja")
CC_BY = "Creative Commons Attribution license (reuse allowed)"


def probe(video_id: str) -> dict | None:
    r = subprocess.run(["yt-dlp", "--dump-json", "--skip-download", "--no-warnings", f"https://www.youtube.com/watch?v={video_id}"],
                       capture_output=True, text=True)
    if r.returncode != 0:
        return None
    try:
        return json.loads(r.stdout)
    except json.JSONDecodeError:
        return None


def download(video_id: str, video_dir: Path) -> Path | None:
    cmd = ["yt-dlp", "-q", "--no-warnings", "-f", FORMAT, "-o", str(video_dir / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={video_id}"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return video_path(video_id, video_dir) if r.returncode == 0 else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(DATA / "candidates.jsonl"))
    ap.add_argument("--video-dir", default=str(DATA / "videos"))
    ap.add_argument("--status", default=str(DATA / "video_status.jsonl"))
    ap.add_argument("--limit", type=int, default=50, help="取得に成功した本数がこの数になったら止める（既存ぶんを含む）")
    ap.add_argument("--max-sec", type=int, default=1200)
    args = ap.parse_args()

    cands = [json.loads(l) for l in open(args.candidates, encoding="utf-8") if l.strip()]
    status_path = Path(args.status)
    done: dict[str, dict] = {}
    if status_path.exists():
        done = {json.loads(l)["id"]: json.loads(l) for l in status_path.open(encoding="utf-8") if l.strip()}
    video_dir = Path(args.video_dir)
    video_dir.mkdir(parents=True, exist_ok=True)
    n_ok = sum(1 for s in done.values() if s["status"] == "ok")
    print(f"候補 {len(cands)} 本、確認済み {len(done)} 本（取得成功 {n_ok}）、目標 {args.limit} 本")

    # domain を順番に回す（偏りを防ぐ）
    by_dom: dict[str, list[dict]] = collections.defaultdict(list)
    for c in cands:
        if c["id"] not in done:
            by_dom[c["domain"]].append(c)
    doms = sorted(by_dom)
    t0 = time.time()
    with status_path.open("a", encoding="utf-8") as f:
        while n_ok < args.limit and any(by_dom.values()):
            for dom in doms:
                if n_ok >= args.limit or not by_dom[dom]:
                    continue
                c = by_dom[dom].pop(0)
                meta = probe(c["id"])
                rec = {"id": c["id"], "domain": dom, "queries": c["queries"], "title": c.get("title"), "channel": c.get("channel"), "url": c["url"]}
                if meta is None:
                    rec.update(status="unavailable")
                elif meta.get("license") != CC_BY:
                    rec.update(status="not_cc_by", license=meta.get("license"))
                elif not meta.get("duration") or meta["duration"] > args.max_sec or meta.get("is_live"):
                    rec.update(status="too_long", duration=meta.get("duration"))
                else:
                    rec.update(title=meta.get("title"), channel=meta.get("channel"), channel_url=meta.get("channel_url"),
                               duration=meta["duration"], upload_date=meta.get("upload_date"), license=meta["license"])
                    p = download(c["id"], video_dir)
                    rec.update(status="ok" if p else "download_failed", path=str(p) if p else None)
                    if p:
                        n_ok += 1
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()
                print(f"[{n_ok}/{args.limit}] {dom} {c['id']} {rec['status']} {(rec.get('title') or '')[:40]}  {time.time() - t0:.0f}s")

    rows = [json.loads(l) for l in status_path.open(encoding="utf-8") if l.strip()]
    print("状態:", dict(collections.Counter(r["status"] for r in rows)))
    ok = [r for r in rows if r["status"] == "ok"]
    print("取得済み domain 別:", dict(collections.Counter(r["domain"] for r in ok)), f"合計 {sum(r['duration'] for r in ok) / 60:.0f} 分")


if __name__ == "__main__":
    main()
