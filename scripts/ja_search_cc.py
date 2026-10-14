"""日本の映像の学習データ候補: YouTube を Creative Commons 絞り込みで検索し、候補の一覧を作る。

- 検索語はベンチマークの subdomain（100 件: 祇園祭、寿司、富士山 …）と domain 名。題材をベンチマークに揃える
- YouTube の検索フィルタ「クリエイティブ・コモンズ」（sp=EgIwAQ%3D%3D）を使う。ライセンスの最終確認は取得時に
  各動画の license フィールド（"Creative Commons Attribution license (reuse allowed)"）で行う
- ベンチマーク（Japanese Video-QA）に含まれる動画は除く（評価データの混入を防ぐ）
- 長さは 30 秒〜20 分（ベンチマークの short / medium と同じ範囲）。ライブ配信のアーカイブ（数時間）は落ちる

出力: data/ja/candidates.jsonl（1 行 1 動画: id, title, channel, duration, queries, domain）

    uv run python scripts/ja_search_cc.py --per-query 50
"""

from __future__ import annotations

import argparse
import functools
import json
import subprocess
import time
import urllib.parse
from pathlib import Path

print = functools.partial(print, flush=True)
OUT = Path("data/ja/candidates.jsonl")
CC_FILTER = "EgIwAQ%3D%3D"  # 「クリエイティブ・コモンズ」フィルタ


def search(query: str, n: int) -> list[dict]:
    url = f"https://www.youtube.com/results?search_query={urllib.parse.quote(query)}&sp={CC_FILTER}"
    cmd = ["yt-dlp", "--flat-playlist", "--dump-json", "--playlist-end", str(n), "--no-warnings", url]
    r = subprocess.run(cmd, capture_output=True, text=True)
    rows = []
    for line in r.stdout.splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-query", type=int, default=50)
    ap.add_argument("--min-sec", type=int, default=30)
    ap.add_argument("--max-sec", type=int, default=1200)
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    from movie_llm.eval.dataset import load_dataframe

    df = load_dataframe()
    exclude = set(df.video_id)
    queries = [(sd, dom) for dom, g in df.groupby("domain") for sd in sorted(g.subdomain.unique())]
    queries += [(dom, dom) for dom in sorted(df.domain.unique())]
    print(f"検索語 {len(queries)} 件、除外する動画 {len(exclude)} 本")

    found: dict[str, dict] = {}
    stats = []
    t0 = time.time()
    for k, (q, dom) in enumerate(queries, 1):
        rows = search(q, args.per_query)
        kept = 0
        for r in rows:
            vid, dur = r.get("id"), r.get("duration")
            if not vid or vid in exclude or dur is None or not (args.min_sec <= dur <= args.max_sec):
                continue
            kept += 1
            if vid in found:
                found[vid]["queries"].append(q)
            else:
                found[vid] = {"id": vid, "title": r.get("title"), "channel": r.get("channel"), "duration": dur,
                              "queries": [q], "domain": dom, "url": f"https://www.youtube.com/watch?v={vid}"}
        stats.append((q, len(rows), kept))
        print(f"[{k}/{len(queries)}] {q}: 検索 {len(rows)} 件、長さ条件を満たす {kept} 件、累計 {len(found)} 本  {time.time() - t0:.0f}s")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for v in found.values():
            f.write(json.dumps(v, ensure_ascii=False) + "\n")
    total = sum(v["duration"] for v in found.values())
    print(f"\n候補 {len(found)} 本、合計 {total / 3600:.1f} 時間 -> {args.out}")
    by_dom: dict[str, int] = {}
    for v in found.values():
        by_dom[v["domain"]] = by_dom.get(v["domain"], 0) + 1
    print("domain 別:", by_dom)
    print("候補が少ない検索語:", [q for q, n, k in stats if k < 5])


if __name__ == "__main__":
    main()
