"""コマンドラインでシーン検索（2 段構成 + 3 段の振り分け）。

    python scripts/demo.py --video samples/xxx.mp4 --query "赤い鳥居の前で参拝している場面" [--ask "何人映っていますか"] [--adapter models/timeloc-lora] [--json]
"""

from __future__ import annotations

import argparse
import json

from movie_llm.demo.pipeline import SceneSearch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--query", required=True, action="append", help="複数回指定できる")
    ap.add_argument("--ask", default=None, help="見つけた場面への追加の質問")
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--max-frames", type=int, default=1024)
    ap.add_argument("--no-verify", action="store_true", help="2 段目を省く（VideoJev だけ）")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    ss = SceneSearch(adapter=args.adapter, stage1_kwargs={"max_frames": args.max_frames})
    for q in args.query:
        r = ss.search(args.video, q, top_k=args.top_k, verify=not args.no_verify)
        if args.json:
            print(json.dumps({k: v for k, v in r.items()}, ensure_ascii=False))
            continue
        print(f"問い: {q}")
        print(f"  {r['band_label']}")
        print(f"  時刻 {r['time_str']}（候補区間 {r['interval_str']}、最大確率 {r['pmax']:.2f}、区間内平均 {r['pmean']:.2f}、一致 {r['agree']}）")
        if r["rationale"]:
            print(f"  根拠: {r['rationale']}")
        for k, c in enumerate(r["candidates"], 1):
            print(f"  候補 {k}: {c['label']}")
        if args.ask:
            print(f"  質問「{args.ask}」→ {ss.ask(args.video, r['interval'][0], r['interval'][1], args.ask)}")
        print(f"  {r['seconds']} 秒")


if __name__ == "__main__":
    main()
