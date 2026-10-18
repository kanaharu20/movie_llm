"""デモ用のサンプル映像を取得する（CC BY ライセンスの日本の動画 3 本、各 1〜2 分、短辺 360px）。

- yt-dlp が必要（`uv tool install yt-dlp` または `brew install yt-dlp`）
- 取得前にライセンスが「Creative Commons Attribution license (reuse allowed)」であることを確認し、違えば取得しない
- 出典を samples/ATTRIBUTION.md に書き出す（CC BY の表示義務）
- このリポジトリは動画ファイルを同梱しない。自分で撮った映像を置く方が望ましい（samples/README.md）

    uv run python scripts/demo_fetch_samples.py [--ids ID ...] [--out samples]
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

CC_BY = "Creative Commons Attribution license (reuse allowed)"
# 学習データ作成時に CC BY を確認した動画から、領域の違う短いものを選んだ
DEFAULT_IDS = ["DdzGeAGSpok", "XOiSYSLoKSc", "_VpWJ9DxTFg"]  # 観光名所（厳島神社）、自然・風景（天橋立）、食文化（お弁当）
FORMAT = "bv*[ext=mp4][vcodec^=avc1][height<=360]/bv*[ext=mp4][height<=360]/bv*[height<=360]/bv*"


def probe(video_id: str) -> dict | None:
    r = subprocess.run(["yt-dlp", "--dump-json", "--skip-download", "--no-warnings", f"https://www.youtube.com/watch?v={video_id}"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    return json.loads(r.stdout.splitlines()[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ids", nargs="*", default=DEFAULT_IDS)
    ap.add_argument("--out", default="samples")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# サンプル映像の出典", "", "いずれも YouTube で CC BY（クリエイティブ・コモンズ 表示）として公開されている動画。取得時にライセンスを確認した。", ""]
    for vid in args.ids:
        meta = probe(vid)
        if meta is None:
            print(f"{vid}: 取得できない（非公開か削除）")
            continue
        if meta.get("license") != CC_BY:
            print(f"{vid}: ライセンスが CC BY ではない（{meta.get('license')}）ので取得しない")
            continue
        dst = out / f"{vid}.mp4"
        if not dst.exists():
            r = subprocess.run(["yt-dlp", "-q", "--no-warnings", "-f", FORMAT, "-o", str(out / "%(id)s.%(ext)s"), f"https://www.youtube.com/watch?v={vid}"])
            if r.returncode != 0 or not dst.exists():
                print(f"{vid}: ダウンロード失敗")
                continue
        print(f"{vid}: {meta.get('title')} / {meta.get('channel')} ({meta.get('duration')} 秒) → {dst}")
        lines.append(f"- `{dst.name}`: 「{meta.get('title')}」 {meta.get('channel')}（{meta.get('channel_url')}）、{meta.get('webpage_url')}、CC BY")
    (out / "ATTRIBUTION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"出典: {out / 'ATTRIBUTION.md'}")


if __name__ == "__main__":
    main()
