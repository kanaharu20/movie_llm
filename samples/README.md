# サンプル映像

デモ（`docker compose up` または `python -m movie_llm.demo.app`）は、このフォルダの `*.mp4` を選択肢に出す。

## 用意の仕方

1. **自分で撮った映像**を置く（推奨。写る人の同意を取り、知らない人の顔が判別できる映像は避ける）
2. または、CC BY ライセンスで公開されている日本の動画を取得する:

```bash
uv run python scripts/demo_fetch_samples.py     # 3 本、計 5 分ほど。yt-dlp が必要
```

取得した動画のライセンスは取得時に確認し、出典は `samples/ATTRIBUTION.md` に書き出す。
このリポジトリには動画ファイルを同梱しない（`.gitignore`）。

## 問いの例

- 「赤い鳥居の前で参拝している場面」
- 「料理を皿に盛り付けている場面」
- 「花火が打ち上がる場面」
