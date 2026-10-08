# movie_llm

未公開映像素材を手元で検索できる小型映像理解モデル。

番組制作会社のディレクターが、外部に出せない未公開の収録素材から「〇〇が△△している場面」を日本語で探せるようになり、素材確認にかかる時間を大幅に減らせる。

プロジェクトの説明は [indx-proposal.md](indx-proposal.md)、これまでの作業の記録は [docs/progress.md](docs/progress.md) を参照。

## 現在の状態

評価基盤と、手元 (MacBook Pro M2 Pro, 16GB) での Qwen3-VL 推論ができた段階。学習・デモはまだ無い。

## セットアップ

[uv](https://docs.astral.sh/uv/) を使う。

```bash
uv sync
```

## Japanese Video-QA の内訳集計

```bash
uv run python scripts/jvqa_stats.py
```

`nablasinc/Japanese_Video-QA` を `data/jvqa/` にダウンロードし、形式・カテゴリ・領域別の内訳を `docs/jvqa_stats.md` に書き出す。結果は [docs/jvqa_stats.md](docs/jvqa_stats.md)。

## 推論（手元での動作確認）

```bash
uv run python scripts/smoke_test.py                        # 合成動画で Qwen3-VL-2B が動くか確認（MPS）
uv run python scripts/jvqa_predict.py --name qwen3vl-2b_f16 --type multi_choice --duration short --limit 20
```

`jvqa_predict.py` は動画が無ければ `yt-dlp` で `data/jvqa/videos/` に取得し（映像のみ・480p以下）、予測を `results/<name>/predictions.jsonl` に追記する。途中で止めても再開できる。`yt-dlp` は `uv tool install yt-dlp` で入れる。

動画のフレーム抜き出しは PyAV で自前に行う（[movie_llm/infer/video.py](movie_llm/infer/video.py)）。transformers 5.x の既定のデコードは torchcodec 前提で、macOS では入れにくいため。

## 評価

```bash
uv run pytest                                   # 採点規則のテスト
uv run python scripts/jvqa_trivial_baselines.py  # 「常に B」などの自明なベースラインを作る
uv run python scripts/jvqa_eval.py --predictions results/always_B/predictions.jsonl --name always_B
```

予測ファイルの形式、GPT-4o 審判の使い方、自明なベースラインの値は [docs/eval.md](docs/eval.md) を参照。

## ライセンス

Apache-2.0。評価に使うデータセットのライセンスは各配布元に従う（Japanese Video-QA の注釈は CC BY 4.0、研究目的限定）。
