# movie_llm

未公開映像素材を手元で検索できる小型映像理解モデル。

番組制作会社のディレクターが、外部に出せない未公開の収録素材から「〇〇が△△している場面」を日本語で探せるようになり、素材確認にかかる時間を大幅に減らせる。

プロジェクトの説明は [indx-proposal.md](indx-proposal.md) を参照。

## 現在の状態

評価ベンチマークの分析段階。モデルの学習・デモはまだ無い。

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

## ライセンス

Apache-2.0。評価に使うデータセットのライセンスは各配布元に従う（Japanese Video-QA の注釈は CC BY 4.0、研究目的限定）。
