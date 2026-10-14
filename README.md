# movie_llm

未公開映像素材を手元で検索できる小型映像理解モデル。

番組制作会社のディレクターが、外部に出せない未公開の収録素材から「〇〇が△△している場面」を日本語で探せるようになり、素材確認にかかる時間を大幅に減らせる。

プロジェクトの説明は [indx-proposal.md](indx-proposal.md)、これまでの作業の記録は [docs/progress.md](docs/progress.md)、保留中の改善案は [docs/ideas.md](docs/ideas.md) を参照。

## 現在の状態

評価基盤、手元 (MacBook Pro M2 Pro, 16GB) での Qwen3-VL 推論、Colab A100 での 4B ベースライン、Perception Test を日本語化した学習データ、自前の GRPO（LoRA）学習ループまでできた段階。GRPO 1 回目はベンチマークを動かせず、2 回目の準備中。経過は [docs/progress.md](docs/progress.md)。デモはまだ無い。

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

## フレームキャッシュと Colab

```bash
uv run python scripts/jvqa_cache_frames.py --frames 16 32 --zip   # 全動画から JPEG フレームを抜き、zip も作る
```

動画のデコードは CPU が重いので、一度抜いたフレームを `data/jvqa/frames/` に置いて使い回す。`jvqa_predict.py --frames-dir data/jvqa/frames` で読む。Colab で 4B のベース記録を取る手順は [notebooks/colab_baseline_4b.ipynb](notebooks/colab_baseline_4b.ipynb)。zip を Drive に上げて上から実行する。

## 評価

```bash
uv run pytest                                   # 採点規則のテスト
uv run python scripts/jvqa_trivial_baselines.py  # 「常に B」などの自明なベースラインを作る
uv run python scripts/jvqa_eval.py --predictions results/always_B/predictions.jsonl --name always_B
```

予測ファイルの形式、GPT-4o 審判の使い方、自明なベースラインの値は [docs/eval.md](docs/eval.md) を参照。

## 学習（GRPO + LoRA）

学習データは Perception Test（CC BY 4.0）を日本語化したもの（[docs/perception_test.md](docs/perception_test.md)）。学習前の 4B に 8 回ずつ答えさせて全正解・全不正解を除き（[docs/pt_difficulty.md](docs/pt_difficulty.md)）、動画単位で train / dev に分ける。

```bash
uv run python scripts/pt_select.py                 # train_grpo2.jsonl / dev_grpo2.jsonl を作る（count は数字で答える形式）
uv run python scripts/train_grpo.py --items data/perception_test/train_grpo2.jsonl --out results/grpo \
    --model Qwen/Qwen3-VL-4B-Instruct --frames 32 --video-dir <動画> --frames-dir <フレーム> --max-steps 300
```

2026-10-14 に用途を「場面の時刻を当てる」（シーン検索）に絞った。経緯と評価の定義は [docs/scene_search.md](docs/scene_search.md)。学習は Colab A100 で行う（[notebooks/colab_grpo2_4b.ipynb](notebooks/colab_grpo2_4b.ipynb)）。仕組みは `movie_llm/train/grpo.py` の冒頭、1 回目の結果は [docs/grpo_run1.md](docs/grpo_run1.md)、2 回目の設計は [docs/grpo_run2.md](docs/grpo_run2.md)。

## 学ぶ（演習）

`learn/` に、本物と同じ構造の GRPO + LoRA を CPU で数秒で回すおもちゃモデルの演習がある（完成版と穴埋め版）。使い方は [learn/README.md](learn/README.md)。

## ライセンス

Apache-2.0。評価に使うデータセットのライセンスは各配布元に従う（Japanese Video-QA の注釈は CC BY 4.0、研究目的限定）。
