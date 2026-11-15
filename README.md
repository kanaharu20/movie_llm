# movie_llm

未公開映像素材を手元で検索できる小型映像理解モデル。

番組制作会社のディレクターが、外部に出せない未公開の収録素材から「〇〇が△△している場面」を日本語で探せるようになり、素材確認にかかる時間を大幅に減らせる。

A small video-understanding model for searching unaired footage in Japanese, on your own machine. Built during an internship at INDX (INDX Summer of Code 2026).

## 結果 / Results

Japanese Video-QA の時刻の課題（722 問）で、場面の時刻を当てた割合。

Share of correctly localized scenes on the timestamp questions of Japanese Video-QA (722 questions).

| 構成 / Setup | 全体 / All | 短い動画 / Short | 長い動画 / Medium |
|---|---|---|---|
| Qwen3-VL-4B（学習前 / base、32 frames） | 55.4% | 77.1% | 32.5% |
| ＋ GRPO + LoRA（自前実装 / self-written、64 frames） | 72.4% | 88.1% | 55.8% |
| VideoJev-4B（区間分類 / interval classifier） | 83.4% | 87.1% | 79.5% |
| **2 段構成 / Two-stage**（VideoJev → 学習済み Qwen3-VL） | **87.5%** | **91.4%** | **83.5%** |

- 生成で時刻を書く方式（GRPO）は長い動画の後半に弱いと分かり、製品の中核を区間分類の VideoJev に乗り換えた。学習済み Qwen3-VL は確認・根拠・質問応答の役にした。
- 2 つのモデルの一致を信頼度にして「自動で返す / 候補を確認 / 無い可能性」の 3 段に振り分ける。自動で返す帯（全体の 56%）の正解率は 95.3%。
- Generating a timestamp as text (GRPO) was weak on the second half of long videos, so the core moved to the interval classifier VideoJev; the trained Qwen3-VL now verifies, explains and answers follow-up questions.
- Agreement between the two models is used as confidence to route each query to auto-answer / human check / probably-absent. The auto-answer band (56% of queries) is 95.3% correct.

詳細 / Details: [docs/videojev_eval.md](docs/videojev_eval.md), [docs/progress.md](docs/progress.md)

プロジェクトの説明は [indx-proposal.md](indx-proposal.md)、これまでの作業の記録は [docs/progress.md](docs/progress.md)、保留中の改善案は [docs/ideas.md](docs/ideas.md) を参照。

## デモ: シーン検索（1 コマンド起動）

「〇〇が△△している場面」を日本語で探し、時刻・根拠・候補を返す。1 段目の VideoJev-4B（区間分類、Apache-2.0）が候補区間を出し、2 段目の学習済み Qwen3-VL-4B（LoRA）が確かめて根拠を書く。信頼度で「自動で返す / 候補を確認 / 無い可能性」の 3 段に振り分ける（Japanese Video-QA 722 問で、自動の帯 56% は 95.3%、全体 87.5%。`docs/videojev_eval.md`）。

GPU 24GB 以上（NVIDIA）と nvidia-container-toolkit が必要。初回はモデルの重み約 18GB を取得する。

```bash
# 1. 学習済み LoRA を models/timeloc-lora/ に置く（Hugging Face: kanaharu20/movie-llm-timeloc-lora）
uv run hf download kanaharu20/movie-llm-timeloc-lora --local-dir models/timeloc-lora
# 2. サンプル映像を samples/ に置く（自分の素材、または CC BY の動画を取得）
uv run python scripts/demo_fetch_samples.py
# 3. 起動 → http://localhost:7860
docker compose up
```

Docker を使わない場合:

```bash
uv sync --extra demo
uv run python -m movie_llm.demo.app --adapter models/timeloc-lora --host 0.0.0.0     # ブラウザ
uv run python scripts/demo.py --video samples/xxx.mp4 --query "赤い鳥居の前で参拝している場面" --ask "何人映っていますか"   # コマンドライン
```

GPU が手元に無い場合は `notebooks/colab_demo.ipynb`（Colab A100 で起動し、共有リンクを出す）。

## 現在の状態

評価基盤、日本語の場面データ（3,475 件）、自前の GRPO（LoRA）学習、VideoJev との比較、2 段構成と信頼度での振り分け、1 コマンドで起動するデモまでできた。次は、自前で撮った未編集素材と「映っていない場面」の問いで精度と「該当なし」の境目を測ること。経過は [docs/progress.md](docs/progress.md)。

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

## ライセンス

Apache-2.0。評価に使うデータセットのライセンスは各配布元に従う（Japanese Video-QA の注釈は CC BY 4.0、研究目的限定）。
