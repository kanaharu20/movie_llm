# AWS で GRPO を回す手順書

AWS は「10 時間を超える学習」だけに使う（評価・選別は Colab、データ作成は手元）。GRPO 3 回目（時刻を当てる課題、根拠つき出力）は 1 ステップ 2〜3 分 × 300〜400 ステップで 10〜20 時間の見込みで、Colab の 1 セッション（最長 24 時間、切断あり）に収まらない。

## 先にインターン先に頼むこと（準備に日数がかかる）

1. **IAM ユーザー**（または一時的なアクセスキー）: EC2 と S3 の操作権限。手元の `~/.aws/credentials` に入れる（git には入れない）
2. **リージョン**: 東京 `ap-northeast-1` か、GPU の空きが多い `us-east-1` / `us-west-2`。指定がなければ `us-west-2`
3. **GPU インスタンスのクォータ**: 新しいアカウントは G 系の vCPU 上限が 0 のことが多い。Service Quotas で次の 2 つを **8 vCPU 以上**に上げてもらう（承認に 1〜2 日）
   - `Running On-Demand G and VT instances`
   - `All G and VT Spot Instance Requests`
4. **S3 バケット** 1 つ（例: `movie-llm-<名前>`）。フレームの zip（Perception Test 用 1GB、日本の映像 1,000 本で 数 GB）とチェックポイントを置く
5. **予算の上限**: AWS Budgets で月 2 万円のアラート。1 回の学習は約 1 万円の見込み

## インスタンス

| 用途 | インスタンス | GPU | 料金の目安（us-west-2） |
|---|---|---|---|
| 4B の GRPO（32 フレーム、G=8、根拠つき） | `g6e.xlarge` | L40S 48GB | オンデマンド 約 $1.9/時、スポット 約 $0.6〜0.9/時 |
| 代替（メモリ 24GB、micro_batch を 1 に） | `g5.xlarge` / `g6.xlarge` | A10G / L4 24GB | 約 $1.0/時 |

- AMI: **Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.x (Ubuntu 22.04)**。NVIDIA ドライバと CUDA が入っている
- ディスク: 200GB（モデル 9GB + フレーム + チェックポイント）
- **スポットを使う。** 途中で止まっても `latest.txt` から再開できる（`train_grpo.py` は同じコマンドの再実行で続きから進む）。止まったら新しいスポットを立て、S3 からチェックポイントを戻して再実行する

## 手順

1. 手元で S3 にフレームを上げる
   ```bash
   aws s3 cp data/ja/frames_f32_s640.zip s3://<bucket>/frames/ja_frames_f32_s640.zip
   aws s3 cp data/jvqa/frames_f32_s640.zip s3://<bucket>/frames/jvqa_frames_f32_s640.zip
   ```
2. インスタンスを立てる（コンソールでも CLI でもよい）。IAM ロールに S3 の読み書きを付ける
3. SSH で入り、`scripts/aws_setup.sh` を実行（uv、リポジトリ、依存、フレームの展開まで）
   ```bash
   curl -fsSL https://raw.githubusercontent.com/kanaharu20/movie_llm/main/scripts/aws_setup.sh | bash -s -- s3://<bucket>
   ```
4. 学習を `tmux` の中で回し、チェックポイントを 10 分ごとに S3 に同期する
   ```bash
   cd ~/movie_llm && tmux new -s grpo
   (while true; do aws s3 sync results/grpo3_time s3://<bucket>/results/grpo3_time --exclude "*.tmp"; sleep 600; done) &
   uv run python scripts/train_grpo.py --items data/ja/train_time.jsonl --out results/grpo3_time \
       --model Qwen/Qwen3-VL-4B-Instruct --frames 32 --video-dir data/ja/videos --frames-dir data/ja/frames \
       --group-size 8 --prompts-per-step 4 --micro-batch 2 --max-new-tokens 96 \
       --lr 1e-5 --beta 0.04 --max-steps 400 --save-every 20 --max-draws 12
   ```
5. 終わったら `results/grpo3_time/step_XXXXX/`（LoRA）と `train_log.jsonl` を S3 経由で手元に戻し、インスタンスを**終了**する（停止ではなく terminate。ディスク代が残らない）

## 評価（Colab でよい）

- 自前 dev: `scripts/pt_difficulty.py --items data/ja/dev_time.jsonl --greedy --adapter <LoRA>`（type=time に対応）
- 本番: `scripts/jvqa_predict.py --task time --adapter <LoRA>` → `scripts/jvqa_eval_time.py`
- 壊れていないか: `scripts/jvqa_predict.py --type multi_choice --adapter <LoRA>`（4 択 220 問）、`pt_difficulty.py --items data/perception_test/dev_grpo2.jsonl --greedy --adapter <LoRA>`

## 費用の見込み（1 回）

| 項目 | 時間 | 費用 |
|---|---|---|
| 学習（スポット $0.8/時） | 15〜20 時間 | $12〜16 |
| 学習（オンデマンド $1.9/時、スポットが取れないとき） | 15〜20 時間 | $29〜38 |
| S3（10GB、1 か月） | — | $0.3 |
| 評価（Colab） | 2 時間 | 約 25 ユニット |
