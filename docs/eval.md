# 評価手順（Japanese Video-QA）

## 採点の仕組み

公式（NABLAS）の採点は、800問すべてを GPT-4o が 1〜3 点で採点する。公式表の Avg は平均点、正解率は 3 点の割合。本リポジトリでは2種類の採点器を持つ。

| 採点器 | 対象 | 用途 |
|---|---|---|
| 規則採点 `movie_llm/eval/rules.py` | multi_choice 241問 + yes_no 16問 | API 不要。学習の報酬にも同じ関数を使う |
| 審判採点 `movie_llm/eval/judge.py` | 全問（公式と同じ GPT-4o・同じプロンプト） | 公式表と比較する正式な数値 |

規則採点は、正解なら 3、不正解なら 1 を返す。モデル出力から選択記号を取り出す規則は、文頭の記号 → 「答え: X」 → 本文中の「(X)」 → 選択肢本文の含有、の順。「AかB」のように2つ以上の記号を挙げた出力は読み取れなかった扱い（1点）にする。

審判採点は公式の `llm_judge_sample.py` と同じシステムプロンプト・入力順・`temperature=0`・JSON 出力で呼ぶ。結果は `results/judge_cache.jsonl` にキャッシュされ、同じ (モデル, 質問, 正解, 選択肢, 予測) には API を呼ばない。

## 予測ファイルの形式

JSONL、1行1問。`idx` は parquet の行番号（0〜799）。

```json
{"idx": 0, "prediction": "A) 江戸の吉原"}
```

## 実行

```bash
# 規則採点だけ（無料）
uv run python scripts/jvqa_eval.py --predictions results/foo/predictions.jsonl --name foo

# 審判採点も行う（.env に OPENAI_API_KEY が必要）
uv run python scripts/jvqa_eval.py --predictions results/foo/predictions.jsonl --name foo --judge

# open と規則採点できない問題だけ審判に回す（API 節約。公式値とは厳密には一致しない）
uv run python scripts/jvqa_eval.py --predictions ... --name foo --judge --judge-only-open

# まず少数で試す
uv run python scripts/jvqa_eval.py --predictions ... --name foo --judge --limit 20

# Vercel AI Gateway 経由で GPT-4o を使う（.env に AI_GATEWAY_API_KEY）
uv run python scripts/jvqa_eval.py --predictions ... --name foo --judge \
    --judge-base-url https://ai-gateway.vercel.sh/v1 --judge-api-key-env AI_GATEWAY_API_KEY --judge-model openai/gpt-4o

# LM Studio / Ollama のローカルモデルを審判にする（キー不要）
uv run python scripts/jvqa_eval.py --predictions ... --name foo --judge \
    --judge-base-url http://localhost:1234/v1 --judge-model <モデル名>
```

## 審判モデルの選び方

- **公式表と比べる数値**を出すときだけ GPT-4o が必要。審判が違うと数ポイント単位でずれる
- **自分の実験どうしの比較**（学習前後、2B と 4B）は、全条件で同じ審判を使えば何でもよい。open の答えは中央値 3 文字と短く、判定は易しい
- 接続先は OpenAI 互換なら何でもよい。Vercel AI Gateway はモデル名が `openai/gpt-4o` のように `提供元/モデル` の形。LM Studio や Ollama はローカルで無料
- キャッシュのキーには接続先とモデル名の両方が入るので、審判を変えると再採点になる（混ざらない）
- 最終的にはローカル審判と GPT-4o の一致率を 50 問程度で測り、高ければ以降はローカル審判で代用する

出力は `results/<name>/items.jsonl`（問題ごとの点）と `results/<name>/report.md`（全体・type・category・domain・duration 別の表）。

## 自明なベースライン（規則採点のみ）

`scripts/jvqa_trivial_baselines.py` で作る。映像を見ずに同じ答えを返した場合の値で、比較表の下限として使う。

| 予測 | multi_choice 正解率 | yes_no 正解率 |
|---|---|---|
| 常に B / 常に はい | 40.7% (98/241) | 68.8% (11/16) |
| 常に A / 常に いいえ | 26.6% (64/241) | 31.2% (5/16) |

選択式の正解が B に偏っているため、「常に B」でチャンスレート(25%)を大きく超える。モデルの選択式スコアはこの 40.7% と比べて評価する。

## 既知のデータ不備

- idx 698: multi_choice だが answer が `9種類`（選択肢 C の本文）。規則採点は本文一致で C に直して扱う
- idx 558: yes_no だが answer が自由記述。規則採点では採点せず、審判に回す
- open 問題は規則採点できない。評価の約7割が審判モデルに依存する
