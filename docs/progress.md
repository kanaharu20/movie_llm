# これまでの作業の記録

2026-10-08 時点。何を、なぜ、どうやったかを順に書く。コードの場所はすべてこのリポジトリからの相対パス。

## 全体像

目標は「小型の映像理解モデル (Qwen3-VL-4B) を日本の映像に特化させ、フロンティアモデルとの差を縮める」こと。そのために、まず **測る仕組み** を作った。学習はまだ始めていない。

```
CLAUDE.md の「最初に取り組むタスク案」
  1. ベンチマークの内訳集計      → 済み
  2. 評価基盤                    → 済み（審判モデルは API キー未設定のため未実行）
  3. 手元で Qwen3-VL の推論確認  → 済み（2B は動く。4B はメモリ不足）
  4. ベースライン評価と弱点分析  → 進行中（動画を取得中）
  5. 学習データ作成の設計        → 未着手
```

## 手順 1: リポジトリを作った

`movie_llm/` はもともとホームディレクトリ全体の git に含まれる未追跡のフォルダだった。ここで `git init` して独立したリポジトリにし、`git@github.com:kanaharu20/movie_llm.git` に接続した。

追加したもの:

- `README.md`: 使い方
- `LICENSE`: Apache-2.0（INDX の要件）
- `pyproject.toml`, `uv.lock`: 依存パッケージ。`uv sync` で同じ環境を再現できる
- `.python-version`: Python 3.12 を使う

## 手順 2: ベンチマークの中身を数えた

**やったこと**: `nablasinc/Japanese_Video-QA` を Hugging Face から取得し、800問を形式・カテゴリ・領域ごとに数えた。

- スクリプト: `scripts/jvqa_stats.py`
- 結果: `docs/jvqa_stats.md`

**分かったこと**:

| 観点 | 数 |
|---|---|
| open（自由記述） | 542問（68%） |
| multi_choice（4択） | 241問（30%） |
| yes_no | 17問（2%） |
| category | spatial 214、count 206、action 184、temporal 158、causal 38 |

- 正誤を機械的に判定できるのは 4択と yes_no の 258問だけ。残り 7割は GPT-4o に採点してもらう必要がある
- 4択の正解は B が 41%。「常に B」と答えるだけで 4割取れる
- ラベルの不備が 2件（4択なのに答えが選択肢の本文、yes_no なのに答えが文章）
- 時刻の表記が 5種類以上混在している

## 手順 3: 採点の仕組みを作った

**やったこと**: 公式の採点サンプル (`llm_judge_sample.py`) を読んで、同じ方式を再現した。

公式は 800問すべてを GPT-4o に「1=不正解, 2=部分的, 3=正解」で採点させている。これに加えて、API を使わない規則採点も作った。

| ファイル | 役割 |
|---|---|
| `movie_llm/eval/dataset.py` | ベンチマークの読み込み。行番号を `idx` として付ける |
| `movie_llm/eval/rules.py` | 規則採点。モデル出力から「C」「はい」などを取り出して正解と比べる |
| `movie_llm/eval/judge.py` | GPT-4o 採点。公式と同じプロンプト。結果はキャッシュして二度呼ばない |
| `movie_llm/eval/report.py` | 集計と Markdown の表 |
| `scripts/jvqa_eval.py` | 採点コマンド |
| `scripts/jvqa_trivial_baselines.py` | 「常に B」などの比較用の予測を作る |
| `tests/` | 採点規則のテスト 11件 |

**規則採点の例**:

```
モデル出力 "C) 9種類"          → C
モデル出力 "**D**"             → D
モデル出力 "トッピングは9種類"  → 選択肢の本文と一致するので C
モデル出力 "AかBのどちらか"     → 判定不能 → 1点
```

**自明なベースライン**（映像を見ずに同じ答えを返した場合）:

| 予測 | 4択の正解率 | yes_no の正解率 |
|---|---|---|
| 常に B / 常に はい | 40.7% | 68.8% |
| 常に A / 常に いいえ | 26.6% | 31.2% |

詳しい使い方は `docs/eval.md`。

### 手順 3 のコードの中身

#### `movie_llm/eval/dataset.py`（ベンチマークの読み込み）

- `load_dataframe()`: Hugging Face から parquet ファイルを `data/jvqa/` に落として pandas の DataFrame にする。先頭に `idx` 列（0〜799 の行番号）を足す。配布データに問題 ID が無いので、この番号を ID 代わりに使う
- `load_items()`: 上の DataFrame を「1問 = 1つの dict」のリストに変える。`choices` は numpy の配列で来るので、普通の `list[str]` に直す。以降のコードはすべてこのリストを使う

```python
items = load_items()
items[2]
# {'idx': 2, 'video_id': 'WYzLvf78-sA', 'question': '粘土で作られた2つのオブジェは…',
#  'answer': 'B', 'type': 'multi_choice', 'choices': ['A) 1回', 'B) 2回', 'C) 3回', 'D) 4回'], ...}
```

#### `movie_llm/eval/rules.py`（規則採点）

モデルの出力文字列から「どの選択肢を選んだか」を取り出し、正解と比べる。正解なら 3、不正解なら 1 を返す（公式の 1〜3 点の尺度に合わせるため）。

- `normalize(text)`: 全角の `Ｃ` を半角の `C` にし、前後の空白を落とす。以降の比較はすべてこの正規化後の文字列で行う
- `split_choices(choices)`: `['A) 1回', 'B) 2回']` を `{'A': '1回', 'B': '2回'}` に分ける。正規表現 `^([A-D])\s*[\)\.．:：、）]\s*(.*)$` で「記号 + 区切り + 本文」に分解している
- `gold_letter(answer, choices)`: 正解ラベルを A〜D に揃える。普通は `'B'` がそのまま返る。idx 698 のように正解が `'9種類'` と本文で書かれている場合は、`split_choices` の本文と突き合わせて `'C'` に直す
- `extract_choice(prediction, choices)`: 本体。次の順で試し、最初に決まったものを返す。どれも当たらなければ `None`
  1. 文頭の記号（`_LEAD`）。`"C) 9種類"` や `"**D**"` から C や D を取る。ただし本文中に他の記号も単独で出てくる場合（`"AかB"`）は採らない。`_STANDALONE` で単独の記号を数えて 1種類のときだけ採用する
  2. 「答え: X」「正解はX」「Answer: (X)」の形（`_TAG`）
  3. 本文中の `(X)` や `X)` が 1種類だけある場合（`_PAREN`）
  4. 選択肢の本文が 1つだけ含まれている場合。`"トッピングは9種類ありました"` なら C の本文 `9種類` が含まれるので C。`"10種類"` のように複数の本文が部分一致する場合は最長一致を採る
- `extract_yes_no(prediction)`: `いいえ`, `はい`, `yes`, `no` のうち文字列中で最初に出てくるものを採る。英単語は前後が英字でない場合だけ数える（`Nothing` を `No` と誤読しないため）
- `score_rule(item, prediction)`: 問題の `type` を見て上の関数を呼び分ける。`multi_choice` なら `extract_choice` と `gold_letter` を比べ、`yes_no` なら `extract_yes_no` と正解を比べる。`open` や正解ラベルが壊れている問題は `None`（審判モデルに回す）

#### `movie_llm/eval/judge.py`（GPT-4o 採点）

公式サンプルと同じ文字列を組み立てて OpenAI API を呼ぶ。

- `SYSTEM_PROMPT`: 公式のシステムプロンプトをそのままコピーしたもの
- `format_choices(choices)`: 公式は Python のリストを `str()` した文字列（`"['A) 1回', 'B) 2回']"`）を渡しているので同じにする。選択肢が無ければ `"なし"`
- `user_prompt(...)`: 「質問: / 解答: / 選択肢: / モデルの回答:」の 4行を公式と同じ並びで作る
- `cache_key(...)`: モデル名・質問・正解・選択肢・予測の 5つを JSON にして SHA-256 を取る。同じ入力なら同じキーになるので、採点済みかどうかをこれで判定する
- `class Judge`:
  - `__init__`: キャッシュファイル `results/judge_cache.jsonl` を読み込んで dict にする。OpenAI のクライアントはここでは作らない（API キーが無くても import できるように）
  - `_get_client()`: 最初に必要になったときだけ `.env` を読んでクライアントを作る
  - `_call(...)`: 実際に API を呼ぶ。`temperature=0`、`response_format={"type": "json_object"}` は公式と同じ。失敗したら 2秒、4秒、8秒…と間隔を倍にして最大5回やり直す
  - `score(...)`: キャッシュにあればそれを返し、なければ `_call` してキャッシュに追記する
  - `score_many(jobs)`: `ThreadPoolExecutor` で 4本並列に `score` を呼ぶ。API 待ちが大半なので並列にすると速い

#### `movie_llm/eval/report.py`（集計）

- `md_table(df)`: DataFrame を Markdown の表にする。pandas の `to_markdown` は別パッケージが要るので自前で書いた
- `summarize(df, by)`: `final_score` 列から、公式表と同じ「Avg, Score 1, Score 2, Score 3」と正解率（3点の割合）を出す。`by="category"` のように指定すると、その列でグループ分けして出す
- `build_report(rows, title)`: 全体、type 別、category 別、domain 別、duration 別の表を並べる。規則採点と審判採点の両方がある問題については一致率も出す。規則採点で記号を読み取れなかった出力の例も載せる（プロンプトの改善に使う）

#### `scripts/jvqa_eval.py`（採点コマンド）

`main()` の流れ:

1. `load_items()` で 800問を読む
2. `read_predictions()` で予測 JSONL を `{idx: 予測文字列}` の dict にする
3. 予測が無い問題は除く（`--all` を付けると空文字として 1点扱いにする）
4. 1問ずつ `score_rule` を呼び、`rule_score` を付ける。読み取った記号も `rule_extracted` として残す
5. `--judge` が付いていれば `Judge.score_many` で審判の点 `judge_score` を付ける。`--judge-only-open` なら規則採点できなかった問題だけ送る
6. `final_score` = 審判の点があればそれ、なければ規則採点の点
7. `results/<name>/items.jsonl` に 1問ずつ書き、`report.md` に表を書く

#### `scripts/jvqa_trivial_baselines.py`

800問すべてに同じ答え（4択は B、yes_no は はい、open は空）を書いた予測ファイルを `results/always_B/predictions.jsonl` に作るだけ。`jvqa_eval.py` に渡せば「常に B」の点が出る。

## 手順 4: 手元で Qwen3-VL を動かした

**やったこと**: MacBook Pro (M2 Pro, 16GB) で Qwen3-VL-2B-Instruct を動かし、動画に質問できることを確かめた。

| ファイル | 役割 |
|---|---|
| `movie_llm/infer/video.py` | PyAV で動画から等間隔に 16 枚のフレームを抜く |
| `movie_llm/infer/qwen3vl.py` | モデルの読み込みと質問。MPS（Mac の GPU）で動く |
| `movie_llm/infer/fetch.py` | yt-dlp で YouTube 動画を取得（映像のみ、480p 以下） |
| `scripts/smoke_test.py` | 合成動画で動作確認 |
| `scripts/jvqa_predict.py` | ベンチマークの問題に答えて `results/<名前>/predictions.jsonl` に書く。途中から再開できる |
| `scripts/jvqa_fetch_videos.py` | 428本の動画をまとめて取得。取れなかった動画を記録する |

**流れ**:

```
動画ファイル
  → PyAV で 16 フレーム抜く（長辺 640px 以下に縮小）
  → Qwen3-VL の processor に「フレーム + 時刻のメタデータ」を渡す
  → 質問文と一緒にモデルへ（4択なら「記号だけ答えて」と指示）
  → 出力文字列を predictions.jsonl に保存
  → scripts/jvqa_eval.py で採点
```

**結果**:

- 合成動画（赤い四角が左→右へ動き、青い丸が2回出る）: 3問とも正解
- ベンチマークの短い4択 17問: 正解率 70.6%（12/17）。1問 約4秒
- category 別: spatial 7/8、temporal 2/3、count 2/5、action 1/1。数を数える問題が弱そう（ただし 17問なので傾向の確認程度）

**4B について**: 重み 8.3GB の読み込みは 16秒で済んだが、推論が進まなかった。この機械では別のプロセスが 10GB 以上使っていてスワップが満杯だったため。4B は他のアプリを閉じて再試行するか、クラウド GPU で動かす。手元の開発は 2B で進める。

詳しくは `docs/local_inference.md`。

### 手順 4 のコードの中身

#### `movie_llm/infer/video.py`（フレームの抜き出し）

transformers 5.x は動画ファイルを torchcodec で読もうとするが Mac では入れにくい。そこで PyAV で自前に読む。

- `probe(path)`: 動画を開いて、フレームレート（fps）、総フレーム数、長さ（秒）、縦横のサイズを返す。総フレーム数がコンテナに書かれていない動画もあるので、その場合は `長さ × fps` で見積もる
- `sample_frames(path, num_frames, max_side)`:
  1. `np.linspace(0, 総フレーム数-1, num_frames)` で等間隔なフレーム番号を決める。たとえば 1000 フレームの動画から 16枚なら 0, 67, 133, …, 999
  2. 動画を先頭から順にデコードし、目当ての番号のフレームだけ取り出す。`max_side=640` なら長辺が 640px を超えるフレームは縮小する（メモリ節約）
  3. 取り出したフレームを `(枚数, 高さ, 幅, 3)` の uint8 配列に積む
  4. `VideoMetadata`（fps と各フレームの元の番号）を一緒に返す。Qwen3-VL はこれを使って「<2.3 seconds>」のような時刻を各フレームの前に挿入する。時刻を正しく渡さないと、時間に関する質問の精度が落ちる

#### `movie_llm/infer/qwen3vl.py`（モデルの読み込みと質問）

- `pick_device(device)`: CUDA があれば `cuda`、Mac の GPU なら `mps`、どちらも無ければ `cpu`。クラウドの GPU でも同じコードが動くようにするため
- `build_prompt(question, qtype, choices)`: 問題の種類ごとに指示文を作る。4択なら選択肢を並べて「記号だけ答えて」、yes_no なら「はいかいいえで」、open なら「簡潔に」。規則採点で読み取りやすい形で答えさせるのが目的
- `GenResult`: 1回の推論の結果。出力文字列、動画トークン数、入力トークン数、かかった秒数
- `class Qwen3VL`:
  - `__init__`:
    - `AutoProcessor.from_pretrained` でトークナイザと画像・動画の前処理器を読む
    - `processor.video_processor.size` を書き換えて、1フレームあたりの画素数の上限を `256 × 28 × 28` にする。これが動画トークン数を決める。小さくすると速くなるが粗くなる
    - `AutoModelForImageTextToText.from_pretrained` で重みを読み、`bfloat16` にして GPU に載せる。`eval()` で推論モードにする
  - `ask(video_path, prompt)`:
    1. `sample_frames` でフレームとメタデータを取る
    2. `messages` を「動画 + 文章」の形で作り、`apply_chat_template` で Qwen の会話形式の文字列にする。この時点では動画の位置に目印のトークンが 1つ入るだけ
    3. `processor(text=..., videos=[frames], video_metadata=[meta], do_sample_frames=False)` でトークン ID と画素のテンソルにする。目印のトークンがここで数百〜千数百個の動画トークンに置き換わる。`do_sample_frames=False` は「もう抜き出し済みなので再サンプリングしない」という指定
    4. `model.generate(..., do_sample=False)` で貪欲法（毎回いちばん確率の高いトークンを選ぶ）で生成する。評価では再現性が要るので乱数を使わない
    5. 出力テンソルのうち入力の長さより後ろが新しく生成された部分なので、そこだけ `decode` して文字列にする
    6. 入力 ID の中で動画トークンの ID と一致する数を数え、動画トークン数として記録する（コストの目安）

#### `movie_llm/infer/fetch.py`（動画の取得）

- `video_path(video_id)`: `data/jvqa/videos/<id>.mp4` などが既にあればそのパス
- `fetch(video_id)`: 無ければ `yt-dlp` を `subprocess` で呼んで取得する。フォーマット指定 `bv*[ext=mp4][vcodec^=avc1][height<=480]/...` は「映像だけ、mp4、H.264、480p 以下」の意味。映像と音声を結合しないので ffmpeg が要らない。失敗したら `None`

#### `scripts/smoke_test.py`

合成動画（`data/samples/synthetic.mp4`）に 3つの質問をして、答えと秒数を表示するだけ。環境が壊れていないかを 30秒で確認するためのもの。

#### `scripts/jvqa_predict.py`（ベンチマークの予測）

`main()` の流れ:

1. `load_items()` で読み、`--idx` `--type` `--duration` `--limit` で対象を絞る
2. `results/<name>/predictions.jsonl` が既にあれば、その中の `idx` は済みとして飛ばす（途中から再開できる）
3. `Qwen3VL` を 1回だけ読み込む
4. 1問ずつ: `fetch` で動画を用意 → `build_prompt` → `model.ask` → 結果を 1行の JSON にして追記。1問ごとに `flush` するので、途中で止めても書いた分は残る
5. 動画が取れない、推論でエラー、のどちらも「その問題を飛ばす」だけで全体は止めない

#### `scripts/jvqa_fetch_videos.py`（全動画の取得）

800問から重複を除いた 428本の `video_id` を、短い動画を先に並べて順に `yt-dlp` で取る。1本ごとに `data/jvqa/video_status.jsonl` に成功か失敗か（と失敗理由）を書く。既にあるファイルと、以前失敗した ID は飛ばす。2秒ずつ間隔を空けている。

## つまずいた点と対処

| 問題 | 対処 |
|---|---|
| `huggingface-hub>=2` を固定すると transformers が入らない | `>=1.0` に緩めた |
| transformers 5.x は動画を torchcodec でデコードしようとし、Mac では入れにくい | PyAV で自前にフレームを抜いて渡すようにした |
| `Qwen3VLVideoProcessor` が torchvision を要求する | torchvision を追加 |
| `scripts/` から `movie_llm` パッケージが import できない | hatchling でインストール可能なパッケージにした |
| YouTube は映像と音声が別配信で、結合に ffmpeg が要る | 映像だけ取る（音声は使わない） |
| ログにリダイレクトすると print が出てこない | `flush=True` にした |
| 評価コマンドが予測の無い問題を 0点扱いしていた | 予測のある問題だけ採点する既定に変えた |
| 4B の推論がメモリ不足で止まった | 手元は 2B に限定 |

## 手順 5: 2B のベースラインを取った

動画の取得が終わった（428本中 390本、91%。残りは非公開・削除・要ログイン）。動画がある 722問に 2B で答えさせ、4択・yes_no は規則採点、open は GPT-4o（Vercel AI Gateway 経由）で採点した。結果の全文は `docs/baseline_2b.md`。

| モデル | Avg | 正解率 | 問数 |
|---|---|---|---|
| Gemini 3 Pro（公式） | 2.61 | 76.3% | 800 |
| Qwen3-VL-4B-Instruct（公式） | 2.19 | 54.5% | 800 |
| Qwen3-VL-2B-Instruct（本リポジトリ、16フレーム） | 1.81 | 36.7% | 722 |

| 区分 | 正解率 | 問数 |
|---|---|---|
| 4択 | 59.1% | 220 |
| yes_no | 75.0% | 12 |
| open | 25.7% | 490 |
| count / temporal / action / spatial / causal | 39.4 / 33.1 / 27.2 / 44.4 / 41.9% | 188 / 145 / 169 / 189 / 31 |
| short / medium | 41.5% / 31.6% | 371 / 351 |

- 4択は「常に B」(40.7%) を大きく超えるが、open は 25.7% と低い。失敗の大半は「白」を「ピンク」のような映像の読み取りの誤り
- count の open 137問で、数が一致 47、過小 60、過大 26。16フレームでは出来事を取りこぼしていると考えられる
- 長い動画（medium）ほど低い。フレーム間隔が広がるためで、フレーム数を増やす実験が必要
- 規則採点と GPT-4o の一致率は 98.3%（4択・yes_no 232問）。不一致 4件はすべて GPT-4o 側の誤り。以後 GPT-4o は open にだけ使う
- 1問 平均 5 秒。800問の予測に約 1時間、GPT-4o 採点は open 約 500問で数百円

## 手順 6: フレーム数を変えて測った

学習に入る前に、入力の設定だけでどれだけ変わるかを確かめた。2B の 4択・yes_no 232問を、フレーム数 16、32、64 で比べた。規則採点のみ。全文は `docs/frames_experiment.md`。

| 区分 | 16フレーム | 32フレーム | 64フレーム |
|---|---|---|---|
| 4択 | 59.1% | 67.3% | 70.9% |
| count | 51.0% | 60.8% | 58.8% |
| temporal | 59.2% | 69.4% | 77.6% |
| medium | 58.4% | 66.3% | 70.3% |
| 1問の秒数 | 5.2 | 8.2 | 18.6 |

- フレームを倍にするだけで 4択が 8 ポイント上がり、64 でさらに 4 ポイント上がる
- temporal は一貫して伸びる。count は 32 で頭打ちで、学習で狙う場所として妥当
- 学習の比較は 32フレームで揃える（64 は 1問 18.6 秒で重い）

## 今動いているもの

- Colab（CPU）で Perception Test の映像取得とフレーム抜き出し（ユーザーが実行中）

## 分かってきた注意点

- **動画の 1割弱（38本）が YouTube から取れない**（非公開・削除・要ログイン）。公式値と厳密には比べられないので、評価できた問題数を必ず併記する
- **自由記述が 7割**なので、公式と同じ数値を出すには GPT-4o の採点が要る。学習中の進捗は規則採点できる 258問で追い、公式比較のときだけ審判を使う
- 4択は「常に B」で 40.7% 取れる。モデルの 4択スコアはこの値を下回ってはいけない

## 手順 7: Colab の A100 で 4B のベースラインを取った

手元では 4B が動かなかったので、Colab Pro（A100）で取った。動画は上げず、手元で抜いた JPEG フレーム（16 と 32、合計 850MB）を Drive 経由で渡した。ノートブックは `notebooks/colab_baseline_4b.ipynb`、結果の全文は `docs/baseline_4b.md`。

| モデル | フレーム | Avg | 正解率 | 問数 |
|---|---|---|---|---|
| Qwen3-VL-4B-Instruct（公式） | 不明 | 2.19 | 54.5% | 800 |
| **Qwen3-VL-4B-Instruct（本リポジトリ）** | 32 | 2.00 | 45.8% | 722 |
| Qwen3-VL-4B-Instruct（本リポジトリ） | 16 | 1.92 | 41.7% | 722 |
| Qwen3-VL-2B-Instruct（本リポジトリ） | 16 | 1.81 | 36.7% | 722 |

- 2B → 4B で 5 ポイント、16 → 32フレームで 4 ポイント
- count は 4B でもほぼ伸びない（39 → 43%）。学習で狙う場所として最有力
- 以後の学習前後の比較は「4B・32フレーム・45.8%」を基準にする
- つまずき: リポジトリが非公開で Colab からクローンできなかった（公開に変更）。T4 では bfloat16 のエミュレーションで 1問 120秒かかった（GPU 世代で float16 に切り替えるよう修正。A100 では 1問 2.5秒）

## 手順 8: 学習データ候補 Perception Test を日本語化した

CC BY 4.0 の動画 QA。train + valid で 26,532 問 / 7,215 本、3択、正解位置は均等。counting 系 6,076 問、sequencing 3,772 問があり、ベンチマークで伸びなかった count と temporal を狙える。詳細と変換の設計は `docs/perception_test.md`。

- 問い 132 種は人手で訳した（`data/perception_test/translations_questions_manual.json`）
- 選択肢 約 4,500 種はローカルの Qwen3-8B（LM Studio）で訳し、英語が残った 306 件を厳しい指示で訳し直し、それでも残った 178 件を Qwen3-32B（Vercel 経由、Apache-2.0 のモデル）で訳し直した。最終的に英字が残るのは RGB、iPad、ノートPC の類だけ
- 文字並べ問題の選択肢（bad、dab、wvVcy など）は訳さず原文のまま
- `scripts/pt_convert.py` で 26,401 問を学習用の形式に変換（count 6,076、spatial 8,041、causal 4,386、temporal 4,186、action 3,712。正解記号は均等）。5,499 問は数を答える open 形式にも変換できる
- 映像は手元では 1MB/s しか出ないので Colab（CPU）で取得してフレームを Drive に置く（`notebooks/colab_pt_frames.ipynb`）
- 次は Colab の A100 で学習前の 4B に 1 問 8 回答えさせ、正解率 2〜8 割の問題だけ残す（`scripts/pt_difficulty.py`）

## 保留している案

友人との議論で出たスコア向上の案（データ増強、規約上問題のない教師モデル、安い審判、学習方法の工夫、出力形式の制約）と Colab の使い分けは `docs/ideas.md` にまとめた。

## 次にやること

1. 学習データの入手元を決める（`docs/ideas.md` の案 1 と 6）。権利の明確な海外の動画 QA データを日本語に訳す案が有力
3. 4B の評価はクラウド GPU で行う
4. 学習データ作成の設計に入る（NHK アーカイブスの規約確認が前提）
