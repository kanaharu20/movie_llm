# Perception Test（学習データ候補）の調査

2026-10-11。DeepMind の Perception Test。映像・注釈ともに CC BY 4.0、コードは Apache-2.0。学習データとして権利面の問題がない。

## 中身（4択ならぬ 3択の動画 QA、train + valid）

| 項目 | 値 |
|---|---|
| 問題数 | 26,532（train 7,392 / valid 19,140） |
| 動画数 | 7,215（train 1,955 / valid 5,260） |
| 選択肢 | 常に 3 つ。正解位置は均等（各 1/3） |
| 動画の長さ | 中央値 23 秒（9〜36 秒）。1080p または 720p、30fps |
| 内容 | 室内で人が物を使って演じる場面（カップ当て、料理の準備、靴紐、物の出し入れなど） |
| area | physics 36%、semantics 30%、abstraction 29%、memory 4% |
| reasoning | descriptive 83%、explanatory 12%、predictive 3%、counterfactual 2% |

## このプロジェクトに効く部分

| 部分集合 | 問題数 | 備考 |
|---|---|---|
| counting 系（object / action / event counting） | 6,076 | うち 5,499 は選択肢が全部数字。「数を答える open 形式」にも変換できる |
| sequencing（順序） | 3,772 | 時間の前後関係。temporal の学習に使える |
| 両方 | 468 | |

ベンチマークで伸びなかった count と、フレーム数に反応した temporal の両方を狙える。

## 翻訳の規模

- 問いの異なりは **132 種類**だけ（上位 20 種で全体の 43%）。テンプレート生成なので、問いは人手で訳せる
- 選択肢の異なりは 6,369 種類（英語の短い句。例: kitchen, preparing tea）。これはローカルの Qwen3-8B（LM Studio）で訳す。数字だけの選択肢 14 種は訳す必要なし
- 規約の問題なし、費用なし

## 注意点

- 映像は 9〜36 秒と短い。ベンチマークの medium（4〜20 分）の弱さはこのデータでは直せない。長い動画は別の入手元が必要
- 室内の演技映像なので、日本の風景や行事の読み取りは学べない。日本の映像の少量追加は別途
- 「Where is the person?」「Is the camera moving or static?」の 2 種で 4,787 問（18%）。易しすぎる問いは問題の選別（学習前モデルの正解率 2〜8 割）で自然に落ちる
- 3択なので「常に同じ記号」で 33% 取れる。学習データでは正解位置を均等に保つ（元データは均等）
- valid の映像は 70GB と重い。まず train（26.5GB、7,392 問）で進め、足りなければ valid を足す

## 変換の設計

1. `mc_all.json`（train + valid を結合、video_id と split を付与）→ 本リポジトリの項目（question, choices, answer, type, category, video_id）に揃える
2. category の対応: counting 系 → count、sequencing → temporal、spatial relations → spatial、action/task → action、explanatory/counterfactual → causal
3. 問いは 132 種の対訳表（人手）、選択肢は Qwen3-8B で訳して対訳表にキャッシュ
4. counting で選択肢が全部数字の問題は、3択版と「数字だけで答える」open 版の両方を作る（open 版は正誤判定が数字一致で確実）
5. 映像から 32 フレームを抜いてキャッシュ（既存の `jvqa_cache_frames.py` を流用）
6. 学習前の 4B に 8 回答えさせて正解率 2〜8 割の問題だけ残す（Colab A100）
