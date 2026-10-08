# 手元での推論メモ（MacBook Pro 14 M2 Pro, 16GB）

## 構成

- torch 2.14 (MPS), transformers 5.19, PyAV 19
- Qwen3-VL-2B-Instruct を bfloat16 で MPS に載せる。重み 4GB
- 動画は PyAV で等間隔に 16 フレーム抜き、長辺 640px 以下に縮小してから processor に渡す
- 1フレームの上限画素数は 256×28×28（video_processor.size.longest_edge を num_frames 倍で設定）

## 実測（2B, 16フレーム）

| 項目 | 値 |
|---|---|
| モデル読み込み | 約5秒 |
| 動画トークン数 | 640（320×240 の合成動画） |
| 1問の推論 | 約2秒（合成動画）、約6秒（ベンチマーク動画 360×450, 36秒） |

合成動画（赤い四角が左→右へ動き、青い丸が2回出る）への回答は3問とも正しかった。ベンチマークの idx 2（粘土のオブジェをオーブンに入れた回数）も正解。

## つまずいた点

- transformers 5.x は動画パスを渡すと torchcodec でデコードしようとし、`load_video_backend` 引数は無視される。PyAV で自前にフレームを抜き、`videos=[配列]`, `video_metadata=[VideoMetadata]`, `do_sample_frames=False` で渡すと動く
- `Qwen3VLVideoProcessor` は torchvision を要求する（デコードには使わないがリサイズに使う）
- `huggingface-hub>=2` を固定すると transformers と衝突する
- YouTube は映像と音声が別配信で、結合には ffmpeg が要る。映像のみ (`bv*`) を取れば ffmpeg 不要

## ベンチマーク20問での実測（2B, 16フレーム, 短い選択式）

`uv run python scripts/jvqa_predict.py --name qwen3vl-2b_f16 --type multi_choice --duration short --limit 20`

| 項目 | 値 |
|---|---|
| 推論できた問題 | 17 / 20（3本は YouTube から消えていた） |
| 1問あたり | 平均 4.1 秒、最大 6.7 秒（動画取得の時間は含まない） |
| 動画トークン数 | 832〜1496 |
| 正解率（規則採点） | 70.6%（12/17） |
| category 別 | spatial 7/8、temporal 2/3、count 2/5、action 1/1 |

17問では統計的に何も言えないが、count が弱いという想定と矛盾しない。公式の 4B 全体値（55%）より高いのは、短い選択式だけを見ているため。

## 4B について

重み 8.3GB (bfloat16)。読み込みは 15.7 秒で終わるが、その後の推論が進まなかった。原因はメモリ不足で、この機械では他のプロセスが常時 10GB 以上使っており、スワップが 28GB 中 27GB 以上使われていた。4B の手元実行は、他のアプリを閉じた状態で再試行するか、クラウド GPU で行う。手元の開発・デバッグは 2B で進める。

## 動画の入手性

20本中3本（15%）が「Video unavailable」「not available」「403」で取れなかった。全428本の取得状況は `scripts/jvqa_fetch_videos.py` が `data/jvqa/video_status.jsonl` に記録する。取れない動画の問題は評価から外れるので、公式値との比較ではその分を明記する必要がある。
