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

## 4B について

重み約 9GB。bfloat16 で 16GB の機械に載るが、フレーム数と解像度を抑える必要がある。確認中。
