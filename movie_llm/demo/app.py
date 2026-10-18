"""シーン検索のデモ（Gradio）。

    python -m movie_llm.demo.app --host 0.0.0.0 --port 7860 [--adapter models/timeloc-lora] [--share]

- 左: 動画（samples/ から選ぶか、アップロード）とその再生、問い
- 右: 振り分けの帯、時刻、根拠、見つけた場面の切り抜き再生、候補 3 つのサムネイル（押すとその候補を再生）、追加の質問
GPU 24GB 以上を想定（VideoJev-4B + Qwen3-VL-4B）。
"""

import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

from movie_llm.demo.confidence import mmss
from movie_llm.demo.pipeline import SceneSearch
from movie_llm.infer.video import clip_video, sample_frames_window

SAMPLES = Path(os.environ.get("MOVIE_LLM_SAMPLES", "samples"))
CLIP_PAD = 5.0  # 切り抜き再生で区間の前後に足す秒数
CLIP_DIR = Path(tempfile.mkdtemp(prefix="movie_llm_clips_"))


def thumbnail(video_path: str, sec: float) -> np.ndarray:
    arr, _ = sample_frames_window(video_path, max(0.0, sec - 0.5), sec + 0.5, 1, 480)
    return arr[0]


def make_clip(video_path: str, start: float, end: float) -> str | None:
    """区間の前後 CLIP_PAD 秒を小さな mp4 に切り出し、そのパスを返す（同じ区間は使い回す）。"""
    s, e = max(0.0, start - CLIP_PAD), end + CLIP_PAD
    key = hashlib.md5(f"{Path(video_path).resolve()}|{s:.1f}|{e:.1f}".encode()).hexdigest()[:16]
    out = CLIP_DIR / f"{key}.mp4"
    if out.exists():
        return str(out)
    try:
        return clip_video(video_path, s, e, str(out))
    except Exception as exc:  # noqa: BLE001
        print(f"切り抜きに失敗: {exc}", flush=True)
        return None


def build(ss: SceneSearch, samples_dir: Path | None = None):
    import gradio as gr

    sdir = Path(samples_dir) if samples_dir else SAMPLES
    sample_files = sorted(str(p) for p in list(sdir.glob("*.mp4")) + list(sdir.glob("*.webm"))) if sdir.exists() else []
    print(f"サンプル映像: {sdir.resolve()} に {len(sample_files)} 本", flush=True)

    def run(sample: str | None, upload: str | None, query: str, top_k: int):
        video = upload or sample
        if not video or not query.strip():
            return "動画と問いを入れてください。", None, [], None
        r = ss.search(video, query.strip(), top_k=int(top_k))
        head = {
            "high": "### ✅ 見つかりました（自動で返せる水準）",
            "mid": "### 🔎 候補を確認してください",
            "low": "### ⚠️ この素材には無い可能性があります",
        }[r["band"]]
        md = [head, f"**時刻: {r['time_str']}**（候補区間 {r['interval_str']}）", ""]
        if r["rationale"]:
            md.append(f"根拠: {r['rationale']}")
        md.append("")
        md.append(f"<small>信頼度: 最大確率 {r['pmax']:.2f} / 区間内の平均 {r['pmean']:.2f} / 2 段目との一致 {'あり' if r['agree'] else 'なし'} / {r['seconds']} 秒</small>")
        clip = make_clip(video, r["interval"][0], r["interval"][1])
        gallery = []
        for k, c in enumerate(r["candidates"], 1):
            try:
                gallery.append((thumbnail(video, (c["start"] + c["end"]) / 2), f"候補 {k}: {c['label']}"))
            except Exception:  # noqa: BLE001
                pass
        state = {"video": video, "start": r["interval"][0], "end": r["interval"][1], "candidates": r["candidates"]}
        return "\n".join(md), clip, gallery, state

    def play_candidate(state: dict | None, evt: gr.SelectData):
        if not state or evt.index is None:
            return None
        cands = state.get("candidates") or []
        i = int(evt.index)
        if i < 0 or i >= len(cands):
            return None
        return make_clip(state["video"], cands[i]["start"], cands[i]["end"])

    def ask(state: dict | None, question: str):
        if not state or not question.strip():
            return "先に場面を探してください。"
        a = ss.ask(state["video"], state["start"], state["end"], question.strip())
        return f"{mmss(state['start'])}〜{mmss(state['end'])} の場面について: {a}"

    with gr.Blocks(title="movie_llm シーン検索") as demo:
        gr.Markdown("# movie_llm シーン検索\n外に出せない映像を、手元の小さなモデルで日本語検索する。1 段目 VideoJev-4B が候補区間を出し、2 段目の学習済み Qwen3-VL-4B が確かめて根拠を書く。")
        with gr.Row():
            with gr.Column(scale=1):
                sample = gr.Dropdown(choices=sample_files, label=f"サンプル映像（{sdir}、{len(sample_files)} 本）", value=sample_files[0] if sample_files else None, allow_custom_value=True)
                player = gr.Video(label="選んだ映像", value=sample_files[0] if sample_files else None, interactive=False)
                upload = gr.Video(label="または動画をアップロード", sources=["upload"])
                query = gr.Textbox(label="探したい場面（日本語）", placeholder="例: 赤い鳥居の前で参拝している場面")
                top_k = gr.Slider(1, 5, value=3, step=1, label="候補の数")
                btn = gr.Button("探す", variant="primary")
            with gr.Column(scale=2):
                out_md = gr.Markdown()
                clip_out = gr.Video(label=f"見つけた場面（候補区間の前後 {CLIP_PAD:.0f} 秒）", interactive=False)
                gallery = gr.Gallery(label="候補（押すとその区間を再生）", columns=3, height=240)
                state = gr.State()
                question = gr.Textbox(label="見つけた場面への質問", placeholder="例: 何人映っていますか")
                ask_btn = gr.Button("質問する")
                answer = gr.Markdown()
        sample.change(lambda p: p or None, [sample], [player])
        btn.click(run, [sample, upload, query, top_k], [out_md, clip_out, gallery, state])
        gallery.select(play_candidate, [state], [clip_out])
        ask_btn.click(ask, [state, question], [answer])
    return demo


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--adapter", default=None, help="学習済み LoRA のフォルダ。無ければ環境変数 MOVIE_LLM_LORA")
    ap.add_argument("--max-frames", type=int, default=1024, help="VideoJev に入れる最大フレーム数（24GB の GPU なら 1024、40GB なら 2048）")
    ap.add_argument("--samples", default=None, help="サンプル映像のフォルダ（既定: 環境変数 MOVIE_LLM_SAMPLES か samples/）")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args()
    ss = SceneSearch(adapter=args.adapter, stage1_kwargs={"max_frames": args.max_frames})
    sdir = Path(args.samples) if args.samples else SAMPLES
    allowed = [str(p.resolve()) for p in (sdir, CLIP_DIR) if p.exists()]
    build(ss, args.samples).launch(server_name=args.host, server_port=args.port, share=args.share, allowed_paths=allowed)


if __name__ == "__main__":
    main()
