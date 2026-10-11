"""Perception Test の問いと選択肢を日本語に訳す（ローカルの Qwen3-8B、LM Studio の OpenAI 互換 API）。

- 訳は data/perception_test/translations.json に {英語: 日本語} でキャッシュする（再実行は未訳分だけ）
- 文字並べの問題（"order of the letters" など）に出てくる選択肢は訳さない
- 数字だけの選択肢は訳さない

    uv run python scripts/pt_translate.py --what questions
    uv run python scripts/pt_translate.py --what options --batch 20
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import time
from pathlib import Path

from openai import OpenAI

print = functools.partial(print, flush=True)
DATA = Path("data/perception_test")
CACHE = DATA / "translations.json"
LETTER_Q = re.compile(r"\bletters?\b|\bword\b", re.I)

def is_letter_sequence(option: str) -> bool:
    """文字並べ問題の選択肢（'bad', 'dab', 'b, a, d', 'gt'）かどうか。英語の語句は False。"""
    toks = [t.strip() for t in option.split(",")]
    return all(re.fullmatch(r"[a-z]{1,5}", t) for t in toks) and option.lower() not in {"yes", "no", "none"}

SYSTEM = """/no_think
あなたは映像理解ベンチマークの翻訳者です。英語の短い文や句を自然な日本語に訳します。
- 動画について問う文です。「the person」は「人物」、「the camera」は「カメラ」と訳す
- 物の名前はカタカナや一般的な日本語にする（cup→カップ、mug→マグカップ、cell-phone→携帯電話）
- 「I don't know」は「わからない」
- 「static or shaking」のような選択肢は「静止または揺れている」のように意味を保つ
- 余計な説明や注釈は付けず、訳だけを返す
出力は必ず JSON で、{"translations": ["訳1", "訳2", ...]} の形。入力と同じ順序・同じ個数で返す。"""


def load_cache() -> dict[str, str]:
    return json.loads(CACHE.read_text()) if CACHE.exists() else {}


def save_cache(c: dict[str, str]) -> None:
    CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=0))


def translate_batch(client: OpenAI, model: str, items: list[str], retries: int = 3) -> list[str]:
    user = "次の英語を日本語に訳してください。\n" + json.dumps({"inputs": items}, ensure_ascii=False)
    for attempt in range(retries):
        r = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "translations",
                    "schema": {
                        "type": "object",
                        "properties": {"translations": {"type": "array", "items": {"type": "string"}}},
                        "required": ["translations"],
                    },
                },
            },
            max_tokens=4000,
        )
        txt = r.choices[0].message.content or ""
        txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
        try:
            out = json.loads(txt)["translations"]
            if len(out) == len(items) and all(isinstance(x, str) for x in out):
                return out
        except Exception:
            pass
        time.sleep(1)
    raise RuntimeError(f"翻訳に失敗: {items[:2]}...")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", choices=["questions", "options"], required=True)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--base-url", default="http://localhost:1234/v1")
    ap.add_argument("--model", default="qwen/qwen3-8b")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    qs = json.load(open(DATA / "mc_all.json"))
    if args.what == "questions":
        targets = sorted({q["question"] for q in qs})
    else:
        # 文字並べの問いに出てくる文字列そのもの（bad, dab など）は訳さない。Yes/No などの語は訳す
        letter_opts = {o for q in qs if LETTER_Q.search(q["question"]) for o in q["options"] if is_letter_sequence(o)}
        targets = sorted({o for q in qs for o in q["options"] if o not in letter_opts and not re.fullmatch(r"\d+", o)})
        print(f"文字並べの選択肢 {len(letter_opts)} 種は訳さない")
    cache = load_cache()
    todo = [t for t in targets if t not in cache]
    if args.limit:
        todo = todo[: args.limit]
    print(f"対象 {len(targets)}、済み {len(targets) - len(todo)}、今回 {len(todo)}")
    client = OpenAI(api_key="none", base_url=args.base_url)
    t0 = time.time()
    for i in range(0, len(todo), args.batch):
        batch = todo[i : i + args.batch]
        out = translate_batch(client, args.model, batch)
        for en, ja in zip(batch, out):
            cache[en] = ja.strip()
        save_cache(cache)
        print(f"  [{i + len(batch)}/{len(todo)}] {time.time() - t0:.0f}s  例: {batch[0]!r} -> {cache[batch[0]]!r}")
    print(f"合計 {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
