"""審判モデルによる採点。公式サンプル (nablas-inc/Japanese_Video-QA/llm_judge_sample.py) を忠実に再現する。

- モデル: gpt-4o、temperature 0、JSON 出力
- システムプロンプトと入力の並びは公式と同一
- 接続先は OpenAI 互換ならどこでもよい（Vercel AI Gateway、LM Studio、Ollama など）。
  base_url と API キーの環境変数名を指定する。公式値と比べるときだけ OpenAI の gpt-4o を使う
- 選択肢は Python のリスト表記 (str(list)) で渡す。open 問題は "なし"
- 結果は JSONL にキャッシュし、同じ入力には API を呼ばない
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

SYSTEM_PROMPT = """
質問と解答、およびモデルの回答が与えられます。解答を踏まえて、モデルの回答の正確さを1から3点で評価してください。
## 評価スケール
1: 誤り・無関係な回答
2: あいまいまたは不十分な回答
3: 正確な回答

## 注意点
- 選択肢が与えられることがあります。その場合は選択肢に沿って回答出来ていることが必須です。できていない場合は1としてください。
- 評価結果は1から3の整数で返してください。

## 出力形式
出力は必ず以下のJSON形式で返してください：
{
    "score": <1から3の整数>
}
"""

DEFAULT_MODEL = "gpt-4o"


def format_choices(choices: list[str]) -> str:
    return str(list(choices)) if choices else "なし"


def user_prompt(question: str, answer: str, prediction: str, choices: str) -> str:
    return f"""質問: {question}
解答: {answer}
選択肢: {choices}

モデルの回答: {prediction}"""


def cache_key(model: str, question: str, answer: str, choices: str, prediction: str) -> str:
    raw = json.dumps([model, question, answer, choices, prediction], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class Judge:
    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        cache_path: Path | str = "results/judge_cache.jsonl",
        max_workers: int = 4,
        max_retries: int = 5,
        base_url: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.api_key_env = api_key_env
        self.cache_path = Path(cache_path)
        self.max_workers = max_workers
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._cache: dict[str, int] = {}
        self._client = None
        self.api_calls = 0
        self._load_cache()

    def _load_cache(self) -> None:
        if not self.cache_path.exists():
            return
        with self.cache_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = int(rec["score"])

    def _append_cache(self, key: str, score: int, meta: dict[str, Any]) -> None:
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            self._cache[key] = score
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "score": score, **meta}, ensure_ascii=False) + "\n")

    def _get_client(self):
        if self._client is None:
            from dotenv import find_dotenv, load_dotenv
            from openai import OpenAI

            load_dotenv(find_dotenv(usecwd=True))
            key = os.getenv(self.api_key_env)
            if not key and not self.base_url:
                raise RuntimeError(f"{self.api_key_env} が設定されていません (.env または環境変数)")
            # ローカルサーバー (LM Studio, Ollama) はキー不要なことが多いので、無ければダミーを入れる
            self._client = OpenAI(api_key=key or "none", base_url=self.base_url)
        return self._client

    @property
    def cache_model_name(self) -> str:
        """キャッシュのキーに使うモデル名。接続先が違えば別物として扱う。"""
        return f"{self.base_url or 'openai'}::{self.model}"

    def _call(self, question: str, answer: str, choices: str, prediction: str) -> int:
        client = self._get_client()
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt(question, answer, prediction, choices)},
                    ],
                    temperature=0,
                    response_format={"type": "json_object"},
                )
                with self._lock:
                    self.api_calls += 1
                score = int(json.loads(resp.choices[0].message.content)["score"])
                if score not in (1, 2, 3):
                    raise ValueError(f"score が 1〜3 でない: {score}")
                return score
            except Exception:
                if attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2
        raise AssertionError("unreachable")

    def score(self, question: str, answer: str, prediction: str, choices: list[str]) -> int:
        choices_str = format_choices(choices)
        key = cache_key(self.cache_model_name, question, answer, choices_str, prediction)
        if key in self._cache:
            return self._cache[key]
        score = self._call(question, answer, choices_str, prediction)
        self._append_cache(key, score, {"model": self.cache_model_name, "question": question, "prediction": prediction})
        return score

    def score_many(self, jobs: list[tuple[str, str, str, list[str]]]) -> list[int]:
        """(question, answer, prediction, choices) のリストを並列に採点する。順序は入力と同じ。"""
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            return list(ex.map(lambda j: self.score(*j), jobs))
