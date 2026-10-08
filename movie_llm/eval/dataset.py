"""Japanese Video-QA の読み込み。

各問題には parquet の行番号 `idx` を付ける（配布データに ID が無いため）。
予測ファイルはこの `idx` で問題と対応づける。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "nablasinc/Japanese_Video-QA"
PARQUET = "data/test-00000-of-00001.parquet"
DEFAULT_DIR = Path("data/jvqa")


def load_dataframe(local_dir: Path | str = DEFAULT_DIR) -> pd.DataFrame:
    path = hf_hub_download(REPO, PARQUET, repo_type="dataset", local_dir=str(local_dir))
    df = pd.read_parquet(path)
    df.insert(0, "idx", range(len(df)))
    return df


def load_items(local_dir: Path | str = DEFAULT_DIR) -> list[dict[str, Any]]:
    df = load_dataframe(local_dir)
    items = df.to_dict(orient="records")
    for it in items:
        it["choices"] = [str(c) for c in it["choices"]]
    return items
