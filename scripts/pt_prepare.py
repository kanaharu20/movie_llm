"""Perception Test の 3択 QA 注釈（小さい zip）を取得し、train + valid を結合した mc_all.json を作る。

映像は含まない（映像は Colab で別途取得してフレームにする）。Colab やクリーンな環境での再生成用。

    uv run python scripts/pt_prepare.py
"""

from __future__ import annotations

import io
import json
import urllib.request
import zipfile
from pathlib import Path

DATA = Path("data/perception_test")
BASE = "https://storage.googleapis.com/dm-perception-test/zip_data/"


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    qs = []
    for split in ("train", "valid"):
        name = f"mc_question_{split}_annotations.zip"
        print("取得", name)
        raw = urllib.request.urlopen(BASE + name, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            d = json.loads(z.read(f"mc_question_{split}.json"))
        for vid, v in d.items():
            for q in v["mc_question"]:
                qs.append({**q, "video_id": vid, "split": split})
    (DATA / "mc_all.json").write_text(json.dumps(qs, ensure_ascii=False))
    print(f"mc_all.json: {len(qs)} 問 / {len({q['video_id'] for q in qs})} 本")


if __name__ == "__main__":
    main()
