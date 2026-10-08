"""規則ベースの採点（multi_choice / yes_no 用）。

審判モデルを使わずに正誤を決める。GRPO の報酬にもこのまま使う。
採点結果は公式と同じ 1〜3 点の尺度に合わせ、正解なら 3、不正解なら 1 を返す。
open 問題や、正解ラベルが壊れている問題には None を返す（審判モデルに回す）。
"""

from __future__ import annotations

import re
import unicodedata

LETTERS = "ABCD"
YES, NO = "はい", "いいえ"

# 'A) 江戸の吉原' / 'B. 大阪' / 'C：東京' のような選択肢表記を (記号, 本文) に分ける
_CHOICE = re.compile(r"^([A-D])\s*[\)\.．:：、）]\s*(.*)$")
# 文頭の記号。前に空白や '**' などの記号があってもよい。直後に英数字が続く場合（"Based" など）は除外
_LEAD = re.compile(r"^[^A-Za-z0-9]*([A-D])(?![A-Za-z0-9])")
# 「答え: C」「正解はB」「Answer: (D)」
_TAG = re.compile(r"(?:答え|回答|正解|解答|選択肢|[Aa]nswer)\s*(?:は|:|：|→)?\s*[\(（]?([A-D])(?![A-Za-z0-9])")
# 本文中の「(C)」「C)」
_PAREN = re.compile(r"[\(（]?([A-D])[\)）]")
# 前後に英数字が無い単独の記号（"AかB" のような二択の検出に使う）
_STANDALONE = re.compile(r"(?<![A-Za-z0-9])([A-D])(?![A-Za-z0-9])")


def normalize(text: str | None) -> str:
    """全角英数を半角に揃え、前後の空白を落とす。"""
    return unicodedata.normalize("NFKC", text or "").strip()


def split_choices(choices: list[str]) -> dict[str, str]:
    """['A) 1回', 'B) 2回'] -> {'A': '1回', 'B': '2回'}"""
    out: dict[str, str] = {}
    for c in choices:
        m = _CHOICE.match(normalize(c))
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def gold_letter(answer: str, choices: list[str]) -> str | None:
    """正解ラベルを A〜D に揃える。正解が選択肢本文で書かれている場合も本文一致で記号に直す。"""
    a = normalize(answer)
    if a in LETTERS:
        return a
    for letter, body in split_choices(choices).items():
        if a == body:
            return letter
    return None


def extract_choice(prediction: str | None, choices: list[str]) -> str | None:
    """モデル出力から選んだ記号 (A〜D) を取り出す。判断できなければ None。

    優先順位: 文頭の記号（本文に他の記号が単独で出てこない場合） > 「答え: X」の形
              > 本文中の「(X)」が1種類だけ > 選択肢本文が1つだけ含まれる
    """
    p = normalize(prediction)
    if not p:
        return None
    m = _LEAD.match(p)
    if m and len({x.group(1) for x in _STANDALONE.finditer(p)}) == 1:
        return m.group(1)
    m = _TAG.search(p)
    if m:
        return m.group(1)
    found = {m.group(1) for m in _PAREN.finditer(p)}
    if len(found) == 1:
        return found.pop()
    bodies = split_choices(choices)
    hits = [letter for letter, body in bodies.items() if len(body) >= 2 and body in p]
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        # 本文が入れ子になっている場合（"1回" と "11回" など）は最長一致を採る
        longest = max(hits, key=lambda l: len(bodies[l]))
        if sum(1 for l in hits if len(bodies[l]) == len(bodies[longest])) == 1:
            return longest
    return None


def extract_yes_no(prediction: str | None) -> str | None:
    """モデル出力を はい / いいえ に正規化する。先に出てきた方を採る。判断できなければ None。"""
    p = normalize(prediction)
    if not p:
        return None
    low = p.lower()
    candidates: list[tuple[int, str]] = []
    for token, label in ((NO, NO), (YES, YES), ("yes", YES), ("no", NO)):
        pos = low.find(token)
        if pos < 0:
            continue
        if token in ("yes", "no"):
            # 英単語は前後が英字でない場合だけ数える（"not", "nothing" を除く）
            before = low[pos - 1] if pos > 0 else " "
            after = low[pos + len(token)] if pos + len(token) < len(low) else " "
            if before.isalpha() or after.isalpha():
                continue
        candidates.append((pos, label))
    if not candidates:
        return None
    return min(candidates)[1]


def score_rule(item: dict, prediction: str | None) -> int | None:
    """問題と予測から 1 / 3 / None を返す。

    - multi_choice: 記号が一致すれば 3、そうでなければ 1
    - yes_no: はい/いいえ が一致すれば 3、そうでなければ 1
    - open または正解ラベルが壊れている問題: None
    """
    t = item["type"]
    if t == "multi_choice":
        gold = gold_letter(item["answer"], item["choices"])
        if gold is None:
            return None
        return 3 if extract_choice(prediction, item["choices"]) == gold else 1
    if t == "yes_no":
        gold = normalize(item["answer"])
        if gold not in (YES, NO):
            return None
        return 3 if extract_yes_no(prediction) == gold else 1
    return None
