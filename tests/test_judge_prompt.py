"""公式サンプルと同じ文字列を組み立てているかの確認（API は呼ばない）。"""

from movie_llm.eval.judge import SYSTEM_PROMPT, format_choices, user_prompt


def test_format_choices_matches_official_style():
    # 公式は str(list) をそのまま渡している
    assert format_choices(["A) 1回", "B) 2回"]) == "['A) 1回', 'B) 2回']"
    assert format_choices([]) == "なし"


def test_user_prompt_layout():
    p = user_prompt("Q?", "C", "C) 蓋をして", "['A) x']")
    assert p == "質問: Q?\n解答: C\n選択肢: ['A) x']\n\nモデルの回答: C) 蓋をして"


def test_system_prompt_key_lines():
    assert "1から3点で評価" in SYSTEM_PROMPT
    assert '"score": <1から3の整数>' in SYSTEM_PROMPT
