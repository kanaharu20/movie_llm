from movie_llm.eval.rules import extract_choice, extract_number, extract_yes_no, gold_letter, score_numeric, score_rule, split_choices

CH = ["A) 7種類", "B) 8種類", "C) 9種類", "D) 10種類"]


def test_split_choices():
    assert split_choices(CH) == {"A": "7種類", "B": "8種類", "C": "9種類", "D": "10種類"}
    assert split_choices(["Ａ）全角", "B. ドット", "C：コロン"]) == {"A": "全角", "B": "ドット", "C": "コロン"}


def test_gold_letter():
    assert gold_letter("C", CH) == "C"
    assert gold_letter("9種類", CH) == "C"  # 正解が本文で書かれている例 (idx 698)
    assert gold_letter("11種類", CH) is None


def test_extract_choice_leading():
    assert extract_choice("C", CH) == "C"
    assert extract_choice("C) 9種類", CH) == "C"
    assert extract_choice("(B)", CH) == "B"
    assert extract_choice("**D**", CH) == "D"
    assert extract_choice("Ｃ", CH) == "C"
    assert extract_choice("  A. 7種類です", CH) == "A"
    assert extract_choice("Dの10種類です", CH) == "D"


def test_extract_choice_tagged():
    assert extract_choice("映像を見ると、答え: B です", CH) == "B"
    assert extract_choice("正解はCです", CH) == "C"
    assert extract_choice("Answer: (D)", CH) == "D"


def test_extract_choice_paren_and_body():
    assert extract_choice("見た限りでは (B) が妥当", CH) == "B"
    assert extract_choice("トッピングは9種類ありました", CH) == "C"
    assert extract_choice("Based on the video, 8種類", CH) == "B"  # 文頭の B は英単語なので無視
    assert extract_choice("10種類", CH) == "D"  # "1種類" が無いので最長一致


def test_extract_choice_none():
    assert extract_choice("", CH) is None
    assert extract_choice("わかりません", CH) is None
    assert extract_choice("AかBのどちらか", CH) is None
    assert extract_choice("(A) または (B)", CH) is None


def test_extract_yes_no():
    assert extract_yes_no("はい") == "はい"
    assert extract_yes_no("いいえ、映っていません") == "いいえ"
    assert extract_yes_no("映像では確認できます。はい。") == "はい"
    assert extract_yes_no("Yes, the man is bowing.") == "はい"
    assert extract_yes_no("No.") == "いいえ"
    assert extract_yes_no("Nothing is shown") is None
    assert extract_yes_no("わかりません") is None
    assert extract_yes_no("") is None


def test_score_rule():
    mc = {"type": "multi_choice", "answer": "C", "choices": CH}
    assert score_rule(mc, "C) 9種類") == 3
    assert score_rule(mc, "B") == 1
    assert score_rule(mc, "") == 1
    broken = {"type": "multi_choice", "answer": "9種類", "choices": CH}
    assert score_rule(broken, "C") == 3
    yn = {"type": "yes_no", "answer": "はい", "choices": []}
    assert score_rule(yn, "はい、見えます") == 3
    assert score_rule(yn, "いいえ") == 1
    yn_broken = {"type": "yes_no", "answer": "一生懸命描いたから", "choices": []}
    assert score_rule(yn_broken, "はい") is None
    op = {"type": "open", "answer": "キャベツ", "choices": []}
    assert score_rule(op, "キャベツ") is None


def test_extract_number():
    assert extract_number("3") == 3
    assert extract_number("３つ") == 3
    assert extract_number("答えは 4 です") == 4
    assert extract_number("三人") == 3
    assert extract_number("十二") == 12
    assert extract_number("二十") == 20
    assert extract_number("0") == 0
    assert extract_number("1.5回転") == 1
    assert extract_number("わかりません") is None
    assert extract_number("") is None


def test_score_numeric():
    assert score_numeric("3", "3つ") == (1.0, True, True)
    assert score_numeric("3", "4") == (0.5, False, True)
    assert score_numeric("3", "2回") == (0.5, False, True)
    assert score_numeric("3", "5") == (0.0, False, True)
    assert score_numeric("3", "たくさん") == (0.0, False, False)
    assert score_numeric("3", "4", partial=0.0) == (0.0, False, True)
    assert score_rule({"type": "numeric", "answer": "3"}, "3個") == 3
    assert score_rule({"type": "numeric", "answer": "3"}, "4個") == 1
