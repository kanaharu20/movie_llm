from movie_llm.eval.timeloc import extract_time, parse_time, score_time, tolerance


def test_parse_time_formats():
    assert parse_time("01:10:000") == 70.0          # 分:秒:ミリ秒
    assert parse_time("00:08:17.000") == 497.0      # 時:分:秒.ミリ秒
    assert parse_time("05:13.000") == 313.0         # 分:秒.ミリ秒
    assert parse_time("00:03:45") == 225.0          # 時:分:秒
    assert parse_time("01:42") == 102.0             # 分:秒
    assert parse_time("00:2:38") == 158.0
    assert parse_time("") is None


def test_parse_time_disambiguate_with_duration():
    # 74 秒の動画に '01:03:00.000' → 時:分:秒 では 3780 秒になるので 分:秒 として読む
    assert parse_time("01:03:00.000", video_duration=74) == 63.0
    assert parse_time("08:18:00", video_duration=781) == 498.0
    assert parse_time("00:08:17.000", video_duration=600) == 497.0


def test_extract_time():
    assert extract_time("根拠: 太鼓が映る\n答え: 01:42") == 102.0
    assert extract_time("答え: 1:42〜1:50") == 102.0
    assert extract_time("1分42秒ごろです") == 102.0
    assert extract_time("約 35 秒") == 35.0
    assert extract_time("答え: 00:01:05") == 65.0
    assert extract_time("分かりません") is None


def test_score_time():
    tol = tolerance(120, 32)  # 1.875 → 最低 5 秒
    assert tol == 5.0
    assert score_time(100.0, 98.0, 104.0, tol) == (1.0, True, 0.0)
    assert score_time(108.0, 98.0, 104.0, tol) == (1.0, True, 4.0)
    s, hit, d = score_time(124.0, 98.0, 104.0, tol)  # 20 秒外れ → 許容 5 を引いて 15 → 1 - 15/30
    assert not hit and abs(s - 0.5) < 1e-9 and d == 20.0
    assert score_time(200.0, 98.0, 104.0, tol)[0] == 0.0
    assert score_time(None, 98.0, 104.0, tol) == (0.0, False, None)
    assert tolerance(1200, 32) == 18.75
