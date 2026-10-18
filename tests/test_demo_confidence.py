from movie_llm.demo.confidence import band, interval_stats, mmss, top_peaks


def test_top_peaks_ignores_flat_floor():
    probs = [0.01] * 20 + [0.3, 0.6, 0.9, 0.95, 0.7, 0.2] + [0.01] * 30 + [0.4, 0.5, 0.3] + [0.01] * 10
    peaks = top_peaks(probs, step=1.0, k=3, suppress_sec=10)
    assert len(peaks) == 2  # 0.01 の平地は候補にならない
    assert peaks[0]["start"] == 21.0 and peaks[0]["end"] == 25.0 and peaks[0]["prob"] == 0.95
    assert peaks[1]["prob"] == 0.5


def test_interval_stats_and_band():
    probs = [0.0] * 10 + [0.9, 0.95, 0.9] + [0.0] * 7
    pmax, pmean = interval_stats(probs, (10.0, 13.0), duration=20.0)
    assert pmax == 0.95 and abs(pmean - 0.9167) < 1e-3
    assert band(0.99, 0.9, True) == "high"
    assert band(0.99, 0.9, False) == "mid"
    assert band(0.99, 0.9, None) == "high"  # 2 段目なしなら確率だけで判定
    assert band(0.95, 0.9, True) == "mid"
    assert band(0.99, 0.3, True) == "low"


def test_mmss():
    assert mmss(0) == "0:00" and mmss(65.4) == "1:05" and mmss(None) == "—"
