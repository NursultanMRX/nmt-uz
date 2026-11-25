from src.latency import measure, percentile


def test_percentile_nearest_rank():
    times = [float(t) for t in range(1, 101)]
    assert percentile(times, 50) == 50.0
    assert percentile(times, 95) == 95.0
    assert percentile([7.0], 95) == 7.0


def test_measure_times_single_sentences_and_a_batch():
    calls = []

    def translate(texts, target, beam):
        calls.append((len(texts), target, beam))
        return ["x"] * len(texts)

    result = measure(translate, ["a", "b", "c", "d"], target="uz", beam=4, warmup=1)
    assert set(result) == {"p50_ms", "p95_ms", "batch_ms_per_sentence", "sentences"}
    assert result["sentences"] == 4
    assert calls[0] == (1, "uz", 4)           # warm-up
    assert calls.count((1, "uz", 4)) == 5     # warm-up + one call per sentence
    assert calls[-1] == (4, "uz", 4)          # the whole set as one batch
