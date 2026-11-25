from src.backtranslate import synthetic_store
from src.training.data import EN2UZ, UZ2EN


class _CharTokenizer:
    def encode(self, texts):
        return [[ord(c) % 1000 for c in t] for t in texts]


def test_uz_mono_trains_only_en2uz_and_drops_broken_outputs():
    real = ["Bugun havo yaxshi", "Kitob öqiyman", "Salom"]
    synth = ["The weather is nice today", "", "Salom"]  # empty and copied outputs
    store, dropped = synthetic_store(real, synth, "uz", _CharTokenizer())
    assert len(store) == 1 and dropped == 2
    assert store.only_dir.tolist() == [EN2UZ]
    assert store.side(0, "uz").tolist() == _CharTokenizer().encode([real[0]])[0]


def test_en_mono_trains_only_uz2en():
    store, _ = synthetic_store(["I read books"], ["Men kitob öqiyman"], "en", _CharTokenizer())
    assert store.only_dir.tolist() == [UZ2EN]
    assert store.side(0, "en").tolist() == _CharTokenizer().encode(["I read books"])[0]
