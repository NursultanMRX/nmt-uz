"""How much shorter is Uzbek in the 2026 alphabet than in the 1995 one?

    uv run python scripts/token_savings.py data/processed/eval/flores_devtest.jsonl

Counts characters of the Uzbek side in both spellings and tokens with our tokenizer. The
1995 token count is an estimate (same characters per token): our tokenizer was trained on
2026 text, so running it on 1995 text overstates the saving; that number is reported only
to show the gap. Writes results/token_savings.json.
"""

import json
import sys
from pathlib import Path

import sentencepiece as spm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_pipeline.alphabet import to_new_latin, to_old_latin


def count(texts: list[str], sp: spm.SentencePieceProcessor) -> dict:
    new = [to_new_latin(t) for t in texts]
    old = [to_old_latin(t) for t in new]
    chars_new, chars_old = sum(map(len, new)), sum(map(len, old))
    tokens_new = sum(len(sp.encode(t)) for t in new)
    return {
        "sentences": len(texts),
        "chars_1995": chars_old,
        "chars_2026": chars_new,
        "chars_saved_pct": round(100 * (1 - chars_new / chars_old), 2),
        "tokens_2026": tokens_new,
        "tokens_1995_estimate": round(tokens_new * chars_old / chars_new),
        "tokens_1995_with_our_tokenizer": sum(len(sp.encode(t)) for t in old),
    }


if __name__ == "__main__":
    rows = [json.loads(line) for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()]
    sp = spm.SentencePieceProcessor(model_file=str(ROOT / "release" / "spm.model"))
    result = count([r["uz"] for r in rows], sp)
    (ROOT / "results" / "token_savings.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
