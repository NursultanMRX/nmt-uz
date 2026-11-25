# Data sources

Audited 2026-11-23. Row counts are before cleaning; the build writes the
after-cleaning numbers to `data/processed/stats.json`.

## Training (parallel en-uz)

| Name | Where | Pairs | License | Domain / notes |
|---|---|---:|---|---|
| NLLB | OPUS `NLLB/v1` moses zip | 30.98M | ODC-By (mined bitext, see NLLB card) | Web-mined with LASER3; largest and noisiest. The HF `allenai/nllb` repo is a loading script that recent `datasets` cannot run, so we pull the OPUS export. |
| wikimedia | OPUS `wikimedia/v20260327` | 1.37M | CC-BY-SA | Wikipedia content translation. Supersedes the `davron04/wikimedia_v20230407_en_uz` mirror. |
| Tanzil | OPUS `Tanzil/v1` | 140.7K | Tanzil terms (non-commercial redistribution) | Religious text, archaic register. Loss weight 0.5. |
| KDE4 / GNOME / Ubuntu / translatewiki / tldr-pages | OPUS | 37.8K / 33.3K / small / 10.0K / 1.4K | GPL / LGPL / CC | Software UI strings and docs, many fragments. Loss weight 0.5. |
| TED2020 | OPUS `TED2020/v1` | 6.3K | CC BY-NC-ND 4.0 | Human-translated talks. Cooldown set. |
| QED | OPUS `QED/v2.0a` | 3.8K | see OPUS | Educational video subtitles. Cooldown set. |
| OPUS-100 en-uz | HF `Helsinki-NLP/opus-100` | 173.2K | unknown (OPUS mix) | Overlaps heavily with the above, dedup removes repeats. |
| mt-dataset-en-uz | HF `ShakhzoDavronov/mt-dataset-en-uz` | 956.7K + 50.4K | not stated | Provenance not documented; columns are `{"text": ...}` structs. |
| mt-dataset-en-uz-cleaned | HF `ShakhzoDavronov/mt-dataset-en-uz-cleaned` | 223.2K + 11.7K | not stated | Filtered subset of the above. |
| English-Uzbek-Translation-1 | HF `ML-Jonibek/...` | 25.4K | not stated | Curated, cooldown candidate. |
| English-Uzbek-Translation | HF `Jonibek21/...` | 48.0K | Apache-2.0 | Same size as the `pythoninverter` copy, almost certainly identical. |
| English-Uzbek-Translation | HF `pythoninverter/...` | 48.0K | Apache-2.0 | See above. |

Not used for training: OPUS `Tatoeba` (514 pairs). The Tatoeba challenge test
sets are built from it, so it stays out to keep that benchmark usable.

## Evaluation

| Name | Where | Size | License |
|---|---|---:|---|
| FLORES+ `uzn_Latn` / `eng_Latn` | HF `openlanguagedata/flores_plus` (gated) | dev 997, devtest 1012 | CC BY-SA 4.0 |

FLORES+ Uzbek is converted to the 2026 alphabet like everything else. Every
FLORES+ sentence (both sides) is used as a blocklist: training pairs that match
it after normalization are dropped.

## Licensing note

The code is released under PolyForm Noncommercial 1.0.0 and the weights under
CC BY-NC 4.0. The data keeps its own licenses; some sources above are
non-commercial (TED2020, Tanzil) or have no stated license, which is one more
reason the model is non-commercial only.
