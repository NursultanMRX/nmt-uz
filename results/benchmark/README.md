# Benchmark

Beam 5. Uzbek scored against the original FLORES+ references (1995 alphabet);
chrF is sacrebleu's default (not chrF++), COMET is wmt22-comet-da.

## FLORES+ devtest, first 200 sentences

Same size as the published results in the OvozifyLabs model card (https://huggingface.co/OvozifyLabs/nllb-200-uz-en-v1).

| System | Params | en→uz BLEU | chrF | COMET | uz→en BLEU | chrF | COMET |
|---|---:|---:|---:|---:|---:|---:|---:|
| nmt-uz-76m (ours) | 76.5M | 18.10 | 56.33 | 0.877 | 26.50 | 55.66 | 0.839 |
| NLLB-200-distilled-600M (our run) | 600M | 13.38 | 53.61 | 0.889 | 30.60 | 58.58 | 0.868 |
| NLLB-200-distilled-1.3B (our run) | 1.3B | 14.13 | 54.35 | 0.898 | 34.39 | 61.85 | 0.883 |
| NLLB-200-uz-en-v1 (fine-tuned), reported | 600M | 20.22 | 59.30 | 0.906 | 34.47 | 62.19 | 0.874 |
| Tahrirchi Tilmoch (commercial), reported | n/a | 19.83 | 58.01 | 0.910 | 33.76 | 61.97 | 0.876 |
| NLLB-200-distilled-600M, reported | 600M | 13.07 | 51.73 | 0.881 | 30.28 | 58.28 | 0.856 |

## FLORES+ devtest, all 1012 sentences

| System | Params | en→uz BLEU | chrF | COMET | uz→en BLEU | chrF | COMET |
|---|---:|---:|---:|---:|---:|---:|---:|
| nmt-uz-76m (ours) | 76.5M | 19.89 | 57.53 | 0.883 | 27.70 | 55.94 | 0.832 |
| NLLB-200-distilled-600M (our run) | 600M | 14.90 | 52.80 | 0.879 | 31.33 | 58.42 | 0.858 |
| NLLB-200-distilled-1.3B (our run) | 1.3B | 17.18 | 55.72 | 0.897 | 34.61 | 61.08 | 0.875 |
