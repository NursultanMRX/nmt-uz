# Benchmark

Beam 5. Uzbek scored against the original FLORES+ references (1995 alphabet);
chrF is sacrebleu's default (not chrF++), COMET is wmt22-comet-da.

## FLORES+ devtest, first 200 sentences

Same size as the published results in the OvozifyLabs model card (https://huggingface.co/OvozifyLabs/nllb-200-uz-en-v1).

| System | Params | en→uz BLEU | chrF | COMET | uz→en BLEU | chrF | COMET |
|---|---:|---:|---:|---:|---:|---:|---:|
| nmt-uz-76m (ours) | 76.5M | 18.10 | 56.33 | 0.877 | 26.50 | 55.66 | 0.839 |
| nmt-uz-76m, ONNX fp32 | 76.5M | 18.13 | 56.30 | 0.877 | 26.50 | 55.67 | 0.839 |
| nmt-uz-76m, ONNX fp16 | 76.5M | 18.13 | 56.30 | 0.877 | 26.50 | 55.67 | 0.839 |
| nmt-uz-76m, ONNX int8 | 76.5M | 17.56 | 56.20 | 0.875 | 26.33 | 55.64 | 0.838 |
| nmt-uz-76m, ONNX int4 | 76.5M | 17.71 | 56.38 | 0.877 | 25.87 | 55.27 | 0.841 |
| NLLB-200-distilled-600M (our run) | 600M | 13.38 | 53.61 | 0.889 | 30.60 | 58.58 | 0.868 |
| NLLB-200-distilled-1.3B (our run) | 1.3B | 14.13 | 54.35 | 0.898 | 34.39 | 61.85 | 0.883 |
| NLLB-200-uz-en-v1 (fine-tuned), reported | 600M | 20.22 | 59.30 | 0.906 | 34.47 | 62.19 | 0.874 |
| Tahrirchi Tilmoch (commercial), reported | n/a | 19.83 | 58.01 | 0.910 | 33.76 | 61.97 | 0.876 |
| NLLB-200-distilled-600M, reported | 600M | 13.07 | 51.73 | 0.881 | 30.28 | 58.28 | 0.856 |

## FLORES+ devtest, all 1012 sentences

| System | Params | en→uz BLEU | chrF | COMET | uz→en BLEU | chrF | COMET |
|---|---:|---:|---:|---:|---:|---:|---:|
| nmt-uz-76m (ours) | 76.5M | 19.89 | 57.53 | 0.883 | 27.70 | 55.94 | 0.832 |
| nmt-uz-76m, ONNX fp32 | 76.5M | 19.88 | 57.54 | 0.883 | 27.70 | 55.93 | 0.832 |
| nmt-uz-76m, ONNX fp16 | 76.5M | 19.88 | 57.54 | 0.883 | 27.70 | 55.93 | 0.832 |
| nmt-uz-76m, ONNX int8 | 76.5M | 19.62 | 57.46 | 0.882 | 27.72 | 55.94 | 0.832 |
| nmt-uz-76m, ONNX int4 | 76.5M | 19.88 | 57.67 | 0.882 | 27.39 | 55.74 | 0.832 |
| NLLB-200-distilled-600M (our run) | 600M | 14.90 | 52.80 | 0.879 | 31.33 | 58.42 | 0.858 |
| NLLB-200-distilled-1.3B (our run) | 1.3B | 17.18 | 55.72 | 0.897 | 34.61 | 61.08 | 0.875 |
