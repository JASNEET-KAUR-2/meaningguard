# Ablations (test split, seed 42)

_Auto-generated. TBD = not run yet._

## A1: binary head only vs multi-task

| Model | Acc | Macro-F1 | Recall:negation | Recall:quantity | Recall:deadline | Recall:intent | Recall:entity | FPR |
|---|---|---|---|---|---|---|---|---|
| Main: MuRIL multi-task (A1 reference) | 99.2% | 0.991 | 98.6% | 99.1% | 100.0% | 98.6% | 100.0% | 1.4% |
| A1: binary head only | 97.1% | 0.971 | 96.4% | 96.4% | 100.0% | 96.4% | 96.4% | 2.1% |

A1 has no dimension head, so its Recall:<dim> columns are detection recall only. Dim-attribution is not defined for it.

## A3: base-encoder swap

| Model | Acc | Macro-F1 | Recall:negation | Recall:quantity | Recall:deadline | Recall:intent | Recall:entity | FPR |
|---|---|---|---|---|---|---|---|---|
| Main: MuRIL (configs/muril_base.yaml) | 99.2% | 0.991 | 98.6% | 99.1% | 100.0% | 98.6% | 100.0% | 1.4% |
| A3: google/muril-base-cased | 99.2% | 0.991 | 98.6% | 99.1% | 100.0% | 98.6% | 100.0% | 1.4% |
| A3: ai4bharat/indic-bert | 97.5% | 0.974 | 96.4% | 96.4% | 100.0% | 100.0% | 96.4% | 2.1% |
| A3: xlm-roberta-base | 96.4% | 0.963 | 96.4% | 96.4% | 96.4% | 96.4% | 96.4% | 2.9% |

Note: `a3_muril` uses the same config as `muril_base`; with identical seed/hardware the two rows should agree up to GPU nondeterminism.

## Commands that produce the TBD cells

- `A1: binary head only` -> run: `python -m meaningguard.train --config ablation/a1_binary_only.yaml`
- `A3: google/muril-base-cased` -> run: `python -m meaningguard.train --config ablation/a3_base_swap_muril.yaml`
- `A3: ai4bharat/indic-bert` -> run: `python -m meaningguard.train --config ablation/a3_base_swap_indicbert.yaml`
- `A3: xlm-roberta-base` -> run: `python -m meaningguard.train --config ablation/a3_base_swap_xlmr.yaml`
