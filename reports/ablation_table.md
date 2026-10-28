# Ablations (test split, seed 42)

_Auto-generated. TBD = not run yet._

## A1: binary head only vs multi-task

| Model | Acc | Macro-F1 | Recall:negation | Recall:quantity | Recall:deadline | Recall:intent | Recall:entity | FPR |
|---|---|---|---|---|---|---|---|---|
| Main: MuRIL multi-task (A1 reference) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A1: binary head only | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

A1 has no dimension head, so its Recall:<dim> columns are detection recall only. Dim-attribution is not defined for it.

## A3: base-encoder swap

| Model | Acc | Macro-F1 | Recall:negation | Recall:quantity | Recall:deadline | Recall:intent | Recall:entity | FPR |
|---|---|---|---|---|---|---|---|---|
| Main: MuRIL (configs/muril_base.yaml) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A3: google/muril-base-cased | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A3: ai4bharat/indic-bert | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| A3: xlm-roberta-base | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

Note: `a3_muril` uses the same config as `muril_base`; with identical seed/hardware the two rows should agree up to GPU nondeterminism.

## Commands that produce the TBD cells

- `Main: MuRIL multi-task (A1 reference)` -> run: `python -m meaningguard.train --config configs/muril_base.yaml`
- `A1: binary head only` -> run: `python -m meaningguard.train --config ablation/a1_binary_only.yaml`
- `Main: MuRIL (configs/muril_base.yaml)` -> run: `python -m meaningguard.train --config configs/muril_base.yaml`
- `A3: google/muril-base-cased` -> run: `python -m meaningguard.train --config ablation/a3_base_swap_muril.yaml`
- `A3: ai4bharat/indic-bert` -> run: `python -m meaningguard.train --config ablation/a3_base_swap_indicbert.yaml`
- `A3: xlm-roberta-base` -> run: `python -m meaningguard.train --config ablation/a3_base_swap_xlmr.yaml`
