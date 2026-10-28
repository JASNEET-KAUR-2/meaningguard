# MeaningGuard

**Detect whether a rewrite of a Hinglish / Romanized-Hindi / Devanagari sentence preserved its meaning, and if not, which dimension changed (negation, quantity, deadline, intent, entity).**

Motivating example: `Refund mat karna` -> `Refund karna` deletes one token and flips "do not refund" into "refund".

```
>>> python -m meaningguard.predict "Refund mat karna" "Refund karna"
{'label': 'changed', 'dimension': 'negation', 'confidence': ...}   # shape of the output; run it after training
```

Model: MuRIL-base encoder -> [CLS] -> two heads (binary changed/preserved + 7-way dimension), loss `CE(A) + 0.5*CE(B)`.
Dataset: 1,200 rule-generated contrast pairs (780 / 140 / 280 train/dev/test), split by seed sentence. Details and all results: [reports/final_report.md](reports/final_report.md).
Model weights: not committed (`*.pt` is git-ignored). HF Hub: `https://huggingface.co/<YOUR-USERNAME>/meaningguard-muril` (placeholder, optional).

## Install
```
pip install -r requirements.txt
```
Python 3.10+, a CUDA GPU with 16 GB is enough (Colab free T4). The data pipeline and `evaluate.py` need only numpy/pandas/scikit-learn.

## One-command reproduce
```
python scripts/extract_seeds.py && python scripts/generate_variants.py && python scripts/split_by_seed.py && python -m meaningguard.train --config configs/muril_base.yaml && python -m meaningguard.evaluate --pred predictions/muril_base_test.jsonl --gold data/processed/test.jsonl
```
`extract_seeds.py` tries COMI-LINGUA from HuggingFace first. To reproduce the committed dataset exactly, add `--force_fallback` to that first command. Step-by-step (baselines, ablations, GitHub push): [RUNBOOK.md](RUNBOOK.md). Colab: `notebooks/colab_train.ipynb`.

## Layout
`scripts/` data pipeline · `meaningguard/` model, training, evaluation, predict · `baselines/` 4 baselines · `ablation/` A1 + A3 configs · `reports/` auto-generated tables and the report · `tests/test_smoke.py`.

## Status
Data pipeline, validation, kappa tooling, evaluation and the dummy / translation baselines have been run. Training, the Qwen and encoder baselines and the ablations need a GPU: their cells are TBD in `reports/*.md` until you run `RUNBOOK.md` steps 4 to 7.

License: MIT.
