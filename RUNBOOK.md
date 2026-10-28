# MeaningGuard RUNBOOK

Chronological commands. Every step below lists the exact command to run.
Run from the repository root (`meaningguard/`). Seed is fixed to 42 everywhere.

## 1. Colab setup cell
Open `notebooks/colab_train.ipynb` in Colab (T4 GPU) and run cells 1 to 3, or paste:
```
from google.colab import drive; drive.mount('/content/drive')
!git clone https://github.com/<YOUR-USERNAME>/meaningguard.git /content/meaningguard
%cd /content/meaningguard
!pip install -q -r requirements.txt
```

## 2. Extract + generate + split + validate
```
python scripts/extract_seeds.py --force_fallback   # drop the flag to try COMI-LINGUA first (changes the dataset)
python scripts/generate_variants.py
python scripts/split_by_seed.py
python scripts/validate_dataset.py
```
Already run locally: 300 seeds -> 1,200 pairs -> 780/140/280, validation PASS (committed under `data/processed/`).

## 3. 20-sample pilot labelling -> Cohen's kappa
`data/annotation_sample_20.csv` is committed (labels withheld). You and your friend each copy it, fill `label` (preserved|changed) and `dimension` (none|negation|quantity|deadline|intent|entity), and save as `annotator_a.csv` / `annotator_b.csv` (columns `pair_id,label,dimension` are enough).
```
python scripts/compute_kappa.py --a annotator_a.csv --b annotator_b.csv --gold data/processed/pairs_raw.jsonl
```
Copy the kappa values into section 4 of `reports/final_report.md` (replace the TBD).

## 4. Train main model
```
python -m meaningguard.train --config configs/muril_base.yaml
```
If it reports CUDA OOM, set `batch_size: 8` and `grad_accum_steps: 4` in the YAML.

## 5. Baselines
```
python baselines/dummy.py
python baselines/zeroshot_encoder.py
python baselines/translation_pipeline.py
python baselines/zeroshot_llm.py
```

## 6. Ablations (A1 and A3 only)
```
python -m meaningguard.train --config ablation/a1_binary_only.yaml
python -m meaningguard.train --config ablation/a3_base_swap_muril.yaml
python -m meaningguard.train --config ablation/a3_base_swap_indicbert.yaml
python -m meaningguard.train --config ablation/a3_base_swap_xlmr.yaml
```

## 7. Evaluate + generate tables
```
python -m meaningguard.evaluate --all
python -m meaningguard.evaluate --errors predictions/muril_base_test.jsonl --gold data/processed/test.jsonl
python -m meaningguard.evaluate --all      # re-inject the taxonomy into final_report.md
```
Then write the one-paragraph analysis under each failure mode in `reports/error_taxonomy.md` and the Discussion in section 6 of the report.

## 8. Push to GitHub
```
git remote add origin https://github.com/<YOUR-USERNAME>/meaningguard.git
git branch -M main
git push -u origin main
```

## 9. Optional: push model to Hugging Face
```
python - <<'PY'
from huggingface_hub import HfApi, login
login()
api = HfApi(); api.create_repo("<YOUR-USERNAME>/meaningguard-muril", exist_ok=True)
api.upload_file(path_or_fileobj="checkpoints/muril_base/best.pt", path_in_repo="best.pt", repo_id="<YOUR-USERNAME>/meaningguard-muril")
PY
```

## 10. Read the report
`reports/final_report.md`
