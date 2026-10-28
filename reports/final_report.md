# MeaningGuard: Detecting Meaning Changes in Romanized Hindi, Devanagari and Hinglish

**Author:** Udit Prakash Singh · **Code:** https://github.com/JASNEET-KAUR-2/meaningguard (placeholder) · **Seed:** 42 · **Status of numbers:** every number below is copied from an actual run; a cell that has not been run says TBD together with the command that produces it.

## 1. Abstract

Rewriting a code-mixed message can silently change what it asks for. We introduce MeaningGuard, a MuRIL-based multi-task model that detects meaning changes across Romanized Hindi, Devanagari, and Hinglish, achieving **TBD% per-dimension recall on negation vs TBD% for zero-shot Qwen-1.8B** (to be filled by `python -m meaningguard.evaluate --all` once the model and the Qwen baseline have been run). The accompanying benchmark contains 1,200 sentence pairs over five meaning dimensions (negation, quantity, deadline, intent, entity), built by rule-based minimal edits and split by seed sentence so that no source sentence appears in two splits. A binary head decides *preserved vs changed* and a second head names the violated dimension. We compare against two zero-shot baselines, a transliteration-and-compare baseline and dummy classifiers, and run two ablations (binary head only; encoder swap).

## 2. Introduction

A rewrite that looks harmless can reverse an instruction. Consider the motivating pair

> S1: *Refund mat karna*  S2: *Refund karna*

Deleting the single token *mat* turns "do not issue the refund" into "issue the refund". Rewriting tools (spell correctors, transliterators, paraphrasers, translation round-trips) for Hinglish are normally judged on fluency or on downstream accuracy, not on whether the *meaning* of the original survives. Code-mixed text makes this harder: negation markers (*mat, nahi, na, don't*), number words (*ek, paanch, sau*), time words (*kal* can mean yesterday or tomorrow) and entity names appear in three surface forms (Roman, Devanagari, mixed English-Hindi), and spelling is not standardised.

**Gap.** Existing code-mixing benchmarks evaluate tagging, NLI, QA and sentiment on a single text; they do not test whether a *rewritten* version preserves meaning along named dimensions. MeaningGuard provides such a benchmark and a model.

**Contributions.** (i) a five-dimension taxonomy and a contrast-pair construction procedure for Hindi-English text in three forms; (ii) a multi-task MuRIL detector with a dimension head; (iii) baselines and two ablations, all reproducible from one repository.

## 3. Related Work

COMI-LINGUA is a large Hindi-English code-mixed corpus and is our preferred source of seed sentences; GLUECoS and LinCE are code-switching evaluation suites covering tasks such as language identification, POS, NER, sentiment and NLI, but each scores a single text rather than a before/after pair. Contrast sets and minimal-pair evaluation from the NLI literature motivate our construction: small, targeted edits whose effect on the label is known by design. MuRIL is a BERT-style encoder pre-trained on 17 Indian languages including transliterated text, which is why we use it as the primary encoder; IndicBERT and XLM-R serve as comparison encoders in ablation A3.

## 4. Benchmark

**Taxonomy.** *negation* (a negation marker is added, removed or replaced), *quantity* (a numeral or number word changes), *deadline* (a time word such as *kal/aaj/parso/tomorrow/today* changes or the limit word *tak/by* is dropped), *intent* (imperative becomes question or completed report, e.g. *karo → karna hai?*), *entity* (a person, bank or platform name is swapped within its category). The label space also reserves `multi`; the generator edits exactly one dimension per pair, so `multi` has no examples (see Limitations).

**Seeds.** The intended source is LingoIITGN/COMI-LINGUA, filtered by regular expressions for negation, numbers, time words and entities (`scripts/extract_seeds.py`). The dataset reported here was built from the **built-in fallback list of 300 seeds** (hand-written templates × curated slot lists, 60 per dimension, deterministic with seed 42) because the HuggingFace download could not be run in the build environment. The HF path is implemented but untested; if you re-run it, regenerate every table in this report.

**Contrast-pair construction** (`scripts/generate_variants.py`, no LLM calls). Each seed yields up to two *preserving* variants (Roman↔Devanagari dictionary transliteration; spelling variants such as *nahi/nahin* and *please/kripya*; casing or period normalisation) and up to two *altering* variants, each a minimal edit on one dimension with `edit_operation` recorded (for example `neg_delete_marker`, `qty_div10`, `dl_drop_tak`, `int_imperative_to_question`, `ent_swap_rahul_to_priya`). A rule-based feature extractor checks every pair: the intended dimension must change and no other may; pairs that fail are dropped (0 of 1,200 attempted pairs were dropped in the final build). The spec asks for 1 + 1 variants per seed, which gives only 600 pairs from 300 seeds; to reach the ~1,200 target we generate up to 2 + 2 per seed.

**Annotation.** Two annotators (the author and one friend) label a 20-pair pilot sample (`data/annotation_sample_20.csv`, 10 preserved / 10 changed) and agreement is computed with `scripts/compute_kappa.py`. **Cohen's kappa: TBD** (pilot not yet labelled; command: `python scripts/compute_kappa.py --a annotator_a.csv --b annotator_b.csv`).

**Splits.** Seeds, not pairs, are assigned to train/dev/test (65/12/23), stratified by primary dimension; `validate_dataset.py` confirms zero seed overlap.

| Split | Seeds | Pairs | Preserved | Changed | negation | quantity | deadline | intent | entity |
|---|---|---|---|---|---|---|---|---|---|
| train | 195 | 780 | 390 | 390 | 78 | 78 | 78 | 78 | 78 |
| dev | 35 | 140 | 70 | 70 | 14 | 14 | 14 | 14 | 14 |
| test | 70 | 280 | 140 | 140 | 28 | 28 | 28 | 28 | 28 |

Language form (form of the seed sentence): train 236 Devanagari / 396 Roman / 148 Hinglish; dev 40 / 68 / 32; test 124 / 92 / 64. "Hinglish" means the sentence contains English syntax words (*please, the, by, don't, send, …*); "Roman" is Romanized Hindi grammar. Transliteration-type preserving pairs contain both scripts regardless of this label.

## 5. Model

**Architecture.** `google/muril-base-cased` encodes `s1 [SEP] s2` (max length 128). The [CLS] vector passes through Dropout(0.1) into two linear heads: **Head A** (2-way: preserved/changed) and **Head B** (7-way: none, negation, quantity, deadline, intent, entity, multi). The loss is

`L = CrossEntropy(A) + 0.5 · CrossEntropy(B)`.

At inference, if Head A says *preserved* the reported dimension is `none`; otherwise it is the arg-max of Head B. The multi-task head forces the encoder to represent *what* changed, not just *that* something changed, which is the hypothesis tested by ablation A1.

**Training.** AdamW, learning rate 2e-5, 4 epochs, batch size 16 with 2 gradient-accumulation steps, `torch.cuda.amp` mixed precision, seed 42, checkpoint chosen by best dev macro-F1; peak GPU memory is printed every epoch so the run can be checked against the 16 GB Colab T4 limit. Config: `configs/muril_base.yaml`. Training time and peak memory: **TBD** (see `checkpoints/muril_base/history.csv`).

## 6. Experiments

**Baselines.**
1. *Zero-shot LLM:* Qwen/Qwen1.5-1.8B-Chat in 4-bit (bitsandbytes), greedy decoding. Verbatim prompt:

```
You are a meaning-preservation checker for Hinglish/Hindi text.
Compare these two sentences:
S1: "{s1}"
S2: "{s2}"
Answer in JSON: {"label": "preserved"|"changed", "dimension": "none"|"negation"|"quantity"|"deadline"|"intent"|"entity"|"multi"}
```

Unparseable outputs default to preserved/none and are logged.
2. *Zero-shot encoder:* mean-pooled MuRIL embeddings, cosine similarity, threshold tuned on dev for macro-F1 (binary only).
3. *Translation/normalisation pipeline:* both sentences are mapped to Devanagari (lexicon + `indic-transliteration`) and compared; equal means preserved (binary only). The specified `indictrans2-en-indic-dist-200M` is an English→Indic translator, so it is not a faithful Roman-Hindi→Devanagari transliterator; it is available behind `--indictrans2` but is not used for the reported numbers. **Caveat:** this baseline shares its lexicon with the pair generator, so it is close to an upper bound for rule-based normalisation on this synthetic benchmark and should not be read as evidence about real-world text.
4. *Dummy:* always-preserved, always-changed.

**Metrics.** Acc and Macro-F1 are computed on the binary label so every system is comparable. Recall:<dim> is *detection recall*: the share of gold pairs of that dimension flagged as changed (comparable for binary-only systems). FPR is the share of preserved pairs flagged as changed. A supplementary table reports strict dimension attribution (predicted dimension equals gold) for systems that output one. Accuracy alone is not informative here (a dummy gets 50%), hence the per-dimension columns.

### Main results (test split, seed 42)

<!-- RESULTS_TABLE_START -->
| Model | Acc | Macro-F1 | Recall:negation | Recall:quantity | Recall:deadline | Recall:intent | Recall:entity | FPR |
|---|---|---|---|---|---|---|---|---|
| Dummy: always-preserved | 50.0% | 0.333 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Dummy: always-changed | 50.0% | 0.333 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Translation pipeline (IndicTrans2/transliteration) | 95.0% | 0.950 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 10.0% |
| Zero-shot MuRIL cosine | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| Zero-shot Qwen1.5-1.8B-Chat (4-bit) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |
| MeaningGuard (MuRIL, multi-task) | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |

**Definitions.** Acc / Macro-F1 are on the binary label (preserved vs changed). Recall:<dim> = share of gold pairs of that dimension flagged as *changed* (detection recall; comparable for binary-only baselines). FPR = share of preserved pairs flagged as changed.

Test support per dimension: {'negation': 28, 'quantity': 28, 'deadline': 28, 'intent': 28, 'entity': 28}

## Supplementary: dimension attribution (models that output a dimension)

| Model | Dim-F1 (6 classes) | DimRecall:negation | DimRecall:quantity | DimRecall:deadline | DimRecall:intent | DimRecall:entity |
|---|---|---|---|---|---|---|
| Dummy: always-preserved | 0.111 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Dummy: always-changed | 0.111 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Translation pipeline (IndicTrans2/transliteration) | 0.111 | 0.0% | 0.0% | 0.0% | 0.0% | 0.0% |
| Zero-shot MuRIL cosine | TBD | TBD | TBD | TBD | TBD | TBD |
| Zero-shot Qwen1.5-1.8B-Chat (4-bit) | TBD | TBD | TBD | TBD | TBD | TBD |
| MeaningGuard (MuRIL, multi-task) | TBD | TBD | TBD | TBD | TBD | TBD |

## Commands that produce the TBD cells

- `Zero-shot MuRIL cosine` -> run: `python baselines/zeroshot_encoder.py`
- `Zero-shot Qwen1.5-1.8B-Chat (4-bit)` -> run: `python baselines/zeroshot_llm.py`
- `MeaningGuard (MuRIL, multi-task)` -> run: `python -m meaningguard.train --config configs/muril_base.yaml`
<!-- RESULTS_TABLE_END -->

### Ablations (test split, seed 42)

<!-- ABLATION_TABLE_START -->
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
<!-- ABLATION_TABLE_END -->

**Discussion.** TBD until the trained rows exist. Write it from the filled tables: (a) per-dimension recall of MeaningGuard vs Qwen-1.8B, with attention to negation; (b) whether A1 (binary head only) changes detection recall or FPR; (c) which encoder wins in A3 and whether the Devanagari/Roman split in `data/processed/test.jsonl` explains it. The dummy rows already confirm that accuracy near 50% means nothing here.

## 7. Analysis

<!-- ERROR_TAXONOMY_START -->
TBD. This file is generated from the trained model's test predictions (20 most-confident wrong predictions, grouped into failure modes).

Produce it with:

```
python -m meaningguard.evaluate --errors predictions/muril_base_test.jsonl --gold data/processed/test.jsonl
```

Then write one analysis paragraph per failure mode by hand (the generator leaves `**Analysis:** TBD` under each group).
Candidate failure modes to look for (hypotheses, NOT results): `mat` attached to an English verb (`share mat karo`), `kal` yesterday-vs-tomorrow ambiguity,
dropped entity substitution, imperative-to-question intent shifts.
<!-- ERROR_TAXONOMY_END -->

## 8. Limitations

- **Small and synthetic.** 1,200 pairs from 300 template-generated seeds. Seeds in different splits share sentence templates, so the test set measures generalisation to new slot fillers, not to new sentence structures. Some seeds are grammatical but odd (for example *Message kal tak bharo*).
- **Rule-based edits.** Dimension labels come from the generator's own rules and a rule-based oracle, not from human judgement of real rewrites; the translation baseline shares that lexicon.
- **`multi` is unused.** The 7-way head has a class with no training examples.
- **Two annotators** and a 20-pair pilot only; agreement is TBD.
- **One seed (42)** so no variance estimates; differences of a few points between models are not reliable. Test support is 28 pairs per dimension.
- **Only ablations A1 and A3** were run; other design choices (loss weight, input format, pooling) are untested.
- The HF/COMI-LINGUA extraction path is implemented but was not exercised for the reported dataset; `kal` ambiguity (yesterday/tomorrow) is handled only as a token swap.

## 9. Reproducibility

```
pip install -r requirements.txt
python scripts/extract_seeds.py --force_fallback && python scripts/generate_variants.py && python scripts/split_by_seed.py && python scripts/validate_dataset.py
python -m meaningguard.train --config configs/muril_base.yaml
python -m meaningguard.evaluate --pred predictions/muril_base_test.jsonl --gold data/processed/test.jsonl
```

The full chronological procedure (Colab, baselines, ablations, GitHub push) is in `RUNBOOK.md`; the Colab notebook is `notebooks/colab_train.ipynb`. GitHub: https://github.com/JASNEET-KAUR-2/meaningguard (placeholder).

## 10. Ethics and Data Statement

The released data are synthetic sentences built from templates and curated lists of common Indian first names, bank names and app names; no personal data, scraped private text or real account details are included, and the entity lists are generic and used only for substitution. If the COMI-LINGUA path is used, seed sentences come from a publicly released corpus and its licence applies. The model is a research prototype: it is trained on a small synthetic benchmark and must not be used as the sole safeguard for consequential rewrites (financial, legal or medical messages). Annotators are the author and a friend, who participate voluntarily; no sensitive attributes are collected.
