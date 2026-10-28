# Error taxonomy

TBD. This file is generated from the trained model's test predictions (20 most-confident wrong predictions, grouped into failure modes).

Produce it with:

```
python -m meaningguard.evaluate --errors predictions/muril_base_test.jsonl --gold data/processed/test.jsonl
```

Then write one analysis paragraph per failure mode by hand (the generator leaves `**Analysis:** TBD` under each group).
Candidate failure modes to look for (hypotheses, NOT results): `mat` attached to an English verb (`share mat karo`), `kal` yesterday-vs-tomorrow ambiguity,
dropped entity substitution, imperative-to-question intent shifts.
