#!/usr/bin/env python
"""Step 17: dummy baselines (always-preserved, always-changed). Per-dimension recall printed as a sanity check.

Run:  python baselines/dummy.py            -> predictions/dummy_always_{preserved,changed}_{dev,test}.jsonl
"""
import json, os, sys
from collections import defaultdict

DIMS = ["negation", "quantity", "deadline", "intent", "entity"]


def main(data_dir="data/processed", out_dir="predictions"):
    os.makedirs(out_dir, exist_ok=True)
    for split in ("dev", "test"):
        gold = [json.loads(l) for l in open(os.path.join(data_dir, split + ".jsonl"), encoding="utf-8") if l.strip()]
        for name, label in (("dummy_always_preserved", "preserved"), ("dummy_always_changed", "changed")):
            with open(os.path.join(out_dir, "%s_%s.jsonl" % (name, split)), "w", encoding="utf-8") as f:
                for g in gold:
                    f.write(json.dumps({"pair_id": g["pair_id"], "label": label, "dimension": "none", "confidence": 1.0}) + "\n")
            if split == "test":
                hit, tot = defaultdict(int), defaultdict(int)
                for g in gold:
                    if g["dimension"] in DIMS:
                        tot[g["dimension"]] += 1
                        # strict dimension recall: predicted dimension is always 'none' -> trivially 0
                        hit[g["dimension"]] += int(g["dimension"] == "none")
                print("%-24s strict per-dimension recall: %s" % (name, {d: round(hit[d] / tot[d], 3) for d in DIMS}))
    print("wrote dummy predictions to %s/" % out_dir)


if __name__ == "__main__":
    main(*sys.argv[1:])
