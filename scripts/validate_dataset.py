#!/usr/bin/env python
"""Step 6: dataset sanity checks. Exits non-zero on FAIL.

Checks: strict schema/enums, no empty fields, no seed leaks across splits, per-dimension counts > 0 in every split,
class balance, language-form distribution, duplicate pair_ids, rule-oracle consistency, total size < 5 MB.
'multi' is part of the 7-way label space but the generator never emits it (edits touch exactly ONE dimension) -> WARN only.

Run:  python scripts/validate_dataset.py
"""
import argparse, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_variants import changed_dims, DIMS

FIELDS = ["pair_id", "seed_id", "s1", "s2", "language_form", "label", "dimension", "edit_operation", "split"]
FORMS = {"roman", "deva", "hinglish"}
DIM_ENUM = {"none", "negation", "quantity", "deadline", "intent", "entity", "multi"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="data/processed")
    args = ap.parse_args()
    fails, warns = [], []
    data = {}
    for sp in ("train", "dev", "test"):
        path = os.path.join(args.dir, sp + ".jsonl")
        if not os.path.exists(path):
            fails.append("missing file %s" % path); continue
        data[sp] = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
    if fails:
        print("FAIL:", fails); sys.exit(1)

    seeds_by_split = {sp: {r["seed_id"] for r in rows} for sp, rows in data.items()}
    for a, b in (("train", "dev"), ("train", "test"), ("dev", "test")):
        inter = seeds_by_split[a] & seeds_by_split[b]
        if inter:
            fails.append("seed leak %s/%s: %d seeds" % (a, b, len(inter)))
    all_ids = Counter(r["pair_id"] for rows in data.values() for r in rows)
    if any(v > 1 for v in all_ids.values()):
        fails.append("duplicate pair_id found")

    for sp, rows in data.items():
        for r in rows:
            if set(r.keys()) != set(FIELDS):
                fails.append("%s %s: schema keys mismatch" % (sp, r.get("pair_id"))); break
            if any(r[k] in ("", None) for k in ("pair_id", "seed_id", "s1", "s2")):
                fails.append("%s %s: empty text field" % (sp, r["pair_id"])); break
            if r["language_form"] not in FORMS or r["dimension"] not in DIM_ENUM or r["label"] not in ("preserved", "changed"):
                fails.append("%s %s: bad enum" % (sp, r["pair_id"])); break
            if r["split"] != sp:
                fails.append("%s %s: split field mismatch" % (sp, r["pair_id"])); break
            if (r["label"] == "preserved") != (r["dimension"] == "none"):
                fails.append("%s %s: label/dimension inconsistent" % (sp, r["pair_id"])); break
            cd = changed_dims(r["s1"], r["s2"])
            exp = set() if r["label"] == "preserved" else {r["dimension"]}
            if cd != exp:
                fails.append("%s %s: oracle disagrees (%s vs %s)" % (sp, r["pair_id"], cd, exp)); break

    print("=== Dataset report ===")
    total = 0
    for sp, rows in data.items():
        total += len(rows)
        dc = Counter(r["dimension"] for r in rows)
        lc = Counter(r["label"] for r in rows)
        fc = Counter(r["language_form"] for r in rows)
        print("%-5s n=%4d seeds=%3d | label %s | forms %s" % (sp, len(rows), len(seeds_by_split[sp]), dict(lc), dict(fc)))
        print("      dims %s" % dict(dc))
        for d in DIMS:
            if dc[d] == 0:
                fails.append("%s has 0 pairs for dimension %s" % (sp, d))
        if dc["multi"] == 0:
            warns.append("%s: 'multi' has 0 examples (reserved class, generator edits one dimension only)" % sp)
        if lc["changed"] and lc["preserved"] and max(lc.values()) / min(lc.values()) > 3:
            warns.append("%s: label imbalance %s" % (sp, dict(lc)))
    size = sum(os.path.getsize(os.path.join(args.dir, sp + ".jsonl")) for sp in data)
    print("total pairs: %d | processed size: %.1f KB" % (total, size / 1024))
    if size > 5 * 1024 * 1024:
        fails.append("processed data >5MB")
    for w in warns:
        print("WARN:", w)
    if fails:
        for f in fails[:20]:
            print("FAIL:", f)
        sys.exit(1)
    print("VALIDATION: PASS")


if __name__ == "__main__":
    main()
