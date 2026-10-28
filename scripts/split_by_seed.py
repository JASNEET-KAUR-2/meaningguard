#!/usr/bin/env python
"""Step 5: split by SEED (not by pair) 65/12/23 -> train/dev/test.  No seed ever appears in two splits.
Seeds are stratified by their primary dimension so every split contains every dimension.

Run:  python scripts/split_by_seed.py
"""
import argparse, json, os, random
from collections import defaultdict, Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", default="data/processed/pairs_raw.jsonl")
    ap.add_argument("--seeds", default="data/seeds.jsonl")
    ap.add_argument("--outdir", default="data/processed")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", default="0.65,0.12,0.23")
    args = ap.parse_args()
    r_tr, r_dev, r_te = [float(x) for x in args.ratios.split(",")]
    rng = random.Random(args.seed)
    seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
    primary = {s["seed_id"]: s.get("intended_dimension") or s["detected_dimensions"][0] for s in seeds}
    pairs = [json.loads(l) for l in open(args.pairs, encoding="utf-8") if l.strip()]
    used = sorted({p["seed_id"] for p in pairs})
    strata = defaultdict(list)
    for sid in used:
        strata[primary.get(sid, "none")].append(sid)
    assign = {}
    for d in sorted(strata):
        ids = sorted(strata[d]); rng.shuffle(ids)
        n = len(ids); n_tr = round(n * r_tr); n_dev = round(n * r_dev)
        for i, sid in enumerate(ids):
            assign[sid] = "train" if i < n_tr else ("dev" if i < n_tr + n_dev else "test")
    out = defaultdict(list)
    for p in pairs:
        p["split"] = assign[p["seed_id"]]
        out[p["split"]].append(p)
    os.makedirs(args.outdir, exist_ok=True)
    for sp in ("train", "dev", "test"):
        rows = out[sp]; rng.shuffle(rows)
        with open(os.path.join(args.outdir, sp + ".jsonl"), "w", encoding="utf-8") as f:
            for p in rows:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
        print("%-5s pairs=%4d seeds=%3d  %s" % (sp, len(rows), len({p['seed_id'] for p in rows}),
                                                 dict(Counter(p['dimension'] for p in rows))))
    # keep pairs_raw.jsonl with the final split field too
    with open(args.pairs, "w", encoding="utf-8") as f:
        for sp in ("train", "dev", "test"):
            for p in out[sp]:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    leak = [s for s in set(a for a in assign) if sum(any(p['seed_id'] == s for p in out[sp]) for sp in out) > 1]
    assert not leak, "seed leak!"
    print("Seed-leak check: OK (0 seeds in more than one split)")


if __name__ == "__main__":
    main()
