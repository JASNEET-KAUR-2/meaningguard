#!/usr/bin/env python
"""Step 7: inter-annotator agreement (Cohen's kappa) + pilot-sample generation.

  python scripts/compute_kappa.py --make_sample          # writes data/annotation_sample_20.csv (10 preserved, 10 changed)
  python scripts/compute_kappa.py --a annotator_a.csv --b annotator_b.csv [--gold data/processed/pairs_raw.jsonl]

Annotator CSV columns: pair_id,label,dimension  (label: preserved|changed ; dimension: none|negation|quantity|deadline|intent|entity|multi)
Per-dimension kappa = one-vs-rest binary kappa on the dimension column.
"""
import argparse, csv, json, random
from collections import defaultdict

DIMS = ["none", "negation", "quantity", "deadline", "intent", "entity", "multi"]


def make_sample(pairs_path, out, seed=42):
    rng = random.Random(seed)
    rows = [json.loads(l) for l in open(pairs_path, encoding="utf-8") if l.strip()]
    pres = [r for r in rows if r["label"] == "preserved"]
    by_dim = defaultdict(list)
    for r in rows:
        if r["label"] == "changed":
            by_dim[r["dimension"]].append(r)
    pick = rng.sample(pres, 10)
    for d in ["negation", "quantity", "deadline", "intent", "entity"]:
        pick += rng.sample(by_dim[d], 2)
    rng.shuffle(pick)
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["pair_id", "s1", "s2", "label", "dimension"])   # label/dimension left blank for annotators
        for r in pick:
            w.writerow([r["pair_id"], r["s1"], r["s2"], "", ""])
    print("wrote %s (20 pairs, gold labels withheld; fill label + dimension columns)" % out)


def read(path):
    with open(path, encoding="utf-8") as f:
        return {r["pair_id"]: (r["label"].strip().lower(), r["dimension"].strip().lower()) for r in csv.DictReader(f)}


def kappa(a, b):
    from sklearn.metrics import cohen_kappa_score
    if len(set(a) | set(b)) < 2:
        return float("nan")   # kappa undefined when both annotators use a single identical class
    return float(cohen_kappa_score(a, b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--make_sample", action="store_true")
    ap.add_argument("--pairs", default="data/processed/pairs_raw.jsonl")
    ap.add_argument("--out", default="data/annotation_sample_20.csv")
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--gold")
    args = ap.parse_args()
    if args.make_sample:
        make_sample(args.pairs, args.out); return
    A, B = read(args.a), read(args.b)
    ids = sorted(set(A) & set(B))
    print("pairs labelled by both: %d" % len(ids))
    la, lb = [A[i][0] for i in ids], [B[i][0] for i in ids]
    da, db = [A[i][1] for i in ids], [B[i][1] for i in ids]
    print("OVERALL label kappa (preserved/changed): %.3f | raw agreement %.1f%%" % (kappa(la, lb), 100 * sum(x == y for x, y in zip(la, lb)) / len(ids)))
    print("OVERALL dimension kappa (7-way):        %.3f | raw agreement %.1f%%" % (kappa(da, db), 100 * sum(x == y for x, y in zip(da, db)) / len(ids)))
    print("Per-dimension kappa (one-vs-rest):")
    for d in DIMS:
        k = kappa([int(x == d) for x in da], [int(x == d) for x in db])
        print("  %-9s %s" % (d, "n/a (dimension absent from both)" if k != k else "%.3f" % k))
    if args.gold:
        gold = {}
        for l in open(args.gold, encoding="utf-8"):
            r = json.loads(l); gold[r["pair_id"]] = (r["label"], r["dimension"])
        for name, X in (("A", A), ("B", B)):
            g = [i for i in ids if i in gold]
            agree = sum(X[i] == gold[i] for i in g)
            print("annotator %s vs gold: %d/%d exact (label+dimension) agree" % (name, agree, len(g)))


if __name__ == "__main__":
    main()
