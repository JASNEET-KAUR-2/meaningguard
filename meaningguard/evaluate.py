#!/usr/bin/env python
"""Step 18/19/20: evaluation + auto-generated tables.

Single run:
  python -m meaningguard.evaluate --pred predictions/muril_base_test.jsonl --gold data/processed/test.jsonl
Rebuild every table from whatever prediction files exist:
  python -m meaningguard.evaluate --all
Error taxonomy dump (20 most-confident wrong predictions):
  python -m meaningguard.evaluate --errors predictions/muril_base_test.jsonl --gold data/processed/test.jsonl

Metric definitions (all computed on the gold file, never invented):
  Acc          binary accuracy on label (preserved / changed)
  Macro-F1     macro-F1 over the 2 binary classes (comparable across ALL models, incl. binary-only baselines)
  Recall:<d>   detection recall: share of gold pairs of dimension d that the model flags as 'changed'
  FPR          share of gold-preserved pairs flagged 'changed'
  Dim-Recall   (supplementary) strict attribution recall: predicted dimension == gold dimension
  Dim-F1       (supplementary) macro-F1 over the 6 dimension classes present in gold (none + 5)
Cells with no prediction file are written as TBD together with the command that produces them.
"""
import argparse, json, os
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

DIMS = ["negation", "quantity", "deadline", "intent", "entity"]
DIM7 = ["none", "negation", "quantity", "deadline", "intent", "entity", "multi"]
LAB = ["preserved", "changed"]
REPORTS = "reports"
PRED_DIR = "predictions"
METRICS_JSON = os.path.join(REPORTS, "metrics.json")

# name -> (display name, command that produces the predictions)
MAIN_ROWS = [
    ("dummy_always_preserved", "Dummy: always-preserved", "python baselines/dummy.py"),
    ("dummy_always_changed", "Dummy: always-changed", "python baselines/dummy.py"),
    ("translation_pipeline", "Translation pipeline (IndicTrans2/transliteration)", "python baselines/translation_pipeline.py"),
    ("encoder_zeroshot", "Zero-shot MuRIL cosine", "python baselines/zeroshot_encoder.py"),
    ("qwen_zeroshot", "Zero-shot Qwen1.5-1.8B-Chat (4-bit)", "python baselines/zeroshot_llm.py"),
    ("muril_base", "MeaningGuard (MuRIL, multi-task)", "python -m meaningguard.train --config configs/muril_base.yaml"),
]
A1_ROWS = [("muril_base", "Main: MuRIL multi-task (A1 reference)", "python -m meaningguard.train --config configs/muril_base.yaml"),
           ("a1_binary_only", "A1: binary head only", "python -m meaningguard.train --config ablation/a1_binary_only.yaml")]
A3_ROWS = [("muril_base", "Main: MuRIL (configs/muril_base.yaml)", "python -m meaningguard.train --config configs/muril_base.yaml"),
           ("a3_muril", "A3: google/muril-base-cased", "python -m meaningguard.train --config ablation/a3_base_swap_muril.yaml"),
           ("a3_indicbert", "A3: ai4bharat/indic-bert", "python -m meaningguard.train --config ablation/a3_base_swap_indicbert.yaml"),
           ("a3_xlmr", "A3: xlm-roberta-base", "python -m meaningguard.train --config ablation/a3_base_swap_xlmr.yaml")]


def _read(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def _norm(p):
    lab = p.get("label", "preserved")
    dim = p.get("dimension", "none")
    if lab == "preserved":
        dim = "none"
    return lab, dim


def compute(pred_rows, gold_rows):
    P = {r["pair_id"]: r for r in pred_rows}
    gold = [g for g in gold_rows if g["pair_id"] in P]
    missing = len(gold_rows) - len(gold)
    yl_true = [g["label"] for g in gold]
    yl_pred = [_norm(P[g["pair_id"]])[0] for g in gold]
    yd_true = [g["dimension"] for g in gold]
    yd_pred = [_norm(P[g["pair_id"]])[1] for g in gold]
    m = {"n": len(gold), "missing_predictions": missing}
    m["acc"] = accuracy_score(yl_true, yl_pred)
    m["macro_f1"] = f1_score(yl_true, yl_pred, labels=LAB, average="macro", zero_division=0)
    pres = [i for i, y in enumerate(yl_true) if y == "preserved"]
    m["fpr"] = float(np.mean([yl_pred[i] == "changed" for i in pres])) if pres else float("nan")
    m["detect_recall"], m["dim_recall"], m["support"] = {}, {}, {}
    for d in DIMS:
        idx = [i for i, y in enumerate(yd_true) if y == d]
        m["support"][d] = len(idx)
        m["detect_recall"][d] = float(np.mean([yl_pred[i] == "changed" for i in idx])) if idx else float("nan")
        m["dim_recall"][d] = float(np.mean([yd_pred[i] == d for i in idx])) if idx else float("nan")
    present = [d for d in DIM7 if d in set(yd_true)]
    m["dim_macro_f1"] = f1_score(yd_true, yd_pred, labels=present, average="macro", zero_division=0)
    m["cm_label"] = confusion_matrix(yl_true, yl_pred, labels=LAB).tolist()
    m["cm_dim"] = confusion_matrix(yd_true, yd_pred, labels=DIM7).tolist()
    return m


def _f(x, pct=True):
    if x is None or (isinstance(x, float) and x != x):
        return "n/a"
    return "%.1f%%" % (100 * x) if pct else "%.3f" % x


HEADER = "| Model | Acc | Macro-F1 | Recall:negation | Recall:quantity | Recall:deadline | Recall:intent | Recall:entity | FPR |\n|---|---|---|---|---|---|---|---|---|\n"


def _row(name, disp, cmd, store):
    m = store.get(name)
    if not m:
        return "| %s | TBD | TBD | TBD | TBD | TBD | TBD | TBD | TBD |\n" % disp, "- `%s` -> run: `%s`\n" % (disp, cmd)
    cells = [_f(m["acc"]), _f(m["macro_f1"], False)] + [_f(m["detect_recall"][d]) for d in DIMS] + [_f(m["fpr"])]
    return "| %s | %s |\n" % (disp, " | ".join(cells)), ""


def _table(rows, store):
    out, todo = HEADER, ""
    for name, disp, cmd in rows:
        r, t = _row(name, disp, cmd, store)
        out += r; todo += t
    return out, todo


def write_tables(store):
    os.makedirs(REPORTS, exist_ok=True)
    t, todo = _table(MAIN_ROWS, store)
    sup = next((m for m in store.values()), None)
    with open(os.path.join(REPORTS, "results_table.md"), "w", encoding="utf-8") as f:
        f.write("# Results (test split)\n\n_Auto-generated by `python -m meaningguard.evaluate --all`. TBD = not run yet; never filled by hand._\n\n")
        f.write(t)
        f.write("\n**Definitions.** Acc / Macro-F1 are on the binary label (preserved vs changed). Recall:<dim> = share of gold pairs of that "
                "dimension flagged as *changed* (detection recall; comparable for binary-only baselines). FPR = share of preserved pairs flagged as changed.\n")
        if sup:
            f.write("\nTest support per dimension: %s\n" % sup["support"])
        f.write("\n## Supplementary: dimension attribution (models that output a dimension)\n\n")
        f.write("| Model | Dim-F1 (6 classes) | DimRecall:negation | DimRecall:quantity | DimRecall:deadline | DimRecall:intent | DimRecall:entity |\n|---|---|---|---|---|---|---|\n")
        for name, disp, cmd in MAIN_ROWS:
            m = store.get(name)
            if m is None:
                f.write("| %s | TBD | TBD | TBD | TBD | TBD | TBD |\n" % disp)
            else:
                f.write("| %s | %s | %s |\n" % (disp, _f(m["dim_macro_f1"], False), " | ".join(_f(m["dim_recall"][d]) for d in DIMS)))
        if todo:
            f.write("\n## Commands that produce the TBD cells\n\n" + todo)
    a1, todo1 = _table(A1_ROWS, store)
    a3, todo3 = _table(A3_ROWS, store)
    with open(os.path.join(REPORTS, "ablation_table.md"), "w", encoding="utf-8") as f:
        f.write("# Ablations (test split, seed 42)\n\n_Auto-generated. TBD = not run yet._\n\n## A1: binary head only vs multi-task\n\n" + a1)
        f.write("\nA1 has no dimension head, so its Recall:<dim> columns are detection recall only. Dim-attribution is not defined for it.\n")
        f.write("\n## A3: base-encoder swap\n\n" + a3)
        f.write("\nNote: `a3_muril` uses the same config as `muril_base`; with identical seed/hardware the two rows should agree up to GPU nondeterminism.\n")
        if todo1 or todo3:
            f.write("\n## Commands that produce the TBD cells\n\n" + todo1 + todo3)
    # confusion matrices
    with open(os.path.join(REPORTS, "confusion_matrices.md"), "w", encoding="utf-8") as f:
        f.write("# Confusion matrices (test split; rows = gold, cols = predicted)\n\n")
        for name, m in store.items():
            f.write("## %s\n\n2x2 label (%s)\n\n```\n%s\n```\n\n7x7 dimension (%s)\n\n```\n%s\n```\n\n" % (
                name, LAB, np.array(m["cm_label"]), DIM7, np.array(m["cm_dim"])))


def _inject(path, tag, body):
    start, end = "<!-- %s_START -->" % tag, "<!-- %s_END -->" % tag
    if not os.path.exists(path):
        return
    t = open(path, encoding="utf-8").read()
    if start in t and end in t:
        a, b = t.index(start) + len(start), t.index(end)
        open(path, "w", encoding="utf-8").write(t[:a] + "\n" + body.strip() + "\n" + t[b:])


def inject_into_report():
    rp = os.path.join(REPORTS, "final_report.md")
    body = lambda f: "".join(l for l in open(os.path.join(REPORTS, f), encoding="utf-8").read().splitlines(True)
                              if not l.startswith("# ") and not l.startswith("_Auto"))
    _inject(rp, "RESULTS_TABLE", body("results_table.md"))
    _inject(rp, "ABLATION_TABLE", body("ablation_table.md"))
    _inject(rp, "ERROR_TAXONOMY", body("error_taxonomy.md") if os.path.exists(os.path.join(REPORTS, "error_taxonomy.md")) else "TBD")


def load_store():
    return json.load(open(METRICS_JSON)) if os.path.exists(METRICS_JSON) else {}


def save_store(store):
    os.makedirs(REPORTS, exist_ok=True)
    json.dump(store, open(METRICS_JSON, "w"), indent=1)


def error_dump(pred_path, gold_path, k=20):
    P = {r["pair_id"]: r for r in _read(pred_path)}
    wrong = []
    for g in _read(gold_path):
        p = P.get(g["pair_id"])
        if not p:
            continue
        lab, dim = _norm(p)
        if lab != g["label"] or dim != g["dimension"]:
            kind = "missed_change" if (g["label"] == "changed" and lab == "preserved") else \
                   "false_alarm" if (g["label"] == "preserved" and lab == "changed") else "wrong_dimension"
            wrong.append((float(p.get("confidence", 0.0)), kind, g, lab, dim))
    wrong.sort(key=lambda x: -x[0])
    top = wrong[:k]
    groups = {}
    for conf, kind, g, lab, dim in top:
        key = "%s | gold dimension: %s | edit op family: %s" % (kind, g["dimension"], (g["edit_operation"] or "").split("_to_")[0][:24])
        groups.setdefault(key, []).append((conf, g, lab, dim))
    os.makedirs(REPORTS, exist_ok=True)
    with open(os.path.join(REPORTS, "error_taxonomy.md"), "w", encoding="utf-8") as f:
        f.write("# Error taxonomy\n\n_Auto-generated from %s: the %d most-confident wrong predictions out of %d total errors. "
                "Failure modes are grouped mechanically by (error type, gold dimension, edit-operation family); the analysis paragraph "
                "of each group must be written by hand after reading the examples._\n\n" % (pred_path, len(top), len(wrong)))
        for i, (key, items) in enumerate(sorted(groups.items(), key=lambda kv: -len(kv[1])), 1):
            f.write("## Failure mode %d: %s  (n=%d of top-%d)\n\n" % (i, key, len(items), k))
            for conf, g, lab, dim in items[:2]:
                f.write("- `%s` (conf %.2f)\n  - S1: %s\n  - S2: %s\n  - gold: %s/%s | edit: %s | model: %s/%s\n" % (
                    g["pair_id"], conf, g["s1"], g["s2"], g["label"], g["dimension"], g["edit_operation"], lab, dim))
            f.write("\n**Analysis:** TBD (write 1 paragraph after reading the examples above).\n\n")
    print("wrote reports/error_taxonomy.md (%d errors total, top %d grouped into %d modes)" % (len(wrong), len(top), len(groups)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred"); ap.add_argument("--gold", default="data/processed/test.jsonl")
    ap.add_argument("--name"); ap.add_argument("--all", action="store_true"); ap.add_argument("--errors")
    args = ap.parse_args()
    if args.errors:
        error_dump(args.errors, args.gold); return
    store = load_store()
    todo = []
    if args.all:
        known = {n for n, _, _ in MAIN_ROWS + A1_ROWS + A3_ROWS}
        for n in sorted(known):
            p = os.path.join(PRED_DIR, "%s_test.jsonl" % n)
            if os.path.exists(p):
                todo.append((n, p))
    elif args.pred:
        name = args.name or os.path.basename(args.pred).replace("_test.jsonl", "").replace(".jsonl", "")
        todo.append((name, args.pred))
    for name, p in todo:
        m = compute(_read(p), _read(args.gold))
        store[name] = m
        print("%-24s n=%d acc=%s macroF1=%s FPR=%s | detect-recall %s" % (
            name, m["n"], _f(m["acc"]), _f(m["macro_f1"], False), _f(m["fpr"]),
            {d: _f(m["detect_recall"][d]) for d in DIMS}))
        if m["missing_predictions"]:
            print("  WARNING: %d gold pairs had no prediction" % m["missing_predictions"])
    save_store(store)
    write_tables(store)
    inject_into_report()
    print("wrote reports/results_table.md, reports/ablation_table.md, reports/confusion_matrices.md")


if __name__ == "__main__":
    main()
