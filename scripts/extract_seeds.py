#!/usr/bin/env python
"""Step 3: extract ~300 seed sentences (~60 per dimension) -> data/seeds.jsonl

1. Try HuggingFace LingoIITGN/COMI-LINGUA, filter for sentences containing negation / number / time / entity cues.
2. If the HF load fails (offline, schema change, too few sentences) fall back to the BUILT-IN seed generator below
   (hand-written templates x curated slot lists, deterministic with seed 42, covers all 5 dims and all 3 forms).

Run:  python scripts/extract_seeds.py [--force_fallback]
"""
import argparse, json, os, random, re, sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_variants import (R2D, DIMS, EN_SYNTAX, detect_dimensions, has_deva, to_deva, to_roman, tokenize, canon)

NEG_RE = re.compile(r"\b(mat|na|nahi|nahin|don't|do not)\b", re.I)
NUM_RE = re.compile(r"\d+|\b(ek|do|teen|char|paanch|das|sau|hazaar|lakh)\b", re.I)
TIME_RE = re.compile(r"\b(kal|aaj|abhi|tak|by|tomorrow|today)\b", re.I)
KNOWN_ENT = {"paytm", "phonepe", "sbi", "hdfc", "icici", "amazon", "flipkart", "google", "whatsapp", "india"}


# ------------------------------------------------------------------ built-in fallback (300 seeds)
def _pairs(table):
    return [(n, v) for n, vs in table.items() for v in vs]


NEG_INF = {"refund": ["karna", "bhejna"], "payment": ["karna", "bhejna"], "form": ["bharna", "bhejna"],
           "link": ["kholna", "bhejna"], "password": ["batana", "likhna"], "otp": ["batana", "bhejna"],
           "report": ["bhejna", "dena"], "file": ["kholna", "rakhna"], "booking": ["karna"], "bill": ["dekhna", "bharna"],
           "photo": ["bhejna", "dekhna"], "message": ["likhna", "bhejna"], "document": ["dena", "rakhna"],
           "card": ["dena", "batana"], "loan": ["lena"], "invoice": ["bhejna", "dena"]}
EN_VERBS = ["send", "share", "submit", "update", "delete", "cancel", "approve", "check"]
EN_NOUNS = ["otp", "password", "report", "invoice", "form", "link", "file", "photo", "message", "document"]
IMP_TABLE = {"bharo": ["form", "bill", "fees"], "bhejo": ["report", "file", "photo", "message", "link", "invoice", "document", "copy"],
             "kholo": ["link", "file", "account", "document"], "likho": ["message", "report", "mail"],
             "dekho": ["report", "bill", "file", "photo", "link", "status"], "rakho": ["file", "copy", "document", "invoice"],
             "batao": ["order status", "account number", "booking status"],
             "karo": ["payment", "refund", "booking", "order cancel", "bill jama", "delivery check", "loan update"]}
PREFIX = ["", "", "", "Please ", "Kripya ", "Zaroor "]


def _cap(s):
    return s[:1].upper() + s[1:]


def build_builtin(rng):
    cands = {d: set() for d in DIMS}
    # ---- negation: marker present (delete / replace) and absent (insert)
    for n, v in _pairs(NEG_INF):
        for p in PREFIX:
            cands["negation"].add(_cap(p + "%s mat %s" % (n, v)))
            cands["negation"].add(_cap(p + "%s %s" % (n, v)))
    for e in EN_VERBS:
        for n in EN_NOUNS:
            cands["negation"].add("Please don't %s %s" % (e, n))
            cands["negation"].add("Please do not %s the %s" % (e, n))
            cands["negation"].add(_cap("%s %s mat karo" % (n, e)))
            cands["negation"].add(_cap("%s %s karo" % (n, e)))
            cands["negation"].add(_cap("%s %s karna" % (n, e)))
    # ---- quantity
    for amt in [50, 100, 150, 200, 300, 500, 1000, 2000, 5000]:
        for v in ["bhejo", "jama karo", "transfer karo", "dena"]:
            cands["quantity"].add("%d rupaye %s" % (amt, v))
        for acct in ["account", "wallet"]:
            cands["quantity"].add("%s me %d rupaye jama karo" % (acct, amt))
    for w in ["ek", "teen", "paanch", "das", "sau"]:
        for n in ["ticket", "copy", "form", "photo", "seat", "file", "package"]:
            for v in ["bhejo", "rakho", "dekho"]:
                cands["quantity"].add("%s %s %s" % (w, n, v))
    # ---- deadline
    for n in ["report", "form", "bill", "payment", "invoice", "document", "fees", "message", "file"]:
        for d in ["kal", "aaj", "parso"]:
            for v in ["bhejo", "jama karo", "bharo", "batao"]:
                cands["deadline"].add(_cap("%s %s tak %s" % (n, d, v)))
    for e in EN_VERBS[:5]:
        for n in ["report", "invoice", "form", "document", "file"]:
            for d in ["tomorrow", "today"]:
                cands["deadline"].add("Please %s the %s by %s" % (e, n, d))
    # ---- intent
    for imp, nouns in IMP_TABLE.items():
        for n in nouns:
            for p in PREFIX[:5]:
                cands["intent"].add(_cap(p + "%s %s" % (n, imp)))
    # ---- entity
    for p in ["rahul", "priya", "amit", "neha", "rohan", "sneha", "vikas", "pooja"]:
        for n in ["report", "link", "message", "invoice", "photo", "document"]:
            cands["entity"].add("%s ko %s bhejo" % (p.capitalize(), n))
    for b in ["SBI", "HDFC", "ICICI", "Axis", "PNB"]:
        for n in ["payment", "refund", "loan", "bill"]:
            cands["entity"].add("%s account me %s jama karo" % (b, n))
    for pl in ["Paytm", "PhonePe", "Flipkart", "Amazon", "Swiggy", "Zomato"]:
        for n in ["order", "payment", "booking", "refund"]:
            cands["entity"].add("%s par %s karo" % (pl, n) if n != "order" else "%s par order cancel karo" % pl)
    for p in ["rahul", "priya", "amit", "neha", "rohan"]:
        for n in ["invoice", "report", "document"]:
            cands["entity"].add("Please send the %s to %s" % (n, p.capitalize()))
    # ---- sample 60 per dimension, then assign forms
    seeds = []
    for d in DIMS:
        pool = sorted(cands[d])
        rng.shuffle(pool)
        chosen = pool[:60]
        for i, text in enumerate(chosen):
            deva = to_deva(text)
            toks = [canon(t[1]) for t in tokenize(text)]
            if i % 3 == 0:
                form = "deva"
            elif any(t in EN_SYNTAX for t in toks):
                form = "hinglish"
            else:
                form = "roman"
            seeds.append({"text_roman": text, "text_deva": deva, "language_form": form, "intended_dimension": d})
    rng.shuffle(seeds)
    out = []
    for i, s in enumerate(seeds):
        out.append({"seed_id": "S%04d" % i, "text_roman": s["text_roman"], "text_deva": s["text_deva"],
                    "source": "builtin", "detected_dimensions": [s["intended_dimension"]],
                    "language_form": s["language_form"], "intended_dimension": s["intended_dimension"]})
    return out


# ------------------------------------------------------------------ HuggingFace extraction
def try_hf(rng, per_dim=60):
    from datasets import load_dataset  # may raise
    ds = load_dataset("LingoIITGN/COMI-LINGUA")
    split = "train" if "train" in ds else list(ds.keys())[0]
    d = ds[split]
    str_cols = [c for c, f in d.features.items() if getattr(f, "dtype", None) == "string"]
    pref = [c for c in str_cols if any(k in c.lower() for k in ("text", "sentence", "hinglish", "roman", "code"))]
    cols = pref or str_cols
    if not cols:
        raise RuntimeError("no string columns in COMI-LINGUA")
    print("HF columns used:", cols)
    seen, pool = set(), []
    for row in d:
        for c in cols:
            t = row[c]
            if not isinstance(t, str):
                continue
            t = " ".join(t.split())
            if not (4 <= len(t.split()) <= 14) or t.lower() in seen:
                continue
            seen.add(t.lower())
            dims = detect_dimensions(t)
            if NEG_RE.search(t) and "negation" not in dims: dims.append("negation")
            if TIME_RE.search(t) and "deadline" not in dims: dims.append("deadline")
            if dims:
                pool.append((t, dims))
    rng.shuffle(pool)
    counts, seeds = Counter(), []
    for t, dims in pool:
        d_pick = min(dims, key=lambda x: counts[x])
        if counts[d_pick] >= per_dim:
            continue
        counts[d_pick] += 1
        deva = has_deva(t)
        seeds.append({"text_roman": to_roman(t) if deva else t, "text_deva": t if deva else to_deva(t),
                      "language_form": "deva" if deva else ("hinglish" if any(canon(x[1]) in EN_SYNTAX for x in tokenize(t)) else "roman"),
                      "source": "COMI-LINGUA", "detected_dimensions": dims})
    return seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/seeds.jsonl")
    ap.add_argument("--force_fallback", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    seeds = []
    if not args.force_fallback:
        try:
            seeds = try_hf(rng)
            print("HF COMI-LINGUA seeds extracted: %d" % len(seeds))
            for i, s in enumerate(seeds):
                s["seed_id"] = "H%04d" % i
            if len(seeds) < 300:
                print("HF yielded <300 seeds; topping up with built-in seeds.")
                extra = build_builtin(rng)[: 300 - len(seeds)]
                seeds += extra
        except Exception as e:
            print("HF load failed (%s: %s) -> using built-in 300-seed fallback" % (type(e).__name__, str(e)[:120]))
            seeds = []
    if not seeds:
        seeds = build_builtin(rng)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for s in seeds:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")
    print("Wrote %d seeds -> %s" % (len(seeds), args.out))
    c = Counter(d for s in seeds for d in s["detected_dimensions"])
    print("Per-dimension seed count:", {d: c[d] for d in DIMS})
    print("Per-form seed count:", dict(Counter(s["language_form"] for s in seeds)))
    print("Per-source seed count:", dict(Counter(s["source"] for s in seeds)))


if __name__ == "__main__":
    main()
