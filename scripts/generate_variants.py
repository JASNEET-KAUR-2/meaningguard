#!/usr/bin/env python
"""Step 4: rule-based contrast-pair generation (NO LLM calls).

For every seed in data/seeds.jsonl produce up to
  * 2 meaning-PRESERVING variants (deterministic: transliteration, spelling / casing / punctuation normalisation)
  * 2 meaning-ALTERING variants (minimal edit on exactly ONE dimension, edit_operation recorded)
Each pair is self-checked with a rule-based feature extractor: the intended dimension must change and no
other dimension may change (preserving pairs: nothing may change). Failed pairs are dropped.

This file also hosts the shared lexicon / feature extractor reused by extract_seeds.py and validate_dataset.py.

Run:  python scripts/generate_variants.py
"""
import argparse, json, os, random, re, sys
from collections import Counter, defaultdict

# --------------------------------------------------------------------------------------
# Lexicon (Roman <-> Devanagari).  One line per word:  roman|devanagari
# --------------------------------------------------------------------------------------
_LEX_TEXT = """
ko|को
ka|का
ki|की
ke|के
se|से
me|में
par|पर
ab|अब
sirf|सिर्फ़
zaroor|ज़रूर
jaldi|जल्दी
please|प्लीज़
kripya|कृपया
hai|है
hain|हैं
tha|था
aap|आप
hum|हम
mujhe|मुझे
sabhi|सभी
yeh|यह
kuch|कुछ
subah|सुबह
shaam|शाम
the|द
by|बाय
to|टू
for|फॉर
with|विद
anyone|एनीवन
tomorrow|टुमॉरो
today|टुडे
kal|कल
aaj|आज
parso|परसों
abhi|अभी
tak|तक
mat|मत
nahi|नहीं
don't|डोंट
do|डू
not|नॉट
ek|एक
teen|तीन
paanch|पाँच
das|दस
sau|सौ
rupaye|रुपये
paise|पैसे
refund|रिफंड
payment|पेमेंट
account|अकाउंट
order|ऑर्डर
delivery|डिलीवरी
ticket|टिकट
form|फॉर्म
fees|फीस
report|रिपोर्ट
file|फाइल
mail|मेल
message|मैसेज
otp|ओटीपी
password|पासवर्ड
link|लिंक
card|कार्ड
bill|बिल
loan|लोन
seat|सीट
booking|बुकिंग
package|पैकेज
invoice|इनवॉइस
document|दस्तावेज़
photo|फोटो
copy|कॉपी
wallet|वॉलेट
number|नंबर
status|स्टेटस
karo|करो
karna|करना
bhejo|भेजो
bhejna|भेजना
jama|जमा
batao|बताओ
batana|बताना
dena|देना
lena|लेना
kholo|खोलो
kholna|खोलना
likho|लिखो
likhna|लिखना
bharo|भरो
bharna|भरना
dekho|देखो
dekhna|देखना
rakho|रखो
rakhna|रखना
kar|कर
diya|दिया
liya|लिया
bhej|भेज
bata|बता
khol|खोल
likh|लिख
bhar|भर
dekh|देख
rakh|रख
send|सेंड
share|शेयर
submit|सबमिट
update|अपडेट
approve|अप्रूव
check|चेक
pay|पे
delete|डिलीट
cancel|कैंसिल
transfer|ट्रांसफर
book|बुक
rahul|राहुल
priya|प्रिया
amit|अमित
neha|नेहा
rohan|रोहन
sneha|स्नेहा
vikas|विकास
pooja|पूजा
sbi|एसबीआई
hdfc|एचडीएफसी
icici|आईसीआईसीआई
axis|एक्सिस
pnb|पीएनबी
paytm|पेटीएम
phonepe|फोनपे
flipkart|फ्लिपकार्ट
amazon|अमेज़न
swiggy|स्विगी
zomato|ज़ोमैटो
"""
R2D = {}
for _l in _LEX_TEXT.strip().splitlines():
    _r, _d = _l.split("|")
    R2D[_r] = _d
DEVA2ROM = {d: r for r, d in R2D.items()}
# Devanagari spelling variants that map back to the same canonical word
DEVA_VARIANTS = {"नहीं": "नही", "में": "मे", "ज़रूर": "जरूर", "प्लीज़": "प्लीज", "पाँच": "पांच", "रुपये": "रूपये"}
for _canon_d, _var in DEVA_VARIANTS.items():
    DEVA2ROM[_var] = DEVA2ROM[_canon_d]
ROMAN_SPELL = {"nahi": "nahin", "nahin": "nahi", "kripya": "please", "please": "kripya", "zaroor": "zarur", "zarur": "zaroor"}
SPELL_CANON = {"nahin": "nahi", "nahee": "nahi", "nhi": "nahi", "zarur": "zaroor", "dont": "don't", "don’t": "don't"}

PERSONS = ["rahul", "priya", "amit", "neha", "rohan", "sneha", "vikas", "pooja"]
BANKS = ["sbi", "hdfc", "icici", "axis", "pnb"]
PLATFORMS = ["paytm", "phonepe", "flipkart", "amazon", "swiggy", "zomato"]
ENT_CATEGORY = {**{e: "person" for e in PERSONS}, **{e: "bank" for e in BANKS}, **{e: "platform" for e in PLATFORMS}}
CATEGORY_MEMBERS = {"person": PERSONS, "bank": BANKS, "platform": PLATFORMS}
DISPLAY = {"rahul": "Rahul", "priya": "Priya", "amit": "Amit", "neha": "Neha", "rohan": "Rohan", "sneha": "Sneha",
           "vikas": "Vikas", "pooja": "Pooja", "sbi": "SBI", "hdfc": "HDFC", "icici": "ICICI", "axis": "Axis",
           "pnb": "PNB", "paytm": "Paytm", "phonepe": "PhonePe", "flipkart": "Flipkart", "amazon": "Amazon",
           "swiggy": "Swiggy", "zomato": "Zomato", "otp": "OTP"}

NEG_MARK = {"mat", "nahi", "na", "don't"}   # plus the bigram "do not" (matched via 'not' after 'do')
NUM_WORDS = {"ek": 1, "teen": 3, "paanch": 5, "das": 10, "sau": 100, "one": 1, "two": 2, "three": 3, "five": 5, "ten": 10}
NUM_CHOICES = ["ek", "teen", "paanch", "das", "sau"]
TIME_TOK = {"kal", "aaj", "parso", "abhi", "tak", "tomorrow", "today", "by"}
IMP = {"karo": "karna", "bhejo": "bhejna", "batao": "batana", "kholo": "kholna", "likho": "likhna",
       "bharo": "bharna", "dekho": "dekhna", "rakho": "rakhna"}
COMPLETED = {"karo": ("kar", "diya"), "bhejo": ("bhej", "diya"), "batao": ("bata", "diya"), "kholo": ("khol", "diya"),
             "likho": ("likh", "diya"), "bharo": ("bhar", "diya"), "dekho": ("dekh", "liya"), "rakho": ("rakh", "diya")}
VERBS = set(IMP) | set(IMP.values()) | {"dena", "lena", "dijiye"}
EN_SYNTAX = {"please", "the", "by", "to", "don't", "do", "not", "tomorrow", "today", "send", "share", "submit",
             "update", "approve", "check", "pay", "delete", "cancel", "for", "with", "anyone"}
DIMS = ["negation", "quantity", "deadline", "intent", "entity"]

# --------------------------------------------------------------------------------------
# Tokenisation / script helpers
# --------------------------------------------------------------------------------------
_TOK_RE = re.compile(r'^([“"(\[]*)(.*?)([.,?!;:”")\]।]*)$')
_DEVA_RE = re.compile(r"[\u0900-\u097F]")


def has_deva(s):
    return bool(_DEVA_RE.search(s))


def tokenize(text):
    toks = []
    for p in text.split():
        m = _TOK_RE.match(p)
        lead, core, trail = m.groups()
        if core == "":
            lead, core, trail = "", p, ""
        toks.append([lead, core, trail])
    return toks


def detok(toks):
    return " ".join(l + c + t for l, c, t in toks)


def canon(core):
    if core in DEVA2ROM:
        return DEVA2ROM[core]
    c = core.lower()
    return SPELL_CANON.get(c, c)


def surf(c, script, like=None):
    if script == "deva":
        return R2D.get(c, c)
    if c in DISPLAY:
        return DISPLAY[c]
    if like and like[:1].isupper():
        return c[:1].upper() + c[1:]
    return c


def script_of(text):
    return "deva" if has_deva(text) else "roman"


# --------------------------------------------------------------------------------------
# Rule-based feature extractor (dimension oracle used for self-check + validation)
# --------------------------------------------------------------------------------------
def features(text):
    cs = [canon(t[1]) for t in tokenize(text)]
    neg = 0
    for i, c in enumerate(cs):
        if c in NEG_MARK or (c == "not" and i > 0 and cs[i - 1] == "do"):
            neg += 1
    qty = []
    for c in cs:
        if c.isdigit():
            qty.append(int(c))
        elif c in NUM_WORDS:
            qty.append(NUM_WORDS[c])
    ddl = tuple(c for c in cs if c in TIME_TOK)
    ent = tuple(c for c in cs if c in ENT_CATEGORY)
    if "?" in text:
        intent = "question"
    elif any(c in ("diya", "liya") for c in cs):
        intent = "completed"
    else:
        intent = "imperative"
    return {"negation": neg, "quantity": tuple(qty), "deadline": ddl, "intent": intent, "entity": ent}


def changed_dims(s1, s2):
    f1, f2 = features(s1), features(s2)
    return {d for d in DIMS if f1[d] != f2[d]}


def detect_dimensions(text):
    """Dimensions *present* in a sentence (used by extract_seeds.py)."""
    f = features(text)
    toks = [canon(t[1]) for t in tokenize(text)]
    out = []
    if f["negation"]:
        out.append("negation")
    if f["quantity"]:
        out.append("quantity")
    if f["deadline"]:
        out.append("deadline")
    if f["entity"]:
        out.append("entity")
    if any(t in IMP for t in toks) or "?" in text:
        out.append("intent")
    return out


# --------------------------------------------------------------------------------------
# Transliteration (dictionary first, indic-transliteration library as optional fallback)
# --------------------------------------------------------------------------------------
try:
    from indic_transliteration import sanscript  # type: ignore
    _HAVE_SANSCRIPT = True
except Exception:  # pragma: no cover
    _HAVE_SANSCRIPT = False


def to_deva(text):
    toks = tokenize(text)
    for t in toks:
        c = canon(t[1])
        if c in R2D:
            t[1] = R2D[c]
        elif _HAVE_SANSCRIPT and t[1].isalpha() and t[1].isascii():
            t[1] = sanscript.transliterate(t[1].lower(), sanscript.ITRANS, sanscript.DEVANAGARI)
    return detok(toks)


def to_roman(text):
    toks = tokenize(text)
    for i, t in enumerate(toks):
        if t[1] in DEVA2ROM:
            c = DEVA2ROM[t[1]]
            t[1] = DISPLAY.get(c, c)
        elif _HAVE_SANSCRIPT and has_deva(t[1]):
            t[1] = sanscript.transliterate(t[1], sanscript.DEVANAGARI, sanscript.ITRANS).lower()
    if toks and toks[0][1][:1].islower():
        toks[0][1] = toks[0][1][:1].upper() + toks[0][1][1:]
    return detok(toks)


# --------------------------------------------------------------------------------------
# Preserving variants (deterministic, no LLM)
# --------------------------------------------------------------------------------------
def preserve_variants(text, script):
    out = []
    # P1: script transliteration
    t1 = to_roman(text) if script == "deva" else to_deva(text)
    out.append(("transliteration_deva_to_roman" if script == "deva" else "transliteration_roman_to_deva", t1))
    # P2: spelling variant, else casing, else terminal-period toggle
    toks = tokenize(text)
    changed = False
    for t in toks:
        if script == "deva" and t[1] in DEVA_VARIANTS:
            t[1] = DEVA_VARIANTS[t[1]]; changed = True
        elif script == "roman" and t[1].lower() in ROMAN_SPELL:
            new = ROMAN_SPELL[t[1].lower()]
            t[1] = new.capitalize() if t[1][:1].isupper() else new; changed = True
    if changed:
        out.append(("spelling_variant", detok(toks)))
    elif script == "roman" and text != text.lower():
        out.append(("casing_lower", text.lower()))
    elif script == "roman" and text[:1].upper() + text[1:] != text:
        out.append(("casing_title", text[:1].upper() + text[1:]))
    else:
        out.append(("period_toggle", text[:-1] if text.endswith(".") else text + "."))
    return out


# --------------------------------------------------------------------------------------
# Altering variants: one dimension at a time
# --------------------------------------------------------------------------------------
def _recap_first(toks, was_upper):
    if was_upper and toks and toks[0][1][:1].islower() and toks[0][1].isascii():
        toks[0][1] = toks[0][1][:1].upper() + toks[0][1][1:]


def _remove_token(toks, idx):
    lead, core, trail = toks[idx]
    del toks[idx]
    if idx > 0 and trail:
        toks[idx - 1][2] += trail


def _neg_positions(toks):
    cs = [canon(t[1]) for t in toks]
    pos = []
    for i, c in enumerate(cs):
        if c in NEG_MARK:
            pos.append((i, 1))
        elif c == "not" and i > 0 and cs[i - 1] == "do":
            pos.append((i - 1, 2))
    return pos, cs


def alter_negation(text, script, rng):
    out = []
    toks = tokenize(text)
    was_upper = toks[0][1][:1].isupper() if toks else False
    pos, cs = _neg_positions(toks)
    if len(pos) == 1:
        i, n = pos[0]
        a = [list(t) for t in toks]
        for _ in range(n):
            _remove_token(a, i)
        _recap_first(a, was_upper)
        out.append(("neg_delete_marker", detok(a)))
        b = [list(t) for t in toks]
        like = b[i][1]
        for _ in range(n - 1):
            del b[i + 1]
        b[i][1] = surf("zaroor", script, like)
        out.append(("neg_replace_with_zaroor", detok(b)))
    elif len(pos) == 0 and toks and cs[-1] in VERBS:
        for word, op in (("mat", "neg_insert_mat"), ("nahi", "neg_insert_nahi")):
            a = [list(t) for t in toks]
            a.insert(len(a) - 1, ["", surf(word, script), ""])
            out.append((op, detok(a)))
    return out


def alter_quantity(text, script, rng):
    toks = tokenize(text)
    idx = None
    for i, t in enumerate(toks):
        c = canon(t[1])
        if c.isdigit() or c in NUM_WORDS:
            idx = i; break
    if idx is None:
        return []
    c = canon(toks[idx][1])
    out = []
    if c.isdigit():
        n = int(c)
        cands = [("qty_div10", n // 10 if n >= 100 and n % 10 == 0 else None),
                 ("qty_times10", n * 10), ("qty_halve", n // 2 if n >= 4 and n % 2 == 0 else None),
                 ("qty_plus_one", n + 1)]
        seen = {n}
        for op, v in cands:
            if v is None or v in seen or v <= 0:
                continue
            seen.add(v)
            a = [list(t) for t in toks]
            a[idx][1] = str(v)
            out.append((op, detok(a)))
    else:
        others = [w for w in NUM_CHOICES if w != c]
        rng.shuffle(others)
        for w in others[:2]:
            a = [list(t) for t in toks]
            a[idx][1] = surf(w, script, toks[idx][1])
            out.append(("qty_word_swap_%s_to_%s" % (c, w), detok(a)))
    return out


def alter_deadline(text, script, rng):
    toks = tokenize(text)
    was_upper = toks[0][1][:1].isupper() if toks else False
    cs = [canon(t[1]) for t in toks]
    swap = {"kal": "aaj", "aaj": "kal", "parso": "kal", "tomorrow": "today", "today": "tomorrow", "abhi": "kal"}
    out = []
    for i, c in enumerate(cs):
        if c in swap:
            a = [list(t) for t in toks]
            a[i][1] = surf(swap[c], script, toks[i][1])
            out.append(("dl_swap_%s_to_%s" % (c, swap[c]), detok(a)))
            break
    for i, c in enumerate(cs):
        if c in ("tak", "by"):
            a = [list(t) for t in toks]
            _remove_token(a, i)
            _recap_first(a, was_upper)
            out.append(("dl_drop_%s" % c, detok(a)))
            break
    return out


def alter_intent(text, script, rng):
    toks = tokenize(text)
    cs = [canon(t[1]) for t in toks]
    if "?" in text or any(c in ("diya", "liya") for c in cs):
        return []
    idx = next((i for i, c in enumerate(cs) if c in IMP), None)
    if idx is None:
        return []
    like = toks[idx][1]
    imp = cs[idx]
    out = []
    a = [list(t) for t in toks]
    trail = a[idx][2]
    a[idx][1] = surf(IMP[imp], script, like)
    a[idx][2] = ""
    a.insert(idx + 1, ["", surf("hai", script), ""])
    a[-1][2] = a[-1][2].rstrip(".") + "?"
    out.append(("int_imperative_to_question", detok(a)))
    b = [list(t) for t in toks]
    v1, v2 = COMPLETED[imp]
    b[idx][1] = surf(v1, script, like)
    b.insert(idx + 1, ["", surf(v2, script), b[idx][2]])
    b[idx][2] = ""
    out.append(("int_imperative_to_completed", detok(b)))
    return out


def alter_entity(text, script, rng):
    toks = tokenize(text)
    idx = next((i for i, t in enumerate(toks) if canon(t[1]) in ENT_CATEGORY), None)
    if idx is None:
        return []
    c = canon(toks[idx][1])
    pool = [e for e in CATEGORY_MEMBERS[ENT_CATEGORY[c]] if e != c]
    rng.shuffle(pool)
    out = []
    for e in pool[:2]:
        a = [list(t) for t in toks]
        a[idx][1] = surf(e, script, toks[idx][1])
        out.append(("ent_swap_%s_to_%s" % (c, e), detok(a)))
    return out


ALTER = {"negation": alter_negation, "quantity": alter_quantity, "deadline": alter_deadline,
         "intent": alter_intent, "entity": alter_entity}


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="data/seeds.jsonl")
    ap.add_argument("--out", default="data/processed/pairs_raw.jsonl")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_alter", type=int, default=2)
    ap.add_argument("--max_preserve", type=int, default=2)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    seeds = [json.loads(l) for l in open(args.seeds, encoding="utf-8") if l.strip()]
    seeds.sort(key=lambda s: s["seed_id"])
    dim_count = Counter()
    stats = Counter()
    pairs = []
    for s in seeds:
        form = s.get("language_form", "deva" if has_deva(s["text_roman"]) else "roman")
        base = s["text_deva"] if form == "deva" else s["text_roman"]
        script = script_of(base)
        k = 0
        cand_pairs = []
        # --- preserving
        for op, t2 in preserve_variants(base, script)[: args.max_preserve]:
            stats["preserve_attempted"] += 1
            if t2 and t2 != base and not changed_dims(base, t2):
                cand_pairs.append(("preserved", "none", op, t2)); stats["preserve_kept"] += 1
            else:
                stats["preserve_dropped"] += 1
        # --- altering (balance dimensions: rarest first)
        dims = sorted(s["detected_dimensions"], key=lambda d: (dim_count[d], DIMS.index(d)))
        n_alt = 0
        for d in dims:
            if n_alt >= args.max_alter:
                break
            for op, t2 in ALTER[d](base, script, rng):
                if n_alt >= args.max_alter:
                    break
                stats["alter_attempted"] += 1
                if t2 and t2 != base and changed_dims(base, t2) == {d}:
                    cand_pairs.append(("changed", d, op, t2)); n_alt += 1; dim_count[d] += 1
                    stats["alter_kept"] += 1
                else:
                    stats["alter_dropped"] += 1
        for label, dim, op, t2 in cand_pairs:
            s1, s2 = base, t2
            if rng.random() < 0.5:     # remove position bias: random order of the two sentences
                s1, s2 = s2, s1
            k += 1
            pairs.append({"pair_id": "%s_%d" % (s["seed_id"], k), "seed_id": s["seed_id"], "s1": s1, "s2": s2,
                          "language_form": form, "label": label, "dimension": dim,
                          "edit_operation": op, "split": "train"})
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print("Seeds read: %d  -> pairs written: %d  (%s)" % (len(seeds), len(pairs), args.out))
    print("Self-check stats:", dict(stats))
    print("Per-dimension counts:", dict(Counter(p["dimension"] for p in pairs)))
    print("Per-form counts:", dict(Counter(p["language_form"] for p in pairs)))
    print("Per-label counts:", dict(Counter(p["label"] for p in pairs)))


if __name__ == "__main__":
    main()
