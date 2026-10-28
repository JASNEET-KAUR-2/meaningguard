#!/usr/bin/env python
"""Step 16: translation / normalisation pipeline baseline.  Both sentences are mapped to ONE script (Devanagari),
then compared back: identical normalised strings -> preserved, otherwise changed.  Binary only.

Backend:
  default      : lexicon + indic-transliteration library (fast, no download).  This is the Step-16 fallback.
  --indictrans2: tries ai4bharat/indictrans2-en-indic-dist-200M via IndicTransToolkit.  NOTE: that checkpoint is an
                 English->Indic translator, so it is not a faithful Roman-Hindi->Devanagari transliterator;
                 it is kept only as an optional, best-effort backend and is NOT used for the reported numbers unless you pass the flag.

Run:  python baselines/translation_pipeline.py   -> predictions/translation_pipeline_{dev,test}.jsonl
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from generate_variants import to_deva, tokenize, canon, has_deva, DEVA2ROM, R2D   # noqa: E402


def normalise(text):
    toks = tokenize(text)
    out = []
    for lead, core, trail in toks:
        c = canon(core)                         # canonicalises spelling variants / both scripts
        out.append(R2D.get(c, to_deva(core) if not has_deva(core) else core) + trail.replace(".", ""))
    return " ".join(out).strip()


def indictrans2_factory():
    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    from IndicTransToolkit import IndicProcessor     # may be missing -> caller falls back
    name = "ai4bharat/indictrans2-en-indic-dist-200M"
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(name, trust_remote_code=True).eval()
    ip = IndicProcessor(inference=True)

    def tr(text):
        b = ip.preprocess_batch([text], src_lang="eng_Latn", tgt_lang="hin_Deva")
        enc = tok(b, return_tensors="pt", padding=True)
        with torch.no_grad():
            g = model.generate(**enc, max_length=64, num_beams=1)
        return ip.postprocess_batch(tok.batch_decode(g, skip_special_tokens=True), lang="hin_Deva")[0]
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data/processed"); ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--indictrans2", action="store_true")
    a = ap.parse_args()
    fn, backend = normalise, "lexicon+indic-transliteration"
    if a.indictrans2:
        try:
            fn, backend = indictrans2_factory(), "indictrans2-en-indic-dist-200M"
        except Exception as e:
            print("IndicTrans2 unavailable (%s) -> falling back to transliteration backend" % type(e).__name__)
    print("backend:", backend)
    os.makedirs(a.out_dir, exist_ok=True)
    for split in ("dev", "test"):
        rows = [json.loads(l) for l in open(os.path.join(a.data_dir, split + ".jsonl"), encoding="utf-8") if l.strip()]
        with open(os.path.join(a.out_dir, "translation_pipeline_%s.jsonl" % split), "w", encoding="utf-8") as f:
            for r in rows:
                same = fn(r["s1"]) == fn(r["s2"])
                f.write(json.dumps({"pair_id": r["pair_id"], "label": "preserved" if same else "changed", "dimension": "none", "confidence": 1.0}) + "\n")
    print("wrote predictions/translation_pipeline_{dev,test}.jsonl")


if __name__ == "__main__":
    main()
