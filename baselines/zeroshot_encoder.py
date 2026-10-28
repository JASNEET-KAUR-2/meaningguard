#!/usr/bin/env python
"""Step 15: zero-shot encoder baseline. MuRIL mean-pooled embeddings of s1 and s2, cosine similarity,
threshold tuned on dev to maximise macro-F1; predict 'changed' if sim < threshold. Binary only (dimension = none).

Run:  python baselines/zeroshot_encoder.py   -> predictions/encoder_zeroshot_{dev,test}.jsonl
"""
import argparse, json, os
import numpy as np
import torch
from sklearn.metrics import f1_score
from transformers import AutoModel, AutoTokenizer


@torch.no_grad()
def embed(texts, tok, model, device, bs=32):
    out = []
    for i in range(0, len(texts), bs):
        enc = tok(texts[i:i + bs], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
        h = model(**enc).last_hidden_state
        m = enc["attention_mask"].unsqueeze(-1).float()
        out.append(((h * m).sum(1) / m.sum(1)).cpu())
    return torch.cat(out)


def sims(rows, tok, model, device):
    a = embed([r["s1"] for r in rows], tok, model, device)
    b = embed([r["s2"] for r in rows], tok, model, device)
    return torch.nn.functional.cosine_similarity(a, b).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="google/muril-base-cased")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="predictions")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModel.from_pretrained(a.model).to(device).eval()
    load = lambda s: [json.loads(l) for l in open(os.path.join(a.data_dir, s + ".jsonl"), encoding="utf-8") if l.strip()]
    dev, test = load("dev"), load("test")
    sd, st = sims(dev, tok, model, device), sims(test, tok, model, device)
    y = ["changed" if r["label"] == "changed" else "preserved" for r in dev]
    best_t, best_f = None, -1
    for t in np.unique(np.round(sd, 4)):
        pred = ["changed" if s < t else "preserved" for s in sd]
        f = f1_score(y, pred, labels=["preserved", "changed"], average="macro", zero_division=0)
        if f > best_f:
            best_t, best_f = float(t), f
    print("threshold tuned on dev: %.4f (dev macro-F1 %.4f)" % (best_t, best_f))
    os.makedirs(a.out_dir, exist_ok=True)
    for name, rows, s in (("dev", dev, sd), ("test", test, st)):
        with open(os.path.join(a.out_dir, "encoder_zeroshot_%s.jsonl" % name), "w", encoding="utf-8") as f:
            for r, v in zip(rows, s):
                lab = "changed" if v < best_t else "preserved"
                f.write(json.dumps({"pair_id": r["pair_id"], "label": lab, "dimension": "none", "confidence": float(abs(v - best_t))}) + "\n")
    print("wrote predictions/encoder_zeroshot_{dev,test}.jsonl")


if __name__ == "__main__":
    main()
