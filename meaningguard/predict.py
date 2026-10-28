#!/usr/bin/env python
"""Step 23: inference API.   is_meaning_preserved(s1, s2) -> {"label", "dimension", "confidence"}
Loads checkpoints/muril_base/best.pt.

CLI:  python -m meaningguard.predict "Refund mat karna" "Refund karna"
"""
import sys
import torch
from transformers import AutoTokenizer

from meaningguard.dataset import ID2DIM, ID2LAB
from meaningguard.model import MeaningGuardModel

_CACHE = {}


class MeaningGuardPredictor:
    def __init__(self, model, tokenizer, max_len=128, device="cpu"):
        self.model, self.tok, self.max_len, self.device = model.to(device).eval(), tokenizer, max_len, device

    @classmethod
    def from_checkpoint(cls, path="checkpoints/muril_base/best.pt", device=None):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ck = torch.load(path, map_location=device)
        cfg = ck["config"]
        model = MeaningGuardModel(cfg["base_model"], cfg["binary_only"], cfg["lambda_dim"])
        model.load_state_dict(ck["state_dict"])
        return cls(model, AutoTokenizer.from_pretrained(cfg["base_model"]), cfg["max_len"], device)

    @torch.no_grad()
    def __call__(self, s1, s2):
        enc = self.tok(s1, s2, truncation=True, max_length=self.max_len, return_tensors="pt")
        enc = {k: v.to(self.device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        o = self.model(enc["input_ids"], enc["attention_mask"], enc.get("token_type_ids"))
        pa = torch.softmax(o["logits_a"].float(), -1)[0]
        lab = int(pa.argmax())
        dim = "none" if o["logits_b"] is None or lab == 0 else ID2DIM[int(o["logits_b"][0].argmax())]
        return {"label": ID2LAB[lab], "dimension": dim, "confidence": float(pa[lab])}


def is_meaning_preserved(s1, s2, checkpoint="checkpoints/muril_base/best.pt"):
    if checkpoint not in _CACHE:
        _CACHE[checkpoint] = MeaningGuardPredictor.from_checkpoint(checkpoint)
    return _CACHE[checkpoint](s1, s2)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit('usage: python -m meaningguard.predict "<s1>" "<s2>"')
    print(is_meaning_preserved(sys.argv[1], sys.argv[2]))
