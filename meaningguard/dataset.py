#!/usr/bin/env python
"""Step 8: PyTorch Dataset + DataLoader.  Input = s1 [SEP] s2 (tokenizer pair encoding), max_length=128.
Returns input_ids, attention_mask, token_type_ids, label_a (0 preserved / 1 changed), label_b (0-6 dimension).

Run (smoke):  python -m meaningguard.dataset --file data/processed/dev.jsonl
"""
import argparse, json
import torch
from torch.utils.data import Dataset, DataLoader

DIM2ID = {"none": 0, "negation": 1, "quantity": 2, "deadline": 3, "intent": 4, "entity": 5, "multi": 6}
ID2DIM = {v: k for k, v in DIM2ID.items()}
LAB2ID = {"preserved": 0, "changed": 1}
ID2LAB = {v: k for k, v in LAB2ID.items()}


def read_jsonl(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


class MeaningPairDataset(Dataset):
    def __init__(self, path_or_rows, tokenizer, max_len=128):
        self.rows = read_jsonl(path_or_rows) if isinstance(path_or_rows, str) else list(path_or_rows)
        self.tok, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        enc = self.tok(r["s1"], r["s2"], truncation=True, max_length=self.max_len, padding="max_length", return_tensors="pt")
        item = {k: v.squeeze(0) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
        if "token_type_ids" not in item:          # e.g. XLM-R tokenizer returns none
            item["token_type_ids"] = torch.zeros_like(item["input_ids"])
        item["label_a"] = torch.tensor(LAB2ID[r["label"]], dtype=torch.long)
        item["label_b"] = torch.tensor(DIM2ID[r["dimension"]], dtype=torch.long)
        return item


def make_loader(path_or_rows, tokenizer, batch_size=16, max_len=128, shuffle=False, seed=42):
    ds = MeaningPairDataset(path_or_rows, tokenizer, max_len)
    g = torch.Generator(); g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=g if shuffle else None)


if __name__ == "__main__":
    from transformers import AutoTokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/processed/dev.jsonl")
    ap.add_argument("--base_model", default="google/muril-base-cased")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.base_model)
    batch = next(iter(make_loader(a.file, tok, batch_size=4)))
    print({k: tuple(v.shape) for k, v in batch.items()})
