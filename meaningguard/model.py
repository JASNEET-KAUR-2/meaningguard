#!/usr/bin/env python
"""Step 9: MeaningGuardModel = encoder -> [CLS] -> Dropout(0.1) -> head_a (2-way) + head_b (7-way).
binary_only=True skips head_b (ablation A1): loss only on head A.
Loss: L = CE(A) + lambda_dim * CE(B)   (lambda_dim = 0.5)

Run (shape check, no download needed):  python -m meaningguard.model
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, PretrainedConfig, AutoConfig


class MeaningGuardModel(nn.Module):
    def __init__(self, base_model="google/muril-base-cased", binary_only=False, lambda_dim=0.5, dropout=0.1, config=None):
        super().__init__()
        if config is not None:                       # offline / tests: build a randomly initialised encoder from a config
            self.encoder = AutoModel.from_config(config)
        else:
            self.encoder = AutoModel.from_pretrained(base_model)
        hidden = self.encoder.config.hidden_size
        self.binary_only, self.lambda_dim = binary_only, lambda_dim
        self.dropout = nn.Dropout(dropout)
        self.head_a = nn.Linear(hidden, 2)
        self.head_b = None if binary_only else nn.Linear(hidden, 7)

    def forward(self, input_ids, attention_mask, token_type_ids=None, label_a=None, label_b=None):
        kw = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None and getattr(self.encoder.config, "type_vocab_size", 0) > 1:
            kw["token_type_ids"] = token_type_ids
        h = self.encoder(**kw).last_hidden_state[:, 0]          # [CLS] pool
        h = self.dropout(h)
        logits_a = self.head_a(h)
        logits_b = None if self.binary_only else self.head_b(h)
        out = {"logits_a": logits_a, "logits_b": logits_b}
        if label_a is not None:
            loss = F.cross_entropy(logits_a.float(), label_a)
            if not self.binary_only and label_b is not None:
                loss = loss + self.lambda_dim * F.cross_entropy(logits_b.float(), label_b)
            out["loss"] = loss
        return out


if __name__ == "__main__":
    from transformers import BertConfig
    cfg = BertConfig(vocab_size=100, hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    for bo in (False, True):
        m = MeaningGuardModel(config=cfg, binary_only=bo)
        ids = torch.randint(0, 100, (4, 16))
        o = m(ids, torch.ones_like(ids), torch.zeros_like(ids), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 2, 3]))
        print("binary_only=%s" % bo, o["logits_a"].shape, None if o["logits_b"] is None else o["logits_b"].shape, float(o["loss"]))
