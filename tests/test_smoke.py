"""Step 25: smoke tests.  Run:  python -m pytest tests/test_smoke.py -q
Torch tests use a tiny randomly-initialised encoder + a toy WordPiece vocab, so no model download is needed."""
import json, os, sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def _tiny(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import BertConfig, BertTokenizerFast
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + list("abcdefghijklmnopqrstuvwxyz") + ["refund", "mat", "karna"]
    vf = tmp_path / "vocab.txt"
    vf.write_text("\n".join(vocab), encoding="utf-8")
    tok = BertTokenizerFast(vocab_file=str(vf), do_lower_case=True)
    cfg = BertConfig(vocab_size=len(vocab), hidden_size=32, num_hidden_layers=2, num_attention_heads=2, intermediate_size=64)
    return torch, tok, cfg


def test_forward_shapes(tmp_path):
    torch, tok, cfg = _tiny(tmp_path)
    from meaningguard.model import MeaningGuardModel
    ids = torch.randint(5, 20, (4, 16))
    for binary_only in (False, True):
        m = MeaningGuardModel(config=cfg, binary_only=binary_only)
        o = m(ids, torch.ones_like(ids), torch.zeros_like(ids), torch.tensor([0, 1, 0, 1]), torch.tensor([0, 1, 2, 3]))
        assert o["logits_a"].shape == (4, 2)
        assert (o["logits_b"] is None) if binary_only else (o["logits_b"].shape == (4, 7))
        assert o["loss"].dim() == 0


def test_dataset_and_predict_keys(tmp_path):
    torch, tok, cfg = _tiny(tmp_path)
    from meaningguard.dataset import MeaningPairDataset
    from meaningguard.model import MeaningGuardModel
    from meaningguard.predict import MeaningGuardPredictor
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data/processed/test.jsonl"), encoding="utf-8")][:2]
    ds = MeaningPairDataset(rows, tok, max_len=128)
    item = ds[0]
    assert item["input_ids"].shape[0] == 128 and set(item) >= {"input_ids", "attention_mask", "token_type_ids", "label_a", "label_b"}
    pred = MeaningGuardPredictor(MeaningGuardModel(config=cfg), tok)
    for r in rows:
        out = pred(r["s1"], r["s2"])
        assert set(out) == {"label", "dimension", "confidence"}
        assert out["label"] in ("preserved", "changed") and 0.0 <= out["confidence"] <= 1.0


def test_generators_are_single_dimension():
    import random
    from generate_variants import ALTER, changed_dims
    rng = random.Random(0)
    cases = {"negation": "Refund mat karna", "quantity": "500 rupaye bhejo", "deadline": "Report kal tak bhejo",
             "intent": "Form bharo", "entity": "Rahul ko report bhejo"}
    for dim, text in cases.items():
        outs = ALTER[dim](text, "roman", rng)
        assert outs, dim
        for op, t2 in outs:
            assert changed_dims(text, t2) == {dim}, (dim, op, t2)


def test_no_seed_leak():
    p = os.path.join(ROOT, "data/processed")
    if not os.path.exists(os.path.join(p, "train.jsonl")):
        pytest.skip("run the data pipeline first")
    seeds = {sp: {json.loads(l)["seed_id"] for l in open(os.path.join(p, sp + ".jsonl"), encoding="utf-8")} for sp in ("train", "dev", "test")}
    assert not (seeds["train"] & seeds["dev"]) and not (seeds["train"] & seeds["test"]) and not (seeds["dev"] & seeds["test"])
