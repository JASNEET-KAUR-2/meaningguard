#!/usr/bin/env python
"""Step 10: config-driven training (YAML).  AdamW lr 2e-5, 4 epochs, batch 16 (x grad_accum), seed 42,
mixed precision (torch.cuda.amp), early-stopping selection on dev macro-F1, peak GPU memory printed each epoch.

Run:  python -m meaningguard.train --config configs/muril_base.yaml
Outputs:
  checkpoints/<run>/best.pt        (state_dict + config)
  checkpoints/<run>/history.csv    (epoch, train_loss, dev_macro_f1, dev recall per dimension, peak_gpu_gb)
  predictions/<run>_{dev,test}.jsonl   (predictions from best.pt)
Dev macro-F1 = macro-F1 over the dimension classes present in dev (multi-task) or over the 2 labels (binary_only).
"""
import argparse, csv, json, os, random, time
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score
from torch.optim import AdamW
from transformers import AutoTokenizer

from meaningguard.dataset import make_loader, read_jsonl, ID2DIM, ID2LAB
from meaningguard.model import MeaningGuardModel

DIMS5 = ["negation", "quantity", "deadline", "intent", "entity"]


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


@torch.no_grad()
def predict_loader(model, loader, device, use_amp):
    model.eval()
    out = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items()}
        with torch.cuda.amp.autocast(enabled=use_amp):
            o = model(b["input_ids"], b["attention_mask"], b["token_type_ids"])
        pa = torch.softmax(o["logits_a"].float(), -1)
        pb = None if o["logits_b"] is None else torch.softmax(o["logits_b"].float(), -1)
        for i in range(pa.size(0)):
            lab = int(pa[i].argmax())
            if pb is None:
                dim, conf = ("none" if lab == 0 else "none"), float(pa[i, lab])
            else:
                dim = ID2DIM[int(pb[i].argmax())]
                conf = float(pa[i, lab])
            out.append((ID2LAB[lab], dim, conf))
    return out


def score(preds, rows, binary_only):
    if binary_only:
        yt = [r["label"] for r in rows]; yp = [p[0] for p in preds]
        f1 = f1_score(yt, yp, labels=["preserved", "changed"], average="macro", zero_division=0)
        rec = {d: float(np.mean([preds[i][0] == "changed" for i, r in enumerate(rows) if r["dimension"] == d] or [float("nan")])) for d in DIMS5}
        return f1, rec
    yt = [r["dimension"] for r in rows]
    yp = [p[1] if p[0] == "changed" else "none" for p in preds]
    present = sorted(set(yt))
    f1 = f1_score(yt, yp, labels=present, average="macro", zero_division=0)
    rec = {d: float(np.mean([yp[i] == d for i, r in enumerate(rows) if r["dimension"] == d] or [float("nan")])) for d in DIMS5}
    return f1, rec


def write_preds(path, preds, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p, r in zip(preds, rows):
            f.write(json.dumps({"pair_id": r["pair_id"], "label": p[0], "dimension": p[1], "confidence": p[2]}) + "\n")


def run(cfg):
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    tok = AutoTokenizer.from_pretrained(cfg["base_model"])
    tr = make_loader(cfg["train_file"], tok, cfg["batch_size"], cfg["max_len"], shuffle=True, seed=cfg["seed"])
    dv = make_loader(cfg["dev_file"], tok, cfg["batch_size"] * 2, cfg["max_len"])
    dev_rows = read_jsonl(cfg["dev_file"])
    model = MeaningGuardModel(cfg["base_model"], cfg["binary_only"], cfg["lambda_dim"]).to(device)
    opt = AdamW(model.parameters(), lr=float(cfg["lr"]))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    accum = int(cfg.get("grad_accum_steps", 1))
    os.makedirs(cfg["output_dir"], exist_ok=True)
    hist_path = os.path.join(cfg["output_dir"], "history.csv")
    best, best_ep = -1.0, -1
    with open(hist_path, "w", newline="") as hf:
        w = csv.writer(hf)
        w.writerow(["epoch", "train_loss", "dev_macro_f1"] + ["dev_recall_" + d for d in DIMS5] + ["peak_gpu_gb", "seconds"])
        for ep in range(1, cfg["epochs"] + 1):
            t0 = time.time(); model.train(); tot, n = 0.0, 0
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            opt.zero_grad()
            for step, b in enumerate(tr, 1):
                b = {k: v.to(device) for k, v in b.items()}
                try:
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        o = model(b["input_ids"], b["attention_mask"], b["token_type_ids"], b["label_a"], b["label_b"])
                        loss = o["loss"] / accum
                    scaler.scale(loss).backward()
                except torch.cuda.OutOfMemoryError:
                    raise SystemExit("CUDA OOM: raise grad_accum_steps (e.g. 4) and lower batch_size (e.g. 8) in the YAML.")
                if step % accum == 0 or step == len(tr):
                    scaler.step(opt); scaler.update(); opt.zero_grad()
                tot += float(o["loss"]); n += 1
            preds = predict_loader(model, dv, device, use_amp)
            f1, rec = score(preds, dev_rows, cfg["binary_only"])
            peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0
            w.writerow([ep, round(tot / n, 4), round(f1, 4)] + [round(rec[d], 4) for d in DIMS5] + [round(peak, 2), round(time.time() - t0, 1)])
            hf.flush()
            print("epoch %d | train_loss %.4f | dev macro-F1 %.4f | dev recall %s | peak GPU %.2f GB" % (ep, tot / n, f1, {k: round(v, 3) for k, v in rec.items()}, peak))
            if f1 > best:
                best, best_ep = f1, ep
                torch.save({"state_dict": model.state_dict(), "config": cfg}, os.path.join(cfg["output_dir"], "best.pt"))
    print("best dev macro-F1 %.4f at epoch %d" % (best, best_ep))
    # reload best and write predictions for dev + test
    ck = torch.load(os.path.join(cfg["output_dir"], "best.pt"), map_location=device)
    model.load_state_dict(ck["state_dict"])
    for split, key in (("dev", "dev_file"), ("test", "test_file")):
        rows = read_jsonl(cfg[key])
        preds = predict_loader(model, make_loader(cfg[key], tok, cfg["batch_size"] * 2, cfg["max_len"]), device, use_amp)
        write_preds("predictions/%s_%s.jsonl" % (cfg["run_name"], split), preds, rows)
    print("wrote predictions/%s_{dev,test}.jsonl" % cfg["run_name"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    a = ap.parse_args()
    run(yaml.safe_load(open(a.config)))
