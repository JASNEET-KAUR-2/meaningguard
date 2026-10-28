#!/usr/bin/env python
"""Step 14: zero-shot LLM baseline - Qwen/Qwen1.5-1.8B-Chat loaded in 4-bit (bitsandbytes), structured-JSON prompt.
Parse failures default to preserved/none and are logged to predictions/qwen_parse_failures.log.

Run:  python baselines/zeroshot_llm.py      -> predictions/qwen_zeroshot_{dev,test}.jsonl
"""
import argparse, json, os, re

PROMPT = '''You are a meaning-preservation checker for Hinglish/Hindi text.
Compare these two sentences:
S1: "{s1}"
S2: "{s2}"
Answer in JSON: {"label": "preserved"|"changed", "dimension": "none"|"negation"|"quantity"|"deadline"|"intent"|"entity"|"multi"}'''
LABELS = {"preserved", "changed"}
DIMS = {"none", "negation", "quantity", "deadline", "intent", "entity", "multi"}


def build_prompt(s1, s2):
    return PROMPT.replace("{s1}", s1).replace("{s2}", s2)     # str.replace: the template contains literal JSON braces


def parse(text):
    m = re.search(r"\{[^{}]*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except Exception:
        lab = re.search(r'"label"\s*:\s*"(\w+)"', m.group(0)); dim = re.search(r'"dimension"\s*:\s*"(\w+)"', m.group(0))
        d = {"label": lab.group(1) if lab else None, "dimension": dim.group(1) if dim else "none"}
    if d.get("label") not in LABELS:
        return None
    dim = d.get("dimension") if d.get("dimension") in DIMS else "none"
    return d["label"], dim


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen1.5-1.8B-Chat")
    ap.add_argument("--data_dir", default="data/processed")
    ap.add_argument("--out_dir", default="predictions")
    ap.add_argument("--max_new_tokens", type=int, default=40)
    a = ap.parse_args()
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4")
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, quantization_config=bnb, device_map="auto").eval()
    os.makedirs(a.out_dir, exist_ok=True)
    log = open(os.path.join(a.out_dir, "qwen_parse_failures.log"), "w", encoding="utf-8")
    for split in ("dev", "test"):
        rows = [json.loads(l) for l in open(os.path.join(a.data_dir, split + ".jsonl"), encoding="utf-8") if l.strip()]
        fails = 0
        with open(os.path.join(a.out_dir, "qwen_zeroshot_%s.jsonl" % split), "w", encoding="utf-8") as f:
            for i, r in enumerate(rows):
                msgs = [{"role": "user", "content": build_prompt(r["s1"], r["s2"])}]
                ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    gen = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False)
                text = tok.decode(gen[0, ids.shape[1]:], skip_special_tokens=True)
                p = parse(text)
                if p is None:
                    fails += 1; p = ("preserved", "none")
                    log.write("%s\t%s\t%r\n" % (split, r["pair_id"], text))
                f.write(json.dumps({"pair_id": r["pair_id"], "label": p[0], "dimension": p[1], "confidence": 1.0}) + "\n")
                if (i + 1) % 50 == 0:
                    print("%s %d/%d (parse failures so far: %d)" % (split, i + 1, len(rows), fails))
        print("%s done | parse failures: %d/%d (defaulted to preserved/none)" % (split, fails, len(rows)))
    log.close()


if __name__ == "__main__":
    main()
