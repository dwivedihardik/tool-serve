#!/usr/bin/env python
"""
Lambda-sweep eval for a LoRA DPO adapter on nvidia/When2Call (MCQ, loglikelihood scoring),
mirroring what lm-evaluation-harness does for your task:

  for each item: score each of the 4 candidate responses by
      sum of token log-probs of the response given the chat-formatted prompt
  prediction = argmax over the 4 scores   (raw "acc"/macro_f1 basis; --score norm for acc_norm basis)

Input: the per-sample jsonl that lm-eval wrote with --log_samples (samples_*.jsonl).
Each line holds the item under "doc" (prompt, choices, target_index, correct_answer) and,
optionally, the harness's own "resps" (used to verify this script reproduces the harness).

Usage:
  python lambda_sweep_when2call.py \
      --adapter path/to/checkpoint-1044 \
      --samples path/to/samples_when2call.jsonl \
      --lambdas 0,0.5,0.7,0.85,1.0 --bs 8 --out results_sweep

Smoke test first:  add --limit 100
"""
import argparse
import json
import os
import random
import time
from collections import defaultdict

import numpy as np
import torch
from peft import PeftModel
from peft.tuners.lora import LoraLayer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Fixed order of `choices` in your data: index == class id
CLASSES = ["direct", "tool_call", "request_for_info", "cannot_answer"]


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def encode_pair(tok, context, continuation):
    """Same logic as lm-eval's HFLM._encode_pair (trailing whitespace of context moves to the continuation)."""
    n_spaces = len(context) - len(context.rstrip())
    if n_spaces > 0:
        continuation = context[-n_spaces:] + continuation
        context = context[:-n_spaces]
    whole = tok.encode(context + continuation, add_special_tokens=False)
    ctx = tok.encode(context, add_special_tokens=False)
    return len(ctx), whole


def parse_ll(r):
    # resps entries look like [['-122.9375', 'False']]
    x = r[0][0] if isinstance(r[0], (list, tuple)) else r[0]
    return float(x)


def load_items(path, tok, limit=None):
    items = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            d = rec.get("doc", rec)
            for key in ("prompt", "choices", "target_index", "correct_answer"):
                if key not in d:
                    raise KeyError(f"doc is missing '{key}'. Keys present: {list(d.keys())}")
            assert len(d["choices"]) == 4
            assert CLASSES[d["target_index"]] == d["correct_answer"], (
                f"choice order assumption broken: target_index={d['target_index']} "
                f"but correct_answer={d['correct_answer']}")
            it = {
                "prompt": d["prompt"],
                "choices": d["choices"],
                "true": d["target_index"],
                "harness_ll": [parse_ll(r) for r in rec["resps"]] if "resps" in rec else None,
            }
            it["_enc"] = [encode_pair(tok, it["prompt"], ch) for ch in it["choices"]]
            items.append(it)
            if limit and len(items) >= limit:
                break
    return items


def stratified_split(items, val_frac, seed):
    rng = random.Random(seed)
    by = defaultdict(list)
    for i, it in enumerate(items):
        by[it["true"]].append(i)
    val, hold = [], []
    for idxs in by.values():
        rng.shuffle(idxs)
        k = int(len(idxs) * val_frac)
        val += idxs[:k]
        hold += idxs[k:]
    return sorted(val), sorted(hold)


# --------------------------------------------------------------------------- #
# Model + lambda scaling
# --------------------------------------------------------------------------- #
def load_model(base, adapter, load_4bit):
    kw = dict(device_map="auto")
    if load_4bit:
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    else:
        kw["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(base, **kw)
    return PeftModel.from_pretrained(model, adapter).eval()  # keep UNMERGED


def make_lambda_setter(model):
    orig = {n: dict(m.scaling) for n, m in model.named_modules() if isinstance(m, LoraLayer)}
    assert orig, "No LoRA layers found - is the adapter loaded and unmerged?"

    def set_lambda(lam):
        for n, m in model.named_modules():
            if isinstance(m, LoraLayer):
                for k in m.scaling:
                    m.scaling[k] = orig[n][k] * lam
    return set_lambda


# --------------------------------------------------------------------------- #
# Loglikelihood scoring
# --------------------------------------------------------------------------- #
@torch.no_grad()
def loglik(model, tok, items, ids, bs):
    """Returns {item_idx: [ll_choice0..ll_choice3]} (sum of continuation token log-probs)."""
    seqs = [(i, k) for i in ids for k in range(4)]
    seqs.sort(key=lambda s: len(items[s[0]]["_enc"][s[1]][1]))
    out = {i: [0.0] * 4 for i in ids}
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for s in range(0, len(seqs), bs):
        chunk = seqs[s:s + bs]
        L = max(len(items[i]["_enc"][k][1]) for i, k in chunk)
        input_ids = torch.full((len(chunk), L), pad, dtype=torch.long)
        attn = torch.zeros((len(chunk), L), dtype=torch.long)
        for b, (i, k) in enumerate(chunk):
            whole = items[i]["_enc"][k][1]
            input_ids[b, :len(whole)] = torch.tensor(whole)
            attn[b, :len(whole)] = 1
        logits = model(input_ids=input_ids.to(model.device), attention_mask=attn.to(model.device)).logits
        for b, (i, k) in enumerate(chunk):
            c, whole = items[i]["_enc"][k]
            n = len(whole)
            lp = torch.log_softmax(logits[b, c - 1:n - 1].float(), dim=-1)
            tgt = torch.tensor(whole[c:n], device=lp.device)
            out[i][k] = lp.gather(-1, tgt[:, None]).sum().item()
    return out


def predict(items, scores, ids, mode):
    preds = {}
    for i in ids:
        ll = np.array(scores[i])
        if mode == "norm":
            ll = ll / np.array([float(len(c)) for c in items[i]["choices"]])
        preds[i] = int(ll.argmax())
    return preds


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def confusion(items, preds, ids):
    cm = np.zeros((4, 4), dtype=int)
    for i in ids:
        cm[items[i]["true"], preds[i]] += 1
    return cm


def custom_hallucination(items, preds, ids):
    """PASTE YOUR HARNESS HALLUCINATION FUNCTION HERE (return float, or None to skip)."""
    return None


def metrics(cm, items, preds, ids):
    rows, cols = cm.sum(1), cm.sum(0)
    f1s = []
    for k in range(4):
        if rows[k] == 0 and cols[k] == 0:
            continue
        p = cm[k, k] / cols[k] if cols[k] else 0.0
        r = cm[k, k] / rows[k] if rows[k] else 0.0
        f1s.append(2 * p * r / (p + r) if (p + r) else 0.0)

    def rec(k):
        return float(cm[k, k] / rows[k]) if rows[k] else float("nan")

    non_tool = rows[[0, 2, 3]].sum()
    overcall = float(cm[[0, 2, 3], 1].sum() / non_tool) if non_tool else float("nan")
    present = [rec(k) for k in range(4) if rows[k] > 0]
    return {
        "acc": float(np.trace(cm) / cm.sum()),
        "macro_f1": float(np.mean(f1s)),
        "macro_recall": float(np.mean(present)),
        "tool_recall": rec(1),
        "rfi_recall": rec(2),
        "cannot_recall": rec(3),
        "tool_overcall_proxy": overcall,  # PROXY: tool predicted when it shouldn't be; not your harness metric
        "hallucination": custom_hallucination(items, preds, ids),
    }


def print_cm(cm, title):
    print(f"\n{title}")
    print("".ljust(24) + "".join(f"pred:{c}".rjust(24) for c in CLASSES))
    for k, c in enumerate(CLASSES):
        print(f"true:{c}".ljust(24) + "".join(str(v).rjust(24) for v in cm[k]))


def print_table(rows, title):
    cols = ["acc", "macro_f1", "macro_recall", "tool_recall", "rfi_recall", "cannot_recall",
            "tool_overcall_proxy", "hallucination"]
    print(f"\n{title}")
    print("lam".rjust(6) + "".join(c.rjust(21) for c in cols))
    for lam, m in rows:
        print(f"{lam:6.2f}" + "".join(("n/a" if m[c] is None else f"{m[c]:.4f}").rjust(21) for c in cols))


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--samples", required=True, help="lm-eval --log_samples jsonl")
    ap.add_argument("--lambdas", default="0,0.5,0.7,0.85,1.0")
    ap.add_argument("--score", choices=["raw", "norm"], default="raw",
                    help="raw = argmax of summed loglik (acc/macro_f1); norm = divided by char length (acc_norm)")
    ap.add_argument("--val-frac", type=float, default=0.3)
    ap.add_argument("--max-halluc", type=float, default=None)
    ap.add_argument("--bs", type=int, default=8, help="sequences per batch (4 per item); lower if OOM")
    ap.add_argument("--check-n", type=int, default=200, help="items used to verify against harness resps")
    ap.add_argument("--limit", type=int, default=None, help="use only first N items (smoke test)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load-4bit", action="store_true")
    ap.add_argument("--out", default="results_sweep")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    lams = [float(x) for x in args.lambdas.split(",")]

    tok = AutoTokenizer.from_pretrained(args.base)
    items = load_items(args.samples, tok, args.limit)
    val_ids, hold_ids = stratified_split(items, args.val_frac, args.seed)
    print(f"{len(items)} items | val={len(val_ids)} (pick lambda) | holdout={len(hold_ids)} (report)")

    model = load_model(args.base, args.adapter, args.load_4bit)
    set_lambda = make_lambda_setter(model)

    cache = {}  # lam -> {idx: [4 lls]}

    def get_scores(lam, ids):
        set_lambda(lam)
        c = cache.setdefault(lam, {})
        missing = [i for i in ids if i not in c]
        if missing:
            t0 = time.time()
            c.update(loglik(model, tok, items, missing, args.bs))
            print(f"  lambda={lam:.2f}: scored {len(missing)} items in {time.time() - t0:.0f}s")
        return c

    # 0) Verify against the harness's own resps (tells you which of lambda=0 / 1 your samples file came from)
    check_ids = [i for i, it in enumerate(items) if it["harness_ll"] is not None][:args.check_n]
    if check_ids:
        print(f"\nHarness check on {len(check_ids)} items:")
        for lam in (0.0, 1.0):
            s = get_scores(lam, check_ids)
            agree = np.mean([int(np.argmax(s[i])) == int(np.argmax(items[i]["harness_ll"])) for i in check_ids])
            diff = np.mean([np.abs(np.array(s[i]) - np.array(items[i]["harness_ll"])).mean() for i in check_ids])
            print(f"  lambda={lam}: prediction agreement with harness = {agree:.3f}, mean |loglik diff| = {diff:.3f}")
        print("  -> the lambda whose agreement is ~0.99+ is the model your samples file was produced with.\n"
              "     If neither is, your prompt/tokenization/dtype differs from the harness; fix before trusting the sweep.")
    else:
        print("\nNo harness 'resps' in the samples file; skipping the reproduction check.")

    # 1) Choose lambda on validation split
    val_rows = []
    for lam in lams:
        s = get_scores(lam, val_ids)
        preds = predict(items, s, val_ids, args.score)
        val_rows.append((lam, metrics(confusion(items, preds, val_ids), items, preds, val_ids)))
    print_table(val_rows, f"VALIDATION split, score={args.score} (use this to choose lambda)")

    def h(m):
        return m["hallucination"] if m["hallucination"] is not None else m["tool_overcall_proxy"]

    if args.max_halluc is not None:
        ok = [r for r in val_rows if h(r[1]) <= args.max_halluc]
        if not ok:
            print(f"WARNING: no lambda met max-halluc={args.max_halluc}; using best tool_recall overall")
        best_lam = max(ok or val_rows, key=lambda r: r[1]["tool_recall"])[0]
    else:
        best_lam = max(val_rows, key=lambda r: r[1]["macro_f1"])[0]
    print(f"\nChosen lambda (validation): {best_lam}")

    # 2) Report on holdout (never used for selection)
    report = sorted(set([0.0, best_lam, 1.0]) & set(lams))
    hold_rows, hold_cms = [], {}
    for lam in report:
        s = get_scores(lam, hold_ids)
        preds = predict(items, s, hold_ids, args.score)
        cm = confusion(items, preds, hold_ids)
        hold_cms[lam] = cm
        hold_rows.append((lam, metrics(cm, items, preds, hold_ids)))
        with open(os.path.join(args.out, f"holdout_preds_lam{lam}.jsonl"), "w") as f:
            for i in hold_ids:
                f.write(json.dumps({"idx": i, "true": CLASSES[items[i]["true"]],
                                    "pred": CLASSES[preds[i]], "ll": s[i]}) + "\n")
    print_table(hold_rows, f"HOLDOUT split, score={args.score} (report these)")
    for lam in report:
        print_cm(hold_cms[lam], f"HOLDOUT confusion matrix, lambda={lam}")

    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump({"chosen_lambda": best_lam, "score": args.score,
                   "validation": {str(l): m for l, m in val_rows},
                   "holdout": {str(l): m for l, m in hold_rows},
                   "holdout_confusion": {str(l): c.tolist() for l, c in hold_cms.items()}}, f, indent=2)
    print(f"\nSaved to {args.out}/")


if __name__ == "__main__":
    main()