"""Source-pointer head on the baseline backbone (train_crossdoc.py), P4 tagged pairs.

The labels (plan v1.2 §3.3) factor as: for value token t in document i,
    dim1(t) = [i's source j is in the pool] AND [t's fact also appears in j]
    dim2(t) = dim1(t) AND [t's value differs from j's value for the same tag]
so the head predicts those three pieces and composes them:
    p_src(i <- j | none)  softmax over the two other documents and a learned "none"
    has(t, j)             t's fact appears in j
    same(t, j)            j carries the same value for t's tag (only where has = 1)
    dim1 = sum_j p_src(i<-j) has(t,j);  dim2 = sum_j p_src(i<-j) has(t,j) (1 - same(t,j))
dim2 => dim1 holds by construction (U7). Each piece gets its own label, derived from
docs.jsonl: source = A for B in Set A and for Bf in Set B, none otherwise; has/same for
every same-event document pair, so the value comparison learns from the whole dataset.

Backbone, CrossDoc layers, stages, optimiser and data split are unchanged from the
baseline, so the comparison isolates the head.

    python train_srcptr.py --data p4_pairs --out /mnt/hdd4/dujddx/tagged_srcptr_p4_20261101
"""
import argparse, collections, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader

from train_crossdoc import MAX_LEN, MODEL, ABCCrossDoc, Pairs, evaluate, token_slices

EPS = 1e-6


# ----------------------------------------------------------------------------- labels
def aux_labels(data_dir, tok_path, pairs):
    """src [P,3] (0/1 = the first/second other channel, 2 = none);
    has, same [P,3,2,L] int8 with -100 outside value tokens (same also where has = 0)."""
    tok = Tokenizer.from_file(tok_path)
    docs = {d["doc_id"]: d for d in map(json.loads, open(Path(data_dir) / "docs.jsonl"))}
    span_cache = {}

    def tok_spans(did):
        if did not in span_cache:
            d = docs[did]
            out = [None] * MAX_LEN
            for k, (a, b) in enumerate(tok.encode(d["text"]).offsets[:MAX_LEN]):
                for sp in d["spans"]:
                    if a < sp["end"] and b > sp["start"]:
                        out[k] = sp; break
            span_cache[did] = out
        return span_cache[did]

    P = len(pairs)
    src = np.full((P, 3), 2, dtype=np.int64)
    has = np.full((P, 3, 2, MAX_LEN), -100, dtype=np.int8)
    same = np.full((P, 3, 2, MAX_LEN), -100, dtype=np.int8)
    for p, pr in enumerate(pairs):
        ds = [docs[x] for x in pr["docs"]]
        roles = [d["role"] if d["event_id"] == pr["event_id"] else "X" for d in ds]
        for i in range(3):
            others = [j for j in range(3) if j != i]
            if (pr["set"] == "A" and roles[i] == "B") or (pr["set"] == "B" and roles[i] == "Bf"):
                src[p, i] = others.index(roles.index("A"))
            spans = tok_spans(pr["docs"][i])
            for k, j in enumerate(others):
                related = ds[j]["event_id"] == ds[i]["event_id"] and ds[j]["branch"] == ds[i]["branch"]
                jval = {sp["tag"]: sp["value"] for sp in ds[j]["spans"]} if related else {}
                for t, sp in enumerate(spans):
                    if sp is None: continue
                    h = int(related and sp["fact"] in ds[j]["facts"])
                    has[p, i, k, t] = h
                    if h and sp["tag"] in jval:
                        same[p, i, k, t] = int(jval[sp["tag"]] == sp["value"])
    return src, has, same


# ----------------------------------------------------------------------------- model
class SrcPtr(nn.Module):
    def __init__(self, model_dir, hidden=768, heads=12, dropout=0.1):
        super().__init__()
        self.enc = ABCCrossDoc(model_dir, dropout=dropout)
        self.enc.head = nn.Identity()                      # baseline head unused
        self.ln = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(hidden * 4, hidden), nn.GELU(), nn.Dropout(dropout))
        self.has = nn.Linear(hidden, 1)
        self.same = nn.Linear(hidden, 1)
        self.src = nn.Sequential(nn.Linear(hidden * 2, hidden), nn.GELU(), nn.Linear(hidden, 1))
        self.null = nn.Parameter(torch.zeros(()))

    def train_stage(self, stage):
        self.enc.train_stage(stage)
        for m in (self.ln, self.attn, self.mlp, self.has, self.same, self.src): m.train()

    def heads(self):
        return [p for n, p in self.named_parameters() if not n.startswith("enc.backbone")]

    def forward_all(self, ids, mask):
        h = self.enc.encode(ids, mask)                     # [B,3,L,H]
        B, N, L, H = h.shape
        u = self.ln(h)
        pad = mask == 0
        valid = mask.unsqueeze(-1).to(h.dtype)             # [B,3,L,1]
        src_logit, has_logit, same_logit = [], [], []
        for i in range(N):
            s_i, h_i, m_i = [], [], []
            for j in (j for j in range(N) if j != i):
                c, _ = self.attn(u[:, i], u[:, j], u[:, j], key_padding_mask=pad[:, j], need_weights=False)
                hi = h[:, i]
                m = self.mlp(torch.cat([hi, c, hi - c, (hi - c).abs()], -1))   # [B,L,H]
                mean = (m * valid[:, i]).sum(1) / valid[:, i].sum(1).clamp(min=1)
                mx = m.masked_fill(pad[:, i].unsqueeze(-1), -1e4).max(1).values
                s_i.append(self.src(torch.cat([mean, mx], -1)).squeeze(-1))
                h_i.append(self.has(m).squeeze(-1)); m_i.append(self.same(m).squeeze(-1))
            s_i.append(self.null.expand(B))
            src_logit.append(torch.stack(s_i, -1))          # [B,3]
            has_logit.append(torch.stack(h_i, 1))           # [B,2,L]
            same_logit.append(torch.stack(m_i, 1))
        src_logit = torch.stack(src_logit, 1).float()       # [B,3(i),3]
        has_logit = torch.stack(has_logit, 1).float()       # [B,3,2,L]
        same_logit = torch.stack(same_logit, 1).float()
        p_src = src_logit.softmax(-1)[..., :2].unsqueeze(-1)            # [B,3,2,1]
        ph, ps = has_logit.sigmoid(), same_logit.sigmoid()
        p1 = (p_src * ph).sum(2)                                        # [B,3,L]
        p2 = (p_src * ph * (1 - ps)).sum(2)
        return dict(src=src_logit, has=has_logit, same=same_logit, p=torch.stack([p1, p2], -1))

    def forward(self, ids, mask):
        """Logits [B,3,L,2] of the composed probabilities, for the shared evaluate()."""
        p = self.forward_all(ids, mask)["p"].clamp(EPS, 1 - EPS)
        return torch.log(p) - torch.log1p(-p)


def losses(o, y, src, has, same):
    out = {}
    for d, name in ((0, "dim1"), (1, "dim2")):
        m = y[..., d] != -100
        out[name] = F.binary_cross_entropy(o["p"][..., d][m].clamp(EPS, 1 - EPS), y[..., d][m].float())
    out["src"] = F.cross_entropy(o["src"].reshape(-1, 3), src.reshape(-1))
    for name, lab in (("has", has), ("same", same)):
        m = lab != -100
        out[name] = (F.binary_cross_entropy_with_logits(o[name][m], lab[m].float())
                     if m.any() else o[name].sum() * 0)
    out["total"] = out["dim1"] + out["dim2"] + out["src"] + 0.5 * (out["has"] + out["same"])
    return out


@torch.no_grad()
def eval_src(model, dl, pairs, src_all, device):
    """Source-pointer accuracy per '<branch>|<set>|<true source>' (none / other)."""
    model.eval()
    acc = collections.defaultdict(lambda: [0, 0])
    for ids, mask, _, ks in dl:
        with torch.autocast("cuda", enabled=(device == "cuda")):
            pred = model.forward_all(ids.to(device), mask.to(device))["src"].argmax(-1).cpu().numpy()
        for b, k in enumerate(ks.tolist()):
            for i in range(3):
                g = src_all[k, i]
                key = f"{pairs[k]['branch']}|{pairs[k]['set']}|{'none' if g == 2 else 'has_source'}"
                acc[key][0] += 1; acc[key][1] += int(pred[b, i] == g)
    return {k: {"n": n, "acc": round(c / n, 4)} for k, (n, c) in sorted(acc.items())}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="p4_pairs")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="/mnt/hdd4/dujddx/tagged_srcptr_p4_20261101")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed); np.random.seed(args.seed)
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    z = dict(np.load(Path(args.data) / "token_labels.npz"))
    pairs = [json.loads(l) for l in open(Path(args.data) / "pairs.jsonl")]
    split = collections.defaultdict(list)
    for i, p in enumerate(pairs): split[p["split"]].append(i)
    if args.limit:
        rng = random.Random(args.seed)
        for s in split: rng.shuffle(split[s]); split[s] = split[s][: args.limit]
    print({s: len(v) for s, v in split.items()}, flush=True)
    tok_path = str(Path(args.model) / "tokenizer.json")
    slices = token_slices(args.data, tok_path, pairs, z["input_ids"])
    src_all, has_all, same_all = aux_labels(args.data, tok_path, pairs)
    # the composed labels must agree with build_pairs' dim1 wherever the source is known
    print("aux labels: src none/has_source",
          int((src_all == 2).sum()), int((src_all < 2).sum()), flush=True)

    dl_val = DataLoader(Pairs(z, split["val"]), batch_size=32, shuffle=False)
    dl_test = DataLoader(Pairs(z, split["test"]), batch_size=32, shuffle=False)
    model = SrcPtr(args.model).to(device)

    def make_opt():
        enc = [p for i, l in enumerate(model.enc.layers) if i >= 8 for p in l.parameters() if p.requires_grad]
        groups = [{"params": [p for p in model.heads() if p.requires_grad], "lr": args.lr_head}]
        if enc: groups.append({"params": enc, "lr": args.lr_enc})
        return torch.optim.AdamW(groups, weight_decay=args.wd)

    log = open(out / f"train_seed{args.seed}.jsonl", "w")
    best = (-1, None)
    start = time.time()
    for stage, n_ep in (("warmup", args.warmup_epochs), ("full", args.epochs - args.warmup_epochs)):
        if n_ep <= 0: continue
        model.train_stage(stage)
        opt = make_opt()
        g = torch.Generator(); g.manual_seed(args.seed + (stage == "full"))
        dl = DataLoader(Pairs(z, split["train"]), batch_size=args.batch, shuffle=True, generator=g,
                        num_workers=2)
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=[x["lr"] for x in opt.param_groups], total_steps=max(1, len(dl) * n_ep),
            pct_start=0.1, anneal_strategy="linear")
        scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
        for ep in range(n_ep):
            run = collections.Counter()
            for step, (ids, mask, y, ks) in enumerate(dl, 1):
                kk = ks.numpy()
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                src = torch.from_numpy(src_all[kk]).to(device)
                has = torch.from_numpy(has_all[kk].astype(np.int64)).to(device)
                same = torch.from_numpy(same_all[kk].astype(np.int64)).to(device)
                with torch.autocast("cuda", enabled=(device == "cuda")):
                    o = model.forward_all(ids, mask)
                ls = losses(o, y, src, has, same)
                opt.zero_grad(set_to_none=True)
                scaler.scale(ls["total"]).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                for k, v in ls.items(): run[k] += float(v)
                if step % 200 == 0:
                    print(f"[{stage}] ep{ep} step{step}/{len(dl)} " +
                          " ".join(f"{k} {v / step:.4f}" for k, v in run.items()), flush=True)
            ev = evaluate(model, dl_val, slices, device)
            ev["src"] = eval_src(model, dl_val, pairs, src_all, device)
            rec = {"stage": stage, "epoch": ep, "train": {k: round(v / len(dl), 5) for k, v in run.items()},
                   "elapsed_s": round(time.time() - start, 1), **ev}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("EVAL", json.dumps({k: v for k, v in rec.items() if k not in ("slices", "train")}), flush=True)
            if ev["score"] > best[0]:
                best = (ev["score"], f"{stage}/{ep}")
                torch.save({"model": model.state_dict(), "stage": stage, "epoch": ep,
                            "val": ev, "args": vars(args)}, out / "best.pt")
            model.train_stage(stage)

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    te = evaluate(model, dl_test, slices, device)
    te["src"] = eval_src(model, dl_test, pairs, src_all, device)
    json.dump({"best_val_score": best[0], "best_at": best[1], "test": te, "args": vars(args),
               "seconds": round(time.time() - start, 1)},
              open(out / f"summary_seed{args.seed}.json", "w"), indent=1, ensure_ascii=False)
    print("TEST", json.dumps({k: v for k, v in te.items() if k != "slices"}), flush=True)
    print("DONE", best, flush=True)


if __name__ == "__main__":
    main()
