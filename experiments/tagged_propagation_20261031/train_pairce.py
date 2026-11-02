"""Pairwise cross-encoder for the P4 tagged pools (v0 data unchanged).

The pool question "which document, if any, is i's parent?" is answered from ordered pairs.
For every target i and candidate j in the pool, one MPNet pass reads
    <s> doc_i </s></s> doc_j </s>        (<= 2 x 160 tokens, MPNet allows 512)
with full attention between the two documents at all 12 layers - unlike the baseline,
whose documents only meet in two added cross-attention blocks after layers 9 and 10.
From that pass:
    score(i <- j)      pair head on <s>
    has(t, j), same(t, j)  token heads on i's tokens
and per target a softmax over its two candidates and a learned "none", composed into
dim1/dim2 exactly as train_srcptr.py does (dim2 => dim1 by construction). Labels, losses,
data split, slices and evaluation are those of train_srcptr.py / train_crossdoc.py.

Weights: the encoder starts from the NEWS-COPY wire-clustering MPNet (trained to match
reprinted news, so it already knows copied text), or from the source-pointer checkpoint's
backbone (--init srcptr). Embeddings and the lowest --freeze layers stay frozen; the data
are small (280 training events).

Training sees one of the six channel orders of each pool per epoch (the pair model is
order-free, so the other five are repeats).

    python train_pairce.py --out /mnt/hdd4/dujddx/tagged_pairce_p4_20261102
"""
import argparse, collections, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, MPNetModel

from train_crossdoc import MAX_LEN, MODEL, Pairs, evaluate, token_slices
from train_srcptr import EPS, aux_labels, eval_src

CK_SRC = "/mnt/hdd4/dujddx/tagged_srcptr_p4_20261101/best.pt"
BOS, EOS, PAD = 0, 2, 1


class PairCE(nn.Module):
    def __init__(self, model_dir, init="newscopy", freeze=4, dropout=0.1):
        super().__init__()
        self.enc = MPNetModel.from_pretrained(model_dir, config=AutoConfig.from_pretrained(model_dir))
        if init == "srcptr":
            sd = torch.load(CK_SRC, map_location="cpu")["model"]
            pre = "enc.backbone."
            self.enc.load_state_dict({k[len(pre):]: v for k, v in sd.items() if k.startswith(pre)}, strict=False)
        for p in self.enc.embeddings.parameters(): p.requires_grad = False
        for i, layer in enumerate(self.enc.encoder.layer):
            for p in layer.parameters(): p.requires_grad = i >= freeze
        H = self.enc.config.hidden_size
        self.drop = nn.Dropout(dropout)
        self.pair = nn.Sequential(nn.Linear(H, H), nn.GELU(), nn.Linear(H, 1))
        self.has = nn.Linear(H, 1)
        self.same = nn.Linear(H, 1)
        self.null = nn.Parameter(torch.zeros(()))
        # exact-match feature (as in reading-comprehension readers): added to the input
        # embedding of every token whose id also occurs in the other document of the pair.
        # The same-value head alone stayed at its base rate (0.89 for match and mismatch).
        self.match = nn.Embedding(2, H)
        nn.init.zeros_(self.match.weight)

    def heads(self):
        return [p for n, p in self.named_parameters() if not n.startswith("enc.")]

    @staticmethod
    def build(ids, mask):
        """[B,3,L] pools -> [B*6, 2L] ordered pairs (i, j), i's tokens first at positions 0..L-1,
        plus the exact-match flags of every position."""
        B, N, L = ids.shape
        seq = torch.full((B, N, N - 1, 2 * L), PAD, dtype=ids.dtype, device=ids.device)
        att = torch.zeros_like(seq)
        match = torch.zeros_like(seq)
        lens = mask.sum(-1)                                   # [B,3]
        for b in range(B):
            for i in range(N):
                li = int(lens[b, i])
                for k, j in enumerate(x for x in range(N) if x != i):
                    lj = int(lens[b, j])
                    ti, tj = ids[b, i, :li], ids[b, j, 1:lj]
                    s = torch.cat([ti, ids.new_tensor([EOS]), tj])
                    seq[b, i, k, :len(s)] = s; att[b, i, k, :len(s)] = 1
                    mi = torch.isin(ti, tj) & (ti > EOS)       # ids 0-2 are <s>, <pad>, </s>
                    mj = torch.isin(tj, ti) & (tj > EOS)
                    match[b, i, k, :li] = mi; match[b, i, k, li + 1:len(s)] = mj
        P = B * N * (N - 1)
        return seq.view(P, 2 * L), att.view(P, 2 * L), match.view(P, 2 * L)

    def forward_all(self, ids, mask):
        B, N, L = ids.shape
        seq, att, match = self.build(ids, mask)
        emb = self.enc.embeddings.word_embeddings(seq) + self.match(match)
        pos = torch.cumsum(att, 1) * att + PAD                # MPNet's padding-aware position ids
        h = self.drop(self.enc(inputs_embeds=emb, position_ids=pos, attention_mask=att).last_hidden_state)
        pair = self.pair(h[:, 0]).view(B, N, N - 1).float()
        tok = h[:, :L]                                        # target's own positions
        has = self.has(tok).squeeze(-1).view(B, N, N - 1, L).float()
        same = self.same(tok).squeeze(-1).view(B, N, N - 1, L).float()
        src = torch.cat([pair, self.null.float().expand(B, N, 1)], -1)                  # [B,3,3]
        p_src = src.softmax(-1)[..., :2].unsqueeze(-1)
        ph, ps = has.sigmoid(), same.sigmoid()
        p1 = (p_src * ph).sum(2); p2 = (p_src * ph * (1 - ps)).sum(2)
        return dict(src=src, has=has, same=same, p=torch.stack([p1, p2], -1))

    def forward(self, ids, mask):
        p = self.forward_all(ids, mask)["p"].clamp(EPS, 1 - EPS)
        return torch.log(p) - torch.log1p(-p)


@torch.no_grad()
def rank_b_over_c(model, z, pairs, docs_role, idx, device):
    """Set A: share of pools where P(B <- A) > P(C <- A) (the two-choice view)."""
    model.eval(); ok = collections.defaultdict(list)
    sel = [k for k in idx if pairs[k]["set"] == "A"]
    for s in range(0, len(sel), 16):
        b = sel[s:s + 16]
        ids = torch.from_numpy(z["input_ids"][b].astype(np.int64)).to(device)
        mk = torch.from_numpy(z["attention_mask"][b].astype(np.int64)).to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            ps = model.forward_all(ids, mk)["src"].float().softmax(-1).cpu().numpy()
        for r, k in enumerate(b):
            roles = [docs_role[d] for d in pairs[k]["docs"]]
            ia, ib, ic = roles.index("A"), roles.index("B"), roles.index("C")
            pa = lambda i: ps[r, i, [j for j in range(3) if j != i].index(ia)]
            ok[pairs[k]["branch"]].append(float(pa(ib) > pa(ic)))
    return {k: round(float(np.mean(v)), 4) for k, v in ok.items()} | {"all": round(float(np.mean(sum(ok.values(), []))), 4)}


def weighted_losses(o, y, src, has, same, w_same0, w_dim2):
    """train_srcptr.losses with class weights: 'value differs' is 11% of same-value labels and
    dim2 = 1 is 2% of tokens, so unweighted both heads settle at the majority answer."""
    out = {}
    for d, name in ((0, "dim1"), (1, "dim2")):
        m = y[..., d] != -100
        t = y[..., d][m].float()
        w = torch.where(t > 0, w_dim2, 1.0) if d == 1 else None
        out[name] = F.binary_cross_entropy(o["p"][..., d][m].clamp(EPS, 1 - EPS), t, weight=w)
    out["src"] = F.cross_entropy(o["src"].reshape(-1, 3), src.reshape(-1))
    for name, lab in (("has", has), ("same", same)):
        m = lab != -100
        t = lab[m].float()
        w = torch.where(t > 0, 1.0, w_same0) if name == "same" else None
        out[name] = F.binary_cross_entropy_with_logits(o[name][m], t, weight=w) if m.any() else o[name].sum() * 0
    out["total"] = out["dim1"] + out["dim2"] + out["src"] + 0.5 * (out["has"] + out["same"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="p4_pairs")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="/mnt/hdd4/dujddx/tagged_pairce_p4_20261102")
    ap.add_argument("--init", choices=["newscopy", "srcptr"], default="newscopy")
    ap.add_argument("--freeze", type=int, default=4, help="lowest encoder layers kept frozen")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--select", choices=["val_loss", "f1_ptr"], default="val_loss")
    ap.add_argument("--w-same0", type=float, default=5.0, help="weight of 'value differs' in the same-value loss")
    ap.add_argument("--w-dim2", type=float, default=5.0, help="weight of dim2 positives")
    ap.add_argument("--min-epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8, help="pools per step (x6 pair passes)")
    ap.add_argument("--lr-enc", type=float, default=2e-5)
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
    docs_role = {d["doc_id"]: (d["role"]) for d in map(json.loads, open(Path(args.data) / "docs.jsonl"))}
    split = collections.defaultdict(list)
    for i, p in enumerate(pairs): split[p["split"]].append(i)
    if args.limit:
        rng = random.Random(args.seed)
        for s in split: rng.shuffle(split[s]); split[s] = split[s][: args.limit]
    # one channel order per pool and epoch: group the six permutations of each pool
    groups = collections.defaultdict(list)
    for k in split["train"]:
        p = pairs[k]; groups[(p["branch"], p["set"], tuple(sorted(p["docs"])))].append(k)
    groups = list(groups.values())
    print({s: len(v) for s, v in split.items()}, f"train pools {len(groups)}", flush=True)
    tok_path = str(Path(args.model) / "tokenizer.json")
    slices = token_slices(args.data, tok_path, pairs, z["input_ids"])
    src_all, has_all, same_all = aux_labels(args.data, tok_path, pairs)

    dl_val = DataLoader(Pairs(z, split["val"]), batch_size=16, shuffle=False)
    dl_test = DataLoader(Pairs(z, split["test"]), batch_size=16, shuffle=False)
    model = PairCE(args.model, init=args.init, freeze=args.freeze).to(device)
    enc = [p for p in model.enc.parameters() if p.requires_grad]
    opt = torch.optim.AdamW([{"params": model.heads(), "lr": args.lr_head},
                             {"params": enc, "lr": args.lr_enc}], weight_decay=args.wd)
    steps = args.epochs * ((len(groups) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[args.lr_head, args.lr_enc], total_steps=steps,
                                                pct_start=0.1, anneal_strategy="linear")
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))
    log = open(out / f"train_seed{args.seed}.jsonl", "w")
    best, bad, start = (-1, None), 0, time.time()
    for ep in range(args.epochs):
        model.train()
        rng = random.Random(args.seed * 100 + ep)
        order = [rng.choice(g) for g in groups]; rng.shuffle(order)
        dl = DataLoader(Pairs(z, order), batch_size=args.batch, shuffle=False, num_workers=2)
        run = collections.Counter()
        for step, (ids, mask, y, ks) in enumerate(dl, 1):
            kk = ks.numpy()
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            src = torch.from_numpy(src_all[kk]).to(device)
            has = torch.from_numpy(has_all[kk].astype(np.int64)).to(device)
            same = torch.from_numpy(same_all[kk].astype(np.int64)).to(device)
            with torch.autocast("cuda", enabled=(device == "cuda")):
                o = model.forward_all(ids, mask)
            ls = weighted_losses(o, y, src, has, same, args.w_same0, args.w_dim2)
            opt.zero_grad(set_to_none=True)
            scaler.scale(ls["total"]).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step()
            for k, v in ls.items(): run[k] += float(v)
            if step % 50 == 0:
                print(f"ep{ep} step{step}/{len(dl)} " + " ".join(f"{k} {v / step:.4f}" for k, v in run.items()),
                      flush=True)
        ev = evaluate(model, dl_val, slices, device)
        ev["src"] = eval_src(model, dl_val, pairs, src_all, device)
        ev["rank_b_over_c"] = rank_b_over_c(model, z, pairs, docs_role, split["val"], device)
        rec = {"epoch": ep, "train": {k: round(v / len(dl), 5) for k, v in run.items()},
               "elapsed_s": round(time.time() - start, 1), **ev}
        # select on F1 plus pointer accuracy where a source exists: F1 can sit at 0 for the
        # first epochs while the pointer is already improving
        ptr = np.mean([v["acc"] for k, v in ev["src"].items() if k.endswith("has_source")])
        sel = ev["score"] + float(ptr) if args.select == "f1_ptr" else -ev["loss"]
        rec["select"] = round(sel, 4); rec["ptr_has_source"] = round(float(ptr), 4)
        if sel > best[0]:
            best, bad = (sel, ep), 0
            torch.save({"model": model.state_dict(), "epoch": ep, "val": ev, "args": vars(args)}, out / "best.pt")
        else:
            bad += 1
        log.write(json.dumps(rec) + "\n"); log.flush()
        print("EVAL", json.dumps({k: v for k, v in rec.items() if k not in ("slices", "src")}), flush=True)
        if bad >= args.patience and ep + 1 >= args.min_epochs: print("early stop", flush=True); break

    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    te = {}
    if split["test"]:                       # data without test events (e.g. pool_train): use eval_heldout.py
        te = evaluate(model, dl_test, slices, device)
        te["src"] = eval_src(model, dl_test, pairs, src_all, device)
        te["rank_b_over_c"] = rank_b_over_c(model, z, pairs, docs_role, split["test"], device)
    json.dump({"best_val_score": best[0], "best_epoch": best[1], "test": te, "args": vars(args),
               "seconds": round(time.time() - start, 1)},
              open(out / f"summary_seed{args.seed}.json", "w"), indent=1, ensure_ascii=False)
    print("TEST", json.dumps({k: v for k, v in te.items() if k not in ("slices",)}), flush=True)
    print("DONE", best, flush=True)


if __name__ == "__main__":
    main()
