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
NLEX = 7


def lex_feats(a, b):
    """Overlap statistics of target token ids a and candidate token ids b (no specials):
    n-gram Jaccard n = 1..4, share of a's / b's 2-grams found in the other, log length ratio."""
    g = lambda t, n: {tuple(t[x:x + n]) for x in range(len(t) - n + 1)}
    jac = lambda x, y: len(x & y) / max(1, len(x | y))
    a2, b2 = g(a, 2), g(b, 2)
    return torch.tensor([*(jac(g(a, n), g(b, n)) for n in (1, 2, 3, 4)),
                         len(a2 & b2) / max(1, len(a2)), len(a2 & b2) / max(1, len(b2)),
                         float(np.log(max(1, len(a)) / max(1, len(b))))])
BOS, EOS, PAD = 0, 2, 1


class PairCE(nn.Module):
    def __init__(self, model_dir, init="newscopy", freeze=4, dropout=0.1, rel=False, lex=False):
        super().__init__()
        self.rel, self.lex = rel, lex
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
        # relational source head (--rel): the score of edge i <- j also sees the reverse edge
        # j <- i, i's other candidates and j's other targets (means, so any pool size works).
        # Copying is relative: "C is close to A" means little until compared with how close B is.
        if rel:
            self.rel_head = nn.Sequential(nn.Linear(5 * H, H), nn.GELU(), nn.Dropout(dropout), nn.Linear(H, 1))
        # lexical edge features (--lex, needs --rel): the token baseline's document-level
        # overlap statistics, computed from the same token ids the encoder reads, added to the
        # edge score through a small MLP over the edge, its reverse and the two neighbour means
        if lex:
            self.lex_head = nn.Sequential(nn.Linear(4 * NLEX, 64), nn.GELU(), nn.Linear(64, 1))
        # decision thresholds on p(dim1), p(dim2), chosen on the validation split for F1
        # after training (calibrate); 0.5 until then
        self.register_buffer("thr", torch.tensor([0.5, 0.5]))

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
        lex = torch.zeros(B, N, N - 1, NLEX, device=ids.device)
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
                    lex[b, i, k] = lex_feats(ti[1:-1].tolist(), tj[:-1].tolist())
        P = B * N * (N - 1)
        return seq.view(P, 2 * L), att.view(P, 2 * L), match.view(P, 2 * L), lex

    def forward_all(self, ids, mask):
        B, N, L = ids.shape
        seq, att, match, lex = self.build(ids, mask)
        emb = self.enc.embeddings.word_embeddings(seq) + self.match(match)
        pos = torch.cumsum(att, 1) * att + PAD                # MPNet's padding-aware position ids
        h = self.drop(self.enc(inputs_embeds=emb, position_ids=pos, attention_mask=att).last_hidden_state)
        if self.rel:
            pair = self.rel_score(h[:, 0].view(B, N, N - 1, -1), lex if self.lex else None).float()
        else:
            pair = self.pair(h[:, 0]).view(B, N, N - 1).float()
        tok = h[:, :L]                                        # target's own positions
        has = self.has(tok).squeeze(-1).view(B, N, N - 1, L).float()
        same = self.same(tok).squeeze(-1).view(B, N, N - 1, L).float()
        src = torch.cat([pair, self.null.float().expand(B, N, 1)], -1)                  # [B,3,3]
        p_src = src.softmax(-1)[..., :2].unsqueeze(-1)
        ph, ps = has.sigmoid(), same.sigmoid()
        p1 = (p_src * ph).sum(2); p2 = (p_src * ph * (1 - ps)).sum(2)
        return dict(src=src, has=has, same=same, p=torch.stack([p1, p2], -1))

    @staticmethod
    def edges(c):
        """[B,N,N-1,D] slot layout -> [B,N,N,D] with E[:, i, j] = edge i <- j (diagonal 0)."""
        B, N, _, D = c.shape
        E = c.new_zeros(B, N, N, D)
        for i in range(N):
            for k, j in enumerate(x for x in range(N) if x != i):
                E[:, i, j] = c[:, i, k]
        return E

    def rel_score(self, c, lex=None):
        """c [B,N,N-1,H] pair <s> states (slot k of target i = k-th other doc) -> [B,N,N-1]."""
        N = c.shape[1]
        nb = lambda E: ((E.sum(2, keepdim=True) - E) / max(1, N - 2),   # i's other candidates
                        (E.sum(1, keepdim=True) - E) / max(1, N - 2))   # j's other targets
        E = self.edges(c); rev = E.transpose(1, 2); tgt, cand = nb(E)
        s = self.rel_head(torch.cat([E, rev, E - rev, tgt, cand], -1)).squeeze(-1)
        if lex is not None:
            X = self.edges(lex.to(c.dtype)); tx, cx = nb(X)
            s = s + self.lex_head(torch.cat([X, X.transpose(1, 2), tx, cx], -1)).squeeze(-1)
        return torch.stack([torch.stack([s[:, i, j] for j in range(N) if j != i], -1) for i in range(N)], 1)

    def forward(self, ids, mask):
        p = self.forward_all(ids, mask)["p"].clamp(EPS, 1 - EPS)
        t = self.thr.clamp(EPS, 1 - EPS)
        return (torch.log(p) - torch.log1p(-p)) - (torch.log(t) - torch.log1p(-t))


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


@torch.no_grad()
def calibrate(model, dl, device):
    """Threshold on p(dim1) that maximises dim1 F1 on dl (the validation split). dim2 stays
    at 0.5 (a val-tuned dim2 threshold of 0.625 lowered dim2 F1 on both held-out sets);
    t2 >= t1 keeps dim2 => dim1."""
    model.eval(); P, Y = [], []
    for ids, mask, y, _ in dl:
        with torch.autocast("cuda", enabled=(device == "cuda")):
            P.append(model.forward_all(ids.to(device), mask.to(device))["p"].float().cpu().numpy())
        Y.append(y.numpy())
    P, Y = np.concatenate(P), np.concatenate(Y)
    grid = np.linspace(0.05, 0.95, 37); thr = []
    for d in (0,):
        v = Y[..., d] != -100; p, g = P[..., d][v], Y[..., d][v]
        f = [2 * ((p > t) & (g == 1)).sum() / max(1, 2 * ((p > t) & (g == 1)).sum() + ((p > t) & (g == 0)).sum()
                                                   + ((p <= t) & (g == 1)).sum()) for t in grid]
        thr.append(float(grid[int(np.argmax(f))]))
    thr.append(max(0.5, thr[0]))
    model.thr.copy_(torch.tensor(thr))
    return thr


def rank_loss(pair, src, margin=1.0):
    """Same-source ranking: for every source j, each true edge i <- j must outscore each
    non-edge i' <- j by a margin. Says only that copies of j look more like j than other
    documents do, which also holds in real news (a wire story and its many reprints).
    pair [B,N,N-1] edge logits, src [B,N] true slot (N-1 = none)."""
    B, N, _ = pair.shape
    E = PairCE.edges(pair.unsqueeze(-1)).squeeze(-1)                  # [B,N,N] i <- j
    Y = PairCE.edges(F.one_hot(src, N)[..., :N - 1].unsqueeze(-1).float()).squeeze(-1)
    off = ~torch.eye(N, dtype=torch.bool, device=pair.device)
    pos = (Y > 0) & off; neg = (Y == 0) & off                          # [B,i,j]
    # pairs (i, i') sharing source j: d = s(i <- j) - s(i' <- j)
    d = E.unsqueeze(2) - E.unsqueeze(1)                                # [B,i,i',j]
    m = pos.unsqueeze(2) & neg.unsqueeze(1)
    return F.softplus(margin - d[m]).mean() if m.any() else pair.sum() * 0


def anti_penalty(pair, pools=None):
    """Mean squared direction gap (s(i <- j) - s(j <- i))^2 over a pool's edges. Keeps the two
    directions of a pair apart only as far as the training signal insists: where nothing in
    the text says who copied whom (Set B: A vs its rewrite Bf), both directions stay close
    and an F1-calibrated threshold keeps both, instead of a confident guess that may flip
    under another writer (it did: DeepSeek Set B dim1 0.60 vs the lexical baseline's 0.69)."""
    E = PairCE.edges(pair.unsqueeze(-1)).squeeze(-1)
    N = E.shape[1]; off = ~torch.eye(N, dtype=torch.bool, device=pair.device)
    G = ((E - E.transpose(1, 2)) ** 2)[:, off]
    if pools is not None:                   # only these pools (Set B in --w-anti-b)
        if not pools.any(): return pair.sum() * 0
        G = G[pools]
    return G.mean()


def weighted_losses(o, y, src, has, same, w_same0, w_dim2, w_rank=0.0, w_anti=0.0, w_anti_b=0.0, set_b=None):
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
    if w_rank:
        out["rank"] = rank_loss(o["src"][..., :-1], src)
        out["total"] = out["total"] + w_rank * out["rank"]
    if w_anti:
        out["anti"] = anti_penalty(o["src"][..., :-1])
        out["total"] = out["total"] + w_anti * out["anti"]
    if w_anti_b:
        # Set B pairs A / Bf are the same story with some values changed: same length, same
        # wording, so the text does not say which one is the rewrite. Set A keeps its direction.
        out["anti_b"] = anti_penalty(o["src"][..., :-1], set_b)
        out["total"] = out["total"] + w_anti_b * out["anti_b"]
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
    ap.add_argument("--rel", action="store_true", help="relational source head")
    ap.add_argument("--lex", action="store_true", help="lexical edge features in the relational head")
    ap.add_argument("--w-rank", type=float, default=0.0, help="weight of the same-source ranking loss")
    ap.add_argument("--w-anti", type=float, default=0.0, help="weight of the direction-gap penalty")
    ap.add_argument("--w-anti-b", type=float, default=0.0, help="direction-gap penalty on Set B pools only")
    ap.add_argument("--init-ckpt", help="start from a trained PairCE state dict (non-strict)")
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
    model = PairCE(args.model, init=args.init, freeze=args.freeze, rel=args.rel, lex=args.lex).to(device)
    if args.init_ckpt:
        miss = model.load_state_dict(torch.load(args.init_ckpt, map_location=device)["model"], strict=False)
        print("init from", args.init_ckpt, miss, flush=True)
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
            set_b = torch.tensor([pairs[k]["set"] == "B" for k in kk], device=device)
            ls = weighted_losses(o, y, src, has, same, args.w_same0, args.w_dim2, args.w_rank, args.w_anti,
                                 args.w_anti_b, set_b)
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

    ck = torch.load(out / "best.pt", map_location=device)
    model.load_state_dict(ck["model"])
    ck["thr"] = calibrate(model, dl_val, device); ck["model"] = model.state_dict()
    torch.save(ck, out / "best.pt")
    print("thresholds (val, F1)", ck["thr"], flush=True)
    te = {}
    if split["test"]:                       # data without test events (e.g. pool_train): use eval_heldout.py
        te = evaluate(model, dl_test, slices, device)
        te["src"] = eval_src(model, dl_test, pairs, src_all, device)
        te["rank_b_over_c"] = rank_b_over_c(model, z, pairs, docs_role, split["test"], device)
    json.dump({"best_val_score": best[0], "best_epoch": best[1], "thr": ck["thr"], "test": te, "args": vars(args),
               "seconds": round(time.time() - start, 1)},
              open(out / f"summary_seed{args.seed}.json", "w"), indent=1, ensure_ascii=False)
    print("TEST", json.dumps({k: v for k, v in te.items() if k not in ("slices",)}), flush=True)
    print("DONE", best, flush=True)


if __name__ == "__main__":
    main()
