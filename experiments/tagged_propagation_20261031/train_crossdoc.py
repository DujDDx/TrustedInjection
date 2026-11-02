"""Baseline: the earlier ABC layered cross-encoder, retrained on the P4 tagged pairs.

Model copied from /mnt/hdd4/dujddx/abc_crossdoc_v1_20261027/abc_crossdoc_train.py
(structure doc: 来源检测_ABC分层交叉编码器结构确认_v1.1_20261026):
  embeddings -> MPNet layers 1-8 -> layer 9 -> CrossDoc9 -> layer 10 -> CrossDoc10
              -> layers 11-12 -> token head
All three channels share the backbone; CrossDoc (query = this document, key/value = the
other two) shares parameters across channels, not between depths. No channel embeddings.

What changed for the new data, nothing else:
  - head 768 -> 2 (dim1 copied, dim2 fabricated) instead of 768 -> 1;
  - loss is masked token BCE on both dims (-100 ignored). The old rank/unique losses were
    built on that dataset's support groups, which these labels do not have;
  - pairs.jsonl already holds all 6 channel orders, so no extra permutation sampling.
Kept: backbone, CrossDoc depths and alpha0, warmup (backbone frozen, CrossDoc + head at
1e-4) then full (layers 9-12 at 1e-5), OneCycle, AMP, clip 1.0, effective batch 8.

    python train_crossdoc.py --data p4_pairs --out /mnt/hdd4/dujddx/tagged_crossdoc_p4_20261101
"""
import argparse, collections, json, random, time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, MPNetModel

MAX_LEN = 160
MODEL = "/mnt/hdd4/models/research/NEWS-COPY-wire-clustering-na"


# ----------------------------------------------------------------------------- model
class CrossDoc(nn.Module):
    """Query = current document; Key/Value = the other two documents only."""

    def __init__(self, hidden=768, heads=12, alpha0=0.01, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, dropout=dropout, batch_first=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha0)))

    def forward(self, h, pad):                       # h [B,3,L,H] ; pad [B,3,L] True=pad
        B, N, L, H = h.shape
        u = self.ln(h)
        out = torch.empty_like(h)
        for i in range(N):
            others = [j for j in range(N) if j != i]
            kv = torch.cat([u[:, j] for j in others], dim=1)              # [B,2L,H]
            kv_pad = torch.cat([pad[:, j] for j in others], dim=1)        # [B,2L]
            o, _ = self.attn(u[:, i], kv, kv, key_padding_mask=kv_pad, need_weights=False)
            out[:, i] = h[:, i] + self.alpha * o
        return out


class ABCCrossDoc(nn.Module):
    def __init__(self, model_dir, crossdoc_after=(9, 10), dropout=0.1, n_out=2):
        super().__init__()
        self.backbone = MPNetModel.from_pretrained(model_dir, config=AutoConfig.from_pretrained(model_dir))
        self.layers = self.backbone.encoder.layer
        self.embeddings = self.backbone.embeddings
        self.rel_bias = self.backbone.encoder.relative_attention_bias
        self.cross = nn.ModuleDict({f"cd{d}": CrossDoc(dropout=dropout) for d in crossdoc_after})
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, n_out))
        self.after = tuple(crossdoc_after)

    def train_stage(self, stage):
        """'warmup': only CrossDoc + head trainable; 'full': layers 9-12 too."""
        for p in self.embeddings.parameters(): p.requires_grad = False
        self.rel_bias.weight.requires_grad = False
        for i, layer in enumerate(self.layers):
            for p in layer.parameters(): p.requires_grad = (stage == "full" and i >= 8)
        # frozen parts stay in eval mode so dropout cannot perturb them
        self.backbone.eval()
        self.cross.train(); self.head.train()
        if stage == "full":
            for i, layer in enumerate(self.layers):
                if i >= 8: layer.train()

    def forward(self, input_ids, attention_mask):
        """input_ids [B,3,L], attention_mask [B,3,L] -> logits [B,3,L,2]."""
        return self.head(self.encode(input_ids, attention_mask))

    def encode(self, input_ids, attention_mask):
        """-> hidden states [B,3,L,H] after layer 12 (CrossDoc applied after 9 and 10)."""
        B, N, L = input_ids.shape
        flat_ids = input_ids.reshape(B * N, L)
        flat_mask = attention_mask.reshape(B * N, L)
        h = self.embeddings(input_ids=flat_ids)
        ext = self.backbone.get_extended_attention_mask(flat_mask, flat_ids.shape)
        pos_bias = self.backbone.encoder.compute_position_bias(h)
        x = h
        for i, layer in enumerate(self.layers):
            x = layer(x, ext, None, pos_bias)[0]
            if i + 1 in self.after:
                pad = (flat_mask == 0).reshape(B, N, L)
                x = self.cross[f"cd{i + 1}"](x.reshape(B, N, L, -1), pad).reshape(B * N, L, -1)
        return x.reshape(B, N, L, -1)


# ----------------------------------------------------------------------------- data
def token_slices(data_dir, tok_path, pairs, ids_arr):
    """Per label token, a slice name for the report: '<branch>|<set>|<slot>|<kind>'.

    slot: A, B, Bf, C, or X (a document from another event). kind: trans (B/Bf value
    copied from A's facts), own (B-only fact), rumor_mirror / rumor_plain (B's rumor value,
    and whether C carries the same rumor), c_mirror (C's mirrored rumor value), val.
    Rebuilt from docs.jsonl with the same tokenizer and overlap rule as build_pairs.py.
    """
    tok = Tokenizer.from_file(tok_path)
    docs = {d["doc_id"]: d for d in map(json.loads, open(Path(data_dir) / "docs.jsonl"))}
    rumor_c = collections.defaultdict(set)              # (event, branch) -> facts mirrored in C
    for d in docs.values():
        if d["role"] == "C":
            for sp in d["spans"]:
                if sp["version"] == "rumor": rumor_c[(d["event_id"], d["branch"])].add(sp["fact"])
    cache = {}

    def kinds(did):
        if did in cache: return cache[did]
        d = docs[did]
        enc = tok.encode(d["text"])
        out = [None] * MAX_LEN
        for k, (a, b) in enumerate(enc.offsets[:MAX_LEN]):
            for sp in d["spans"]:
                if a < sp["end"] and b > sp["start"]:
                    if d["role"] in ("B", "Bf"):
                        if sp["fact"] not in d["a_ids"]: kind = "own"
                        elif sp["version"] == "rumor":
                            mir = sp["fact"] in rumor_c[(d["event_id"], d["branch"])]
                            kind = "rumor_mirror" if mir else "rumor_plain"
                        else: kind = "trans"
                    elif d["role"] == "C" and sp["version"] == "rumor": kind = "c_mirror"
                    else: kind = "val"
                    out[k] = kind
                    break
        cache[did] = (enc.ids[:MAX_LEN], out)
        return cache[did]

    sl = np.empty(ids_arr.shape, dtype=object)
    for p, pr in enumerate(pairs):
        for c, did in enumerate(pr["docs"]):
            ids, ks = kinds(did)
            assert list(ids_arr[p, c, :len(ids)]) == ids, f"token mismatch {did}"
            d = docs[did]
            slot = "X" if d["event_id"] != pr["event_id"] else d["role"]
            for t, k in enumerate(ks):
                if k is not None: sl[p, c, t] = f"{pr['branch']}|{pr['set']}|{slot}|{k}"
    return sl


class Pairs(Dataset):
    def __init__(self, z, idx):
        self.z, self.idx = z, idx

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        k = self.idx[i]
        return (torch.from_numpy(self.z["input_ids"][k].astype(np.int64)),
                torch.from_numpy(self.z["attention_mask"][k].astype(np.int64)),
                torch.from_numpy(np.stack([self.z["dim1"][k], self.z["dim2"][k]], -1).astype(np.int64)),
                k)


def masked_bce(logits, y):
    """logits, y [...,2]; y == -100 is ignored. Mean over labelled tokens, per dim, summed."""
    tot = 0.0
    for d in range(2):
        m = y[..., d] != -100
        if m.any():
            tot = tot + F.binary_cross_entropy_with_logits(logits[..., d][m], y[..., d][m].float())
    return tot


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, dl, slices, device):
    model.eval()
    acc = collections.defaultdict(lambda: [0, 0, 0, 0])   # slice -> [n, d1 ok, d2 ok, d1 pred=1]
    conf = {d: collections.Counter() for d in (0, 1)}     # dim -> tp/fp/fn
    loss = 0.0; nb = 0
    for ids, mask, y, ks in dl:
        ids, mask, y = ids.to(device), mask.to(device), y.to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            lg = model(ids, mask)
        lg = lg.float()
        loss += float(masked_bce(lg, y)); nb += 1
        pred = (lg > 0).long().cpu().numpy(); yy = y.cpu().numpy()
        for b, k in enumerate(ks.tolist()):
            for c, t in zip(*np.nonzero(yy[b, :, :, 0] != -100)):
                s = slices[k, c, t]
                a = acc[s]; a[0] += 1
                a[1] += int(pred[b, c, t, 0] == yy[b, c, t, 0])
                a[2] += int(pred[b, c, t, 1] == yy[b, c, t, 1])
                a[3] += int(pred[b, c, t, 0])
                for d in (0, 1):
                    p_, g_ = pred[b, c, t, d], yy[b, c, t, d]
                    conf[d]["tp"] += int(p_ == 1 and g_ == 1)
                    conf[d]["fp"] += int(p_ == 1 and g_ == 0)
                    conf[d]["fn"] += int(p_ == 0 and g_ == 1)
    f1 = {}
    for d, name in ((0, "dim1"), (1, "dim2")):
        c = conf[d]
        f1[name] = 2 * c["tp"] / max(1, 2 * c["tp"] + c["fp"] + c["fn"])
    sl = {s: {"n": a[0], "dim1_acc": round(a[1] / a[0], 4), "dim2_acc": round(a[2] / a[0], 4)}
          for s, a in sorted(acc.items())}
    return {"loss": loss / max(nb, 1), "dim1_f1": round(f1["dim1"], 4), "dim2_f1": round(f1["dim2"], 4),
            "score": round((f1["dim1"] + f1["dim2"]) / 2, 4), "slices": sl}


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="p4_pairs")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="/mnt/hdd4/dujddx/tagged_crossdoc_p4_20261101")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--warmup-epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--lr-head", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="cap train/val pairs (smoke test)")
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
    slices = token_slices(args.data, str(Path(args.model) / "tokenizer.json"), pairs, z["input_ids"])

    dl_val = DataLoader(Pairs(z, split["val"]), batch_size=32, shuffle=False)
    dl_test = DataLoader(Pairs(z, split["test"]), batch_size=32, shuffle=False)

    model = ABCCrossDoc(args.model).to(device)

    def make_opt():
        enc = [p for i, l in enumerate(model.layers) if i >= 8 for p in l.parameters() if p.requires_grad]
        heads = [p for p in list(model.cross.parameters()) + list(model.head.parameters()) if p.requires_grad]
        groups = [{"params": heads, "lr": args.lr_head}]
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
            run = 0.0
            for step, (ids, mask, y, _) in enumerate(dl, 1):
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                with torch.autocast("cuda", enabled=(device == "cuda")):
                    lg = model(ids, mask)
                loss = masked_bce(lg.float(), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()
                run += float(loss)
                if step % 200 == 0:
                    print(f"[{stage}] ep{ep} step{step}/{len(dl)} loss {run / step:.4f}", flush=True)
            ev = evaluate(model, dl_val, slices, device)
            rec = {"stage": stage, "epoch": ep, "train_loss": round(run / max(len(dl), 1), 5),
                   "elapsed_s": round(time.time() - start, 1), **ev}
            log.write(json.dumps(rec) + "\n"); log.flush()
            print("EVAL", json.dumps({k: v for k, v in rec.items() if k != "slices"}), flush=True)
            if ev["score"] > best[0]:
                best = (ev["score"], f"{stage}/{ep}")
                torch.save({"model": model.state_dict(), "stage": stage, "epoch": ep,
                            "val": ev, "args": vars(args)}, out / "best.pt")
            model.train_stage(stage)

    # test once, with the checkpoint chosen on val
    model.load_state_dict(torch.load(out / "best.pt", map_location=device)["model"])
    te = evaluate(model, dl_test, slices, device)
    json.dump({"best_val_score": best[0], "best_at": best[1], "test": te, "args": vars(args),
               "seconds": round(time.time() - start, 1)},
              open(out / f"summary_seed{args.seed}.json", "w"), indent=1, ensure_ascii=False)
    print("TEST", json.dumps({k: v for k, v in te.items() if k != "slices"}), flush=True)
    print("DONE", best, flush=True)


if __name__ == "__main__":
    main()
