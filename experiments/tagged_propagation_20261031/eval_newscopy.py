"""Real-news check: does a model trained on the P4 synthetic pools see derivation in real
newspapers? NEWS-COPY test split (historical US newspapers, OCR), report only.

Pairs
  pos    two articles from the same NEWS-COPY cluster (reprints of one wire story)
  hard   anchor + the most 2-gram-similar article of a different cluster on the same
         newspaper day (NEWS-COPY "query")
  easy   anchor + a random article from another day
Each pair is a three-channel pool with a random article from another day as the third
document, like Set B's X slot.

Label caveat: NEWS-COPY positives are siblings of a wire that is not in the pool. Under
plan v1.2 §3.3 such a pair is not "copied" (Set C: source absent -> 0), so this measures
whether the model sees a derivation relation in real text at all, not dim1 exactly.
The backbone (NEWS-COPY-wire-clustering) was trained on NEWS-COPY clustering, so its
plain cosine is an in-domain reference, not a fair rival.

Scores
  jac2       document 2-gram Jaccard (whole article)
  cosine     backbone mean-pooled cosine (first 256 tokens)
  srcptr     max(p_src(1<-2), p_src(2<-1)) from the source-pointer model (first 160 tokens)
  crossdoc   mean dim1 probability over all tokens of both articles, baseline model

    python eval_newscopy.py [--n 600]
"""
import argparse, collections, json, random, re

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModel, AutoTokenizer

from train_crossdoc import MAX_LEN, MODEL, ABCCrossDoc
from train_srcptr import SrcPtr

PARQUET = ("/mnt/hdd4/workspace/TrustedInjection_latent_20261019/data/"
           "news_copy_eval_chenghao_8886f0e/data/test-00000-of-00001.parquet")
CK_SRC = "/mnt/hdd4/dujddx/tagged_srcptr_p4_20261101/best.pt"
CK_BASE = "/mnt/hdd4/dujddx/tagged_crossdoc_p4_20261101/best.pt"


def clean(t):
    t = re.sub(r"-\s*\n\s*", "", t)                  # OCR line-break hyphenation
    return re.sub(r"\s+", " ", t).strip()


def words(t): return re.findall(r"[a-z']+|\d+", t.lower())
def grams(w, n=2): return {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
def jac(a, b): return len(a & b) / max(1, len(a | b))


def build_pairs(n, seed=0):
    d = pd.read_parquet(PARQUET)
    d["text"] = d["article"].map(clean)
    d = d[d["text"].str.split().str.len() >= 40].reset_index(drop=True)
    rng = random.Random(seed)
    g2 = [grams(words(t)) for t in d["text"]]
    by_cluster = collections.defaultdict(list)
    for i, c in enumerate(d["cluster"]): by_cluster[c].append(i)
    by_query = collections.defaultdict(list)
    for i, q in enumerate(d["query"]): by_query[q].append(i)
    multi = [c for c, v in by_cluster.items() if len(v) >= 2]
    rng.shuffle(multi)
    out = []
    for c in multi[:n]:
        a, b = rng.sample(by_cluster[c], 2)
        q = d["query"][a]
        cand = [i for i in by_query[q] if d["cluster"][i] != c]
        if not cand: continue
        hard = max(cand, key=lambda i: jac(g2[a], g2[i]))
        other = [i for i in rng.sample(range(len(d)), 50) if d["query"][i] != q]
        easy, x = other[0], other[1]
        for kind, j in (("pos", b), ("hard", hard), ("easy", easy)):
            out.append(dict(kind=kind, a=d["text"][a], b=d["text"][j], x=d["text"][x],
                            jac2=jac(g2[a], g2[j])))
    return out


@torch.no_grad()
def score(pairs, dev):
    tok = AutoTokenizer.from_pretrained(MODEL)
    sp = SrcPtr(MODEL).to(dev); sp.load_state_dict(torch.load(CK_SRC, map_location=dev)["model"]); sp.eval()
    bm = ABCCrossDoc(MODEL).to(dev); bm.load_state_dict(torch.load(CK_BASE, map_location=dev)["model"]); bm.eval()
    enc = AutoModel.from_pretrained(MODEL).to(dev).eval()
    for s in range(0, len(pairs), 16):
        b = pairs[s:s + 16]
        texts = [t for p in b for t in (p["a"], p["b"], p["x"])]
        e = tok(texts, padding="max_length", truncation=True, max_length=MAX_LEN, return_tensors="pt").to(dev)
        ids = e["input_ids"].view(len(b), 3, MAX_LEN); mask = e["attention_mask"].view(len(b), 3, MAX_LEN)
        with torch.autocast("cuda", enabled=(dev == "cuda")):
            src = sp.forward_all(ids, mask)["src"].float().softmax(-1)      # [B,3(i),3]: others order + none
            lg = bm(ids, mask).float()
        # channel 0 <- 1 is src[:,0,0] (others of 0 = [1,2]); channel 1 <- 0 is src[:,1,0] (others of 1 = [0,2])
        p_src = torch.maximum(src[:, 0, 0], src[:, 1, 0]).cpu().numpy()
        m = mask[:, :2].float()
        d1 = ((lg[:, :2, :, 0].sigmoid() * m).sum((1, 2)) / m.sum((1, 2))).cpu().numpy()
        e2 = tok([t for p in b for t in (p["a"], p["b"])], padding=True, truncation=True, max_length=256,
                 return_tensors="pt").to(dev)
        h = enc(**e2).last_hidden_state; mk = e2["attention_mask"].unsqueeze(-1)
        v = torch.nn.functional.normalize((h * mk).sum(1) / mk.sum(1), dim=-1).view(len(b), 2, -1)
        cos = (v[:, 0] * v[:, 1]).sum(-1).cpu().numpy()
        for k, p in enumerate(b):
            p.update(srcptr=float(p_src[k]), crossdoc=float(d1[k]), cosine=float(cos[k]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600, help="clusters to sample")
    ap.add_argument("--out", default="/mnt/hdd4/dujddx/newscopy_eval_20261102.jsonl")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pairs = build_pairs(a.n)
    print(collections.Counter(p["kind"] for p in pairs))
    score(pairs, dev)
    with open(a.out, "w") as f:
        for p in pairs: f.write(json.dumps({k: v for k, v in p.items() if k not in ("a", "b", "x")}) + "\n")
    print(f"\n{'score':10s} {'pos vs hard':>12s} {'pos vs easy':>12s}   mean pos / hard / easy")
    for s in ("jac2", "cosine", "srcptr", "crossdoc"):
        v = {k: np.array([p[s] for p in pairs if p["kind"] == k]) for k in ("pos", "hard", "easy")}
        auc = lambda neg: roc_auc_score([1] * len(v["pos"]) + [0] * len(v[neg]), np.r_[v["pos"], v[neg]])
        print(f"{s:10s} {auc('hard'):12.3f} {auc('easy'):12.3f}   "
              f"{v['pos'].mean():.3f} / {v['hard'].mean():.3f} / {v['easy'].mean():.3f}")


if __name__ == "__main__":
    main()
