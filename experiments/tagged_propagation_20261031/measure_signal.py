"""How much "who copied whom" signal the P4 documents carry, measured with simple text
features and no fine-tuning. Report only.

Unit: one (event, branch) with its four documents A, C, B, Bf. Text is the tagged text
(values as <Tag>), so only wording counts; in the Duplicate branch values are identical
anyway.

  T1  Set A question. Given A, which of B and C was derived from A?
      Rule per feature: "the one more similar to A"; plus a logistic regression on the
      feature differences, 5-fold grouped by event.
  T2  Set B question. Given A and Bf (same facts, Bf rewrote A), which is derived?
      Symmetric similarity cannot answer this, so: logistic regression on per-document
      style features and on asymmetric coverage, both orders, grouped CV.
  T3  Control: is B more similar to A than Bf is? Both derive from A; Bf has no extra
      fact, so this shows how much an inserted fact disturbs the rewrite.

    python measure_signal.py --docs p4_pairs/docs.jsonl [--emb]
"""
import argparse, collections, json, re

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

MODEL = "/mnt/hdd4/models/research/NEWS-COPY-wire-clustering-na"
SENT = re.compile(r"(?<=[.!?])\s+")


def toks(s):
    return [w for w in re.findall(r"<[A-Za-z_]+>|[A-Za-z']+|\d+", s.lower())]


def grams(t, n):
    return {tuple(t[i:i + n]) for i in range(len(t) - n + 1)}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def lcs(a, b):
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def sent_align(x, y):
    """Mean over x's sentences of the best 2-gram Jaccard with any sentence of y."""
    xs = [grams(toks(s), 2) for s in SENT.split(x) if s.strip()]
    ys = [grams(toks(s), 2) for s in SENT.split(y) if s.strip()]
    return float(np.mean([max(jac(a, b) for b in ys) for a in xs]))


def sym_feats(x, y, emb=None):
    tx, ty = toks(x), toks(y)
    f = {f"jac{n}": jac(grams(tx, n), grams(ty, n)) for n in (1, 2, 3, 4)}
    f["lcs"] = lcs(tx, ty) / max(1, min(len(tx), len(ty)))
    f["sent_align"] = (sent_align(x, y) + sent_align(y, x)) / 2
    if emb is not None: f["mpnet_cos"] = float(emb[x] @ emb[y])
    return f


def style(x):
    t = toks(x); ss = [s for s in SENT.split(x) if s.strip()]
    return [len(t), len(ss), len(t) / max(1, len(ss)), len(set(t)) / max(1, len(t)),
            sum(w.startswith("<") for w in t) / max(1, len(t)), x.count(","), x.count(" and ")]


def cover(x, y, n=3):
    gx = grams(toks(x), n)
    return len(gx & grams(toks(y), n)) / max(1, len(gx))


def embed(texts):
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL); m = AutoModel.from_pretrained(MODEL).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m.to(dev)
    out = {}
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = texts[i:i + 64]
            enc = tok(b, padding=True, truncation=True, max_length=256, return_tensors="pt").to(dev)
            h = m(**enc).last_hidden_state
            mk = enc["attention_mask"].unsqueeze(-1)
            v = (h * mk).sum(1) / mk.sum(1)
            v = torch.nn.functional.normalize(v, dim=-1).cpu().numpy()
            out.update(zip(b, v))
    return out


def cv_acc(X, y, groups):
    accs = []
    for tr, te in GroupKFold(5).split(X, y, groups):
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
        clf.fit(X[tr], y[tr]); accs.append(clf.score(X[te], y[te]))
    return float(np.mean(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="p4_pairs/docs.jsonl")
    ap.add_argument("--emb", action="store_true", help="add NEWS-COPY MPNet cosine / embeddings")
    args = ap.parse_args()

    q = collections.defaultdict(dict)
    for d in map(json.loads, open(args.docs)):
        q[(d["event_id"], d["branch"])][d["role"]] = d["text_tagged"]
    units = [(k, v) for k, v in sorted(q.items()) if len(v) == 4]
    print(f"units {len(units)} (event x branch with A, B, C, Bf)")
    emb = embed(sorted({t for _, v in units for t in v.values()})) if args.emb else None

    # ---- T1: B vs C relative to A
    rows, rule = [], collections.defaultdict(list)
    for (eid, br), v in units:
        fb, fc = sym_feats(v["A"], v["B"], emb), sym_feats(v["A"], v["C"], emb)
        for k in fb:
            rule[k].append(1.0 if fb[k] > fc[k] else 0.5 if fb[k] == fc[k] else 0.0)
        rows.append((eid, [fb[k] for k in fb], [fc[k] for k in fb], list(fb)))
    names = rows[0][3]
    print("\nT1  Set A: which of B, C copied A?  (chance 0.50)")
    print(f"  {'feature':12s} {'rule acc':>8s}   mean sim A-B / A-C")
    for i, k in enumerate(names):
        mb = np.mean([r[1][i] for r in rows]); mc = np.mean([r[2][i] for r in rows])
        print(f"  {k:12s} {np.mean(rule[k]):8.3f}   {mb:.3f} / {mc:.3f}")
    X = np.array([np.subtract(r[1], r[2]) for r in rows] + [np.subtract(r[2], r[1]) for r in rows])
    y = np.array([1] * len(rows) + [0] * len(rows))
    g = np.array([r[0] for r in rows] * 2)
    print(f"  {'LR all':12s} {cv_acc(X, y, g):8.3f}   (5-fold, grouped by event)")

    # ---- T2: direction between A and Bf
    X, y, g = [], [], []
    for (eid, br), v in units:
        a, f = v["A"], v["Bf"]
        for first, second, lab in ((a, f, 0), (f, a, 1)):          # lab 1 = first is derived
            feat = list(np.subtract(style(first), style(second)))
            feat += [cover(first, second) - cover(second, first)]
            if emb is not None: feat += list(emb[first] - emb[second])
            X.append(feat); y.append(lab); g.append(eid)
    X, y, g = np.array(X), np.array(y), np.array(g)
    print("\nT2  Set B: A or Bf, which is derived?  (chance 0.50)")
    print(f"  style + coverage      {cv_acc(X[:, :8], y, g):.3f}")
    if emb is not None: print(f"  + MPNet embedding     {cv_acc(X, y, g):.3f}")
    print(f"  mean words A / Bf     {np.mean([len(toks(v['A'])) for _, v in units]):.1f} / "
          f"{np.mean([len(toks(v['Bf'])) for _, v in units]):.1f}")

    # ---- T3: B vs Bf similarity to A
    print("\nT3  similarity to A: B / Bf / C  (3-gram Jaccard, sentence alignment)")
    for k in ("jac3", "sent_align"):
        vals = {r: np.mean([sym_feats(v["A"], v[r])[k] for _, v in units]) for r in ("B", "Bf", "C")}
        print(f"  {k:12s} " + "  ".join(f"{r} {x:.3f}" for r, x in vals.items()))


if __name__ == "__main__":
    main()
