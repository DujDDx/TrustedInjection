"""Token-level lexical baselines for the P4 pools, scored exactly like the models.

Same input as the model (plain text, no tags, no fact ids), same factorisation as
train_srcptr.py / train_pairce.py:
    (1) source    which other document in the pool, if any, is target i's parent
    (2) has(t,j)  does the fact of value token t also appear in candidate j
    (3) same(t,j) is t's value unchanged in j
    dim1 = [src = j] * has(t, j);  dim2 = dim1 * (1 - same(t, j))

Two versions, both on word-level lexical features (measure_signal.py helpers):
  rule  src = most 2-gram-similar candidate among those whose content i contains more of
        than the reverse (child = superset), none if 2-gram Jaccard < tau;
        has = share of the 2-grams in a +-5 word window around t found in j >= theta;
        same = t's word occurs in j.
  lr    logistic regressions on the training split: src on ordered pair features,
        has on token + pair features, same on token features. src = argmax, none if
        p < tau; has = p >= theta; same = p >= 0.5.
tau and theta are grid-searched on the training split, once for mean F1 and once for
mean accuracy of dim1/dim2. Word predictions go to subwords through character offsets.

    python baseline_token.py --train p4_pairs --sets p4_pairs heldout/qwen_pairs heldout/ds_pairs \
        --out results_baseline/baseline_token.json
"""
import argparse, collections, json, re
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tokenizers import Tokenizer

from measure_signal import SENT, grams, jac, lcs
from train_crossdoc import MAX_LEN, MODEL, token_slices
from train_srcptr import aux_labels

WORD = re.compile(r"[A-Za-z']+|\d+(?:[.,]\d+)*")
WIN = 5
TOK = str(Path(MODEL) / "tokenizer.json")


# ----------------------------------------------------------------------------- features
class Doc:
    def __init__(self, text, tok):
        ms = list(WORD.finditer(text))
        self.w = [m.group().lower() for m in ms]
        self.wset = set(self.w)
        bounds = [m.end() for m in SENT.finditer(text)]
        self.sid = [sum(b <= m.start() for b in bounds) for m in ms]
        ns = max(self.sid) + 1 if ms else 1
        self.sents = [[w for w, s in zip(self.w, self.sid) if s == k] for k in range(ns)]
        self.sg2 = [grams(s, 2) for s in self.sents]
        self.g = {n: grams(self.w, n) for n in (1, 2, 3, 4)}
        starts = np.array([m.start() for m in ms]); ends = np.array([m.end() for m in ms])
        self.sub = np.full(MAX_LEN, -1)
        for k, (a, b) in enumerate(tok.encode(text).offsets[:MAX_LEN]):
            if b <= a or not ms: continue
            hit = np.nonzero((starts < b) & (ends > a))[0]
            self.sub[k] = hit[0] if len(hit) else int(np.argmin(np.minimum(abs(starts - b), abs(ends - a))))


def pair_feats(i, j):
    """Ordered features of (target i, candidate j)."""
    li, lj = len(i.w), len(j.w)
    c = lcs(i.w, j.w)
    cij = len(i.g[2] & j.g[2])
    sa = lambda x, y: float(np.mean([max((jac(a, b) for b in y.sg2), default=0) for a in x.sg2]))
    return np.array([jac(i.g[1], j.g[1]), jac(i.g[2], j.g[2]), jac(i.g[3], j.g[3]), jac(i.g[4], j.g[4]),
                     c / max(1, li), c / max(1, lj),
                     cij / max(1, len(i.g[2])), cij / max(1, len(j.g[2])),   # i in j, j in i
                     np.log(max(1, li) / max(1, lj)), sa(i, j), sa(j, i)])


PF = ["jac1", "jac2", "jac3", "jac4", "lcs_i", "lcs_j", "cov_i_in_j", "cov_j_in_i", "log_len_ratio",
      "salign_ij", "salign_ji"]


def word_feats(i, j):
    """[n_words, 5]: word in j, window 2-gram coverage without / with t, best sentence
    2-gram Jaccard, word in the best-aligned sentence of j."""
    out = np.zeros((max(1, len(i.w)), 5))
    for p, w in enumerate(i.w):
        lo, hi = max(0, p - WIN), min(len(i.w) - 1, p + WIN)
        bg = [(q, (i.w[q], i.w[q + 1])) for q in range(lo, hi)]
        ctx = [g for q, g in bg if q not in (p - 1, p)]
        inc = [g for q, g in bg if q in (p - 1, p)]
        sims = [jac(i.sg2[i.sid[p]], s) for s in j.sg2]
        best = int(np.argmax(sims)) if sims else 0
        out[p] = [w in j.wset,
                  np.mean([g in j.g[2] for g in ctx]) if ctx else 0.0,
                  np.mean([g in j.g[2] for g in inc]) if inc else 0.0,
                  max(sims) if sims else 0.0,
                  w in set(j.sents[best]) if j.sents else 0.0]
    return out


def load(data_dir, tok):
    d = Path(data_dir)
    z = dict(np.load(d / "token_labels.npz"))
    pairs = [json.loads(l) for l in open(d / "pairs.jsonl")]
    docs = {r["doc_id"]: r for r in map(json.loads, open(d / "docs.jsonl"))}
    src, has, same = aux_labels(data_dir, TOK, pairs)
    D = {k: Doc(v["text"], tok) for k, v in docs.items()}
    cache = {}
    P = len(pairs)
    pf = np.zeros((P, 3, 2, len(PF)), np.float32); tf = np.zeros((P, 3, 2, MAX_LEN, 5), np.float32)
    for p, pr in enumerate(pairs):
        for i in range(3):
            for k, j in enumerate([j for j in range(3) if j != i]):
                key = (pr["docs"][i], pr["docs"][j])
                if key not in cache:
                    a, b = D[key[0]], D[key[1]]
                    wf = word_feats(a, b)
                    cache[key] = (pair_feats(a, b), np.where((a.sub >= 0)[:, None], wf[np.maximum(a.sub, 0)], 0))
                pf[p, i, k], tf[p, i, k] = cache[key]
    y = np.stack([z["dim1"], z["dim2"]], -1).astype(np.int64)            # [P,3,L,2]
    slices = token_slices(data_dir, TOK, pairs, z["input_ids"])
    roles = [[docs[x]["role"] if docs[x]["event_id"] == pr["event_id"] else "X" for x in pr["docs"]] for pr in pairs]
    return dict(pairs=pairs, src=src, has=has, same=same, pf=pf, tf=tf, y=y, slices=slices, roles=roles,
                split=np.array([pr["split"] for pr in pairs]))


# ----------------------------------------------------------------------------- scoring
def rule_scores(S):
    sc = S["pf"][..., PF.index("jac2")].copy()
    child = S["pf"][..., PF.index("cov_j_in_i")] > S["pf"][..., PF.index("cov_i_in_j")]
    return dict(src=np.where(child, sc, -np.inf), rank=S["pf"][..., PF.index("jac2")],
                has=S["tf"][..., 1], same=S["tf"][..., 0] > 0)


def lr_fit(S, m):
    pf, tf = S["pf"][m], S["tf"][m]
    src = S["src"][m]
    ks = np.arange(2)[None, None, :]
    X = pf.reshape(-1, len(PF)); Y = (src[..., None] == ks).reshape(-1)
    f_src = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(X, Y)
    hl = S["has"][m]; sel = hl != -100                                     # [n,3,2,L]
    Xh = np.concatenate([tf[sel], np.broadcast_to(pf[:, :, :, None, :], hl.shape + (len(PF),))[sel]], -1)
    f_has = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xh, hl[sel])
    sl = S["same"][m]; sel2 = sl != -100
    f_same = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tf[sel2], sl[sel2])
    return f_src, f_has, f_same


def lr_scores(S, fs):
    f_src, f_has, f_same = fs
    pf, tf = S["pf"], S["tf"]
    ps = f_src.predict_proba(pf.reshape(-1, len(PF)))[:, 1].reshape(pf.shape[:3])
    ph = np.zeros(tf.shape[:4], np.float32); pm = np.zeros(tf.shape[:4], bool)
    for a in range(0, len(tf), 1000):                                       # chunks: memory
        t, f = tf[a:a + 1000], pf[a:a + 1000]
        Xh = np.concatenate([t, np.broadcast_to(f[:, :, :, None, :], t.shape[:4] + (len(PF),))], -1)
        ph[a:a + 1000] = f_has.predict_proba(Xh.reshape(-1, Xh.shape[-1]))[:, 1].reshape(t.shape[:4])
        pm[a:a + 1000] = f_same.predict_proba(t.reshape(-1, 5))[:, 1].reshape(t.shape[:4]) >= 0.5
    return dict(src=ps, rank=ps, has=ph, same=pm)


def predict(sc, tau, theta):
    s = sc["src"]
    k = np.argmax(s, -1); best = np.take_along_axis(s, k[..., None], -1)[..., 0]
    src = np.where(best >= tau, k, 2)                                      # [P,3]
    kk = np.minimum(src, 1)[..., None, None]
    h = np.take_along_axis(sc["has"], kk, 2)[:, :, 0] >= theta              # [P,3,L]
    m = np.take_along_axis(sc["same"], kk, 2)[:, :, 0]
    d1 = h & (src != 2)[..., None]
    return src, np.stack([d1, d1 & ~m], -1).astype(np.int64)


def metrics(pred, y, sel):
    out = {}
    for d, name in ((0, "dim1"), (1, "dim2")):
        g = y[sel][..., d]; p = pred[sel][..., d]; v = g != -100
        g, p = g[v], p[v]
        tp = int(((p == 1) & (g == 1)).sum()); fp = int(((p == 1) & (g == 0)).sum()); fn = int(((p == 0) & (g == 1)).sum())
        out[f"{name}_f1"] = round(2 * tp / max(1, 2 * tp + fp + fn), 4)
        out[f"{name}_acc"] = round(float((p == g).mean()), 4)
        out[f"{name}_acc_all0"] = round(float((g == 0).mean()), 4)
    out["score"] = round((out["dim1_f1"] + out["dim2_f1"]) / 2, 4)
    out["acc"] = round((out["dim1_acc"] + out["dim2_acc"]) / 2, 4)
    return out


def tune(sc, S, sel, obj):
    s = sc["src"][sel]; s = s[np.isfinite(s)]
    taus = np.unique(np.quantile(s, np.linspace(0, 1, 41))) if s.size else [0.0]
    thetas = np.linspace(0, 1, 21)
    best = None
    sub = {k: v[sel] for k, v in sc.items()}
    for tau in taus:
        for th in thetas:
            _, pr = predict(sub, tau, th)
            r = metrics(pr, S["y"][sel], slice(None))[obj]
            if best is None or r > best[0]: best = (r, float(tau), float(th))
    return best


def report(sc, S, sel, tau, theta):
    src, pred = predict(sc, tau, theta)
    out = metrics(pred, S["y"], sel)
    acc = collections.defaultdict(lambda: [0, 0])
    for p in np.nonzero(sel)[0]:
        pr = S["pairs"][p]
        for i in range(3):
            g = S["src"][p, i]
            key = f"{pr['branch']}|{pr['set']}|{'none' if g == 2 else 'has_source'}"
            acc[key][0] += 1; acc[key][1] += int(src[p, i] == g)
    out["src"] = {k: round(c / n, 4) for k, (n, c) in sorted(acc.items())}
    rk = collections.defaultdict(list)
    for p in np.nonzero(sel)[0]:
        pr = S["pairs"][p]
        if pr["set"] != "A": continue
        r = S["roles"][p]; ia, ib, ic = r.index("A"), r.index("B"), r.index("C")
        sco = lambda i: sc["rank"][p, i, [j for j in range(3) if j != i].index(ia)]
        rk[pr["branch"]].append(1.0 if sco(ib) > sco(ic) else 0.5 if sco(ib) == sco(ic) else 0.0)
    out["rank_b_over_c"] = {k: round(float(np.mean(v)), 4) for k, v in rk.items()} | \
                           {"all": round(float(np.mean(sum(rk.values(), []))), 4)}
    sl = collections.defaultdict(lambda: [0, 0, 0])
    for p, i, t in zip(*np.nonzero((S["y"][..., 0] != -100) & sel[:, None, None])):
        a = sl[S["slices"][p, i, t]]; a[0] += 1
        a[1] += int(pred[p, i, t, 0] == S["y"][p, i, t, 0]); a[2] += int(pred[p, i, t, 1] == S["y"][p, i, t, 1])
    out["slices"] = {k: {"n": n, "dim1_acc": round(a / n, 4), "dim2_acc": round(b / n, 4)}
                     for k, (n, a, b) in sorted(sl.items())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="p4_pairs")
    ap.add_argument("--sets", nargs="+", default=["p4_pairs"])
    ap.add_argument("--out", default="results_baseline/baseline_token.json")
    ap.add_argument("--dump", help="dir: save test predictions [P,3,L,2] per version and set")
    ap.add_argument("--versions", nargs="+", default=["rule", "lr"])
    a = ap.parse_args()
    tok = Tokenizer.from_file(TOK)
    T = load(a.train, tok); tr = T["split"] == "train"
    print(f"train {a.train}: {int(tr.sum())} pairs", flush=True)
    fs = lr_fit(T, tr)
    res = {"features": PF, "versions": {}}
    sets = {s: (T if s == a.train else load(s, tok)) for s in a.sets}
    for ver in a.versions:
        sc_tr = rule_scores(T) if ver == "rule" else lr_scores(T, fs)
        for obj in ("score", "acc"):
            r, tau, th = tune(sc_tr, T, tr, obj)
            name = f"{ver}|tuned_on_train_for_{'f1' if obj == 'score' else 'acc'}"
            res["versions"][name] = {"tau": tau, "theta": th, "train": metrics(predict(sc_tr, tau, th)[1], T["y"], tr)}
            print(f"\n== {name}: tau {tau:.4f} theta {th:.2f}  train {obj} {r:.4f}", flush=True)
            for s, S in sets.items():
                sel = S["split"] == "test"
                sc = sc_tr if S is T else (rule_scores(S) if ver == "rule" else lr_scores(S, fs))
                o = report(sc, S, sel, tau, th)
                if a.dump:
                    Path(a.dump).mkdir(parents=True, exist_ok=True)
                    np.savez_compressed(Path(a.dump) / f"{name.replace('|', '_')}_{Path(s).name}.npz",
                                        idx=np.nonzero(sel)[0], pred=predict(sc, tau, th)[1][sel].astype(np.int8))
                res["versions"][name][s] = o
                print(f"  {s:20s} dim1 F1 {o['dim1_f1']:.3f} acc {o['dim1_acc']:.3f} (all-0 {o['dim1_acc_all0']:.3f})  "
                      f"dim2 F1 {o['dim2_f1']:.3f} acc {o['dim2_acc']:.3f} (all-0 {o['dim2_acc_all0']:.3f})  "
                      f"B>C {o['rank_b_over_c']['all']:.3f}", flush=True)
                print("    src " + "  ".join(f"{k} {v:.2f}" for k, v in o["src"].items()), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
