"""Where do a pair model's dim1 errors come from? Report only.

dim1 = sum_j p_src(i<-j) * has(t, j). Each dim1 error on a test token is charged to the
pointer (argmax source is wrong) or to has (pointer right, has wrong). Then each part is
replaced by the truth in turn, which gives the dim1 F1 a perfect pointer / a perfect has
would reach with the other part unchanged.

    python diag_dim1.py --ckpt /mnt/hdd4/dujddx/tagged_pairce_match_p4_20261102/best.pt \
        --sets p4_pairs heldout/qwen_pairs heldout/ds_pairs
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
import torch

from train_crossdoc import MODEL
from train_pairce import PairCE
from train_srcptr import aux_labels

TOK = str(Path(MODEL) / "tokenizer.json")


@torch.no_grad()
def dump(model, z, idx, device):
    out = collections.defaultdict(list)
    for s in range(0, len(idx), 16):
        b = idx[s:s + 16]
        ids = torch.from_numpy(z["input_ids"][b].astype(np.int64)).to(device)
        mk = torch.from_numpy(z["attention_mask"][b].astype(np.int64)).to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            o = model.forward_all(ids, mk)
        out["src"].append(o["src"].float().softmax(-1).cpu().numpy())
        out["has"].append(o["has"].float().sigmoid().cpu().numpy())
        out["p1"].append(o["p"][..., 0].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in out.items()}


def f1(pred, gold):
    tp = int(((pred == 1) & (gold == 1)).sum()); fp = int(((pred == 1) & (gold == 0)).sum())
    fn = int(((pred == 0) & (gold == 1)).sum())
    return round(2 * tp / max(1, 2 * tp + fp + fn), 4)


def compose(src_p, has_p):
    """src_p [n,3,3] probs (2 = none), has_p [n,3,2,L] -> soft dim1 [n,3,L]."""
    return (src_p[..., :2, None] * has_p).sum(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--save", help="npz prefix for the dumped outputs")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location=dev)
    m = PairCE(MODEL, rel=ck.get("args", {}).get("rel", False), lex=ck.get("args", {}).get("lex", False)).to(dev)
    m.load_state_dict(ck["model"], strict=False); m.eval()
    for d in a.sets:
        z = dict(np.load(Path(d) / "token_labels.npz"))
        pairs = [json.loads(l) for l in open(Path(d) / "pairs.jsonl")]
        idx = [i for i, p in enumerate(pairs) if p["split"] == "test"]
        src, has, _ = aux_labels(d, TOK, pairs)
        src, has = src[idx], has[idx]
        o = dump(m, z, idx, dev)
        if a.save: np.savez_compressed(f"{a.save}_{Path(d).name}.npz", idx=np.array(idx), **o)
        y = z["dim1"][idx].astype(np.int64); v = y != -100
        true_src = np.eye(3)[src]                                       # [n,3,3]
        true_has = np.where(has == 1, 1.0, 0.0)
        pred = (o["p1"] > 0.5).astype(int)
        print(f"\n== {d}: {len(idx)} pairs")
        print(f"  dim1 F1 model {f1(pred[v], y[v])}"
              f"  | true pointer + model has {f1((compose(true_src, o['has']) > 0.5)[v], y[v])}"
              f"  | model pointer + true has {f1((compose(o['src'], true_has) > 0.5)[v], y[v])}")
        # hard attribution of errors
        am = o["src"].argmax(-1)                                        # [n,3]
        ptr_ok = (am == src)[..., None].repeat(y.shape[-1], -1)
        err = (pred != y) & v
        kinds = collections.Counter()
        for g in (0, 1):
            e = err & (y == g)
            kinds[f"{'FN' if g else 'FP'} pointer wrong"] += int((e & ~ptr_ok).sum())
            kinds[f"{'FN' if g else 'FP'} pointer right, has wrong"] += int((e & ptr_ok).sum())
        tot = sum(kinds.values())
        print("  errors " + "  ".join(f"{k} {n} ({n / tot:.0%})" for k, n in kinds.items()))
        # dim1 F1 with the best threshold on this set (upper bound of calibration alone)
        best = max((f1((o["p1"] > t)[v].astype(int), y[v]), t) for t in np.linspace(0.05, 0.95, 19))
        print(f"  dim1 F1 at best threshold on this set {best[0]} (t={best[1]:.2f})")


if __name__ == "__main__":
    main()
