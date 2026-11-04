"""dim1 / dim2 of single pair models and of their average, against the token baseline,
with a paired bootstrap over events.

The ensemble averages p(dim1), p(dim2) of the given checkpoints; its dim1 threshold is
chosen on the v0 validation split for F1 (as train_pairce.calibrate does for one model),
dim2 stays at 0.5. Baseline predictions come from baseline_token.py --dump.

    python eval_ensemble.py --ckpts A/best.pt B/best.pt C/best.pt \
        --sets p4_pairs heldout/qwen_pairs heldout/ds_pairs \
        --baseline /mnt/hdd4/dujddx/diag/baseline_preds/lr_tuned_on_train_for_f1 --out ens.json
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
import torch

from train_crossdoc import MODEL
from train_pairce import PairCE


@torch.no_grad()
def probs(model, z, idx, device):
    out = []
    for s in range(0, len(idx), 16):
        b = idx[s:s + 16]
        ids = torch.from_numpy(z["input_ids"][b].astype(np.int64)).to(device)
        mk = torch.from_numpy(z["attention_mask"][b].astype(np.int64)).to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            out.append(model.forward_all(ids, mk)["p"].float().cpu().numpy())
    return np.concatenate(out)                                          # [n,3,L,2]


def counts(pred, y):
    v = y != -100
    p, g = pred[v], y[v]
    return np.array([((p == 1) & (g == 1)).sum(), ((p == 1) & (g == 0)).sum(), ((p == 0) & (g == 1)).sum()])


def f1c(c):
    return 2 * c[0] / max(1, 2 * c[0] + c[1] + c[2])


def best_thr(p, y):
    grid = np.linspace(0.05, 0.95, 37)
    return float(grid[int(np.argmax([f1c(counts((p > t).astype(int), y)) for t in grid]))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--val-data", default="p4_pairs", help="pairs whose val split sets the ensemble threshold")
    ap.add_argument("--baseline", help="prefix of baseline_token.py --dump files")
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out")
    ap.add_argument("--dump", help="directory for per-set predictions of the ensemble")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    data = {}
    for d in dict.fromkeys([a.val_data] + a.sets):
        z = dict(np.load(Path(d) / "token_labels.npz"))
        pairs = [json.loads(l) for l in open(Path(d) / "pairs.jsonl")]
        data[d] = (z, pairs)
    P = collections.defaultdict(dict); thr1 = []
    for c in a.ckpts:
        ck = torch.load(c, map_location=dev)
        m = PairCE(MODEL, rel=ck.get("args", {}).get("rel", False), lex=ck.get("args", {}).get("lex", False)).to(dev)
        m.load_state_dict(ck["model"], strict=False); m.eval()
        thr1.append(float(m.thr[0]))
        z, pairs = data[a.val_data]
        P[c]["val"] = probs(m, z, [i for i, p in enumerate(pairs) if p["split"] == "val"], dev)
        for d in a.sets:
            z, pairs = data[d]
            P[c][d] = probs(m, z, [i for i, p in enumerate(pairs) if p["split"] == "test"], dev)
        del m; torch.cuda.empty_cache()
    z, pairs = data[a.val_data]
    yval = np.stack([z["dim1"], z["dim2"]], -1)[[i for i, p in enumerate(pairs) if p["split"] == "val"]].astype(int)
    ens_val = np.mean([P[c]["val"] for c in a.ckpts], 0)
    t_ens = best_thr(ens_val[..., 0], yval[..., 0])
    systems = {f"seed:{Path(c).parent.name}": (c, t) for c, t in zip(a.ckpts, thr1)}
    systems["ensemble"] = (None, t_ens)
    res = {"thr": {k: v[1] for k, v in systems.items()}}
    rng = np.random.default_rng(0)
    for d in a.sets:
        z, pairs = data[d]
        idx = [i for i, p in enumerate(pairs) if p["split"] == "test"]
        y = np.stack([z["dim1"], z["dim2"]], -1)[idx].astype(int)
        ev = np.array([pairs[i]["event_id"] for i in idx]); evs = sorted(set(ev))
        rows = [np.nonzero(ev == e)[0] for e in evs]
        preds = {}
        for name, (c, t) in systems.items():
            p = np.mean([P[k][d] for k in a.ckpts], 0) if c is None else P[c][d]
            preds[name] = np.stack([(p[..., 0] > t), (p[..., 1] > max(0.5, t))], -1).astype(int)
        if a.baseline:
            b = np.load(f"{a.baseline}_{Path(d).name}.npz")
            assert list(b["idx"]) == idx, "baseline rows differ"
            preds["baseline"] = b["pred"].astype(int)
        if a.dump:
            Path(a.dump).mkdir(parents=True, exist_ok=True)
            pe = np.mean([P[k][d] for k in a.ckpts], 0)
            np.savez_compressed(Path(a.dump) / f"ensemble_{Path(d).name}.npz", idx=np.array(idx),
                                event_id=ev, prob=pe.astype(np.float16), pred=preds["ensemble"].astype(np.int8),
                                thr=np.array([t_ens, max(0.5, t_ens)]))
        # per-event counts so a bootstrap sample is a sum of rows
        C = {n: {dim: np.array([counts(pr[r][..., dim], y[r][..., dim]) for r in rows]) for dim in (0, 1)}
             for n, pr in preds.items()}
        out = {}
        for n in preds:
            out[n] = {f"dim{dim + 1}_f1": round(f1c(C[n][dim].sum(0)), 4) for dim in (0, 1)}
        if a.baseline:
            samp = rng.integers(0, len(evs), (a.boot, len(evs)))
            for n in preds:
                if n == "baseline": continue
                for dim in (0, 1):
                    diff = np.array([f1c(C[n][dim][s].sum(0)) - f1c(C["baseline"][dim][s].sum(0)) for s in samp])
                    out[n][f"dim{dim + 1}_vs_baseline"] = {
                        "diff": round(out[n][f"dim{dim + 1}_f1"] - out["baseline"][f"dim{dim + 1}_f1"], 4),
                        "ci95": [round(float(np.quantile(diff, 0.025)), 4), round(float(np.quantile(diff, 0.975)), 4)],
                        "p_le_0": round(float((diff <= 0).mean()), 4)}
        res[d] = out
        print(f"\n== {d}: {len(evs)} events")
        for n, o in out.items():
            s = f"  {n:45s} dim1 {o['dim1_f1']:.3f}  dim2 {o['dim2_f1']:.3f}"
            if "dim1_vs_baseline" in o:
                v = o["dim1_vs_baseline"]
                s += f"  | dim1 - baseline {v['diff']:+.3f}  95% CI [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]"
            print(s, flush=True)
    print("thresholds", res["thr"])
    if a.out: json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
