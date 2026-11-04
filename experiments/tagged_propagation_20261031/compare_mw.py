"""Multi-writer ensemble against V1 and the token baseline, overall and per set, with
paired event-level bootstrap and a dim1 threshold scan (diagnostic only).

    python compare_mw.py mw5 > results_mw/comparisons/compare_mw5.txt
"""
import json, sys
import numpy as np

SETS = (("p4_pairs", "p4_pairs"), ("heldout/qwen_pairs", "qwen_pairs"), ("heldout/ds_pairs", "ds_pairs"))


def f1c(c):
    return 2 * c[0] / max(2 * c[0] + c[1] + c[2], 1)


def cnt(p, y):
    return np.array([((p == 1) & (y == 1)).sum(), ((p == 1) & (y == 0)).sum(), ((p == 0) & (y == 1)).sum()])


def boot(pa, pb, y, ev, rng, n=2000):
    evs = sorted(set(ev)); rows = [np.nonzero(ev == e)[0] for e in evs]
    a = np.array([cnt(pa[r], y[r]) for r in rows]); b = np.array([cnt(pb[r], y[r]) for r in rows])
    samp = rng.integers(0, len(evs), (n, len(evs)))
    d = np.array([f1c(a[s].sum(0)) - f1c(b[s].sum(0)) for s in samp])
    return f1c(a.sum(0)) - f1c(b.sum(0)), np.quantile(d, .025), np.quantile(d, .975)


def main():
    tag = sys.argv[1]; rng = np.random.default_rng(0)
    for d, name in SETS:
        pairs = [json.loads(l) for l in open(f"{d}/pairs.jsonl")]
        y1 = np.load(f"{d}/token_labels.npz")["dim1"]
        m = np.load(f"results_mw/predictions/{tag}/ensemble_{name}.npz")
        v = np.load(f"results_v1/predictions/ensemble_{name}.npz")
        bm = np.load(f"results_mw/baseline/lr_tuned_on_train_for_f1_{name}.npz")
        bv = np.load(f"results_baseline/predictions/lr_tuned_on_train_for_f1_{name}.npz")
        idx = list(m["idx"]); assert idx == list(v["idx"]) == list(bm["idx"]) == list(bv["idx"])
        y = y1[idx]; sets = np.array([pairs[i]["set"] for i in idx]); ev = np.array([pairs[i]["event_id"] for i in idx])
        S = {"MW": m["pred"][..., 0], "V1": v["pred"][..., 0], "baseMW": bm["pred"][..., 0], "baseV1": bv["pred"][..., 0]}
        print(f"\n== {name}: {len(set(ev))} events; thr MW {m['thr'][0]:.3f} V1 {v['thr'][0]:.3f}")
        for s in ("all", "A", "B", "C"):
            k = np.ones(len(idx), bool) if s == "all" else sets == s
            line = f"  {s:3s}"
            for n, p in S.items():
                tp, fp, fn = cnt(p[k], y[k])
                line += f" {n} F1 {f1c((tp, fp, fn)):.3f} P {tp / max(tp + fp, 1):.3f} R {tp / max(tp + fn, 1):.3f} FP {fp} |"
            print(line)
            if s == "C": continue
            for a, b in (("MW", "baseMW"), ("MW", "V1"), ("V1", "baseV1")):
                x, lo, hi = boot(S[a][k], S[b][k], y[k], ev[k], rng)
                print(f"      {a} - {b}: {x:+.3f} [{lo:+.3f}, {hi:+.3f}]")
        pr = m["prob"][..., 0].astype(float)
        scan = {round(float(t), 3): round(f1c(cnt((pr > t).astype(int), y)), 3) for t in np.arange(.3, .651, .025)}
        print("  dim1 threshold scan:", scan)


if __name__ == "__main__":
    main()
