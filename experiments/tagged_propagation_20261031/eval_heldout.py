"""Evaluate one trained pair cross-encoder on several test sets with the same metrics.

  v0           p4_pairs test split (prompt p00, Qwen) - the training distribution
  qwen_heldout held-out prompts, Qwen                  - unseen prompts
  ds_heldout   held-out prompts, DeepSeek              - unseen prompts and writer
All three cover the same 60 test-split events. Besides dim1/dim2 F1, the source pointer
and the B-over-C ranking, dim2 on B's rumor values is split into its two steps: was A
chosen as B's source, and given that, did the same-value head call the value different.

    python eval_heldout.py --ckpt /mnt/hdd4/dujddx/tagged_pairce_match_p4_20261102/best.pt \
        --sets p4_pairs heldout/qwen_pairs heldout/ds_pairs
"""
import argparse, collections, json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from train_crossdoc import MODEL, Pairs, evaluate, token_slices
from train_pairce import PairCE, rank_b_over_c
from train_srcptr import aux_labels, eval_src


@torch.no_grad()
def dim2_steps(model, z, pairs, docs, idx, device):
    """Rumor Set A, B's rumor tokens: P(pointer picks A), P(value judged different | A picked),
    and composed dim2 accuracy."""
    model.eval(); n = picked = diff_given = diff_n = d2_ok = 0
    sel = [k for k in idx if pairs[k]["set"] == "A" and pairs[k]["branch"] == "rumor"]
    for s in range(0, len(sel), 16):
        b = sel[s:s + 16]
        ids = torch.from_numpy(z["input_ids"][b].astype(np.int64)).to(device)
        mk = torch.from_numpy(z["attention_mask"][b].astype(np.int64)).to(device)
        with torch.autocast("cuda", enabled=(device == "cuda")):
            o = model.forward_all(ids, mk)
        src = o["src"].float().argmax(-1).cpu().numpy()
        same = o["same"].float().sigmoid().cpu().numpy(); p2 = o["p"][..., 1].float().cpu().numpy()
        for r, k in enumerate(b):
            roles = [docs[d]["role"] for d in pairs[k]["docs"]]
            ia, ib = roles.index("A"), roles.index("B")
            ka = [j for j in range(3) if j != ib].index(ia)
            for t in np.nonzero(z["dim2"][k][ib] == 1)[0]:
                n += 1; d2_ok += int(p2[r, ib, t] > 0.5)
                if src[r, ib] == ka:
                    picked += 1; diff_n += 1; diff_given += int(same[r, ib, ka, t] < 0.5)
    return {"rumor_tokens": n, "pointer_picks_A": round(picked / max(1, n), 4),
            "value_differs_given_A": round(diff_given / max(1, diff_n), 4),
            "dim2_recall": round(d2_ok / max(1, n), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", nargs="+", required=True)
    ap.add_argument("--out")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = PairCE(MODEL).to(dev)
    m.load_state_dict(torch.load(a.ckpt, map_location=dev)["model"]); m.eval()
    tok = str(Path(MODEL) / "tokenizer.json")
    res = {}
    for d in a.sets:
        z = dict(np.load(Path(d) / "token_labels.npz"))
        pairs = [json.loads(l) for l in open(Path(d) / "pairs.jsonl")]
        docs = {x["doc_id"]: x for x in map(json.loads, open(Path(d) / "docs.jsonl"))}
        idx = [i for i, p in enumerate(pairs) if p["split"] == "test"]
        slices = token_slices(d, tok, pairs, z["input_ids"])
        src_all, _, _ = aux_labels(d, tok, pairs)
        dl = DataLoader(Pairs(z, idx), batch_size=16, shuffle=False)
        ev = evaluate(m, dl, slices, dev)
        ev["src"] = eval_src(m, dl, pairs, src_all, dev)
        ev["rank_b_over_c"] = rank_b_over_c(m, z, pairs, {k: v["role"] for k, v in docs.items()}, idx, dev)
        ev["dim2_steps"] = dim2_steps(m, z, pairs, docs, idx, dev)
        ev["n_pairs"] = len(idx); ev["n_events"] = len({pairs[i]["event_id"] for i in idx})
        res[d] = ev
        print(f"\n== {d}: {ev['n_pairs']} pairs, {ev['n_events']} events")
        print(f"  dim1 F1 {ev['dim1_f1']:.3f}  dim2 F1 {ev['dim2_f1']:.3f}  B-over-C {ev['rank_b_over_c']}")
        print(f"  source pointer: " + "  ".join(f"{k} {v['acc']:.2f}" for k, v in ev["src"].items()))
        print(f"  dim2 steps: {ev['dim2_steps']}")
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
