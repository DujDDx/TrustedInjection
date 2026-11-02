"""Held-out test documents: test-split events only, held-out prompts, Qwen or DeepSeek.

The model is trained on v0 (one prompt, Qwen). To test generalization, the documents of
the 60 test-split events are written again, both branches, with the four held-out
prompts of prompt_pool.py (four distinct per event and branch) by
  qwen      the local writer used for training      -> unseen prompts
  deepseek  a writer never used for training        -> unseen prompts and unseen writer
Fact allocation, materials, audit and retries are those of gen_docs.py, plus one audit
rule: no reference to the prompt itself ("according to the material"), which DeepSeek
wrote in 6 of 152 pilot documents. Output is gen_docs.py's record format, so
build_pairs.py assembles and labels it unchanged.

    python gen_heldout.py --writer qwen --out heldout/qwen_docs.jsonl
    python gen_heldout.py --writer deepseek --out heldout/ds_docs.jsonl --shard 0/4
    python gen_heldout.py --writer qwen --splits train val --pool train --out pool_train/docs_s0.jsonl --shard 0/3
"""
import argparse, json, random, re, zlib
from pathlib import Path

import gen_docs as G
from gen_docs import SENT, audit, plan, tags_of
from pilot_prompts import writer_fn
import prompt_pool as P

HERE = Path(__file__).resolve().parent
LEAK = re.compile(r"\b(the|this|that) material\b|\bmaterial (provided|above|below)\b|\bplaceholder|\bbracket", re.I)


def split_events(splits):
    """Events of the given build_pairs.py splits (same event-level 70/15/15 split as v0)."""
    pairs = [json.loads(l) for l in open(HERE / "p4_pairs" / "pairs.jsonl")]
    keep = {p["event_id"] for p in pairs if p["split"] in splits}
    return [e for e in map(json.loads, open(HERE / "events_p4v2.jsonl")) if e["event_id"] in keep]


def gen(write, prompt, role, lines, seed, mat_tags, a_tags=None, b_tags=None):
    last = None
    for k in range(G.TRIES):
        txt = write(prompt.replace("__MATERIAL__", G.material(lines)), seed + k)
        probs = audit(txt, role, mat_tags, a_tags, b_tags)
        if LEAK.search(txt or ""): probs = probs + ["refers to the prompt material"]
        if not probs: return txt.strip(), seed + k
        last = probs
    return None, last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--writer", choices=["qwen", "deepseek"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    # training data with the training prompts: --splits train val --pool train
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--pool", choices=["train", "heldout"], default="heldout")
    ap.add_argument("--skip", nargs="*", default=[], help="docs files whose events are already done")
    a = ap.parse_args()
    done = {json.loads(l)["event_id"] for f in a.skip for l in open(f)}
    si, sn = map(int, a.shard.split("/"))
    write = writer_fn(a.writer)
    pool = P.TRAIN if a.pool == "train" else P.HELDOUT
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    fo = open(out, "w")
    for n, ev in enumerate(split_events(set(a.splits))):
        if n % sn != si: continue
        eid = ev["event_id"]
        if eid in done: continue
        ftxt = {f["id"]: f["text"] for f in ev["facts"]}
        for branch in ("dup", "rumor"):
            rng = random.Random(f"{eid}-{branch}")                      # gen_docs.py draws
            a_ids, c_ids, b_only = plan(ev, rng)
            base = zlib.crc32(f"{eid}-{branch}".encode()) % 90000 + 150000
            pr = P.assign(eid, branch, pool)
            a_lines = [ftxt[i] for i in a_ids]; rng.shuffle(a_lines)
            a_tags = tags_of(" ".join(a_lines))
            A, a_seed = gen(write, P.prompt(pr["A"]), "A", a_lines, base, a_tags)
            if A is None:
                print(f"  {eid}/{branch}: A failed {a_seed} -> drop branch", flush=True); continue
            recs = [("A", A, a_seed, a_ids, a_lines, a_tags)]
            c_lines = [ftxt[i] for i in c_ids]; rng.shuffle(c_lines)
            c_tags = tags_of(" ".join(c_lines))
            C, c_seed = gen(write, P.prompt(pr["C"]), "C", c_lines, base + 1000, c_tags)
            recs.append(("C", C, c_seed, c_ids, c_lines, c_tags))
            sents = [s for s in SENT.split(A) if s.strip()]
            b_tags = tags_of(" ".join(ftxt[i] for i in b_only))
            b_lines = sents[:]; rng.shuffle(b_lines)
            for f in [ftxt[i] for i in b_only]: b_lines.insert(rng.randrange(len(b_lines) + 1), f)
            B, b_seed = gen(write, P.prompt(pr["B"]), "B", b_lines, base + 2000, a_tags | b_tags, a_tags, b_tags)
            recs.append(("B", B, b_seed, a_ids + b_only, b_lines, a_tags | b_tags))
            bf_lines = sents[:]; rng.shuffle(bf_lines)
            Bf, f_seed = gen(write, P.prompt(pr["Bf"]), "Bf", bf_lines, base + 3000, a_tags, a_tags)
            recs.append(("Bf", Bf, f_seed, a_ids, bf_lines, a_tags))
            for role, t, seed, facts, lines, tags in recs:
                if t is None: continue
                fo.write(json.dumps(dict(event_id=eid, branch=branch, role=role, seed=seed, text=t,
                                         facts=facts, material=lines, tags=sorted(tags), prompt=pr[role],
                                         writer=a.writer), ensure_ascii=False) + "\n")
            fo.flush()
            print(f"  {eid}/{branch}: " + " ".join(f"{r}{'ok' if t else 'FAIL'}" for r, t, *_ in recs), flush=True)


if __name__ == "__main__":
    main()
