"""Pilot: per-document prompts from prompt_pool.py, v0 materials. Report only.

Same fact allocation, materials, audit and retries as gen_docs.py; the only change is
that each of A, C, B, Bf gets its own prompt (four distinct per event). Writer is the
local Qwen (training data) or DeepSeek (held-out writer, test documents only).

    python pilot_prompts.py --pool train   --writer qwen     --n 40 --out pilot_prompts/qwen_train.jsonl
    python pilot_prompts.py --pool heldout --writer deepseek --n 10 --out pilot_prompts/ds_heldout.jsonl
"""
import argparse, json, os, random, sys, zlib
from pathlib import Path

import gen_docs as G
from gen_docs import SENT, audit, plan, tags_of
from pilot_bgen import events_sample
import prompt_pool as P

HERE = Path(__file__).resolve().parent


def writer_fn(name):
    """deepseek: the API writer (test only). Any other name: whatever model the local
    llama-server at TI_BASE is serving; the name is only the label kept in the records."""
    if name != "deepseek":
        return lambda text, seed: G.complete(text, max_tokens=G.MAX_TOK, temperature=G.TEMP, seed=seed)
    sys.path.insert(0, str(HERE / "p4v2"))
    from gen_events_ds import chat                      # key from DEEPSEEK_API_KEY or ./dskey
    return lambda text, seed: chat([{"role": "user", "content": text}], temperature=G.TEMP,
                                   max_tokens=G.MAX_TOK, json_mode=False)[0]


def gen(write, prompt, role, lines, seed, mat_tags, a_tags=None, b_tags=None):
    last = None
    for k in range(G.TRIES):
        txt = write(prompt.replace("__MATERIAL__", G.material(lines)), seed + k)
        probs = audit(txt, role, mat_tags, a_tags, b_tags)
        if not probs: return txt.strip(), k + 1, []
        last = probs
    return None, G.TRIES, last


def run(pool, writer, n, out, branch="dup"):
    write = writer_fn(writer)
    pids = P.TRAIN if pool == "train" else P.HELDOUT
    out.parent.mkdir(parents=True, exist_ok=True)
    fo = open(out, "w")
    for ev in events_sample(n):
        eid = ev["event_id"]
        ftxt = {f["id"]: f["text"] for f in ev["facts"]}
        rng = random.Random(f"{eid}-{branch}")                     # gen_docs.py draws
        a_ids, c_ids, b_only = plan(ev, rng)
        base = zlib.crc32(f"{eid}-{branch}".encode()) % 90000 + 130000
        pr = P.assign(eid, branch, pids)
        a_lines = [ftxt[i] for i in a_ids]; rng.shuffle(a_lines)
        a_tags = tags_of(" ".join(a_lines))
        A, at, ap = gen(write, P.prompt(pr["A"]), "A", a_lines, base, a_tags)
        recs = [("A", A, at, ap, a_lines)]
        if A is not None:
            c_lines = [ftxt[i] for i in c_ids]; rng.shuffle(c_lines)
            C, ct, cp = gen(write, P.prompt(pr["C"]), "C", c_lines, base + 1000, tags_of(" ".join(c_lines)))
            sents = [s for s in SENT.split(A) if s.strip()]
            b_tags = tags_of(" ".join(ftxt[i] for i in b_only))
            b_lines = sents[:]; rng.shuffle(b_lines)
            for f in [ftxt[i] for i in b_only]: b_lines.insert(rng.randrange(len(b_lines) + 1), f)
            B, bt, bp = gen(write, P.prompt(pr["B"]), "B", b_lines, base + 2000, a_tags | b_tags, a_tags, b_tags)
            bf_lines = sents[:]; rng.shuffle(bf_lines)
            Bf, ft, fp = gen(write, P.prompt(pr["Bf"]), "Bf", bf_lines, base + 3000, a_tags, a_tags)
            recs += [("C", C, ct, cp, c_lines), ("B", B, bt, bp, b_lines), ("Bf", Bf, ft, fp, bf_lines)]
        for role, t, att, probs, lines in recs:
            fo.write(json.dumps({"event_id": eid, "branch": branch, "writer": writer, "role": role,
                                 "prompt": pr[role], "text": t, "attempts": att, "problems": probs,
                                 "material": lines}, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"  {eid}: " + " ".join(f"{r}[{pr[r]}]{'ok' if t else 'FAIL'}({a})" for r, t, a, _, _ in recs),
              flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", choices=["train", "heldout"], required=True)
    ap.add_argument("--writer", choices=["qwen", "deepseek"], required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.pool, a.writer, a.n, Path(a.out))


if __name__ == "__main__":
    main()
