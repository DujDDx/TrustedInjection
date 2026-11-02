"""Pilot: can B / Bf be generated so that copying A leaves a trace? Report only.

measure_signal.py showed the P4 documents cap "which of B, C copied A" at about 0.71 and
"A or Bf, which is derived" at about 0.58, whatever the model. The cause is generation:
B/Bf get A's sentences shuffled and the same "write in your own words; do not copy
sentences" prompt as A and C, so a third of B's keep less of A than C does.

Variants, all on the Duplicate branch of the same events (values do not affect wording):
  v0  current P4 documents (read from docs_p4_raw.jsonl, nothing generated)
  v1  B/Bf read A's whole text in its own order (B's fact inserted at a sentence
      boundary); prompt unchanged. A and C reused from v0.
  v2  v1 + the shared prompt loses "Write in your own words; do not copy sentences from
      the material." for every role, so A and C are regenerated too.
  v3  v1, but B/Bf get their own prompt: rewrite another outlet's report, keeping its
      structure. A and C reused from v0.

Fact allocation, seeds and the audit are those of gen_docs.py.

    python pilot_bgen.py --variant v1 --n 40 --out pilot_bgen/v1.jsonl
    python pilot_bgen.py --report pilot_bgen
"""
import argparse, collections, json, random, re, zlib
from pathlib import Path

import numpy as np

import gen_docs as G
from gen_docs import SENT, audit, complete, plan, tags_of
from measure_signal import grams, jac, lcs, sent_align, toks

HERE = Path(__file__).resolve().parent
OWN_WORDS = " Write in your own words; do not copy sentences from the material."

PROMPT_V2 = G.PROMPT.replace(OWN_WORDS, "")

PROMPT_V3 = """You are a news reporter. Below is a report published by another outlet. Write your outlet's version of it.

Words in angle brackets, such as <City>, stand for details you are not shown. Copy each one exactly, brackets included. Never replace them with names or numbers, never create new ones, and write no digits.
Every sentence must contain at least one bracketed word. Use every bracketed word in the material at least once. Do not add any fact that is not in the material. Keep the report's structure and the order of its information; edit the wording as a copy editor would.

__MATERIAL__

Return only the report."""


PROMPT_V4 = PROMPT_V3.replace(
    "Keep the report's structure and the order of its information; edit the wording as a copy editor would.",
    "Republish it with light edits: keep its sentences and their order, change some words and phrasing "
    "in each sentence, and work any information that is not yet in the report into it.")


P_REWRITE = 0.5
PROMPT_SENT = """You are a news reporter. Rewrite the sentence below in your own words.

Words in angle brackets, such as <City>, stand for details you are not shown. Copy each one exactly, brackets included. Never replace them with names or numbers, never create new ones, and write no digits. Do not add any fact.

SENTENCE
__S__

Return only the rewritten sentence."""


def rewrite_sentence(s, seed):
    """v5: one sentence rewritten by the model; same tags, one sentence, no digits."""
    want = tags_of(s)
    for k in range(G.TRIES):
        t = complete(PROMPT_SENT.replace("__S__", s.strip()), max_tokens=150,
                     temperature=G.TEMP, seed=seed + k).strip()
        if (t and tags_of(t) == want and not re.search(r"\d", t)
                and len([x for x in SENT.split(t) if x.strip()]) == 1 and t.strip() != s.strip()):
            return t, k + 1
    return None, G.TRIES


def events_sample(n):
    evs = [json.loads(l) for l in open(HERE / "events_p4v2.jsonl") if l.strip()]
    step = max(1, len(evs) // n)
    return evs[::step][:n]                      # every k-th event: spans all 8 domains


def gen_with(prompt, role, mat, seed, mat_tags, a_tags=None, b_tags=None):
    last = None
    for k in range(G.TRIES):
        txt = complete(prompt.replace("__MATERIAL__", mat), max_tokens=G.MAX_TOK,
                       temperature=G.TEMP, seed=seed + k)
        probs = audit(txt, role, mat_tags, a_tags, b_tags)
        if not probs: return txt.strip(), k + 1
        last = probs
    return None, G.TRIES


def run(variant, n, out):
    v0 = {(d["event_id"], d["role"]): d for d in map(json.loads, open(HERE / "docs_p4_raw.jsonl"))
          if d["branch"] == "dup"}
    out.parent.mkdir(parents=True, exist_ok=True)
    fo = open(out, "w")
    for ev in events_sample(n):
        eid = ev["event_id"]
        if (eid, "A") not in v0: continue
        ftxt = {f["id"]: f["text"] for f in ev["facts"]}
        rng = random.Random(f"{eid}-dup")
        a_ids, c_ids, b_only = plan(ev, rng)
        base = zlib.crc32(f"{eid}-dup".encode()) % 90000 + 50000      # new seed block
        recs = {}
        if variant == "v2":                     # A and C regenerated with the loosened prompt
            for role, ids, off in (("A", a_ids, 0), ("C", c_ids, 1000)):
                lines = [ftxt[i] for i in ids]; rng.shuffle(lines)
                t, att = gen_with(PROMPT_V2, role, G.material(lines), base + off, tags_of(" ".join(lines)))
                recs[role] = (t, att)
            if recs["A"][0] is None:
                print(f"  {eid}: A failed"); continue
            a_txt = recs["A"][0]
        else:
            a_txt = v0[(eid, "A")]["text"]
            recs["A"] = (a_txt, 0); recs["C"] = (v0[(eid, "C")]["text"], 0)
        a_tags = tags_of(" ".join(ftxt[i] for i in a_ids))
        b_tags = tags_of(" ".join(ftxt[i] for i in b_only))
        sents = [s for s in SENT.split(a_txt) if s.strip()]
        ins = random.Random(f"{eid}-ins").randrange(len(sents) + 1)
        b_sents = sents[:ins] + [ftxt[i] for i in b_only] + sents[ins:]
        if variant == "v5":
            for role, off in (("B", 2000), ("Bf", 3000)):
                r = random.Random(f"{eid}-{role}-v5")
                out_s, att = [], 0
                for k, s in enumerate(sents):
                    if r.random() < P_REWRITE:
                        t, a = rewrite_sentence(s, base + off + 10 * k); att += a
                        out_s.append(t)
                    else:
                        out_s.append(s.strip())
                if role == "B":
                    fs = [rewrite_sentence(ftxt[i], base + off + 500 + j) for j, i in enumerate(b_only)]
                    att += sum(a for _, a in fs)
                    out_s[ins:ins] = [t for t, _ in fs]
                txt = None if any(x is None for x in out_s) else " ".join(out_s)
                recs[role] = (txt, att)
            for role, (t, att) in recs.items():
                fo.write(json.dumps({"event_id": eid, "variant": variant, "role": role, "text": t,
                                     "attempts": att}, ensure_ascii=False) + "\n")
            fo.flush(); print(f"  {eid}: " + " ".join(f"{r}{'ok' if t else 'FAIL'}" for r, (t, _) in recs.items()), flush=True)
            continue
        prompt = PROMPT_V4 if variant == "v4" else PROMPT_V3 if variant == "v3" else (PROMPT_V2 if variant == "v2" else G.PROMPT)
        head = "REPORT\n" if variant in ("v3", "v4") else "MATERIAL\n"
        for role, body, off, mt in (("B", " ".join(b_sents), 2000, a_tags | b_tags),
                                    ("Bf", a_txt, 3000, a_tags)):
            t, att = gen_with(prompt, role, head + body, base + off, mt, a_tags,
                              b_tags if role == "B" else None)
            recs[role] = (t, att)
        for role, (t, att) in recs.items():
            fo.write(json.dumps({"event_id": eid, "variant": variant, "role": role, "text": t,
                                 "attempts": att}, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"  {eid}: " + " ".join(f"{r}{'ok' if t else 'FAIL'}({a})" for r, (t, a) in recs.items()),
              flush=True)


def tag_order_tau(x, y):
    """Kendall tau of the shared tags' first-occurrence order in x and y."""
    ox = [t for t in dict.fromkeys(re.findall(r"<[A-Za-z_]+>", x))]
    oy = [t for t in dict.fromkeys(re.findall(r"<[A-Za-z_]+>", y))]
    sh = [t for t in ox if t in oy]
    if len(sh) < 2: return 0.0
    py = {t: oy.index(t) for t in sh}
    c = d = 0
    for i in range(len(sh)):
        for j in range(i + 1, len(sh)):
            if py[sh[i]] < py[sh[j]]: c += 1
            else: d += 1
    return (c - d) / (c + d)


def report(d):
    v0 = collections.defaultdict(dict)
    for x in map(json.loads, open(HERE / "docs_p4_raw.jsonl")):
        if x["branch"] == "dup": v0[x["event_id"]][x["role"]] = x["text"]
    runs = {"v0": None}
    for p in sorted(Path(d).glob("v*.jsonl")):
        runs[p.stem] = [json.loads(l) for l in open(p)]
    ids = None
    for v, rows in runs.items():
        if rows is not None:
            s = {r["event_id"] for r in rows}; ids = s if ids is None else ids & s
    feats = {"jac2": lambda x, y: jac(grams(toks(x), 2), grams(toks(y), 2)),
             "jac3": lambda x, y: jac(grams(toks(x), 3), grams(toks(y), 3)),
             "lcs": lambda x, y: lcs(toks(x), toks(y)) / max(1, min(len(toks(x)), len(toks(y)))),
             "sent_align": lambda x, y: (sent_align(x, y) + sent_align(y, x)) / 2,
             "tag_order": tag_order_tau}
    print(f"events {len(ids)} (Duplicate branch)\n")
    print(f"{'variant':7s} {'audit':>6s} {'tries':>5s} {'words A/C/B/Bf':>16s} | " +
          " ".join(f"{k:>10s}" for k in feats) + " | A-B / A-C / A-Bf jac3")
    for v, rows in runs.items():
        by = collections.defaultdict(dict); att = []; fails = 0; tot = 0
        if rows is None:
            for e in ids: by[e] = v0[e]
        else:
            for r in rows:
                if r["event_id"] not in ids: continue
                tot += 1
                if r["text"] is None: fails += 1; continue
                by[r["event_id"]][r["role"]] = r["text"]
                if r["attempts"]: att.append(r["attempts"])
        full = [e for e in ids if len(by[e]) == 4]
        acc = {}
        for k, f in feats.items():
            acc[k] = np.mean([1. if f(by[e]["A"], by[e]["B"]) > f(by[e]["A"], by[e]["C"]) else
                              .5 if f(by[e]["A"], by[e]["B"]) == f(by[e]["A"], by[e]["C"]) else 0.
                              for e in full])
        w = "/".join(f"{np.mean([len(toks(by[e][r])) for e in full]):.0f}" for r in ("A", "C", "B", "Bf"))
        j = " / ".join(f"{np.mean([feats['jac3'](by[e]['A'], by[e][r]) for e in full]):.2f}"
                       for r in ("B", "C", "Bf"))
        aud = f"{1 - fails / tot:.2f}" if tot else "  -  "
        tries = f"{np.mean(att):.2f}" if att else "  -  "
        print(f"{v:7s} {aud:>6s} {tries:>5s} {w:>16s} | " + " ".join(f"{acc[k]:10.3f}" for k in feats) +
              f" | {j}")
    print("\nfeature columns: share of events where B is more similar to A than C is (chance 0.5)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["v1", "v2", "v3", "v4", "v5"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out")
    ap.add_argument("--report")
    a = ap.parse_args()
    if a.report: report(a.report)
    else: run(a.variant, a.n, Path(a.out))


if __name__ == "__main__":
    main()
