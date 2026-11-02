"""Pilot: make C a realistic independent report. Report only.

Against COUNTER real news, the current C (v0) looks like a partly-to-fully derived copy of
A (2-gram Jaccard 0.28, 28% of its sentences copied; real independent reports: 0.06 and
2%). Cause: A and C are both written from the same fact-card sentences. A, B, Bf and the
shared prompt stay exactly as in v0; only the wording of C's material changes. Two options,
tested separately:

  c1  each fact card line given to C is first reworded by the model into completely
      different wording (same tags, one sentence, no digits)
  c2  each fact card line is first turned into terse reporter's notes (fragments separated
      by semicolons, same tags, no full sentence)

C's fact allocation, line shuffle and the audit are those of gen_docs.py; the C prompt is
gen_docs.PROMPT unchanged. Rows for A, B, Bf are copied from docs_p4_raw.jsonl so the
output holds full quartets.

    python pilot_cgen.py --variant c1 --n 40 --out pilot_cgen/c1.jsonl
"""
import argparse, json, random, re, zlib
from pathlib import Path

import gen_docs as G
from gen_docs import SENT, complete, plan, tags_of
from measure_signal import grams, jac, toks
from pilot_bgen import events_sample, gen_with

HERE = Path(__file__).resolve().parent

RULES = ("Words in angle brackets, such as <City>, stand for details you are not shown. Copy each "
         "one exactly, brackets included. Never replace them with names or numbers, never create "
         "new ones, and write no digits. Do not add any fact.")

PROMPT_REWORD = f"""Below is one fact from a news story. Restate the same fact in completely different wording: different verbs, different nouns where possible, a different sentence structure.

{RULES}

FACT
__S__

Return only the restated fact, as one sentence."""

PROMPT_NOTES = f"""Below is one fact from a news story. Turn it into a reporter's terse notes: short fragments separated by semicolons, not a sentence, no articles where they can be dropped.

{RULES}

FACT
__S__

Return only the notes, on one line."""

MAX_SHARED_2GRAM = 0.25     # reworded line must not keep much of the card wording


def transform(line, variant, seed):
    """One card line -> reworded sentence (c1) or notes (c2). None after TRIES failures."""
    want = tags_of(line)
    prompt = (PROMPT_REWORD if variant == "c1" else PROMPT_NOTES).replace("__S__", line.strip())
    for k in range(G.TRIES):
        t = complete(prompt, max_tokens=150, temperature=G.TEMP, seed=seed + k).strip()
        t = t.splitlines()[0].strip() if t else t
        if not t or tags_of(t) != want or re.search(r"\d", t):
            continue
        if jac(grams(toks(t), 2), grams(toks(line), 2)) > MAX_SHARED_2GRAM:
            continue
        if variant == "c1" and len([x for x in SENT.split(t) if x.strip()]) != 1:
            continue
        return t, k + 1
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
        rng = random.Random(f"{eid}-dup")                      # same draws as gen_docs.py
        a_ids, c_ids, b_only = plan(ev, rng)
        rng.shuffle([ftxt[i] for i in a_ids])                  # A's shuffle, to keep rng in step
        c_lines = [ftxt[i] for i in c_ids]
        rng.shuffle(c_lines)
        base = zlib.crc32(f"{eid}-dup".encode()) % 90000 + 70000          # new seed block
        new_lines, att_t = [], 0
        for k, line in enumerate(c_lines):
            t, a = transform(line, variant, base + 10 * k); att_t += a
            new_lines.append(t)
        if any(x is None for x in new_lines):
            c_txt, att = None, 0
            print(f"  {eid}: material transform failed")
        else:
            c_txt, att = gen_with(G.PROMPT, "C", G.material(new_lines), base + 1000,
                                  tags_of(" ".join(new_lines)))
        for role in ("A", "B", "Bf"):
            fo.write(json.dumps({"event_id": eid, "variant": variant, "role": role,
                                 "text": v0[(eid, role)]["text"], "attempts": 0}) + "\n")
        fo.write(json.dumps({"event_id": eid, "variant": variant, "role": "C", "text": c_txt,
                             "attempts": att, "material_attempts": att_t, "material": new_lines,
                             "cards": c_lines}, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"  {eid}: C {'ok' if c_txt else 'FAIL'} (material tries {att_t}, doc tries {att})", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["c1", "c2"], required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    run(a.variant, a.n, Path(a.out))


if __name__ == "__main__":
    main()
