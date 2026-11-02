"""Pilot: matched prompts within an event, independent notes for A and C. Report only.

v0 already makes C independent by shuffling the card lines and giving C a different fact
subset, but A and C still read the same card sentences, so C shares A's wording (against
COUNTER real news it looks partly-to-fully derived). Here each independent writer gets
its own notes:

  - per event, one writing prompt w (from a pool) is used for all four documents and one
    rewording prompt r (from a pool) for every card line; prompts rotate across events;
  - A's card lines and C's card lines are each reworded with r, with different seeds, so
    A and C are produced by byte-identical prompts and differ only in seed and facts;
  - B's own fact is reworded the same way (no document reads a raw card);
  - B / Bf read A's sentences exactly as in gen_docs.py.

A reworded line must keep the tags, be one sentence with no digits, share at most
MAX_SHARED_2GRAM of the card's 2-grams, use at most RARE_SLACK more rare words than the card,
and pass a same-meaning check by the model at temperature 0.

    python pilot_matched.py --n 40 --out pilot_matched/m1.jsonl [--shard 0/2]
"""
import argparse, collections, hashlib, json, random, re, zlib
from pathlib import Path

import gen_docs as G
from gen_docs import SENT, complete, plan, tags_of
from measure_signal import grams, jac, toks
from pilot_bgen import events_sample, gen_with

HERE = Path(__file__).resolve().parent

_RULES = G.PROMPT[G.PROMPT.index("Words in angle brackets"):]          # tag rules .. end, unchanged
WRITE = {
    "w0": G.PROMPT,
    "w1": "You are a wire-service reporter. Write a brief, factual news item for the wire based only on "
          "the material below. Lead with the most important fact.\n\n" + _RULES,
    "w2": "You are a reporter for a regional newspaper. Write a short news story for local readers based "
          "only on the material below, in a natural, readable style.\n\n" + _RULES,
    "w3": "You are a news editor writing a short online news update based only on the material below. "
          "Keep the sentences clear and direct.\n\n" + _RULES,
}

_TAGRULE = ("Words in angle brackets, such as <City>, stand for details you are not shown. Copy each one "
            "exactly, brackets included. Never replace them with names or numbers, never create new ones, "
            "and write no digits. Do not add or drop any detail.")
REWORD = {
    "r0": "Below is one fact from a reporter's notes. Restate it the way a different reporter would write it "
          "in their own notes: plain everyday news English, a different sentence structure, exactly the same "
          "meaning. Keep titles, roles and the names of things accurate.\n\n" + _TAGRULE +
          "\n\nFACT\n__S__\n\nReturn only the restated fact, as one sentence.",
    # r1 (fact-sheet style) dropped: two rewordings of one card were 30% identical at temp 0.8
    "r1_dropped": "Rewrite the fact below as one plain sentence in your own words, as it might appear on a news "
          "agency's fact sheet. Keep exactly the same meaning and the same level of detail, and use common "
          "words.\n\n" + _TAGRULE + "\n\nFACT\n__S__\n\nReturn only the sentence.",
}
JUDGE = """Do the two sentences below state the same fact with the same details (who, what, roles and titles, quantities, places, times)? Words in angle brackets are placeholders and must match.

1. __A__
2. __B__

Answer YES or NO only."""

MAX_SHARED_2GRAM = 0.40
REWORD_TEMP = 1.1       # 0.8 -> 1.1 lowers overlap between two rewordings of one card 0.43 -> 0.40
REWORD_TRIES = 12       # one failed line drops the whole event
RARE_SLACK = 1          # extra rare words allowed over the card (cards are ~12 words)


def _common():
    c = collections.Counter(w for x in map(json.loads, open(HERE / "docs_p4_raw.jsonl"))
                            if x["role"] == "A" for w in re.findall(r"[a-z]+", x["text"].lower()))
    return {w for w, _ in c.most_common(3000)}


COMMON = _common()


def rare(t):
    """Count of words outside the 3,000 most common words of the v0 A documents."""
    return sum(w not in COMMON for w in re.findall(r"[a-z]+", re.sub(r"<[^>]+>", "", t).lower()))


def pick(pool, key):
    ks = sorted(k for k in pool if not k.endswith("_dropped"))
    return ks[int(hashlib.md5(key.encode()).hexdigest(), 16) % len(ks)]


def reword(line, r, seed, stats):
    want = tags_of(line)
    for k in range(REWORD_TRIES):
        t = complete(REWORD[r].replace("__S__", line.strip()), max_tokens=150,
                     temperature=REWORD_TEMP, seed=seed + k).strip()
        t = t.splitlines()[0].strip() if t else t
        why = None
        if not t or tags_of(t) != want: why = "tags"
        elif re.search(r"\d", t): why = "digit"
        elif len([x for x in SENT.split(t) if x.strip()]) != 1: why = "sentences"
        elif jac(grams(toks(t), 2), grams(toks(line), 2)) > MAX_SHARED_2GRAM: why = "too_close"
        elif rare(t) > rare(line) + RARE_SLACK: why = "rare_words"
        else:
            v = complete(JUDGE.replace("__A__", line.strip()).replace("__B__", t), max_tokens=5,
                         temperature=0.0, seed=seed + k).strip().upper()
            if not v.startswith("YES"): why = "meaning"
        if why is None:
            stats["ok"] += 1
            return t
        stats[why] += 1
    stats["line_failed"] += 1
    return None


def run(n, out, shard):
    si, sn = map(int, shard.split("/"))
    out.parent.mkdir(parents=True, exist_ok=True)
    fo = open(out, "w")
    stats = collections.Counter()
    for idx, ev in enumerate(events_sample(n)):
        if idx % sn != si: continue
        eid = ev["event_id"]
        ftxt = {f["id"]: f["text"] for f in ev["facts"]}
        rng = random.Random(f"{eid}-dup")
        a_ids, c_ids, b_only = plan(ev, rng)
        w, r = pick(WRITE, f"{eid}-w"), pick(REWORD, f"{eid}-r")
        base = zlib.crc32(f"{eid}-dup".encode()) % 90000 + 110000          # new seed block
        a_cards = [ftxt[i] for i in a_ids]; rng.shuffle(a_cards)
        c_cards = [ftxt[i] for i in c_ids]; rng.shuffle(c_cards)
        a_mat = [reword(x, r, base + 10 * k, stats) for k, x in enumerate(a_cards)]
        c_mat = [reword(x, r, base + 5000 + 10 * k, stats) for k, x in enumerate(c_cards)]
        b_mat = [reword(ftxt[i], r, base + 7000 + 10 * k, stats) for k, i in enumerate(b_only)]
        rec = dict(event_id=eid, w=w, r=r)
        if None in a_mat or None in c_mat or None in b_mat:
            print(f"  {eid}: rewording failed", flush=True); continue
        a_tags = tags_of(" ".join(a_mat)); b_tags = tags_of(" ".join(b_mat))
        A, at = gen_with(WRITE[w], "A", G.material(a_mat), base, a_tags)
        if A is None:
            print(f"  {eid}: A failed", flush=True); continue
        C, ct = gen_with(WRITE[w], "C", G.material(c_mat), base + 1000, tags_of(" ".join(c_mat)))
        sents = [s for s in SENT.split(A) if s.strip()]
        r2 = random.Random(f"{eid}-matched")
        b_lines = sents[:]; r2.shuffle(b_lines)
        for f in b_mat: b_lines.insert(r2.randrange(len(b_lines) + 1), f)
        B, bt = gen_with(WRITE[w], "B", G.material(b_lines), base + 2000, a_tags | b_tags, a_tags, b_tags)
        bf_lines = sents[:]; r2.shuffle(bf_lines)
        Bf, ft = gen_with(WRITE[w], "Bf", G.material(bf_lines), base + 3000, a_tags, a_tags)
        mats = {"A": a_mat, "C": c_mat, "B": b_lines, "Bf": bf_lines}
        cards = {"A": a_cards, "C": c_cards, "B": [ftxt[i] for i in b_only], "Bf": []}
        for role, t, att in (("A", A, at), ("C", C, ct), ("B", B, bt), ("Bf", Bf, ft)):
            fo.write(json.dumps({**rec, "role": role, "text": t, "attempts": att,
                                 "material": mats[role], "cards": cards[role]}, ensure_ascii=False) + "\n")
        fo.flush()
        print(f"  {eid} [{w},{r}]: " + " ".join(f"{k}{'ok' if t else 'FAIL'}"
                                              for k, t in (("A", A), ("C", C), ("B", B), ("Bf", Bf))), flush=True)
    print("reword stats", dict(stats), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    run(a.n, Path(a.out), a.shard)


if __name__ == "__main__":
    main()
