"""P1: generate the four documents per event per branch (plan v1.2 §2, §6).

Roles (v1.2 names): A = the source, C = an independent report, B = derives from A and
adds its own facts, Bf = derives from A only.

    A   f1 + n-2m-1 other common facts          -> n-2m facts
    C   A's facts + m of the remaining 2m       -> n-m   facts
    B   A's sentences (shuffled) + m B-only facts -> n-m  facts
    Bf  A's sentences (shuffled)                -> n-2m facts

All four documents use one byte-identical prompt; only the seed and MATERIAL differ
(v1.1 §6.1). f1 is the subject fact: it never becomes a B-only fact and is always given
to A (v1.2 U8), so anaphora like "the company" always has an antecedent.

Audit is deterministic (v1.1 §6.3 + v1.2 §2.1). A failed document is retried with a new
seed only, up to 6 times; the prompt is never amended, so the four roles stay identical.
If A fails 6 times the whole (event, branch) is dropped and the reason recorded.

    python3 gen_docs.py --events events_pilot.jsonl --out docs.jsonl [--limit 2] [--resume]
Environment: TI_BASE (default http://127.0.0.1:8818), TI_MODEL.
"""
import hashlib
import json
import os
import random
import re
import sys
import zlib

# 模型客户端与上一个实验共用一份，避免两套实现漂移
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "provenance_sym_20261030"))
from llm_client import complete  # noqa: E402

PROMPT = """You are a news reporter. Write a short news report based only on the material below.

Words in angle brackets, such as <City>, stand for details you are not shown. Copy each one exactly, brackets included. Never replace them with names or numbers, never create new ones, and write no digits.
Every sentence must contain at least one bracketed word. Use every bracketed word in the material at least once. Do not add any fact that is not in the material. Write in your own words; do not copy sentences from the material.

MATERIAL
__MATERIAL__

Return only the report."""

TAG = re.compile(r"<[A-Za-z_]+>")
SENT = re.compile(r"(?<=[.!?])\s+")
TRIES = 6
MAX_TOK = 500
TEMP = 0.8


def tags_of(text):
    return set(TAG.findall(text))


def audit(text, role, mat_tags, a_tags=None, b_tags=None):
    """Return a list of problems; empty means the document passes."""
    p = []
    t = text.strip()
    if not t:
        return ["empty"]
    if re.search(r"\d", t):
        p.append("contains a digit")
    used = tags_of(t)
    if used - mat_tags:
        p.append(f"tags not in material {sorted(used - mat_tags)}")
    if mat_tags - used:
        p.append(f"material tags missing {sorted(mat_tags - used)}")
    sents = [s for s in SENT.split(t) if s.strip()]
    bare = [i for i, s in enumerate(sents) if not TAG.search(s)]
    if bare:
        p.append(f"sentences without a tag: {bare}")
    w = len(t.split())
    if not 25 <= w <= 140:
        p.append(f"length {w} words")
    if "MATERIAL" in t or "```" in t or re.search(r"(?m)^\s*[-*]\s", t):
        p.append("markdown or material echo")
    if role == "B":
        allowed = (a_tags or set()) | (b_tags or set())
        if used - allowed:
            p.append(f"tags outside A and B-only {sorted(used - allowed)}")
        if (b_tags or set()) - used:
            p.append(f"B-only tags missing {sorted(b_tags - used)}")
    if role == "Bf" and a_tags is not None and used != a_tags:
        p.append(f"Bf tags differ from A {sorted(used ^ a_tags)}")
    return p


def plan(ev, rng):
    """Allocate facts for one (event, branch). Returns (a_ids, c_ids, b_only)."""
    ids = [f["id"] for f in ev["facts"]]
    n, m = ev["n"], ev["m"]
    pool = [i for i in ids if i != "f1"]          # U8: f1 never becomes B-only
    b_only = sorted(rng.sample(pool, m))
    common = [i for i in ids if i not in b_only]
    rest = [i for i in common if i != "f1"]
    a_ids = ["f1"] + sorted(rng.sample(rest, n - 2 * m - 1))
    left = [i for i in common if i not in a_ids]
    c_ids = a_ids + sorted(rng.sample(left + b_only, m))
    assert len(a_ids) == n - 2 * m and len(c_ids) == n - m and len(left) == m
    return a_ids, c_ids, b_only


def material(lines):
    return "MATERIAL\n" + "\n".join(f"- {x}" for x in lines)


def gen(role, lines, seed, mat_tags, a_tags=None, b_tags=None):
    """Draw up to TRIES candidates; return (text, used_seed, problems, attempts)."""
    last = None
    for k in range(TRIES):
        txt = complete(PROMPT.replace("__MATERIAL__", material(lines)),
                       max_tokens=MAX_TOK, temperature=TEMP, seed=seed + k)
        probs = audit(txt, role, mat_tags, a_tags, b_tags)
        if not probs:
            return txt.strip(), seed + k, [], k + 1
        last = probs
    return None, seed, last, TRIES


def main():
    a = sys.argv
    ev_path = a[a.index("--events") + 1] if "--events" in a else "events_pilot.jsonl"
    out = a[a.index("--out") + 1] if "--out" in a else "docs.jsonl"
    limit = int(a[a.index("--limit") + 1]) if "--limit" in a else 10 ** 6
    # 换一个种子块重抽某个分支（计划允许 6 个种子；种子块用尽后重开一块，等价于把
    # 预算从 6 扩到 12，只在整分支失败时用，不在"过与不过"之间挑质量）
    offset = int(a[a.index("--seed-offset") + 1]) if "--seed-offset" in a else 0
    events = [json.loads(l) for l in open(ev_path) if l.strip()][:limit]
    print(f"{len(events)} events x 2 branches x 4 documents; prompt_sha="
          f"{hashlib.sha256(PROMPT.encode()).hexdigest()[:16]}")

    # --resume: keep what is already in --out and skip those (event, branch) pairs, so a
    # crash hours into the 3,200-document run does not start it over. Seeds are derived
    # from the event id, so a resumed branch draws exactly what a fresh run would.
    resume = "--resume" in a and os.path.exists(out)
    done = set()
    if resume:
        done = {(r["event_id"], r["branch"]) for r in map(json.loads, open(out)) if r}
        print(f"resume: {len(done)} branches already in {out}")
    stats = {"ok": 0, "fail": 0, "dropped": 0, "retried": 0}
    with open(out, "a" if resume else "w") as fo:
        for ev in events:
            ftxt = {f["id"]: f["text"] for f in ev["facts"]}
            for branch in ("dup", "rumor"):
                if (ev["event_id"], branch) in done:
                    continue
                rng = random.Random(f"{ev['event_id']}-{branch}")
                a_ids, c_ids, b_only = plan(ev, rng)
                base = zlib.crc32(f"{ev['event_id']}-{branch}".encode()) % 90000 + offset

                # ---- A ----
                a_lines = [ftxt[i] for i in a_ids]
                rng.shuffle(a_lines)
                a_tags = tags_of(" ".join(a_lines))
                a_txt, a_seed, probs, att = gen("A", a_lines, base, a_tags)
                stats["retried"] += att - 1
                if a_txt is None:
                    stats["dropped"] += 1
                    print(f"  {ev['event_id']}/{branch}: A failed 6x ({probs}) -> drop branch")
                    continue
                stats["ok"] += 1
                a_sents = [s for s in SENT.split(a_txt) if s.strip()]

                recs = [dict(event_id=ev["event_id"], branch=branch, role="A", seed=a_seed,
                             text=a_txt, facts=a_ids, material=a_lines,
                             tags=sorted(a_tags))]

                # ---- C: A's facts plus m of the remaining 2m ----
                c_lines = [ftxt[i] for i in c_ids]
                rng.shuffle(c_lines)
                c_tags = tags_of(" ".join(c_lines))
                c_txt, c_seed, probs, att = gen("C", c_lines, base + 1000, c_tags)
                stats["retried"] += att - 1
                if c_txt is None:
                    stats["fail"] += 1
                    print(f"  {ev['event_id']}/{branch}: C failed 6x ({probs})")
                else:
                    stats["ok"] += 1
                    recs.append(dict(event_id=ev["event_id"], branch=branch, role="C",
                                     seed=c_seed, text=c_txt, facts=c_ids,
                                     material=c_lines, tags=sorted(c_tags)))

                # ---- B: A's sentences (shuffled) with the B-only facts inserted ----
                b_tags = tags_of(" ".join(ftxt[i] for i in b_only))
                b_lines = a_sents[:]
                rng.shuffle(b_lines)
                for f in [ftxt[i] for i in b_only]:
                    b_lines.insert(rng.randrange(len(b_lines) + 1), f)
                b_txt, b_seed, probs, att = gen("B", b_lines, base + 2000,
                                               a_tags | b_tags, a_tags, b_tags)
                stats["retried"] += att - 1
                if b_txt is None:
                    stats["fail"] += 1
                    print(f"  {ev['event_id']}/{branch}: B failed 6x ({probs})")
                else:
                    stats["ok"] += 1
                    recs.append(dict(event_id=ev["event_id"], branch=branch, role="B",
                                     seed=b_seed, text=b_txt, facts=a_ids + b_only,
                                     material=b_lines, tags=sorted(a_tags | b_tags)))

                # ---- Bf: A's sentences only ----
                bf_lines = a_sents[:]
                rng.shuffle(bf_lines)
                bf_txt, bf_seed, probs, att = gen("Bf", bf_lines, base + 3000, a_tags, a_tags)
                stats["retried"] += att - 1
                if bf_txt is None:
                    stats["fail"] += 1
                    print(f"  {ev['event_id']}/{branch}: Bf failed 6x ({probs})")
                else:
                    stats["ok"] += 1
                    recs.append(dict(event_id=ev["event_id"], branch=branch, role="Bf",
                                     seed=bf_seed, text=bf_txt, facts=a_ids,
                                     material=bf_lines, tags=sorted(a_tags)))

                for r in recs:
                    fo.write(json.dumps(r, ensure_ascii=False) + "\n")
                fo.flush()
                print(f"  {ev['event_id']}/{branch}: A{a_ids} C{c_ids} b_only{b_only} "
                      f"-> {len(recs)}/4 docs")

    tot = stats["ok"] + stats["fail"]
    print(f"\ndocs {stats['ok']}/{tot} = {stats['ok'] / max(1, tot):.1%}; "
          f"branches dropped {stats['dropped']}")
    sys.exit(0 if stats["ok"] / max(1, tot) >= 0.90 else 1)


if __name__ == "__main__":
    main()
