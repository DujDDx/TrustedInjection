"""P2: fill values, assemble the three sets, split by event, label every token.

Plan v1.2 §3 (fill + labels), §7.3 (split), §8 (token labels).

Fill (per branch, once per document):
  A   truth everywhere
  C   truth, except each of B's Rumor facts is mirrored with p=1/2: every value of that
      fact in C becomes B's rumor value (U5)
  B   its own facts truth; 1-2 of the transcribed facts filled with rumor values
  Bf  truth (U4)

Labels (§3.3):
  Set A  A, C -> (0,0); B transcribed -> (1, rumor?1:0); B own facts -> (0,0)
  Set B  A, X -> (0,0); Bf -> (1,0)
  Set C  X1, X2 -> (0,0); D (=B or Bf) transcribed -> (0,0)  (U7: not copied, so not
         fabricated); B own facts -> (0,0)

    python3 build_pairs.py --events events_pilot.jsonl --docs docs_raw.jsonl --out-dir .
"""
import collections
import json
import os
import random
import re
import sys
from itertools import permutations

import numpy as np
from tokenizers import Tokenizer

TAG = re.compile(r"<[A-Za-z_]+>")
MAXLEN = 160
ART = re.compile(r"(?i)\b(a|an)\s+$")


def article_for(val):
    """'a' or 'an' by the sound of the value, not its first letter: an 8m, an 11,000,
    an 18-month, a 110; a university, a one-off, an hour."""
    m = re.match(r"\d+", val)
    if m:
        lead = m.group()
        return "an" if lead[0] == "8" or lead in ("11", "18") else "a"
    w = val.lower()
    if w.startswith(("uni", "use", "usu", "uti", "eu", "one", "once")):
        return "a"
    if w.startswith(("hour", "honest", "honour", "heir")):
        return "an"
    return "an" if w[:1] in "aeiou" else "a"


def fill(text, values):
    """Replace every <Tag> with values[tag]. Returns (filled_text, spans).

    The article fix runs before the span is recorded, so offsets stay consistent: a
    replacement that turns "a" into "an" only shifts text before the value.
    """
    s, spans, i = "", [], 0
    for m in TAG.finditer(text):
        s += text[i:m.start()]
        tag = m.group(0)
        val = values[tag]
        mm = ART.search(s)
        if mm:
            want = article_for(val)
            if mm.group(1).isupper() or mm.group(1)[0].isupper():
                want = want.capitalize()
            if mm.group(1) != want:
                s = s[:mm.start(1)] + want + s[mm.end(1):]
        start = len(s)
        s += val
        spans.append({"start": start, "end": len(s), "tag": tag, "value": val})
        i = m.end()
    s += text[i:]
    return s, spans


def label_doc(tok, text, spans, d1_of, d2_of):
    """Tokenise and label. A token overlapping a value span at all counts as inside it
    (§8.1); everything else is -100 so it never enters the loss or the metrics (§8.2)."""
    enc = tok.encode(text)
    ids, offs = enc.ids[:MAXLEN], enc.offsets[:MAXLEN]
    d1 = np.full(MAXLEN, -100, dtype=np.int8)
    d2 = np.full(MAXLEN, -100, dtype=np.int8)
    for k, (a, b) in enumerate(offs):
        for sp in spans:
            if a < sp["end"] and b > sp["start"]:
                d1[k], d2[k] = d1_of(sp), d2_of(sp)
                break
    n = len(ids)
    ids = ids + [tok.token_to_id("<pad>")] * (MAXLEN - n)
    mask = [1] * n + [0] * (MAXLEN - n)
    return ids, mask, d1, d2


def span_dims(pair_set, role, sp, a_ids):
    """(dim1, dim2) for one value span of one document inside one pair (§3.3)."""
    is_rumor = sp["version"] == "rumor"
    if pair_set == "A":
        if role in ("A", "C"):
            return 0, 0
        if role == "B":
            if sp["fact"] not in a_ids:        # B's own fact
                return 0, 0
            return 1, (1 if is_rumor else 0)
    if pair_set == "B":
        if role in ("A", "X"):
            return 0, 0
        if role == "Bf":
            return 1, 0
    if pair_set == "C":
        return 0, 0
    return 0, 0


def main():
    a = sys.argv
    ev_path = a[a.index("--events") + 1] if "--events" in a else "events_pilot.jsonl"
    doc_path = a[a.index("--docs") + 1] if "--docs" in a else "docs_raw.jsonl"
    out_dir = a[a.index("--out-dir") + 1] if "--out-dir" in a else "."
    tok_path = a[a.index("--tok") + 1] if "--tok" in a else os.path.expanduser(
        "~/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots")
    if os.path.isdir(tok_path):
        sub = sorted(d for d in os.listdir(tok_path)
                     if os.path.isdir(os.path.join(tok_path, d)))
        tok_path = os.path.join(tok_path, sub[0], "tokenizer.json")
    tok = Tokenizer.from_file(tok_path)
    print(f"tokenizer {tok_path}")

    events = {e["event_id"]: e for e in map(json.loads, open(ev_path)) if e}
    by = {}
    for d in map(json.loads, open(doc_path)):
        by[(d["event_id"], d["branch"], d["role"])] = d

    # ---- split by event 70/15/15 (§7.3) ----
    ev_ids = sorted(events)
    random.Random(20261101).shuffle(ev_ids)
    c1, c2 = round(0.70 * len(ev_ids)), round(0.85 * len(ev_ids))
    split_of = {e: ("train" if i < c1 else "val" if i < c2 else "test")
                for i, e in enumerate(ev_ids)}
    print("split:", collections.Counter(split_of.values()))

    docs_out, events_out, filled = [], [], {}

    # ---- pass 1: decide the fact allocation and B's rumor facts ----
    # Mirroring is decided globally, not per fact: with 26 rumor facts a per-fact coin
    # flip lands outside the plan's 50%±5pp band about a third of the time (measured
    # 11/26 = 42.3% on the first run). Taking exactly half still decides *which* facts
    # at random, and it is closer to "randomly take half of them" than a coin flip is.
    plan = {}
    for ev in events.values():
        eid = ev["event_id"]
        for branch in ("dup", "rumor"):
            roles = {r: by.get((eid, branch, r)) for r in ("A", "C", "B", "Bf")}
            if any(v is None for v in roles.values()):
                print(f"  {eid}/{branch}: incomplete, skipped")
                continue
            rng = random.Random(f"fill-{eid}-{branch}")
            a_ids = roles["A"]["facts"]
            b_only = [f for f in roles["B"]["facts"] if f not in a_ids]
            if branch == "rumor":
                k = 1 if len(a_ids) <= 2 else rng.choice([1, 2])
                k = min(k, max(1, len(a_ids) - 1))      # B keeps >=1 truthful fact
                rumor_facts = set(rng.sample(a_ids, k))
            else:
                rumor_facts = set()
            plan[(eid, branch)] = (roles, a_ids, b_only, rumor_facts)

    all_rumor = sorted((e, b, f) for (e, b), v in plan.items() for f in v[3])
    rng_m = random.Random(20261101)
    mirror_set = set(rng_m.sample(all_rumor, round(0.5 * len(all_rumor))))
    print(f"mirroring {len(mirror_set)}/{len(all_rumor)} rumor facts")

    # ---- pass 2: fill and label ----
    for (eid, branch), (roles, a_ids, b_only, rumor_facts) in plan.items():
        ev = events[eid]
        ftxt = {f["id"]: f["text"] for f in ev["facts"]}
        fact_of_tag = {t: fid for fid, txt in ftxt.items() for t in TAG.findall(txt)}
        mirror = {f for (e, b, f) in mirror_set if e == eid and b == branch}
        if True:
            def values_for(role):
                out = {}
                for tag, meta in ev["tags"].items():
                    f = fact_of_tag.get(tag)
                    use_rumor = ((role == "C" and f in mirror) or
                                 (role == "B" and f in rumor_facts))
                    out[tag] = meta["rumor"] if use_rumor else meta["truth"]
                return out

            for role, doc in roles.items():
                vals = values_for(role)
                text, spans = fill(doc["text"], vals)
                for sp in spans:
                    sp["fact"] = fact_of_tag.get(sp["tag"])
                    sp["version"] = ("rumor" if vals[sp["tag"]] ==
                                     ev["tags"][sp["tag"]]["rumor"] else "truth")
                n = len(tok.encode(text).ids)
                if n > MAXLEN:
                    print(f"  {eid}/{branch}/{role}: {n} tokens > {MAXLEN}, dropped")
                    continue
                d = {"doc_id": f"{eid}-{branch}-{role}", "event_id": eid,
                     "branch": branch, "role": role, "split": split_of[eid],
                     "text_tagged": doc["text"], "text": text, "spans": spans,
                     "facts": doc["facts"], "a_ids": a_ids, "ntok": n}
                filled[(eid, branch, role)] = d
                docs_out.append(d)
            events_out.append({"event_id": eid, "branch": branch, "split": split_of[eid],
                               "a_ids": a_ids, "c_ids": roles["C"]["facts"],
                               "b_only": b_only, "rumor_facts": sorted(rumor_facts),
                               "mirrored_to_c": sorted(mirror)})

    doc_by_id = {d["doc_id"]: d for d in docs_out}
    pool = {}
    for d in docs_out:
        if d["role"] in ("A", "C"):
            pool.setdefault((d["branch"], d["split"]), []).append(d["doc_id"])

    def pick_x(branch, split, eid, k):
        cand = sorted(x for x in pool[(branch, split)] if not x.startswith(eid + "-"))
        if len(cand) < k:
            return None
        return random.Random(f"x-{eid}-{branch}-{k}").sample(cand, k)

    # ---- assemble pairs (§3.2) ----
    pairs, skipped = [], 0
    rng_d = random.Random("setc")
    for ev in events.values():
        eid = ev["event_id"]
        for branch in ("dup", "rumor"):
            if any((eid, branch, r) not in filled for r in ("A", "C", "B", "Bf")):
                continue
            split = split_of[eid]
            for perm in permutations(["A", "B", "C"]):
                pairs.append({"event_id": eid, "branch": branch, "split": split,
                              "set": "A",
                              "docs": [f"{eid}-{branch}-{r}" for r in perm]})
            x = pick_x(branch, split, eid, 1)
            if x:
                for perm in permutations(["A", "Bf", "X"]):
                    pairs.append({"event_id": eid, "branch": branch, "split": split,
                                  "set": "B",
                                  "docs": [f"{eid}-{branch}-{r}" if r != "X" else x[0]
                                           for r in perm]})
            else:
                skipped += 1
            xs = pick_x(branch, split, eid, 2)
            if xs:
                d_role = rng_d.choice(["B", "Bf"])
                for perm in permutations(["X1", d_role, "X2"]):
                    pairs.append({"event_id": eid, "branch": branch, "split": split,
                                  "set": "C",
                                  "docs": [(f"{eid}-{branch}-{r}" if r not in ("X1", "X2")
                                            else (xs[0] if r == "X1" else xs[1]))
                                           for r in perm]})
            else:
                skipped += 1

    # ---- strict 1:1:1:1:1:1 (§3.2) ----
    # 只要有一个分支被丢弃，六类就会不齐。用户 2026-11-01 决定六类必须严格相等，
    # 所以在组装后按最小类降采样：随机（固定种子）决定去掉哪些对，不是按事件砍。
    cnt = collections.Counter((p["branch"], p["set"]) for p in pairs)
    cap = min(cnt.values()) if cnt else 0
    if len(set(cnt.values())) > 1:
        order = list(range(len(pairs)))
        random.Random("balance").shuffle(order)
        keep, seen = set(), collections.Counter()
        for i in order:
            k = (pairs[i]["branch"], pairs[i]["set"])
            if seen[k] < cap:
                seen[k] += 1
                keep.add(i)
        print(f"balance: 六类不等 {dict(cnt)}，按最小类 {cap} 降采样，去掉 "
              f"{len(pairs) - len(keep)} 对")
        pairs = [p for i, p in enumerate(pairs) if i in keep]

    # ---- tokenise every pair ----
    rows = []
    for pr in pairs:
        trio = [doc_by_id.get(x) for x in pr["docs"]]
        if any(t is None for t in trio):
            continue
        chans = []
        for doc in trio:
            # a doc used in the X slot keeps its own role (A or C); Set B/C label every
            # value 0 anyway, so span_dims only needs the pair-set there.
            chans.append(label_doc(
                tok, doc["text"], doc["spans"],
                lambda sp, d=doc, s=pr["set"]: span_dims(s, d["role"], sp, d["a_ids"])[0],
                lambda sp, d=doc, s=pr["set"]: span_dims(s, d["role"], sp, d["a_ids"])[1]))
        rows.append((pr, chans))

    ids_arr = np.array([[c[0] for c in ch] for _, ch in rows], dtype=np.int32)
    mask_arr = np.array([[c[1] for c in ch] for _, ch in rows], dtype=np.int8)
    d1_arr = np.array([[c[2] for c in ch] for _, ch in rows], dtype=np.int8)
    d2_arr = np.array([[c[3] for c in ch] for _, ch in rows], dtype=np.int8)

    with open(os.path.join(out_dir, "docs.jsonl"), "w") as f:
        for d in docs_out:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "events.jsonl"), "w") as f:
        for e in events_out:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    with open(os.path.join(out_dir, "pairs.jsonl"), "w") as f:
        for pr, _ in rows:
            f.write(json.dumps(pr, ensure_ascii=False) + "\n")
    np.savez_compressed(os.path.join(out_dir, "token_labels.npz"),
                        input_ids=ids_arr, attention_mask=mask_arr,
                        dim1=d1_arr, dim2=d2_arr)

    # ---- report ----
    cls = collections.Counter((p["branch"], p["set"]) for p, _ in rows)
    print(f"\npairs {len(rows)}; docs {len(docs_out)}; X-slot skips {skipped}")
    for k in sorted(cls):
        print(f"  {k[0]:6s} Set {k[1]}: {cls[k]}")
    print("  split:", dict(collections.Counter(p["split"] for p, _ in rows)))
    lab = int((d1_arr != -100).sum())
    print(f"  label tokens {lab}/{ids_arr.size} ({lab / ids_arr.size:.1%}); "
          f"dim1=1 {int((d1_arr == 1).sum())}; dim2=1 {int((d2_arr == 1).sum())}")
    assert not ((d2_arr == 1) & (d1_arr != 1)).any(), "dim2=1 requires dim1=1 (U7)"
    rum = [e for e in events_out if e["branch"] == "rumor"]
    if rum:
        m = sum(len(e["mirrored_to_c"]) for e in rum)
        r = sum(len(e["rumor_facts"]) for e in rum)
        print(f"  mirror ratio {m}/{r} = {m / max(1, r):.1%} (target 50% ±5pp)")


if __name__ == "__main__":
    main()
