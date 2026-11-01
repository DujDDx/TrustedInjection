"""P0 gate (plan v1.1 §4.3 + v1.2 §4): deterministic checks on the hand-written events.

    python3 validate_events.py events_pilot.jsonl
"""
import collections
import json
import re
import sys

TAG = re.compile(r"<[A-Za-z_]+>")
CATS = {"date", "place", "agent", "quantity", "name"}
REQUIRED = {"date", "place", "agent", "quantity"}


def check(e):
    p = []
    facts = [f["text"] for f in e["facts"]]
    n, m = len(facts), e["m"]
    if n != e["n"]:
        p.append("n field does not match the number of facts")
    if not (n in (5, 6) and (m == 1 or (n == 6 and m == 2))):
        p.append(f"n={n}, m={m} outside the allowed grid")

    per_fact = [TAG.findall(t) for t in facts]
    used = [t for tags in per_fact for t in tags]
    owner = collections.Counter(t for tags in per_fact for t in set(tags))
    for t, c in owner.items():
        if c > 1:
            p.append(f"{t} appears in {c} facts (each tag must belong to one fact)")
    defined = set(e["tags"])
    if set(used) - defined:
        p.append(f"undefined tags {sorted(set(used) - defined)}")
    if defined - set(used):
        p.append(f"defined but unused tags {sorted(defined - set(used))}")

    # U8: f1 is the subject fact; it must name the subject with an agent, name or place tag
    f1_cats = {e["tags"][t]["cat"] for t in per_fact[0] if t in e["tags"]}
    if not f1_cats & {"agent", "name", "place"}:
        p.append("f1 does not name the subject (no agent/name/place tag)")

    for i, (t, tags) in enumerate(zip(facts, per_fact), 1):
        if not 1 <= len(tags) <= 3:
            p.append(f"f{i} has {len(tags)} tags")
        if re.search(r"\d", TAG.sub("", t)):
            p.append(f"f{i} has a digit outside a tag")
    if sum(len(t) >= 2 for t in per_fact) < 2:
        p.append("fewer than two facts carry two or more tags")
    if not 6 <= len(defined) <= 11:
        p.append(f"{len(defined)} tags (6-11 required)")

    cats = {v["cat"] for v in e["tags"].values()}
    if cats - CATS:
        p.append(f"unknown categories {sorted(cats - CATS)}")
    if REQUIRED - cats:
        p.append(f"missing categories {sorted(REQUIRED - cats)}")

    for k, v in e["tags"].items():
        tr, ru = str(v["truth"]), str(v["rumor"])
        if not tr or not ru:
            p.append(f"{k} empty value")
        if tr == ru:
            p.append(f"{k} truth equals rumor")
        if re.match(r"(?i)(the|a|an)\s", tr) or re.match(r"(?i)(the|a|an)\s", ru):
            p.append(f"{k} value starts with an article")
        # a decimal point inside a number is fine; any other full stop would break the
        # sentence splitter used by the audit
        for name, val in (("truth", tr), ("rumor", ru)):
            if "." in re.sub(r"\d\.\d", "", val):
                p.append(f"{k} {name} contains a full stop")
        if "<" in tr + ru or ">" in tr + ru:
            p.append(f"{k} value contains an angle bracket")

    # the seed text should state every truth value (it is the archived source event)
    for k, v in e["tags"].items():
        if str(v["truth"]) not in e["seed_text"]:
            p.append(f"{k} truth '{v['truth']}' not in seed_text")
    return p


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    bad = 0
    ids = collections.Counter(r["event_id"] for r in rows)
    for k, c in ids.items():
        if c > 1:
            print(f"duplicate event_id {k}")
            bad += 1
    for r in rows:
        p = check(r)
        if p:
            bad += 1
            print(r["event_id"], "; ".join(p))
    grid = collections.Counter((r["n"], r["m"]) for r in rows)
    dom = collections.Counter(r["domain"] for r in rows)
    tense = collections.Counter(r["tense"] for r in rows)
    ntags = [len(r["tags"]) for r in rows]
    print(f"\nevents {len(rows)}  failing {bad}")
    print("  (n, m):", dict(grid))
    print("  domains:", dict(dom))
    print("  tense:", dict(tense))
    print(f"  tags per event: min {min(ntags)} max {max(ntags)} mean {sum(ntags) / len(ntags):.1f}")
    cats = collections.Counter(v["cat"] for r in rows for v in r["tags"].values())
    print("  tag categories:", dict(cats))
    names = collections.Counter(k for r in rows for k in r["tags"])
    print("  most common tag names:", names.most_common(8))
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
