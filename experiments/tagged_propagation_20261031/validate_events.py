"""P0 gate (plan v1.1 §4.3 + v1.2 §4): deterministic checks on the hand-written events.

    python3 validate_events.py events_pilot.jsonl

Per-event checks cover the grid, tag ownership, categories, values and the seed text.
File-level checks (v2, 2026-11-01) cover what only shows up across events: rumor direction
(the first P4 table had 98% of numeric rumors below the truth, so "the larger number is the
source" solved Rumor·Set A), repeated fact templates inside a domain (v1.1 §4.2), reused
proper names and duplicate subjects.
"""
import collections
import json
import re
import sys

TAG = re.compile(r"<[A-Za-z_]+>")
CATS = {"date", "place", "agent", "quantity", "name"}
REQUIRED = {"date", "place", "agent", "quantity"}
# v1.1 §4.2: eight domains in rotation
DOMAINS = {"academic_conference", "business", "sports", "public_policy", "science",
           "disaster_accident", "culture", "health"}
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]
# connectives that tie a fact to another fact; facts are shuffled (v1.2 §2.4) and any one
# of f2..fn may be missing from a document, so each must stand on its own given f1
LINKERS = re.compile(r"(?i)\b(also|too|this|these|those|another|then|meanwhile|however|"
                     r"moreover|furthermore|additionally|likewise|besides)\b")
NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
# v1.1 §4.1: every number in a fact is a tag, spelt-out ones included
NUMWORDS = re.compile(
    r"(?i)\b(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|"
    r"seventy|eighty|ninety|hundreds?|thousands?|millions?|billions?|dozens?|half|double|"
    r"twice|triple|thrice|quarter|fiftieth|hundredth|centenary|decades?)\b")


def appears(val, text):
    """Word-boundary match: `5 percent` is not in `3.5 percent`, `21` is not in `2021`."""
    pat = r"(?<![\w.])" + re.escape(val) + r"(?![\w]|\.\d)"
    return re.search(pat, text) is not None


def shape(v):
    """Surface form with numbers and month names abstracted, e.g. '#m pounds', '# MONTH'."""
    s = NUM.sub("#", v)
    # only a value that is a date as a whole ('12 May', 'May', 'May 2027'), not 'June Ralston'
    mo = "|".join(MONTHS)
    return re.sub(rf"^(# )?({mo})( #)?$", lambda m: f"{m.group(1) or ''}MONTH{m.group(3) or ''}", s)


def check(e):
    p = []
    for k in ("event_id", "domain", "topic", "tense", "n", "m", "seed_text", "facts", "tags"):
        if k not in e:
            p.append(f"missing field {k}")
    if not isinstance(e["facts"], list) or not e["facts"] or not isinstance(e["tags"], dict):
        p.append("facts must be a non-empty list and tags a dict")
    else:
        for f in e["facts"]:
            if not isinstance(f, dict) or not isinstance(f.get("text"), str) or "id" not in f:
                p.append("every fact needs id and text")
                break
        for k, v in e["tags"].items():
            if not isinstance(v, dict) or not {"cat", "truth", "rumor"} <= set(v):
                p.append(f"{k} needs cat, truth and rumor")
    if p:
        return p
    if e["domain"] not in DOMAINS:
        p.append(f"domain {e['domain']!r} not one of the eight")
    if e["tense"] not in ("future", "past"):
        p.append(f"tense {e['tense']!r}")

    facts = [f["text"] for f in e["facts"]]
    n, m = len(facts), e["m"]
    if n != e["n"]:
        p.append("n field does not match the number of facts")
    if [f["id"] for f in e["facts"]] != [f"f{i}" for i in range(1, n + 1)]:
        p.append("fact ids are not f1..fn")
    # 2026-11-01: the (6, 2) cell is dropped
    if not (n in (5, 6) and m == 1):
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
        # A reads only n-2m facts and must reach 25 words (the pilot lost E0015/rumor at 23)
        if not 7 <= len(t.split()) <= 24:
            p.append(f"f{i} has {len(t.split())} words (7-24)")
        if len(tags) != len(set(tags)):
            p.append(f"f{i} repeats a tag")
        if re.search(r"\d", TAG.sub("", t)):
            p.append(f"f{i} has a digit outside a tag")
        if not t.endswith(".") or re.search(r"[?!;]", t) or t.count(".") > 1:
            p.append(f"f{i} must be one sentence ending in a full stop")
        if not (t[:1].isupper() or t.startswith("<")):
            p.append(f"f{i} starts lowercase")
        # v1.1 §4.1: every name in a fact is a tag; a capitalised word after the first is a
        # name left in the open
        words = TAG.sub("x", t).split()
        bare = [w for w in words[1:] if w[:1].isupper() and w.strip(",.'") != "I"]
        if bare:
            p.append(f"f{i} has untagged names {bare}")
        # a bare day number reads as "on day 16"; it always comes as <X_Day> <X_Month>
        for mm in re.finditer(r"<([A-Za-z_]*Day[A-Za-z_]*)>(\s*<[A-Za-z_]+>)?", t):
            nxt = (mm.group(2) or "").strip()
            if f"<{mm.group(1)}>" in e["tags"] and e["tags"][f"<{mm.group(1)}>"].get("cat") == "date" \
                    and "Month" not in nxt:
                p.append(f"f{i}: <{mm.group(1)}> must be followed directly by a Month tag")
        nw = NUMWORDS.findall(TAG.sub("", t))
        if nw:
            p.append(f"f{i} has untagged number words {nw}")
        if i > 1 and LINKERS.search(TAG.sub("", t)):
            p.append(f"f{i} leans on another fact ({LINKERS.search(TAG.sub('', t)).group()!r})")
    # a value that carries its unit must not be followed by the unit again:
    # "with <Samples> samples" + "61 samples" fills to "61 samples samples"
    for i, t in enumerate(facts, 1):
        for mm in re.finditer(r"(<[A-Za-z_]+>)\s+([A-Za-z-]+)", t):
            v = e["tags"].get(mm.group(1))
            nxt = mm.group(2).lower().rstrip("s")
            if v and any(str(v[s]).split()[-1].lower().rstrip("s") == nxt
                         for s in ("truth", "rumor") if str(v.get(s, "")).strip()):
                p.append(f"f{i} repeats the unit of {mm.group(1)} after the tag")
    if sum(len(t) >= 2 for t in per_fact) < 2:
        p.append("fewer than two facts carry two or more tags")
    if not 6 <= len(defined) <= 11:
        p.append(f"{len(defined)} tags (6-11 required)")

    cats = {v.get("cat") for v in e["tags"].values()}
    if cats - CATS:
        p.append(f"unknown categories {sorted(cats - CATS, key=str)}")
    if REQUIRED - cats:
        p.append(f"missing categories {sorted(REQUIRED - cats)}")

    seen = {}
    for k, v in e["tags"].items():
        tr, ru = str(v.get("truth", "")), str(v.get("rumor", ""))
        if not tr or not ru:
            p.append(f"{k} empty value")
            continue
        if tr == ru:
            p.append(f"{k} truth equals rumor")
        for name, val in (("truth", tr), ("rumor", ru)):
            if re.match(r"(?i)(the|a|an)\s", val):
                p.append(f"{k} {name} starts with an article")
            # a decimal point inside a number is fine; any other full stop would break the
            # sentence splitter used by the audit
            if "." in re.sub(r"\d\.\d", "", val) or re.search(r"[?!;|<>\"]", val):
                p.append(f"{k} {name} contains forbidden punctuation")
            if val != val.strip() or "  " in val or len(val) > 40:
                p.append(f"{k} {name} spacing or length")
            prev = seen.get(val.lower())
            if prev and prev[0] != k:
                p.append(f"{k} {name} '{val}' equals {prev[0]} {prev[1]}")
            seen.setdefault(val.lower(), (k, name))
        # same type (v1.1 §5): same unit, same notation, same digit/no-digit form
        if (shape(tr) != tr or shape(ru) != ru) and shape(tr) != shape(ru):
            p.append(f"{k} truth/rumor differ in form: '{tr}' vs '{ru}'")
        if re.fullmatch(r"(1[5-9]|20)\d\d", tr) and "Year" not in k:
            p.append(f"{k} holds a year; the tag name must say Year")
        if tr in MONTHS and "Month" not in k:
            p.append(f"{k} holds a month; the tag name must say Month")
        # v1.1 §4.2: a date is split into <Year>, <Month>, <Day>, each its own tag
        if v.get("cat") == "date":
            for name, val in (("truth", tr), ("rumor", ru)):
                if re.fullmatch(r"\d{4}", val):
                    ok = "Year" in k
                elif val in MONTHS:
                    ok = "Month" in k
                elif re.fullmatch(r"\d{1,2}", val) and 1 <= int(val) <= 31:
                    ok = "Day" in k
                else:
                    ok = False
                if not ok:
                    p.append(f"{k} {name} '{val}': a date tag holds only a year (tag name "
                             f"with Year), a month name (Month) or a day number 1-31 (Day)")
                    break
        # v1.1 §4.2: values are proper names or numbers
        for name, val in (("truth", tr), ("rumor", ru)):
            if not re.search(r"\d", val) and not val[:1].isupper():
                p.append(f"{k} {name} '{val}' is neither a proper name nor a number")
                break

    # the seed text is the archived source story: every truth value, no rumor value
    seed = e["seed_text"]
    w = len(seed.split())
    if not 35 <= w <= 130:
        p.append(f"seed_text {w} words (35-130)")
    for k, v in e["tags"].items():
        if not appears(str(v.get("truth", "")), seed):
            p.append(f"{k} truth '{v.get('truth')}' not in seed_text")
        if appears(str(v.get("rumor", "")), seed):
            p.append(f"{k} rumor '{v.get('rumor')}' appears in seed_text")
    return p


def direction(a, b):
    """(kind, truth, rumor) as comparable numbers, or None."""
    if re.fullmatch(r"\d{4}", a) and re.fullmatch(r"\d{4}", b):
        return "year", int(a), int(b)
    if a in MONTHS and b in MONTHS:
        return "month", MONTHS.index(a), MONTHS.index(b)
    # '12 May' and 'May 2027' order by the whole date, not by the first number in them
    mo = "|".join(MONTHS)
    da, db = (re.fullmatch(rf"(\d+) ({mo})", s) for s in (a, b))
    if da and db:
        return ("date", MONTHS.index(da[2]) * 32 + int(da[1]),
                MONTHS.index(db[2]) * 32 + int(db[1]))
    ma, mb = (re.fullmatch(rf"({mo}) (\d{{4}})", s) for s in (a, b))
    if ma and mb:
        return ("date", int(ma[2]) * 12 + MONTHS.index(ma[1]),
                int(mb[2]) * 12 + MONTHS.index(mb[1]))
    x, y = NUM.search(a), NUM.search(b)
    if x and y and shape(a) == shape(b):
        return "number", float(x.group().replace(",", "")), float(y.group().replace(",", ""))
    return None


def file_checks(rows):
    """Problems that only exist across events. Returns a list of strings."""
    p = []
    # rumor direction must carry no signal: about half above, half below the truth
    d = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        for v in r["tags"].values():
            got = direction(str(v["truth"]), str(v["rumor"]))
            if got and got[1] != got[2]:
                d[got[0]][got[2] > got[1]] += 1
    for kind, (lo, hi) in sorted(d.items()):
        tot = lo + hi
        share = hi / tot
        print(f"  rumor direction {kind}: above {hi}, below {lo} ({share:.0%} above)")
        if tot >= 20 and not 0.40 <= share <= 0.60:
            p.append(f"rumor direction for {kind} is {share:.0%} above the truth (40-60% required)")

    fut = sum(r["tense"] == "future" for r in rows) / len(rows)
    if len(rows) >= 20 and not 0.40 <= fut <= 0.60:
        p.append(f"future events are {fut:.0%} (40-60% required)")

    # v1.1 §4.2: no repeated sentence frame within a domain
    frames = collections.defaultdict(list)
    for r in rows:
        for f in r["facts"]:
            frames[(r["domain"], TAG.sub("<>", f["text"]).lower())].append(r["event_id"])
    for (dom, fr), ids in frames.items():
        if len(ids) > 1:
            p.append(f"frame repeated in {dom}: {fr!r} {ids}")
    cross = collections.Counter(fr for (_, fr) in frames)
    rep = sum(1 for c in cross.values() if c > 1)
    print(f"  frames shared across domains: {rep}")

    # the same named entity should not keep coming back as the truth
    reuse = collections.defaultdict(set)
    for r in rows:
        for v in r["tags"].values():
            if v["cat"] in ("agent", "name"):
                reuse[str(v["truth"]).lower()].add(r["event_id"])
    for val, ids in sorted(reuse.items()):
        if len(ids) > 2:
            p.append(f"agent/name truth '{val}' used in {len(ids)} events {sorted(ids)}")
    places = collections.Counter(str(v["truth"]).lower() for r in rows
                                 for v in r["tags"].values() if v["cat"] == "place")
    for val, c in places.items():
        if c > max(4, len(rows) // 50):
            p.append(f"place truth '{val}' used {c} times")

    # two events with the same subject are the same story
    subj = collections.defaultdict(list)
    for r in rows:
        f1 = r["facts"][0]["text"]
        key = tuple(sorted(str(r["tags"][t]["truth"]).lower() for t in TAG.findall(f1)
                           if t in r["tags"] and r["tags"][t]["cat"] in ("agent", "name")))
        if key:
            subj[key].append(r["event_id"])
    for key, ids in subj.items():
        if len(ids) > 1:
            p.append(f"same f1 subject {key} in {ids}")
    return p


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    bad = 0
    ids = collections.Counter(r.get("event_id") for r in rows)
    for k, c in ids.items():
        if c > 1:
            print(f"duplicate event_id {k}")
            bad += 1
    for r in rows:
        p = check(r)
        if p:
            bad += 1
            print(r.get("event_id"), "; ".join(p))
    print()
    fp = file_checks(rows) if not bad else ["(file-level checks skipped until events pass)"]
    for x in fp:
        print("FILE", x)
    grid = collections.Counter((r["n"], r["m"]) for r in rows)
    dom = collections.Counter(r["domain"] for r in rows)
    tense = collections.Counter(r["tense"] for r in rows)
    ntags = [len(r["tags"]) for r in rows]
    print(f"\nevents {len(rows)}  failing {bad}  file-level problems {len(fp) if not bad else '-'}")
    print("  (n, m):", dict(grid))
    print("  domains:", dict(dom))
    print("  tense:", dict(tense))
    print(f"  tags per event: min {min(ntags)} max {max(ntags)} mean {sum(ntags) / len(ntags):.1f}")
    cats = collections.Counter(v["cat"] for r in rows for v in r["tags"].values())
    print("  tag categories:", dict(cats))
    names = collections.Counter(k for r in rows for k in r["tags"])
    print("  most common tag names:", names.most_common(8))
    sys.exit(1 if bad or fp else 0)


if __name__ == "__main__":
    main()
