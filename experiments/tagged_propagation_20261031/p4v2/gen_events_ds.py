"""P4 event table through the DeepSeek API (plan v1.2 + v1.1 §4-5, rules in BRIEF.md).

    python3 gen_events_ds.py slots              # 50 subtopics per domain -> slots_<domain>.json
    python3 gen_events_ds.py events [domain..]  # one call per slot -> batch_<k>_<domain>.jsonl
    python3 gen_events_ds.py balance            # rumor-direction coin flips, in place

Why it is built this way. The Haiku attempt let the writer choose its own stories and it
fell back to filling a handful of templates. Here the script fixes every slot up front
(subtopic, country, tense, n, id), the model writes exactly one event per call, and every
event must pass validate_events.check() plus the domain-level checks against what is already
accepted (sentence frames, reused entities, same subject) before it is kept. Validator
messages are sent back to the model, up to RETRIES times.

Rumor direction is not left to the model: models write rumors below the truth (the first
table had 98% below). `balance` flips a fixed coin per numeric/date tag and swaps truth and
rumor when the coin disagrees, rewriting the seed text to match, so the direction carries
no information whatever the model did.

Key: DEEPSEEK_API_KEY, else the file dskey at the repository root. Thinking is disabled.
"""
import collections
import glob
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import validate_events as V  # noqa: E402

API = "https://api.deepseek.com/chat/completions"
MODEL = "deepseek-flash"
RETRIES = 4
LOG = os.path.join(HERE, "ds_log.jsonl")

DOMAINS = [  # (batch, key, scope given to the writer, n5, n6)
    (1, "academic_conference", "academic conferences, congresses, workshops, symposia and "
        "summits in any field", 33, 17),
    (2, "business", "companies and markets: earnings, mergers, plants, recalls, layoffs, "
        "funding, retail, energy, shipping, airlines, banking, startups", 33, 17),
    (3, "sports", "many sports: results, transfers, records, injuries, venues, doping, "
        "coaching changes, tournaments, governing bodies", 33, 17),
    (4, "public_policy", "governments, councils, regulators, courts, elections: budgets, "
        "laws, rulings, planning, transport, housing, tax, negotiations", 33, 17),
    (5, "science", "research findings, studies, space missions, expeditions, labs, "
        "discoveries, research funding, technology", 32, 18),
    (6, "disaster_accident", "storms, floods, earthquakes, wildfires, droughts, industrial "
        "and transport accidents, collapses, fires, spills, rescues, relief; future events "
        "can be forecasts, warnings or planned drills", 32, 18),
    (7, "culture", "film, music, theatre, books, art, museums, festivals, heritage, awards, "
        "auctions, tours, exhibitions", 32, 18),
    (8, "health", "hospitals, clinics, outbreaks, vaccination, drug trials and approvals, "
        "waiting times, public health measures, medical staffing", 32, 18),
]
COUNTRIES = ["United Kingdom", "United States", "India", "Nigeria", "Brazil", "Japan",
             "Germany", "France", "Kenya", "Mexico", "Indonesia", "Canada", "Australia",
             "South Africa", "Spain", "Italy", "Netherlands", "Poland", "Turkey", "Egypt",
             "South Korea", "Vietnam", "Argentina", "Chile", "Sweden", "Norway",
             "Philippines", "Ghana", "New Zealand", "Ireland"]

_lock = threading.Lock()


def key():
    k = os.environ.get("DEEPSEEK_API_KEY")
    if not k:
        with open(os.path.join(HERE, "..", "..", "..", "dskey")) as f:
            k = f.read().strip()
    return k


def chat(messages, temperature=0.9, max_tokens=2500, json_mode=True):
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens, "stream": False, "thinking": {"type": "disabled"}}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key()}"})
    last = None
    for a in range(8):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                d = json.loads(r.read().decode())
            msg = d["choices"][0]["message"]
            if msg.get("reasoning_content"):
                raise RuntimeError("thinking was not disabled")
            return msg["content"], d.get("usage", {})
        except (urllib.error.URLError, TimeoutError, KeyError, json.JSONDecodeError) as e:
            last = e
            # 429 at 32 workers: back off with jitter so the threads do not retry in step
            slow = isinstance(e, urllib.error.HTTPError) and e.code == 429
            time.sleep((10 if slow else 3) * (a + 1) + random.random() * 5)
    raise RuntimeError(f"API failed: {last}")


# ---------------------------------------------------------------- slots

def make_slots():
    for k, dom, scope, n5, n6 in DOMAINS:
        path = os.path.join(HERE, f"slots_{dom}.json")
        if os.path.exists(path):
            print(f"{dom}: slots exist")
            continue
        prompt = (f"List 50 clearly different kinds of news story for the domain '{dom}' "
                  f"({scope}). Each must be a distinct situation, not the same story with a "
                  f"different noun. Return JSON: {{\"topics\": [{{\"topic\": snake_case, "
                  f"\"angle\": one sentence describing the story}}, ...]}} with exactly 50.")
        out, _ = chat([{"role": "user", "content": prompt}], temperature=1.0, max_tokens=5000)
        uniq = {}
        for t in json.loads(out)["topics"]:
            uniq.setdefault(t["topic"], t)
        topics = list(uniq.values())[:50]
        assert len(topics) == 50, (dom, len(uniq))
        rng = random.Random(f"slots-{dom}")
        ns = [5] * n5 + [6] * n6
        rng.shuffle(ns)
        tenses = ["future", "past"] * 25
        rng.shuffle(tenses)
        off = (k - 1) * 4
        slots = [dict(event_id=f"E{2001 + (k - 1) * 50 + i}", domain=dom, topic=t["topic"],
                      angle=t["angle"], country=COUNTRIES[(off + i) % len(COUNTRIES)],
                      tense=tenses[i], n=ns[i]) for i, t in enumerate(topics)]
        json.dump(slots, open(path, "w"), indent=1, ensure_ascii=False)
        print(f"{dom}: {len(slots)} slots")


# ---------------------------------------------------------------- events

RULES = open(os.path.join(HERE, "BRIEF.md")).read()
RULES = RULES[RULES.index("## What an event is"):RULES.index("## Example")]
EXAMPLE = open(os.path.join(HERE, "example_event.json")).read()

SYSTEM = f"""You write one event for a dataset of invented news stories. Follow these rules exactly.

{RULES}
Example of a valid event:
{EXAMPLE}

Return only the JSON object of the event."""


def ask(slot):
    tense = ("future: the story is about something announced or planned"
             if slot["tense"] == "future" else "past: the story has already happened")
    return (f"Write event {slot['event_id']}.\n"
            f"domain: {slot['domain']}\ntopic: {slot['topic']} ({slot['angle']})\n"
            f"set in: {slot['country']} (use real places there; invent people and most "
            f"organisations)\ntense: {tense}\nn = {slot['n']} facts, m = 1\n"
            f"Write the seed_text first as a real news report, then extract the facts. "
            f"Make each fact a sentence frame of its own: do not use generic frames like "
            f"\"The scheme costs <Cost>.\" or \"<Body> approved the plan.\"")


class Registry:
    """What is already accepted: frames per domain, entities and subjects overall."""

    def __init__(self, rows):
        self.frames = collections.defaultdict(set)
        self.ents = collections.Counter()
        self.places = collections.Counter()
        self.subjects = set()
        for r in rows:
            self.add(r)

    @staticmethod
    def frame(t):
        return V.TAG.sub("<>", t).lower()

    @staticmethod
    def subject(e):
        f1 = e["facts"][0]["text"]
        return tuple(sorted(str(e["tags"][t]["truth"]).lower() for t in V.TAG.findall(f1)
                            if t in e["tags"] and e["tags"][t]["cat"] in ("agent", "name")))

    def problems(self, e):
        p = []
        for f in e["facts"]:
            if self.frame(f["text"]) in self.frames[e["domain"]]:
                p.append(f"{f['id']} repeats a sentence frame already used in this domain; "
                         f"rewrite it with a different structure")
        frs = [self.frame(f["text"]) for f in e["facts"]]
        if len(set(frs)) < len(frs):
            p.append("two facts share a sentence frame")
        for k, v in e["tags"].items():
            val = str(v["truth"]).lower()
            if v["cat"] in ("agent", "name") and self.ents[val] >= 2:
                p.append(f"{k} truth '{v['truth']}' is already used by other events; invent "
                         f"another")
            if v["cat"] == "place" and self.places[val] >= 7:
                p.append(f"{k} truth '{v['truth']}' is overused; pick another place")
        s = self.subject(e)
        if s and s in self.subjects:
            p.append(f"the f1 subject {s} is already the subject of another event")
        return p

    def add(self, e):
        for f in e["facts"]:
            self.frames[e["domain"]].add(self.frame(f["text"]))
        for v in e["tags"].values():
            if v["cat"] in ("agent", "name"):
                self.ents[str(v["truth"]).lower()] += 1
            if v["cat"] == "place":
                self.places[str(v["truth"]).lower()] += 1
        s = self.subject(e)
        if s:
            self.subjects.add(s)


def normalise(e, slot):
    """Fields the script owns; the model cannot get these wrong."""
    e = {k: e.get(k) for k in ("seed_text", "facts", "tags")}
    out = {k: slot[k] for k in ("event_id", "domain", "topic", "tense", "n")}
    out["m"] = 1
    out.update(e)
    if isinstance(out["facts"], list):
        out["facts"] = [{"id": f"f{i}", "text": (f.get("text") if isinstance(f, dict)
                                                 else str(f))}
                        for i, f in enumerate(out["facts"], 1)]
    return out


def write_one(slot, reg):
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ask(slot)}]
    last = None
    for attempt in range(RETRIES + 1):
        out, usage = chat(msgs)
        try:
            e = normalise(json.loads(out), slot)
            p = V.check(e)
            if not p:
                with _lock:
                    p = reg.problems(e)
                    if not p:
                        reg.add(e)
        except (json.JSONDecodeError, AttributeError, TypeError, KeyError) as x:
            e, p = None, [f"output is not a valid event object ({type(x).__name__}: {x})"]
        with _lock, open(LOG, "a") as f:
            f.write(json.dumps({"event_id": slot["event_id"], "attempt": attempt,
                                "problems": p, "usage": usage}) + "\n")
        if not p:
            return e, attempt
        last = p
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": "The validator rejected it:\n- " + "\n- ".join(p)
                  + "\nFix every problem and return the whole corrected event as JSON."}]
    return None, last


def run_slot(slot, path, reg):
    try:
        e, info = write_one(slot, reg)
    except RuntimeError as x:          # one slot's API trouble must not stop the pool
        e, info = None, [str(x)]
    if e is None:
        print(f"  {slot['event_id']} FAILED: {info[:3]}", flush=True)
        return slot["event_id"]
    with _lock, open(path, "a") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  {slot['event_id']} ok after {info} retries", flush=True)
    return None


def make_events(only):
    """All open slots of all domains in one pool (WORKERS, default 32). Slots of the same
    domain run side by side; the registry check and insert happen under one lock at accept
    time, so a clash just sends that event back for another attempt."""
    rows = [json.loads(l) for p in glob.glob(os.path.join(HERE, "batch_*.jsonl"))
            for l in open(p) if l.strip()]
    reg = Registry(rows)
    done = {r["event_id"] for r in rows}
    jobs = []
    for k, dom, *_ in DOMAINS:
        if only and dom not in only:
            continue
        slots = json.load(open(os.path.join(HERE, f"slots_{dom}.json")))
        slots = slots[:int(os.environ.get("LIMIT", len(slots)))]   # LIMIT=5 for a pilot
        path = os.path.join(HERE, f"batch_{k}_{dom}.jsonl")
        jobs += [(s, path) for s in slots if s["event_id"] not in done]
    print(f"{len(jobs)} open slots", flush=True)
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 32))) as ex:
        fails = [x for x in ex.map(lambda j: run_slot(*j, reg), jobs) if x]
    print(f"{len(fails)} slots failed {fails}")


# ---------------------------------------------------------------- balance

def swap_ok(e, k):
    """Truth and rumor of tag k can swap: the truth occurs once in the seed and no other
    value contains it, so a plain substitution rewrites exactly this value."""
    v = e["tags"][k]
    tr, ru = str(v["truth"]), str(v["rumor"])
    pat = r"(?<![\w.])" + re.escape(tr) + r"(?![\w]|\.\d)"
    if len(re.findall(pat, e["seed_text"])) != 1:
        return False
    for k2, v2 in e["tags"].items():
        if k2 != k and any(re.search(pat, str(v2[s])) or
                           re.search(r"(?<![\w.])" + re.escape(ru) + r"(?![\w]|\.\d)",
                                     str(v2[s])) for s in ("truth", "rumor")):
            return False
    got = V.direction(tr, ru)
    if got and got[0] == "year":       # the truth year must still fit the tense
        new = int(ru)
        if (e["tense"] == "future" and new < 2026) or (e["tense"] == "past" and new > 2026):
            return False
    return True


def balance():
    for path in sorted(glob.glob(os.path.join(HERE, "batch_*.jsonl"))):
        rows = [json.loads(l) for l in open(path) if l.strip()]
        swapped = 0
        for e in rows:
            for k in sorted(e["tags"]):
                v = e["tags"][k]
                got = V.direction(str(v["truth"]), str(v["rumor"]))
                if not got or got[1] == got[2]:
                    continue
                want_above = random.Random(f"{e['event_id']}{k}").random() < 0.5
                if (got[2] > got[1]) == want_above or not swap_ok(e, k):
                    continue
                tr, ru = str(v["truth"]), str(v["rumor"])
                pat = r"(?<![\w.])" + re.escape(tr) + r"(?![\w]|\.\d)"
                seed = re.sub(pat, lambda _: ru, e["seed_text"])
                trial = json.loads(json.dumps(e))
                trial["seed_text"] = seed
                trial["tags"][k].update(truth=ru, rumor=tr)
                if not V.check(trial):
                    e.update(trial)
                    swapped += 1
        with open(path, "w") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"{os.path.basename(path)}: swapped {swapped}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "slots":
        make_slots()
    elif cmd == "events":
        make_events(sys.argv[2:])
    elif cmd == "balance":
        balance()
    else:
        print(__doc__)
