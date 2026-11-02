"""P4 event table through the DeepSeek API (plan v1.2 + v1.1 §4-5, rules in BRIEF.md).

    python3 gen_events_ds.py slots              # 50 subtopics per domain -> slots_<domain>.json
    python3 gen_events_ds.py events [domain..]  # one call per slot -> batch_<k>_<domain>.jsonl
    python3 gen_events_ds.py repair [ids..]     # rework existing events in place (THINK=medium)

Why it is built this way. The Haiku attempt let the writer choose its own stories and it
fell back to filling a handful of templates. Here the script fixes every slot up front
(subtopic, country, tense, n, id), the model writes exactly one event per call, and every
event must pass validate_events.check(), the domain-level checks against the other events
(sentence frames, reused entities, same subject) and the assigned rumor directions before it
is kept. Problems are sent back to the model, up to RETRIES times.

Rumor direction is assigned, not left to the model (models write rumors below the truth; the
first table had 98% below). A fixed coin per (event, tag) says whether the rumor must be
higher or lower than the truth, and the model writes the rumor to match. An earlier version
swapped truth and rumor afterwards instead; that broke the truth story (an IPO priced "above
the range" below it, a road reopening before it closed), so it is gone.

Key: DEEPSEEK_API_KEY, else the file dskey at the repository root. Thinking is disabled
unless THINK is set (THINK=medium for repair: careful editing, coherence across facts).
"""
import collections
import glob
import http.client
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
THINK = os.environ.get("THINK")         # None = thinking off; "low"/"medium"/"high"

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


def chat(messages, temperature=0.9, max_tokens=2500, json_mode=True, think=None):
    think = think if think is not None else THINK
    body = {"model": MODEL, "messages": messages, "temperature": temperature,
            "max_tokens": max_tokens + (12000 if think else 0), "stream": False,
            "thinking": {"type": "enabled" if think else "disabled"}}
    if think:
        body["reasoning_effort"] = think
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    req = urllib.request.Request(API, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json",
                                          "Authorization": f"Bearer {key()}"})
    last = None
    for a in range(8):
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read().decode())
            msg = d["choices"][0]["message"]
            if msg.get("reasoning_content") and not think:
                raise RuntimeError("thinking was not disabled")
            return msg["content"], d.get("usage", {})
        except (urllib.error.URLError, http.client.HTTPException, OSError, KeyError,
                json.JSONDecodeError) as e:     # IncompleteRead killed a 400-event run
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


# ---------------------------------------------------------------- shared checks

RULES = open(os.path.join(HERE, "BRIEF.md")).read()
RULES = RULES[RULES.index("## What an event is"):RULES.index("## Example")]
EXAMPLE = open(os.path.join(HERE, "example_event.json")).read()

SYSTEM = f"""You write one event for a dataset of invented news stories. Follow these rules exactly.

{RULES}
Example of a valid event:
{EXAMPLE}

Return only the JSON object of the event."""


def want_higher(event_id, tag):
    """The assigned rumor direction: a fixed coin per (event, tag)."""
    return random.Random(f"{event_id}{tag}").random() < 0.5


def direction_problems(e):
    p = []
    for k, v in e["tags"].items():
        got = V.direction(str(v["truth"]), str(v["rumor"]))
        if got and got[1] != got[2] and (got[2] > got[1]) != want_higher(e["event_id"], k):
            p.append(f"{k}: the rumor must be {'higher' if want_higher(e['event_id'], k) else 'lower'}"
                     f" than the truth ('{v['truth']}'); change the rumor")
    return p


def direction_hint(e):
    """Directions for the tags an event already has (new tags are checked after)."""
    hs = [f"{k} rumor {'higher' if want_higher(e['event_id'], k) else 'lower'}"
          for k, v in e["tags"].items() if V.direction(str(v["truth"]), str(v["rumor"]))]
    return "; ".join(hs) or "none yet"


class Registry:
    """Frames, entities and subjects of every event, keyed by event, so that a repaired
    event replaces its old version and is compared only with the others."""

    def __init__(self, rows):
        self.ev = {}
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

    def entry(self, e):
        return dict(domain=e["domain"],
                    frames={self.frame(f["text"]) for f in e["facts"]},
                    ents=[str(v["truth"]).lower() for v in e["tags"].values()
                          if v["cat"] in ("agent", "name")],
                    places=[str(v["truth"]).lower() for v in e["tags"].values()
                            if v["cat"] == "place"],
                    subject=self.subject(e))

    def problems(self, e):
        others = [x for i, x in self.ev.items() if i != e["event_id"]]
        frames = set().union(*(x["frames"] for x in others if x["domain"] == e["domain"]))
        ents = collections.Counter(v for x in others for v in set(x["ents"]))
        places = collections.Counter(v for x in others for v in x["places"])
        subjects = {x["subject"] for x in others if x["subject"]}
        p = []
        for f in e["facts"]:
            if self.frame(f["text"]) in frames:
                p.append(f"{f['id']} repeats a sentence frame already used in this domain; "
                         f"rewrite it with a different structure")
        frs = [self.frame(f["text"]) for f in e["facts"]]
        if len(set(frs)) < len(frs):
            p.append("two facts share a sentence frame")
        for k, v in e["tags"].items():
            val = str(v["truth"]).lower()
            if v["cat"] in ("agent", "name") and ents[val] >= 2:
                p.append(f"{k} truth '{v['truth']}' is already used by other events; invent "
                         f"another")
            if v["cat"] == "place" and places[val] >= 7:
                p.append(f"{k} truth '{v['truth']}' is overused; pick another place")
        s = self.subject(e)
        if s and s in subjects:
            p.append(f"the f1 subject {s} is already the subject of another event")
        return p

    def add(self, e):
        self.ev[e["event_id"]] = self.entry(e)


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


def accept(msgs, slot, reg):
    """Ask, check, send problems back; return (event, attempts) or (None, problems)."""
    last = None
    for attempt in range(RETRIES + 1):
        out, usage = chat(msgs)
        try:
            e = normalise(json.loads(out), slot)
            p = V.check(e)
            if not p:
                p = direction_problems(e)
            if not p:
                with _lock:
                    p = reg.problems(e)
                    if not p:
                        reg.add(e)
        except (json.JSONDecodeError, AttributeError, TypeError, KeyError) as x:
            e, p = None, [f"output is not a valid event object ({type(x).__name__}: {x})"]
        with _lock, open(LOG, "a") as f:
            f.write(json.dumps({"event_id": slot["event_id"], "attempt": attempt,
                                "think": THINK, "problems": p, "usage": usage}) + "\n")
        if not p:
            return e, attempt
        last = p
        msgs += [{"role": "assistant", "content": out},
                 {"role": "user", "content": "The validator rejected it:\n- " + "\n- ".join(p)
                  + "\nFix every problem and return the whole corrected event as JSON."}]
    return None, last


# ---------------------------------------------------------------- events

def ask(slot):
    tense = ("future: the story is about something announced or planned"
             if slot["tense"] == "future" else "past: the story has already happened")
    return (f"Write event {slot['event_id']}.\n"
            f"domain: {slot['domain']}\ntopic: {slot['topic']} ({slot['angle']})\n"
            f"set in: {slot['country']} (use real places there; invent people and most "
            f"organisations)\ntense: {tense}\nn = {slot['n']} facts, m = 1\n"
            f"Write the seed_text first as a real news report, then extract the facts. "
            f"Make each fact a sentence frame of its own: do not use generic frames like "
            f"\"The scheme costs <Cost>.\" or \"<Body> approved the plan.\" Rumor directions "
            f"are assigned per tag after you write; the validator will tell you which to flip.")


def run_slot(slot, path, reg):
    try:
        e, info = accept([{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": ask(slot)}], slot, reg)
    except RuntimeError as x:          # one slot's API trouble must not stop the pool
        e, info = None, [str(x)]
    if e is None:
        print(f"  {slot['event_id']} FAILED: {info[:3]}", flush=True)
        return slot["event_id"]
    with _lock, open(path, "a") as f:
        f.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"  {slot['event_id']} ok after {info} retries", flush=True)
    return None


def batch_files():
    return {dom: os.path.join(HERE, f"batch_{k}_{dom}.jsonl") for k, dom, *_ in DOMAINS}


def make_events(only):
    """All open slots of all domains in one pool (WORKERS, default 16). Slots of the same
    domain run side by side; the registry check and insert happen under one lock at accept
    time, so a clash just sends that event back for another attempt."""
    rows = [json.loads(l) for p in batch_files().values() if os.path.exists(p)
            for l in open(p) if l.strip()]
    reg = Registry(rows)
    done = {r["event_id"] for r in rows}
    jobs = []
    for dom, path in batch_files().items():
        if only and dom not in only:
            continue
        slots = json.load(open(os.path.join(HERE, f"slots_{dom}.json")))
        slots = slots[:int(os.environ.get("LIMIT", len(slots)))]   # LIMIT=5 for a pilot
        jobs += [(s, path) for s in slots if s["event_id"] not in done]
    print(f"{len(jobs)} open slots", flush=True)
    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 16))) as ex:
        fails = [x for x in ex.map(lambda j: run_slot(*j, reg), jobs) if x]
    print(f"{len(fails)} slots failed {fails}")


# ---------------------------------------------------------------- repair

REPAIR = """Repair event {eid} so that it follows every rule. Keep its story, subject, country, tense and number of facts; change only what the rules require.

Known problems:
{problems}

Also, whatever the list says:
- Make the truth values one coherent story across all facts (dates in order, parts not above totals, comparisons true). Rumors need not be plausible.
- Rumor directions for the current numeric/date tags: {hints}. Tags you create get directions from the validator.
- Rewrite seed_text as a natural news report of the truth story that states every truth value verbatim and no rumor value; it must agree with every fact.

Current event:
{event}"""


def repair(ids):
    review = {}
    rp = os.path.join(HERE, os.environ.get("REVIEW", "review_p4v2.jsonl"))
    if os.path.exists(rp):
        for l in open(rp):
            r = json.loads(l)
            review[r["event_id"]] = [f"{i.get('kind')} {i.get('where')}: {i.get('detail')}"
                                     for i in r["issues"] if i.get("kind") != "category"]
    files = batch_files()
    rows = {dom: [json.loads(l) for l in open(p)] for dom, p in files.items()}
    allrows = [e for rs in rows.values() for e in rs]
    # every repaired event is saved the moment it passes, so a crash loses only the events
    # in flight; a restart skips what is already done (delete the file to start over)
    prog = os.path.join(HERE, "repair_done.jsonl")
    done = {}
    if os.path.exists(prog):
        for l in open(prog):
            r = json.loads(l)
            done[r["event_id"]] = r
    reg = Registry([done.get(e["event_id"], e) for e in allrows])
    todo = [e for e in allrows if (not ids or e["event_id"] in ids) and e["event_id"] not in done]
    print(f"{len(done)} already repaired, {len(todo)} to go", flush=True)

    def one(e):
        probs = V.check(e) + review.get(e["event_id"], [])
        msg = REPAIR.format(eid=e["event_id"], problems="\n".join(f"- {x}" for x in probs)
                            or "- none found by the checks", hints=direction_hint(e),
                            event=json.dumps(e, ensure_ascii=False))
        try:
            new, info = accept([{"role": "system", "content": SYSTEM},
                                {"role": "user", "content": msg}], e, reg)
        except Exception as x:          # noqa: BLE001 - one event must not stop the pool
            new, info = None, [f"{type(x).__name__}: {x}"]
        if new:
            with _lock, open(prog, "a") as f:
                f.write(json.dumps(new, ensure_ascii=False) + "\n")
        print(f"  {e['event_id']} {'ok after ' + str(info) if new else 'FAILED ' + str(info[:3])}",
              flush=True)
        return e["event_id"], new

    with ThreadPoolExecutor(int(os.environ.get("WORKERS", 16))) as ex:
        fails = [i for i, v in ex.map(one, todo) if v is None]
    for l in open(prog) if os.path.exists(prog) else []:
        r = json.loads(l)
        done[r["event_id"]] = r
    for dom, rs in rows.items():
        with open(files[dom], "w") as f:
            for e in rs:
                f.write(json.dumps(done.get(e["event_id"], e), ensure_ascii=False) + "\n")
    print(f"repaired {len(done)} in total, failed this run {len(fails)} {fails}")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "slots":
        make_slots()
    elif cmd == "events":
        make_events(sys.argv[2:])
    elif cmd == "repair":
        repair(set(sys.argv[2:]))
    else:
        print(__doc__)
