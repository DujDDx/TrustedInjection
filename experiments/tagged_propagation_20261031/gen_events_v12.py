"""P4 的事件起草器：为正式规模批量产出 Phase 0/1 事件（计划 v1.2 §4）。

P0 的 20 个事件是手写的，单个约 1.5KB JSON；400 个要 578KB 手写内容，不现实。这里用
写作者起草，再用 `validate_events.check` 做全自动校验——这个任务比结构模拟那条线容易，
因为标签是显式的、Truth 值必须出现在种子新闻里，错了立刻能查出来。

**拆成两次调用**：一次只写"种子新闻 + 带标签的事实"（不写值），一次只给这些标签填
truth/rumor。一次调用同时满足十来条约束时实测 0/4 通过——每轮换一条违反，带反馈重抽
不收敛；拆开后每半的任务简单得多，取值那半还能机械修复（去冠词、对齐大小写）。

用法：
  python3 gen_events_v12.py --n 40 --out events_p4.jsonl [--start 0]
环境：TI_BASE（默认指向 dujddxServer 的 llama.cpp）
"""
import json
import os
import random
import re
import sys
import zlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "provenance_sym_20261030"))
from llm_client import complete_json          # noqa: E402
from validate_events import check             # noqa: E402

TAG = re.compile(r"<[A-Za-z_]+>")

DOMAINS = [
    "an academic conference", "a listed company's earnings", "a football transfer",
    "a city council vote", "a scientific paper", "a natural disaster",
    "a film festival", "a hospital announcement", "a court ruling",
    "a central bank decision", "a museum exhibition", "a university appointment",
    "a factory opening", "an airline route change", "a union negotiation",
    "a charity campaign", "a product recall", "an election result",
    "a transport strike", "a housing development", "a farm subsidy",
    "a telecoms outage", "a book prize", "a space launch",
    "a bank branch closure", "a music tour", "a school inspection",
    "a fisheries dispute", "a wildfire response", "a vaccine rollout",
    "a railway upgrade", "a brewery takeover", "a port expansion",
    "a planning inquiry", "a doping case", "a census release",
    "a power plant shutdown", "a film box office", "a startup funding round",
    "a heritage listing",
]

P_STRUCT = """Write one short factual news event, then break it into __N__ numbered facts.

The event is about __DOMAIN__. Tense: __TENSE__.

Here is a complete worked example for a different subject. Copy its *shape*, not its
subject, its tags or its values.

{"domain": "a library closure", "tense": "past", "n": 5, "m": 1,
 "seed_text": "Hull City Council said on 4 February that the Riverside Library in Hull would close in March. The building, which opened in 1978, serves about 12,000 readers a year. Council leader Ann Pike said the closure would save 340,000 pounds.",
 "facts": [
   {"id": "f1", "text": "The <Library> in <City> will close in <Month>."},
   {"id": "f2", "text": "<Council> announced the closure on <Day>."},
   {"id": "f3", "text": "The building opened in <Year> and serves about <Readers> readers a year."},
   {"id": "f4", "text": "<Person_A> is the council leader."},
   {"id": "f5", "text": "The closure would save <Savings>."}],
 "tags": {
   "<Library>": {"cat": "name"}, "<City>": {"cat": "place"},
   "<Month>": {"cat": "date"}, "<Council>": {"cat": "agent"},
   "<Day>": {"cat": "date"}, "<Year>": {"cat": "date"},
   "<Readers>": {"cat": "quantity"}, "<Person_A>": {"cat": "agent"},
   "<Savings>": {"cat": "quantity"}}}

Rules for the facts:
- exactly __N__ facts, ids f1..fn in order;
- **every tag is written inline in a fact sentence**, exactly as in the example: the tag
  name in angle brackets stands where the value will go. A fact with no tag is invalid;
- f1 states the subject and must contain at least one of: a person, an organisation, a
  place, or an event name (as a tag);
- each fact contains 1 to 3 tags; **a tag appears in exactly one fact** — if you need to
  refer to the same entity again, use a plain noun ("the company", "the court", "the
  city"), never the tag twice; at least two facts carry two or more tags;
- 6 to 11 distinct tags in total;
- categories used across the event must include date, place, agent and quantity;
- no digits outside tags.

Rules for the seed text:
- 2 to 4 plain news sentences, no headline, past or future tense as requested;
- it must read like a real wire story and be consistent with every fact.

Output only the JSON object with the keys shown. Every tag value is just {"cat": "..."}.
Do not invent values yet."""

P_VALUES = """Below is a news event and the tags it uses. For each tag give two values.

EVENT
__SEED__

TAGS
__TAGS__

Rules:
- "truth" must appear **character-for-character** in the event text above (same case);
- "rumor" must be plausible and of the same kind (a year for a year, a city for a city, a
  number of similar magnitude for a number), different from truth, and **must not appear
  anywhere in the event text**;
- neither value may start with an article ("the", "a", "an");
- neither value may contain a full stop (a decimal point inside a number is fine);
- keep values short: a name, a place, a date or a figure, not a clause.

Output only JSON: {"<Tag>": {"truth": "...", "rumor": "..."}, ...}"""


def shape(v):
    if re.fullmatch(r"\d{4}", v):
        return "year"
    if re.fullmatch(r"[\d,]+", v):
        return "int"
    if re.fullmatch(r"[\d.,]+", v):
        return "num"
    if re.fullmatch(r"[\d.,]+\s?[%a-zA-Z]*", v):
        return "num+unit"
    if v[:1].isupper():
        return "proper"
    return "other"


def normalize(e):
    """把模型常见的几种形状统一成 P0 的 schema。"""
    f = e.get("facts")
    if isinstance(f, list) and f and isinstance(f[0], str):
        e["facts"] = [{"id": f"f{i + 1}", "text": s} for i, s in enumerate(f)]
    elif isinstance(f, dict):
        e["facts"] = [{"id": k, "text": v if isinstance(v, str) else str(v)}
                      for k, v in f.items()]
    t = e.get("tags")
    if isinstance(t, list):
        e["tags"] = {x.get("name") or x.get("tag"): {"cat": x.get("cat", "name")}
                     for x in t if isinstance(x, dict)}
    return e


def struct_problems(e, n):
    """只看结构，不看值（值还没填）。"""
    p = []
    facts = [f["text"] for f in e["facts"]]
    if len(facts) != n:
        return [f"{len(facts)} 条事实，要求 {n}"]
    per = [TAG.findall(t) for t in facts]
    owner = {}
    for i, tags in enumerate(per, 1):
        for t in set(tags):
            if t in owner:
                p.append(f"{t} 同时出现在 f{owner[t]} 与 f{i}")
            owner[t] = i
    defined = set(e.get("tags") or {})
    used = set(owner)
    if used - defined:
        p.append(f"未定义的标签 {sorted(used - defined)}")
    if defined - used:
        p.append(f"定义了但没用的标签 {sorted(defined - used)}")
    f1c = {e["tags"][t].get("cat") for t in per[0] if t in e["tags"]}
    if not f1c & {"agent", "name", "place"}:
        p.append("f1 没有点明主体（缺 agent/name/place 标签）")
    for i, tags in enumerate(per, 1):
        if not 1 <= len(tags) <= 3:
            p.append(f"f{i} 有 {len(tags)} 个标签")
        if re.search(r"\d", TAG.sub("", facts[i - 1])):
            p.append(f"f{i} 有标签外的数字")
    if sum(len(set(t)) >= 2 for t in per) < 2:
        p.append("少于两条事实带两个以上标签")
    if not 6 <= len(defined) <= 11:
        p.append(f"{len(defined)} 个标签（要求 6-11）")
    cats = {v.get("cat") for v in e["tags"].values()}
    if not {"date", "place", "agent", "quantity"} <= cats:
        p.append(f"缺类别 {sorted({'date', 'place', 'agent', 'quantity'} - cats)}")
    return p


def repair_values(e):
    """机械修掉两类可修的取值问题：去掉冠词、对齐种子新闻里的大小写。"""
    seed = e["seed_text"]
    low = seed.lower()
    for v in e["tags"].values():
        for key in ("truth", "rumor"):
            v[key] = re.sub(r"(?i)^(the|a|an)\s+", "", str(v.get(key, ""))).strip()
        tr = str(v["truth"])
        if tr and tr not in seed and tr.lower() in low:
            i = low.index(tr.lower())
            v["truth"] = seed[i:i + len(tr)]
    return e


def value_problems(e):
    seed = e["seed_text"]
    p = []
    for k, v in e["tags"].items():
        tr, ru = str(v.get("truth", "")), str(v.get("rumor", ""))
        if not tr or not ru:
            p.append(f"{k} 值为空")
            continue
        if tr == ru:
            p.append(f"{k} truth 与 rumor 相同")
        if tr not in seed:
            p.append(f"{k} truth '{tr}' 不在种子新闻里")
        if ru in seed:
            p.append(f"{k} rumor '{ru}' 出现在种子新闻里")
        if shape(tr) != shape(ru):
            p.append(f"{k} 两类不同：{shape(tr)} vs {shape(ru)}")
        if re.match(r"(?i)(the|a|an)\s", tr) or re.match(r"(?i)(the|a|an)\s", ru):
            p.append(f"{k} 值以冠词开头")
    return p


def draft(domain, n, tense, seed, tries=4):
    last = None
    for t in range(tries):
        prompt = (P_STRUCT.replace("__DOMAIN__", domain).replace("__N__", str(n))
                  .replace("__TENSE__", tense))
        if last:
            prompt += ("\n\nYour previous attempt broke these rules; fix them and output "
                       "the whole JSON again:\n- " + "\n- ".join(last[:6]))
        try:
            e = complete_json(prompt, max_tokens=1400, temperature=0.85, seed=seed + t)
        except Exception as ex:                # noqa: BLE001
            last = [f"{type(ex).__name__}: {str(ex)[:80]}"]
            print(f"    结构调用失败：{last[0]}")
            continue
        e = normalize(e)
        e["domain"], e["tense"], e["n"], e["m"] = domain, tense, n, 1
        probs = struct_problems(e, n)
        if probs:
            last = probs
            print(f"    第 {t + 1} 次结构不合格：{probs[:3]}")
            continue

        tags = "\n".join(f"{k}  ({v.get('cat')})" for k, v in e["tags"].items())
        try:
            vals = complete_json(P_VALUES.replace("__SEED__", e["seed_text"])
                                 .replace("__TAGS__", tags),
                                 max_tokens=700, temperature=0.8, seed=seed + 100 + t)
        except Exception as ex:                # noqa: BLE001
            last = [f"values: {type(ex).__name__}: {str(ex)[:80]}"]
            print(f"    取值调用失败：{last[0]}")
            continue
        # 取值那半常把键写成不带尖括号（Court 而不是 <Court>），按三种写法找
        low = {str(k).strip("<>").lower(): v for k, v in vals.items()}
        for k in e["tags"]:
            got = (vals.get(k) or vals.get(k.strip("<>"))
                   or low.get(k.strip("<>").lower()) or {})
            if not isinstance(got, dict):
                got = {}
            e["tags"][k]["truth"] = str(got.get("truth", ""))
            e["tags"][k]["rumor"] = str(got.get("rumor", ""))
        repair_values(e)
        probs = value_problems(e) + check(e)
        if not probs:
            return e, []
        last = probs
        print(f"    第 {t + 1} 次取值不合格：{probs[:3]}")
    return None, last


def main():
    a = sys.argv
    total = int(a[a.index("--n") + 1]) if "--n" in a else 40
    out = a[a.index("--out") + 1] if "--out" in a else "events_p4.jsonl"
    start = int(a[a.index("--start") + 1]) if "--start" in a else 0
    rng = random.Random(20261101)
    domains = DOMAINS[:]
    rng.shuffle(domains)

    seen = set()
    if start and os.path.exists(out):
        for line in open(out):
            e = json.loads(line)
            f1t = TAG.findall(e["facts"][0]["text"])
            if f1t:
                seen.add(str(e["tags"][f1t[0]]["truth"]).lower())

    ok = 0
    with open(out, "a" if start else "w") as fo:
        for i in range(start, start + total):
            n = 6 if i % 3 == 0 else 5
            dom = domains[i % len(domains)]
            tense = "past" if i % 5 < 3 else "future"
            print(f"\n== E{i + 1001:04d}  n={n}  {tense}  {dom}")
            e, probs = draft(dom, n, tense, seed=zlib.crc32(f"{dom}-{i}".encode()) % 90000)
            if e is None:
                print(f"    放弃：{probs[:3]}")
                continue
            f1t = TAG.findall(e["facts"][0]["text"])
            subj = str(e["tags"][f1t[0]]["truth"]).lower() if f1t else ""
            if subj in seen:
                print(f"    主体 '{subj}' 重复，丢弃")
                continue
            seen.add(subj)
            e["event_id"] = f"E{i + 1001:04d}"
            fo.write(json.dumps(e, ensure_ascii=False) + "\n")
            fo.flush()
            ok += 1
            print(f"    写出 {e['event_id']}（{subj}）")
    print(f"\n合计 {ok}/{total} 个事件")


if __name__ == "__main__":
    main()
