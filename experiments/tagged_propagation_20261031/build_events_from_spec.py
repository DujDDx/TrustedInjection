"""把紧凑规格展开成 P0 的事件 JSON，并逐条校验。

事实表由人写（用户 2026-11-01 要求），生成器只负责每个事件那 8 篇正文。手写完整 JSON
太容易错（一个标签写错位置，8 篇全废），所以这里用逐行规格：

    ### E1001 | academic_conference | future
    f1 | The <Conference> <Year> will be held in <City>. | <Conference>=name:ACL|EMNLP, <Year>=date:2027|1996, <City>=place:Kyoto|Shizuoka
    f2 | ...

- `### 事件ID | 领域 | 时态`，n 由 f 行数决定，m 固定 1。
- 每条事实：`id | 带标签的句子 | 标签定义`，标签定义写成 `名称=类别:真值|错值`。
- **种子新闻由真值代入事实后拼成**（P0 是手写的，这里改成机械拼接，理由见下）。

为什么种子新闻可以机械拼：它只用于校验"每个真值都有出处"，不进入训练数据（计划 §8.3
的产物里没有它）。手写 400 条种子新闻是纯负担，而真值本来就是我定的，出处不会错。

    python3 build_events_from_spec.py specs/p4_01.txt [...] --out events_p4.jsonl
"""
import json
import re
import sys

from validate_events import check

TAG = re.compile(r"<[A-Za-z_]+>")


def parse(path):
    events, cur = [], None
    for raw in open(path):
        line = raw.rstrip("\n")
        if not line.strip():
            continue
        if line.startswith("###"):
            head = [x.strip() for x in line[3:].split("|")]
            cur = {"event_id": head[0], "domain": head[1], "tense": head[2],
                   "m": 1, "facts": [], "tags": {}}
            events.append(cur)
            continue
        if line.lstrip().startswith("#"):
            continue
        if cur is None:
            raise SystemExit(f"{path}: 事实行出现在事件头之前：{line[:40]}")
        # 只切前两个竖线：标签定义里还有 truth|rumor 的竖线
        parts = [x.strip() for x in line.split("|", 2)]
        if len(parts) != 3:
            raise SystemExit(f"{path}: 事实行格式不对（要 3 段）：{line[:60]}")
        fid, text, defs = parts
        cur["facts"].append({"id": fid, "text": text})
        for d in defs.split(";"):
            d = d.strip()
            if not d:
                continue
            m = re.fullmatch(r"(<[A-Za-z_]+>)\s*=\s*(\w+):(.+)\|(.+)", d)
            if not m:
                raise SystemExit(f"{path}: 标签定义格式不对：{d}")
            cur["tags"][m.group(1)] = {"cat": m.group(2),
                                       "truth": m.group(3).strip(),
                                       "rumor": m.group(4).strip()}
    return events


def make_seed(e):
    """真值代入事实，拼成种子新闻。

    以 "The <X>" 开头的句子，若 X 的真值是大写开头的专名，去掉开头的 "The"——
    否则会拼出 "The NeurIPS 2028 will be held" 这种病句。这条只影响种子新闻的可读性，
    不改变任何标签或取值。
    """
    out = []
    for f in e["facts"]:
        s = f["text"]
        m = re.match(r"^The (<[A-Za-z_]+>)", s)
        if m and e["tags"][m.group(1)]["truth"][:1].isupper():
            s = s[4:]
        for t in set(TAG.findall(s)):
            s = s.replace(t, e["tags"][t]["truth"])
        # 机械拼接会把 "a indoor" 这类冠词配错，按值的首音改回（种子只用于校验，
        # 但读起来不该是病句）
        def fix(m):
            w = m.group(2).lower()
            if w[:1] in "aeiou" and not w.startswith(("one", "uni", "eu", "uk", "us")):
                return "an " + m.group(2)
            return "a " + m.group(2)
        s = re.sub(r"\b(a|an)\s+([A-Za-z][\w'-]*)", fix, s)
        out.append(s)
    return " ".join(out)


def appears(val, text):
    """值是否真的出现在文本里，按词边界判。

    朴素的子串判断会把 `5 percent` 判成出现在 `3.5 percent` 里、把 `21` 判成出现在
    `2021` 里、把 `2` 判成出现在 `12 December` 里——这些都不是同一个值。数字两侧
    不许紧邻数字或小数点。
    """
    pat = r"(?<![\d.])" + re.escape(val) + r"(?![\d.])"
    return re.search(pat, text) is not None


def extra(e):
    """规格展开后追加的检查。"""
    p = []
    seed = e["seed_text"]
    for k, v in e["tags"].items():
        if v["truth"] == v["rumor"]:
            p.append(f"{k} 真值与错值相同")
        if appears(v["rumor"], seed):
            p.append(f"{k} 错值 '{v['rumor']}' 出现在种子新闻里")
    if len(e["facts"]) != e["n"]:
        p.append("n 与事实条数不符")
    return p


def repair(e):
    """机械修掉可修的：去掉值开头的冠词。改完 truth 仍出现在种子里（去冠词不破坏包含）。"""
    for v in e["tags"].values():
        for key in ("truth", "rumor"):
            v[key] = re.sub(r"(?i)^(the|a|an)\s+", "", str(v[key])).strip()
    e["seed_text"] = make_seed(e)
    return e


PLACES = ["Aberdeen", "Belfast", "Bolton", "Cardiff", "Chester", "Derby", "Dundee",
          "Exeter", "Glasgow", "Hull", "Ipswich", "Leeds", "Norwich", "Oxford",
          "Perth", "Reading", "Swansea", "Truro", "Wrexham", "York"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December"]
PERSONS = ["Alice Munro", "Brian Kerr", "Clara Voss", "Derek Lam", "Elena Ruiz",
           "Farid Haddad", "Grace Okafor", "Hugo Lind", "Iris Chen", "Jonas Berg",
           "Kavita Rao", "Liam Doyle", "Marta Silva", "Nils Ek", "Owen Pryce",
           "Petra Novak", "Quinn Fraser", "Rosa Diaz", "Sven Holm", "Tara Mehta"]
ORGS = ["Northgate Trust", "Riverside Council", "Meridian Group", "Harbour Board",
        "Civic Society", "County Federation", "Regional Assembly", "Coastal Forum"]


def _shape(v):
    if v in MONTHS:
        return "month"
    if re.fullmatch(r"\d{4}", v):
        return "year"
    if re.fullmatch(r"\d{1,2} [A-Z][a-z]+", v):
        return "daymonth"
    if re.fullmatch(r"[\d.,]+ ?%", v):
        return "pct"
    if re.fullmatch(r"[\d.,]+ ?[a-z]+", v):
        return "amount"
    if re.fullmatch(r"[\d,]+", v):
        return "int"
    return "proper"


def _fresh(shape, cat, used, cur):
    """按形状和类别挑一个没用过的值，**量级贴着原值**。

    第一版对整数用了固定池（120…910），把"3 小时"改成了"120 小时"、"4 艘船"改成了
    "120 艘船"——量级离谱到一眼假。现在按原值就近取。
    """
    cur_s = str(cur)
    if shape == "month":
        pool = [m for m in MONTHS if m != cur_s]
    elif shape == "year":
        y = int(cur_s)
        pool = [str(y + d) for d in (2, -2, 3, -3, 5, -5, 8, -8)]
    elif shape == "daymonth":
        day, mon = cur_s.split(" ", 1)
        pool = [f"{d} {mon}" for d in ("3", "11", "19", "27") if d != day]
        pool += [f"{day} {m}" for m in MONTHS if m != mon]
    elif shape == "pct":
        n = float(re.sub(r"[^\d.]", "", cur_s))
        pool = [f"{max(1, round(n + d))} percent" for d in (3, -3, 6, -6, 9, -9)]
    elif shape == "amount":
        m = re.fullmatch(r"([\d.,]+) ?([a-z]+)", cur_s)
        num, unit = (m.group(1), m.group(2)) if m else (cur_s, "")
        n = float(num.replace(",", "").rstrip("m") or 0)
        pool = [f"{max(1, round(n * f))}m {unit}".replace("m m", "m").strip()
                for f in (0.7, 1.3, 0.85, 1.15)]
    elif shape == "int":
        n = int(re.sub(r"[^\d]", "", cur_s) or 0)
        pool = [str(max(1, n + d)) for d in (1, -1, 2, -2, 3, -3, 5, -5)]
    elif cat == "agent":
        pool = PERSONS + ORGS
    elif cat == "place":
        pool = PLACES
    else:
        pool = PERSONS
    for x in pool:
        if x.lower() not in used:
            return x
    return None


def dedupe(e):
    """同一事件内值不得重复，错值也不得等于别的标签的真值。

    实测 400 个事件里有 15 个真值重复（两个标签都写 London）、9 个错值撞到别的标签的
    真值、28 个错值互相重复。填值阶段会把它们变成"同一篇里两个矛盾的取值"，所以必须
    在事件层面清掉：优先改错值，真值重复时改后出现的那个。
    """
    tags = e["tags"]
    changed = []
    for _ in range(6):
        truths = {}
        for k, v in tags.items():
            truths.setdefault(v["truth"].lower(), []).append(k)
        dup_truth = {k for ks in truths.values() if len(ks) > 1 for k in ks[1:]}
        rumors = {}
        for k, v in tags.items():
            rumors.setdefault(v["rumor"].lower(), []).append(k)
        used = set(truths) | set(rumors)
        fixed = False
        for k, v in tags.items():
            bad_rumor = (len(rumors[v["rumor"].lower()]) > 1
                         or (v["rumor"].lower() in truths
                             and truths[v["rumor"].lower()] != [k]))
            if k in dup_truth:
                new = _fresh(_shape(v["truth"]), v["cat"], used, v["truth"])
                if new:
                    used.discard(v["truth"].lower())
                    v["truth"] = new
                    used.add(new.lower())
                    changed.append(f"{k} 真值→{new}")
                    fixed = True
            if bad_rumor:
                new = _fresh(_shape(v["rumor"]), v["cat"], used, v["rumor"])
                if new:
                    v["rumor"] = new
                    used.add(new.lower())
                    changed.append(f"{k} 错值→{new}")
                    fixed = True
        if not fixed:
            break
    return e, changed


def main():
    out = "events_p4.jsonl"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    paths = [a for a in sys.argv[1:] if a.endswith(".txt")]
    events = []
    for p in paths:
        events += parse(p)

    bad = 0
    for e in events:
        e["n"] = len(e["facts"])
        try:
            e = repair(e)
            e, _fixes = dedupe(e)
            e["seed_text"] = make_seed(e)
            probs = check(e) + extra(e)
        except KeyError as ex:
            probs = [f"事实里的标签 {ex} 没有定义"]
        except Exception as ex:            # noqa: BLE001
            probs = [f"{type(ex).__name__}: {str(ex)[:60]}"]
        if probs:
            bad += 1
            print(f"{e['event_id']}  {len(probs)} 处问题：{probs[:4]}")
    print(f"\n事件 {len(events)}，不合格 {bad}")

    if not bad:
        with open(out, "w") as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        print(f"写出 {out}")
    else:
        print("有不合格事件，未写出")
        sys.exit(1)


if __name__ == "__main__":
    main()
