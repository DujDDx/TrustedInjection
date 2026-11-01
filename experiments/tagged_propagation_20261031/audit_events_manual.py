"""人工复核用的启发式审计：查 validate_events 查不到的东西。

validate_events 只管结构（标签归属、类别覆盖、裸数字、真值在种子里）。它查不出：
  - 机械拼接的种子新闻里冠词用错（"a apple"）、"the" 后跟专名（"the Thames Water"）；
  - 种子里的空格/标点毛病（" ."、双空格、句子小写开头）；
  - 错值虽然同类但量级离谱（年份差 30 年）；
  - 两个事件其实是同一个故事。

    python3 audit_events_manual.py events_p4.jsonl
"""
import collections
import json
import re
import sys

TAG = re.compile(r"<[A-Za-z_]+>")
VOWEL = set("aeiou")


def article_problems(seed):
    p = []
    for m in re.finditer(r"\b(a|an)\s+([A-Za-z][A-Za-z-]*)", seed):
        art, word = m.group(1), m.group(2)
        w = word.lower()
        if w in ("one", "once", "uniform", "union", "university", "uk", "us", "european"):
            continue                      # 发音例外，不判
        if w.startswith("hour") or w.startswith("honest"):
            continue
        want = "an" if w[0] in VOWEL else "a"
        if art != want:
            p.append(f"冠词：'{art} {word}' 应为 '{want}'")
    return p


def the_proper(seed, tags):
    """'the' 后面跟专名（大写开头的 name/agent 值）。"""
    p = []
    for m in re.finditer(r"\b[Tt]he\s+([A-Z][\w'-]*)", seed):
        w = m.group(1)
        if w in ("The",):
            continue
        for k, v in tags.items():
            if str(v["truth"]) == w and v["cat"] in ("name", "agent"):
                p.append(f"'the {w}' 后面是专名（{k}）")
    return p


def punctuation(seed):
    p = []
    if "  " in seed:
        p.append("双空格")
    if re.search(r"\s+[.,;:]", seed):
        p.append("标点前有空格")
    for s in re.split(r"(?<=[.!?])\s+", seed):
        if s and s[0].islower():
            p.append(f"句子小写开头：{s[:40]}")
    if not seed.rstrip().endswith((".", "!", "?")):
        p.append("种子不以句号结尾")
    return p


def magnitude(tags):
    """错值与真值量级差太远（年份差 > 15 年、数字差 > 5 倍）。"""
    p = []
    for k, v in tags.items():
        tr, ru = str(v["truth"]), str(v["rumor"])
        if re.fullmatch(r"\d{4}", tr) and re.fullmatch(r"\d{4}", ru):
            if abs(int(tr) - int(ru)) > 15:
                p.append(f"{k} 年份差 {abs(int(tr) - int(ru))} 年（{tr} vs {ru}）")
        a = re.fullmatch(r"([\d,]+)", tr)
        b = re.fullmatch(r"([\d,]+)", ru)
        if a and b:
            x, y = int(a.group(1).replace(",", "")), int(b.group(1).replace(",", ""))
            if x and y and max(x, y) / min(x, y) > 5:
                p.append(f"{k} 数量差 {max(x, y) / min(x, y):.1f} 倍（{tr} vs {ru}）")
    return p


def main():
    evs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    print(f"事件 {len(evs)}")

    tot = collections.Counter()
    shown = collections.Counter()
    for e in evs:
        seed = e["seed_text"]
        probs = (article_problems(seed) + the_proper(seed, e["tags"])
                 + punctuation(seed) + magnitude(e["tags"]))
        for x in probs:
            key = x.split("：")[0].split(" ")[0]
            tot[key] += 1
            if shown[key] < 3:
                shown[key] += 1
                print(f"  {e['event_id']}  {x}")

    print("\n问题计数：", dict(tot))

    # 近重复事件
    def grams(t):
        w = [x.strip(".,").lower() for x in t.split()]
        return {tuple(w[i:i + 5]) for i in range(max(0, len(w) - 4))}
    g = [(e["event_id"], grams(e["seed_text"])) for e in evs]
    near = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            a, b = g[i][1], g[j][1]
            if a and b and len(a & b) / len(a | b) > 0.6:
                near += 1
                if near <= 5:
                    print(f"  近重复：{g[i][0]} / {g[j][0]}")
    print(f"近重复事件对 {near}")

    # 长度分布
    lens = [len(e["seed_text"].split()) for e in evs]
    lens.sort()
    print(f"种子词数：中位 {lens[len(lens) // 2]}，最短 {lens[0]}，最长 {lens[-1]}")


if __name__ == "__main__":
    main()
