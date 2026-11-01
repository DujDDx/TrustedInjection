"""P2 的事件素材起草器：为每个事件生成一份事实包（通稿 6–8 条 + 自采 13/5/3 条）。

为什么要有这一步：`struct_p1.py` 的事实是手写的单事件，P2 要 20 个不同事件。事实包
必须满足三条硬条件，否则下游的句级标签会失效：

  1. **每条事实自带锚点**（必须原样保留的字面），且锚点真的出现在这条笔记里；
  2. **数字在整个包内唯一**——两条事实都含 `12%` 时，按锚点反查事实键会有两个候选，
     标签就落不准（P1 的 `resolve_fact_key` 靠的就是数字唯一）；
  3. **自采事实不复述通稿事实**——复述会把 ND/PD 的包含率顶上去（P1 实测：ND 给三条
     共同事实时 cont2 = 0.217，目标 0.10，降到两条才达标）。

起草用写作者模型（Qwen3.8-27B），起草完机械校验，不合格就带着违规项重抽。**事实是
模型编的，不是真人核过的**——本数据集只用于来源结构，不用于事实问答，方法节必须写明
这一点（另见 P1 事件自身也不满足算术自洽：850 人 = 6% 隐含 14,167 员工，而通稿另有
4,800 员工的数字）。

用法：
  python3 gen_events.py --n 24 --out events_p2.jsonl
环境变量：TI_BASE（默认 http://127.0.0.1:8818）
"""
import json
import random
import re
import sys
import unicodedata

from llm_client import complete_json

DOMAINS = [
    "a regional electricity supplier", "a supermarket chain", "a bus operator",
    "a hospital trust", "a water company", "a housebuilder", "an airline",
    "a telecoms operator", "a steel maker", "a university", "a port authority",
    "a dairy cooperative", "a betting company", "a waste contractor",
    "a train operating company", "a pharmacy chain", "a car plant",
    "a local council", "a housing association", "a newspaper group",
    "a chemicals plant", "a logistics firm", "a wind farm developer",
    "a credit union", "a brewery", "a furniture retailer",
]

PROMPT = """You are building a synthetic news event for a research dataset on how news
travels between outlets. Everything is fictional; invent names freely.

The event concerns __DOMAIN__.

Produce exactly these fact notes:

- "wire": __NWIRE__ facts that a wire service would file about this event. These are the
  story: the numbers, the decision, the quote, the market reaction.
- "nd": 13 facts an INDEPENDENT outlet would have from its own reporting, not in the wire:
  analyst notes, union statements, a local MP, a customer group, company history,
  what happens next.
- "pd": 5 facts a PARTIALLY DERIVED outlet would add from a local angle.
- "wd": 3 facts a DERIVED outlet would add as local colour.

Hard rules:
- every fact is a telegraphic note, at most 16 words, no full sentences, no verbs of
  reporting in the note itself;
- **every "wire", "nd" and "pd" fact must be locatable by an anchor**: a figure of two or
  more characters (12%, 4,800, 3bn, 12.4p, March 2027) or a proper name (Unite, Deeside,
  Marian Holt). A lone digit is not enough ("3 new routes" is rejected). Filler with
  neither is rejected — write "Unite to ballot 2,400 members on strike action", not
  "Unions call plan premature";
- numbers in digits with units exactly as they must appear later: 12%, 310m GBP, 4.26bn
  GBP, 12.4p, 4,800, 4 November, March 2027;
- **no anchor may be reused across facts, and no anchor may appear inside another
  fact's note.** If "2027" is an anchor of one fact, no other fact may mention 2027; if
  you need two dates, use different months (March 2027 and January 2027 are fine, two
  facts both dated March 2027 are not);
- the nd, pd and wd facts must NOT restate a wire fact: no overlapping claim, no second
  figure for the same quantity;
- for each fact give "anchors": the 1-3 literals that must survive verbatim if a
  journalist rewrites the note. **An anchor must be a figure, a proper name, a place or a
  date copied character-for-character out of the note. A role or a category is not an
  anchor**: "Unite" and "Deeside" are anchors, "Unions" and "Local MP" are not, and
  "15%" is an anchor while "15" alone is not.
- "wd" facts may have zero anchors (pure colour), but "wire", "nd" and "pd" must each
  have at least one.

Output only JSON:
{"domain": "...", "company": "...",
 "wire": {"f1": {"note": "...", "anchors": ["..."]}, ...},
 "nd":   {"n1": {"note": "...", "anchors": ["..."]}, ...},
 "pd":   {"p1": {"note": "...", "anchors": ["..."]}, ...},
 "wd":   {"w1": {"note": "...", "anchors": []}, ...}}"""

NUM = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?(?:%|\s?(?:bn|m|p)\b)?")


def norm(s):
    return re.sub(r"\s+", "", s)


def nums(s):
    return [norm(x) for x in NUM.findall(s)]


def words(s):
    return [w for w in ("".join(c for c in t if not unicodedata.category(c).startswith("P"))
                        for t in s.split()) if w]


def repair(b):
    """把可机械修复的锚点问题就地修掉，避免为格式问题重抽整包。

    三类修法，都是**加强**约束而不是放宽：
      - 笔记里的数字没声明成锚点 → 补成锚点（数字本来就该原样保留）；
      - 锚点只是大小写/空格写法不同 → 换成笔记里的实际写法；
      - 锚点根本不在笔记里（模型给的是概念词，如 "share price"）→ 丢掉，它本来也
        没法用来定位这句；
      - 一个锚点是另一个锚点的子串（`4` ⊂ `4 November`、`310m` ⊂ `310m GBP`）→ 只留
        长的。短的那个会匹配到任何含该数字的句子，反而让事实键反查多出候选。
    """
    for grp in ("wire", "nd", "pd", "wd"):
        for f in (b.get(grp) or {}).values():
            note = f.get("note", "") or ""
            keep, have = [], set()
            for a in f.get("anchors") or []:
                if norm(a) in norm(note):
                    keep.append(a)
                    have.add(norm(a))
                    continue
                m = re.search(re.escape(a), note, re.I)
                if m:
                    keep.append(m.group(0))
                    have.add(norm(m.group(0)))
            for lit in NUM.findall(note):
                if norm(lit) not in have:
                    keep.append(lit)
                    have.add(norm(lit))
            # 单个数字是合法锚点（"3 stores" 的 3），只有单字符的**非数字**锚点才丢。
            keep = [a for a in keep
                    if (len(norm(a)) >= 2 or nums(a))
                    and not any(norm(a) != norm(b) and norm(a) in norm(b) for b in keep)]
            f["anchors"] = keep
    return b


def check(b, nwire):
    """返回 (errors, warnings)。errors 非空则丢弃重抽。"""
    err, warn = [], []
    want = {"wire": nwire, "nd": 13, "pd": 5, "wd": 3}
    for grp, k in want.items():
        got = b.get(grp) or {}
        if len(got) != k:
            err.append(f"{grp} 有 {len(got)} 条，要求 {k}")
        for key, f in got.items():
            note, anc = f.get("note", ""), f.get("anchors") or []
            if not note.strip():
                err.append(f"{grp}.{key} 笔记为空")
                continue
            if len(words(note)) > 18:
                warn.append(f"{grp}.{key} 笔记 {len(words(note))} 词，偏长")
            for a in anc:
                if norm(a) not in norm(note):
                    err.append(f"{grp}.{key} 锚点 '{a}' 不在笔记里")
            if grp != "wd" and not anc:
                err.append(f"{grp}.{key} 没有锚点")
            if grp != "wd":
                # 锚点可以是专名，不必是数字。要求"必须含数字"试过三轮，27 条事实里
                # 总有四五条是纯文字的（"CEO to step down"），模型过不了；而专名做
                # 锚点同样能定位，还更自然。只挡掉单字符与纯角色词。
                if not anc:
                    err.append(f"{grp}.{key} 没有可用锚点（既无数字也无专名）")
            # 笔记里的数字必须被某个锚点覆盖。用"包含"而不是"相等"：锚点常写成
            # `4.26bn GBP`，而数字正则只取到 `4.26bn`。
            extra = [x for x in nums(note)
                     if not any(norm(x) in norm(a) for a in anc)]
            if extra:
                err.append(f"{grp}.{key} 笔记有未声明为锚点的数字 {extra}")

    # 锚点歧义只报警告，不拒稿。
    #    试过三轮把它做成硬约束：同一个专名（议员选区、工会）本来就会出现在好几条
    #    事实里，逼模型做到"每条事实的锚点在全包内唯一"等于逼它换着写法绕开同一个人
    #    物，写出来的东西不像新闻。真正的解法在生成端——句→事实的归属用分值匹配，
    #    并把"最高分不唯一"的句子报出来（见 struct_p2.py 的 assign_facts）。
    flat = [(f"{grp}.{key}", f) for grp in ("wire", "nd", "pd", "wd")
            for key, f in (b.get(grp) or {}).items()]
    for name_a, fa in flat:
        anc = [a for a in (fa.get("anchors") or [])]
        if not anc:
            continue
        for name_b, fb in flat:
            if name_a == name_b:
                continue
            if all(norm(a) in norm(fb.get("note", "")) for a in anc):
                warn.append(f"{name_a} 的锚点 {anc} 也全在 {name_b} 的笔记里（归属需靠分值）")
                break
    return err, warn


def draft(domain, nwire, seed, tries=4):
    rng = random.Random(seed)
    last = None
    for t in range(tries):
        p = PROMPT.replace("__DOMAIN__", domain).replace("__NWIRE__", str(nwire))
        if last:
            p += ("\n\nYour previous attempt broke these rules; fix them and output the "
                  "whole JSON again:\n- " + "\n- ".join(last[:8]))
        try:
            b = complete_json(p, max_tokens=3800, temperature=0.8 + 0.1 * t)
        except Exception as e:      # noqa: BLE001
            # 一次解析失败不能让整批停掉（实测 e11 在第 1 次就吐了半截 JSON，
            # 异常直接冒到 main 把 10 个已写事件之后的全丢了）。
            last = [f"{type(e).__name__}: {str(e)[:80]}"]
            print(f"   第 {t + 1} 次调用失败：{last[0]}")
            continue
        if not isinstance(b, dict):
            last = ["输出不是 JSON 对象"]
            print(f"   第 {t + 1} 次不合格：输出不是对象")
            continue
        b["n_wire"] = nwire
        b["domain"] = b.get("domain") or domain
        repair(b)
        err, warn = check(b, nwire)
        if not err:
            return b, warn
        last = err
        print(f"   第 {t + 1} 次不合格：{err[:4]}")
    return None, last


def main():
    n = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 24
    out = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
           else "events_p2.jsonl")
    rng = random.Random(20261031)
    domains = DOMAINS[:]
    rng.shuffle(domains)
    ok = 0
    with open(out, "w") as fo:
        for i in range(n):
            nwire = (6, 7, 8)[i % 3]          # 让取整噪声在 20 个事件上摊开
            dom = domains[i % len(domains)]
            print(f"\n== e{i + 1:02d}  {nwire} 条通稿  {dom}")
            try:
                b, warn = draft(dom, nwire, seed=20261031 + i)
            except Exception as e:          # noqa: BLE001
                print(f"   放弃（异常）：{type(e).__name__}: {str(e)[:100]}")
                continue
            if b is None:
                print(f"   放弃：{warn[:3]}")
                continue
            b["id"] = f"e{i + 1:02d}"
            for w in warn[:3]:
                print(f"   提示：{w}")
            fo.write(json.dumps(b, ensure_ascii=False) + "\n")
            fo.flush()
            ok += 1
            print(f"   写出 {b['id']}（{b.get('company', '?')}）")
    print(f"\n合计 {ok}/{n} 个事件写入 {out}")


if __name__ == "__main__":
    main()
