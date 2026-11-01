"""P1（传播结构模拟计划 §8）：单事件原型 v4。

迭代记录（每一版都按 p1_measure.py 的 COUNTER 口径实测）：

  v1  事实表写成整句新闻稿 → 模型直接照抄（PD 的 f5/n1/n2 逐字等于事实表）；
      提示词里带 outlet/句数，各文档提示词并不逐字相同；ND 只有 7 句。
  v2  事实改成电报式笔记；提示词固定，outlet/长度写进材料块；ND 拉长。
      仍有：改写太重（句级相似度双峰 1.0 / 0.05，τ=0.30 照抄比例 0.40 vs 0.64）、
      导语保留与句序保持恒为 1.0（0.53 / 0.64）、PD/ND 偏短。
  v3  改写改成"最小改动" → 矫枉过正：相似度 0.54–0.70，jac2 0.56–0.65（目标 0.35），
      new=0（目标 0.24）。原因是 WD 没有任何新增句，且改写几乎是同义替换。
  v4  三处修正：
      a) 长度按通稿句数推算：WD=0.62n、PD=0.92n、ND=1.85n，不再写死；
      b) 改写按相似度带选候选：同一提示词抽两次，取与原文 3-gram 相似度最接近
         0.40 的一版（真实改写集中在 0.30–0.50）；
      c) WD 补一条自采句（各报的本地补充），对应 COUNTER 的 new=0.24。

只做一个事件，目的是看结构能不能对上 COUNTER，不是产出训练数据。
复制/删除/重排由程序做，LLM 只做"读材料写报道"和"改写这几句"，提示词里从不出现
"照抄""改写自第几句"这类角色差异（v1.1 §4.4 的教训）。

用法（在服务器上跑，写作者走本地 llama.cpp，思考已在客户端关闭）：
  python3 struct_p1.py [out.json]
环境变量：TI_BASE（默认 http://127.0.0.1:8818）、TI_MODEL（默认 local-model）
测量（COUNTER 口径，不重新生成）：
  python3 p1_measure.py out.json
"""
import hashlib
import json
import random
import re
import sys
import unicodedata

from llm_client import complete_json

# 只在直接运行时读命令行。struct_p2 会 import 本模块复用提示词与布局，那时 sys.argv
# 里是 struct_p2 自己的参数（实测直接崩在 int('events_p2.jsonl')）。
if __name__ == "__main__":
    OUT = sys.argv[1] if len(sys.argv) > 1 else "p1_event.json"
    SEED = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].lstrip("-").isdigit() else 20261031
else:
    OUT, SEED = "p1_event.json", 20261031

# 唯一的文档提示词。所有文档逐字相同，差别只在 __MATERIAL__。
PROMPT = """You are a news reporter. You have been given the material below for one story.

Write the news report.

Rules:
- use only facts stated in the material; do not add any number, name, date or claim;
- the lines "outlet:" and "length:" are instructions about the assignment, not facts:
  never mention them, and never write a sentence about the report itself;
- keep every number, name, place and date exactly as written in the material: write
  numbers in digits, never in words;
- do not copy the material word for word; write your own sentences;
- write in full newspaper prose: complete sentences, with attribution and connective
  clauses where they read naturally; do not write telegram-style fragments;
- one fact per sentence;
- output only JSON: {"headline": "...", "sentences": [{"fact": "f1", "text": "..."}]}

Material:
__MATERIAL__"""

# 改写提示词。与文档提示词是两种操作，不参与"文档提示词一致"的判定。
R_PROMPT = """Rewrite each sentence below. The result must read differently from the
original, but keep the same information and much of the original wording.

Rules:
- keep every number, name, place and date exactly as written;
- do not merge or split sentences; return one rewrite per input sentence, same order;
- do not add or drop any fact;
- output only JSON: {"rewrites": ["...", "..."]}

Sentences:
__SENTS__"""

# ---------------------------------------------------------------- 事件
WIRE = {
    "f1": "revenue +7% y/y to 4.26bn GBP; quarter to 30 September",
    "f2": "pre-tax profit -12% to 310m GBP",
    "f3": "850 job cuts, about 6% of workforce",
    "f4": "two gas plants to close by March 2027, one in Wales, one in Kent",
    "f5": "CEO Marian Holt: cost pressure sharper than expected",
    "f6": "Ofgem opened investigation into billing system",
    "f7": "dividend held at 12.4p a share",
    "f8": "shares -4.2% in morning trading",
}

ND_OWN = {
    "n1": "Kestrel Securities analyst: job cuts save 90m GBP a year",
    "n2": "Unite to ballot members on industrial action",
    "n3": "Kent plant employs 410 staff; local supply chain exposed",
    "n4": "MP for Deeside asks for urgent meeting with the company",
    "n5": "1,200 staff left under voluntary scheme since 2024",
    "n6": "Ofgem to report preliminary findings by end of 2026",
    "n7": "shares closed at 268p, down 11p",
    "n8": "customer group logged 3,400 billing complaints in six months",
    "n9": "talks with unions start on 4 November",
    "n10": "Ofgem fined the company 1.8m GBP in 2023 over billing",
    "n11": "company: no compulsory redundancies before January 2027",
    "n12": "Kestrel Securities cut its price target to 240p from 290p",
    "n13": "the company employs 4,800 staff in the UK",
}

PD_OWN = {
    "p1": "Kestrel Securities analyst: job cuts save 90m GBP a year",
    "p2": "Unite to ballot members on industrial action",
    "p3": "MP for Deeside asks for urgent meeting with the company",
    "p4": "company: customer bills unchanged until April 2027",
    "p5": "council leader: seeks assurances on the Kent plant",
}

# WD 的本地补充句（各报自己加的一句，通稿里没有）
WD_OWN = {
    "w1": "company declines to comment on the timing of the plant closures",
    "w2": "union branch secretary: members are angry but not surprised",
    "w3": "local chamber of commerce: the town has lost 300 manufacturing jobs since 2020",
}

ANCHORS = {
    "f1": ["7%", "4.26bn", "30 September"],
    "f2": ["12%", "310m"],
    "f3": ["850", "6%"],
    "f4": ["March 2027", "Wales", "Kent"],
    "f5": ["Marian Holt"],
    "f6": ["Ofgem"],
    "f7": ["12.4p"],
    "f8": ["4.2%"],
    "n1": ["Kestrel Securities", "90m"],
    "n2": ["Unite"],
    "n3": ["410"],
    "n4": ["Deeside"],
    "n5": ["1,200", "2024"],
    "n6": ["Ofgem", "2026"],
    "n7": ["268p", "11p"],
    "n8": ["3,400"],
    "n9": ["4 November"],
    "n10": ["1.8m", "2023"],
    "n11": ["January 2027"],
    "n12": ["240p", "290p"],
    "n13": ["4,800"],
    "p1": ["Kestrel Securities", "90m"],
    "p2": ["Unite"],
    "p3": ["Deeside"],
    "p4": ["April 2027"],
    "p5": ["Kent"],
    "w1": [],
    "w2": [],
    "w3": ["300", "2020"],
}

ALL_FACTS = {**WIRE, **ND_OWN, **PD_OWN, **WD_OWN}

# COUNTER 比例（计划 §4）
R_WD, R_PD, R_ND = 0.62, 0.92, 1.85
COPY_WD, REWRITE_WD = 0.47, 0.29
SIM_TARGET = 0.40


# ---------------------------------------------------------------- 度量
def toks(s):
    out = []
    for w in s.split():
        w = "".join(c for c in w if not unicodedata.category(c).startswith("P")).lower()
        if w:
            out.append(w)
    return out


def grams(ts, n=3):
    if len(ts) < n:
        return {tuple(ts)} if ts else set()
    return {tuple(ts[i:i + n]) for i in range(len(ts) - n + 1)}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def sim(a, b):
    return jac(grams(toks(a)), grams(toks(b)))


# ---------------------------------------------------------------- 生成
def material(outlet, length, facts):
    lines = [f"outlet: {outlet}", f"length: {length} sentences", "notes:"]
    lines += [f"{k}: {v}" for k, v in facts.items()]
    return "\n".join(lines)


def gen_doc(outlet, length, facts, want=None, tries=4, tol=1):
    want = want or length
    best = None
    for _ in range(tries):
        d = complete_json(PROMPT.replace("__MATERIAL__", material(outlet, length, facts)),
                          max_tokens=1800)
        sents = [(s.get("fact", "?"), s["text"].strip()) for s in d.get("sentences", [])]
        if abs(len(sents) - want) <= tol:
            return d.get("headline", ""), sents
        if best is None or abs(len(sents) - want) < abs(len(best[1]) - want):
            best = (d.get("headline", ""), sents)
    return best


def rewrite(sents, draws=2, target=SIM_TARGET, cap=None, floor=None):
    """同一提示词抽 draws 次，每句取与原文相似度最接近 target 的候选。

    cap/floor 是硬边界：改写句必须落在 [0.22, 0.48)，否则会被相似度判成"照抄"或
    "新增"，把 copy/new 比例顶偏。实测不加边界时 WD 的 new 在 0.20–0.40 之间跳。
    """
    cands = []
    for _ in range(draws):
        r = complete_json(R_PROMPT.replace("__SENTS__", "\n".join(
            f"{i + 1}. {s}" for i, s in enumerate(sents))), max_tokens=900)["rewrites"]
        cands.append([x.strip() for x in r])
    out = []
    for i, s in enumerate(sents):
        best, pick = None, None
        anybest, anypick = None, None
        for c in cands:
            if i >= len(c):
                continue
            v = sim(c[i], s)
            d = abs(v - target)
            if anybest is None or d < anybest:      # 无视边界的兜底候选
                anybest, anypick = d, c[i]
            if cap is not None and v >= cap:
                continue
            if floor is not None and v < floor:
                continue
            if best is None or d < best:
                best, pick = d, c[i]
        # 没有候选落在边界内时，取最接近目标的那个，**不能退回原句**——
        # 退回原句会把"改写句"变成逐字照抄，PD 的 copy 比例会从 0 跳到 0.14。
        out.append(pick if pick is not None else (anypick if anypick is not None else s))
    return out


def wd_layout(n, keep_lead, scramble, rng, exclude_lead=False):
    """按 COUNTER 比例算出 (复制句下标, 改写句下标, 输出顺序)。

    总句数 = 0.62n，其中复制 0.47、自采固定 1 句（本地补充），余下改写。
    自采句必须算在总句数里，v4 把它额外加上，导致句数比只有 0.50。

    exclude_lead=True 时，"不保留导语"就真的不用通稿第 1 句——否则它会落进改写集合，
    改写后相似度若仍 ≥0.50，测出来的 lead 还是 1，导语保留比例降不下来（P2 实测
    0.75，目标 0.53）。P1 不用这个开关，保持已报告结果不变。
    """
    total = max(4, round(R_WD * n))
    copy_n = max(1, min(total - 2, round(COPY_WD * total)))
    rewrite_n = max(1, total - copy_n - 1)
    if keep_lead:
        copy_idx = [0] + sorted(rng.sample(range(1, n), min(copy_n - 1, n - 1)))
    else:
        copy_idx = sorted(rng.sample(range(1, n), min(copy_n, n - 1)))
    rest = [i for i in range(n) if i not in copy_idx]
    if exclude_lead and not keep_lead:
        rest = [i for i in rest if i != 0]
    rewrite_idx = sorted(rng.sample(rest, min(rewrite_n, len(rest))))
    order = sorted(copy_idx + rewrite_idx)
    if scramble and len(copy_idx) >= 2:
        last = copy_idx[-1]
        order = [last] + [i for i in order if i != last]
    return copy_idx, rewrite_idx, order


# ---------------------------------------------------------------- 锚点自检
NUM = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?(?:%|\s?(?:bn|m|p)\b)?")


def _norm(s):
    return re.sub(r"\s+", "", s)


def anchor_report(fact_key, text):
    pool = ALL_FACTS.get(fact_key, "")
    miss = [a for a in ANCHORS.get(fact_key, []) if _norm(a) not in _norm(text)]
    extra = [n for n in [_norm(x) for x in NUM.findall(text)]
             if n not in set(_norm(x) for x in NUM.findall(pool))]
    return miss, extra


def resolve_fact_key(text, pool, fallback="?"):
    """按锚点回填事实键。

    模型经常不照抄材料里的键：PD 的材料写 p1..p5，它返回 f1/f2/f3（实测 u33 三句
    全错位）。标签一旦落错事实，句级标签就废了。这里用声明好的锚点反查——哪条事实的
    锚点全部出现在这句里，就归给哪条。
    """
    best, bk = 0, None
    for k in pool:
        need = ANCHORS.get(k, [])
        if not need:
            continue
        hit = sum(1 for a in need if _norm(a) in _norm(text))
        if hit == len(need) and hit > best:
            best, bk = hit, k
    return bk or fallback


def fact_integrity(variants):
    """一条事实的全部锚点必须落在同一句里。

    v3 实测 4/37：模型把含两个数字的事实拆成两句（850 一句、6% 一句），每句都缺
    一半锚点。句级标签要求"一句一事实"，这种拆句会让标签失去唯一落点，必须报出来。
    同时检查同一事实是否被多句重复承载（会导致同一标签落到多句）。
    """
    issues = []
    for name, v in variants.items():
        by_fact = {}
        for s in v["sentences"]:
            by_fact.setdefault(s["fact"], []).append(s["text"])
        for k, texts in by_fact.items():
            need = ANCHORS.get(k, [])
            if not need:
                continue
            holders = [t for t in texts
                       if all(_norm(a) in _norm(t) for a in need)]
            if not holders:
                issues.append((name, k, "锚点被拆到多句或缺失", texts))
            elif len(holders) > 1:
                issues.append((name, k, f"{len(holders)} 句重复承载", texts))
    return issues


# ---------------------------------------------------------------- 主流程
def main():
    rng = random.Random(SEED)
    rec = {"event": "p1_energy", "seed": SEED,
           "prompt_sha": hashlib.sha256(PROMPT.encode()).hexdigest()[:16],
           "wire": {}, "variants": {}}

    # 1. 通稿
    #    固定 8 句（tol=0）：P1 只有三个事件，句数的整数取整会主导方差——
    #    7 句时 WD 是 4 句（复制 2/4=0.50），8 句时是 5 句（2/5=0.40），
    #    包含率随之在 0.57 与 0.65 之间跳。P1 先固定 n 把取整噪声隔离掉，
    #    只留模型措辞的方差；P2 再放开 6–8 句。
    hl, w = gen_doc("Pennine Wire Service", 8, WIRE, want=8, tol=0)
    if len(w) != 8:
        raise RuntimeError(f"W 只写了 {len(w)} 句，要求 8 句")
    w = [(resolve_fact_key(t, WIRE, f), t) for f, t in w]
    n = len(w)
    print(f"== W 通稿：{n} 句  {hl[:60]}")
    for k, t in w:
        print(f"   [{k}] {t}")
    rec["wire"] = {"headline": hl, "sentences": w}
    print(f"   派生目标句数：WD {round(R_WD * n)} / PD {round(R_PD * n)} / ND {round(R_ND * n)}")

    # 2. WD：复制 + 轻改写 + 1 条本地自采；导语保留与句序由程序随机化
    outlets = ["The Northern Echo", "The Yorkshire Post", "Western Mail"]
    own_ids = list(WD_OWN)
    for j, (keep_lead, scramble) in enumerate(((True, False), (False, True), (True, True))):
        copy_idx, rewrite_idx, order = wd_layout(n, keep_lead, scramble, rng)
        # 复制句也要一道"外观编辑"：COUNTER 的 copy@0.7 只有 0.35、copy@0.5 是 0.47，
        # 说明真实照抄多数落在 0.50–0.70，不是逐字 1.0（报纸会改称谓、加电头）。
        # 逐字搬运会把 WD 的包含率顶到 0.65（目标 0.57）。
        cp = rewrite([w[i][1] for i in copy_idx], draws=2, target=0.80,
                     cap=0.95, floor=0.55)
        cmap = {i: cp[k] for k, i in enumerate(copy_idx)}
        for i in copy_idx:                      # 编辑过头就不算照抄了，退回原文
            if sim(cmap[i], w[i][1]) < 0.50:
                cmap[i] = w[i][1]
        rw = rewrite([w[i][1] for i in rewrite_idx], draws=3,
                     target=0.35, cap=0.48, floor=0.22)
        rmap = {i: rw[k] for k, i in enumerate(rewrite_idx)}
        own_id = own_ids[j]
        _, own_sents = gen_doc(outlets[j], 1, {own_id: WD_OWN[own_id]}, want=1)
        own_txt = own_sents[0][1] if own_sents else WD_OWN[own_id]
        out = [(w[i][0], cmap[i], "copied") if i in copy_idx
               else (w[i][0], rmap[i], "rewritten") for i in order]
        out.append((own_id, own_txt, "own"))
        name = f"WD_{j + 1}"
        rec["variants"][name] = {"role": "WD", "outlet": outlets[j], "sentences": [
            {"fact": f, "text": t, "origin": o} for f, t, o in out]}
        print(f"\n== {name}（{outlets[j]}）：复制 {copy_idx} 改写 {rewrite_idx} "
              f"顺序 {order} + 自采 1")
        for f, t, o in out:
            print(f"   [{f}/{o}] {t[:78]}")

    # 3. PD：程序借 2 句改写，其余全部自采
    #    原先给 PD 通稿 f3/f5 的**笔记**，模型把这两条写得几乎等于通稿
    #    （实测相似度 0.15–0.54），PD 包含率被顶到 0.30（目标 0.19）。
    #    真实 PD 是从通稿正文取材，不是从同一份笔记——笔记只给自采部分。
    pd_total = round(R_PD * n)
    borrow_idx = [0, min(2, n - 1)]
    rws = rewrite([w[i][1] for i in borrow_idx], draws=3,
                  target=0.30, cap=0.45, floor=0.22)
    hl_pd, pd_own = gen_doc("The Northern Echo", pd_total - len(borrow_idx), PD_OWN,
                            want=pd_total - len(borrow_idx))
    pd = [(w[borrow_idx[0]][0], rws[0], "rewritten")]
    pd += [(resolve_fact_key(t, PD_OWN, f), t, "own") for f, t in pd_own[:len(pd_own) // 2]]
    pd.append((w[borrow_idx[1]][0], rws[1], "rewritten"))
    pd += [(resolve_fact_key(t, PD_OWN, f), t, "own") for f, t in pd_own[len(pd_own) // 2:]]
    rec["variants"]["PD"] = {"role": "PD", "headline": hl_pd, "sentences": [
        {"fact": f, "text": t, "origin": o} for f, t, o in pd]}
    print(f"\n== PD：借用 1 句改写 + 自采 {len(pd) - 1} 句")
    for f, t, o in pd:
        print(f"   [{f}/{o}] {t[:78]}")

    # 4. ND：两条共同事实 + 十二条自采，不看 W
    #    v4 给三条共同事实，包含率 0.217（目标 0.10）——独立报道里被复述的通稿内容
    #    只占约 0.12，三条太多。降到两条，同时补两条自采把长度撑到 0.62n 的 1.85 倍。
    nd_total = round(R_ND * n)
    nd_pool = {**{k: WIRE[k] for k in ("f2", "f3")}, **ND_OWN}
    hl_nd, nd = gen_doc("Western Mail", nd_total, nd_pool, want=nd_total, tol=0)
    nd = [(resolve_fact_key(t, nd_pool, f), t) for f, t in nd]
    rec["variants"]["ND"] = {"role": "ND", "headline": hl_nd, "sentences": [
        {"fact": f, "text": t, "origin": "own"} for f, t in nd]}
    print(f"\n== ND：独立写 {len(nd)} 句")
    for f, t in nd:
        print(f"   [{f}] {t[:78]}")

    # 5. 锚点自检
    print("\n== 锚点自检（缺失锚点 / 多余数字）")
    bad = tot = 0
    for name, v in rec["variants"].items():
        for s in v["sentences"]:
            k = s["fact"]
            if k not in ALL_FACTS:
                print(f"   {name:6s} [{k}] 未知事实键")
                continue
            tot += 1
            miss, extra = anchor_report(k, s["text"])
            if miss or extra:
                bad += 1
                print(f"   {name:6s} [{k}] 缺失 {miss} 多余数字 {extra}")
    print(f"   合计 {bad}/{tot} 句有锚点问题")
    rec["anchor_missing"], rec["anchor_total"] = bad, tot

    # 6. 一句一事实校验
    issues = fact_integrity(rec["variants"])
    print(f"\n== 一句一事实校验：{len(issues)} 处异常")
    for name, k, why, texts in issues:
        print(f"   {name:6s} [{k}] {why}")
        for t in texts:
            print(f"        {t[:90]}")
    rec["fact_integrity_issues"] = [
        {"variant": n, "fact": k, "why": w} for n, k, w, _ in issues]

    with open(OUT, "w") as f:
        f.write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
    print(f"\n写出 {OUT}")


if __name__ == "__main__":
    main()
