"""P1 产出的事后审计：标签与操作是否一致（计划 §5.2 第 2 项）。

**判据是单向的**，不是"origin 必须等于相似度分段"。origin 说的是来源（这句从父文档
来的，还是记者自己采的），相似度说的是表象；两者本来就不必相等——一句重度改写的
派生句相似度可以低于 0.20，一句独立采写的同事实报道相似度可以到 0.3–0.4。要求两者
相等，等于把"派生"重新定义成"像派生"，正是本数据集要避免的循环。

因此只查两条硬约束：
  - origin=copied  → 相似度必须 ≥ 0.50（说是照抄就必须看起来像照抄）
  - origin=own     → 相似度必须 < 0.50（说是自采就不能是从父文档搬的）
  - origin=rewritten → 不作硬约束，只统计它落在哪一段（这是数据集的难例来源）

比对基准随文档而变：有 parent 的（HOP2）与父文档比，其余与通稿比。origin 说的是
"这句从谁那里来"，所以二跳文档的"照抄"必须像它的父文档，而不是像通稿——父文档自己
也可能是改写句。

同时核两件事：
  - 事实键与内容是否对得上（该事实声明的锚点必须全在这句里，且不得多出数字）；
  - 同一事实是否被多句重复承载（会让同一标签落到多句）。

被篡改的句子锚点必然对不上——那正是篡改的定义，单列为"篡改后锚点不符（预期）"，
不计入异常。

用法：
  python3 audit_p1.py p1_g11.json [more.json ...]
"""
import json
import re
import sys
import unicodedata

COPY_TAU = 0.50

# 与 struct_p1.py 保持一致（审计脚本刻意不 import，避免共用同一份错误）
ANCHORS = {
    "f1": ["7%", "4.26bn", "30 September"], "f2": ["12%", "310m"],
    "f3": ["850", "6%"], "f4": ["March 2027", "Wales", "Kent"],
    "f5": ["Marian Holt"], "f6": ["Ofgem"], "f7": ["12.4p"], "f8": ["4.2%"],
    "n1": ["Kestrel Securities", "90m"], "n2": ["Unite"], "n3": ["410"],
    "n4": ["Deeside"], "n5": ["1,200", "2024"], "n6": ["Ofgem", "2026"],
    "n7": ["268p", "11p"], "n8": ["3,400"], "n9": ["4 November"],
    "n10": ["1.8m", "2023"], "n11": ["January 2027"], "n12": ["240p", "290p"],
    "n13": ["4,800"], "p1": ["Kestrel Securities", "90m"], "p2": ["Unite"],
    "p3": ["Deeside"], "p4": ["April 2027"], "p5": ["Kent"],
    "w1": [], "w2": [], "w3": ["300", "2020"],
}
NUM = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?(?:%|\s?(?:bn|m|p)\b)?")


def toks(s):
    return [w for w in ("".join(c for c in t if not unicodedata.category(c).startswith("P")).lower()
                        for t in s.split()) if w]


def grams(ts, n=3):
    if len(ts) < n:
        return {tuple(ts)} if ts else set()
    return {tuple(ts[i:i + n]) for i in range(len(ts) - n + 1)}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def sim(a, b):
    return jac(grams(toks(a)), grams(toks(b)))


def norm(s):
    return re.sub(r"\s+", "", s)


def nums(s):
    return [norm(x) for x in NUM.findall(s)]


def has_anchor(text, a):
    """锚点是否在这句里。按**词**比对而不是按子串。

    `310m GBP` 与 `GBP 310m` 是同一件事，子串比对会把语序变化判成缺失——P2 实测
    "锚点不全"里有相当一部分是这类假阳性（还有 `community centre` / `center` 这类
    拼写变体，那个是真不同，仍然算缺失）。
    """
    tt = set(toks(text))
    ta = toks(a)
    return bool(ta) and all(t in tt for t in ta)


def audit(path):
    d = json.load(open(path))
    # P1 的锚点写死在本文件里（刻意不 import 生成器，避免共用同一份错误）；
    # P2 的事件各有自己的事实包，锚点随文件给出，优先用文件里的。
    A = d.get("anchors") or ANCHORS
    w = [t for _, t in d["wire"]["sentences"]]
    wg = [(t, grams(toks(t))) for t in w]
    # 比对基准：有 parent 的用父文档，否则用通稿
    refs = {}
    for name, v in d["variants"].items():
        p = v.get("parent")
        texts = [s["text"] for s in d["variants"][p]["sentences"]] if p else w
        refs[name] = (p or "wire", [(t, grams(toks(t))) for t in texts])
    hops = [f"{n}→{r[0]}" for n, r in refs.items() if r[0] != "wire"]
    print(f"\n=== {path}   W={len(w)} 句   prompt_sha={d.get('prompt_sha')}"
          + (f"   基准换父文档: {', '.join(hops)}" if hops else ""))

    conf = {}          # (origin, observed) -> n
    viol, keybad, dup = [], [], []
    tamper_ok_n = 0    # 篡改句锚点不符：预期行为，单独计
    for name, v in d["variants"].items():
        _, ref = refs[name]
        seen = {}
        for i, s in enumerate(v["sentences"]):
            txt, org, k = s["text"], s.get("origin", "?"), s["fact"]
            best = max((jac(grams(toks(txt)), g), t) for t, g in ref)[0] if ref else 0.0
            obs = ("copied" if best >= COPY_TAU else
                   "rewritten" if best >= 0.20 else "own")
            conf[(org, obs)] = conf.get((org, obs), 0) + 1
            # 单向硬约束。被篡改/被错引的句子不参与：改了数字或追加了来源之后，它本来
            # 就不再像通稿（实测 e02 一句 0.538→0.438 只是因为句尾多了 ", Finch Capital
            # said."），这是这两类操作的**定义**，不是标签错误。
            if not (s.get("tampered") or s.get("forged")):
                if org == "copied" and best < COPY_TAU:
                    viol.append((name, i, "说是照抄但相似度 %.3f < 0.50" % best, txt))
                if org == "own" and best >= COPY_TAU:
                    viol.append((name, i, "说是自采但相似度 %.3f ≥ 0.50" % best, txt))
            # 事实键核对
            if k in A:
                need = A[k]
                miss = need and not all(has_anchor(txt, a) for a in need)
                if miss:
                    if s.get("tampered"):
                        tamper_ok_n += 1          # 篡改后锚点不符是设计，不是错
                    else:
                        keybad.append((name, i, k, "锚点不全", txt))
                if len(seen.get(k, [])) == 0:
                    seen[k] = [i]
                else:
                    seen[k].append(i)
                    dup.append((name, k, seen[k], txt))
            else:
                keybad.append((name, i, k, "未知事实键", txt))

    print("   origin × 实际类别（混淆表，行=origin，列=实际；rewritten 行跨段是正常的）")
    print(f"   {'':12s} {'copied':>8s} {'rewritten':>10s} {'own':>6s}")
    for org in ("copied", "rewritten", "own"):
        row = [conf.get((org, o), 0) for o in ("copied", "rewritten", "own")]
        print(f"   {org:12s} {row[0]:8d} {row[1]:10d} {row[2]:6d}")
    hard = len(viol)
    tot = sum(conf.values())
    chk = tot - sum(1 for v in d["variants"].values() for s in v["sentences"]
                    if s.get("tampered") or s.get("forged"))
    print(f"   硬约束违反 {hard}/{chk}（照抄不像照抄 / 自采像照抄；"
          f"篡改或错引的 {tot - chk} 句不计）")

    print(f"   越界 {len(viol)} 句")
    for name, i, why, txt in viol[:8]:
        print(f"      {name} #{i} {why}  {txt[:70]}")

    # ---- 篡改 / 错引 / 二跳继承 ----
    tstat = {"tamper_ok": 0, "tamper_bad": 0, "forge_ok": 0, "forge_bad": 0,
             "inherit_ok": 0, "inherit_bad": 0}
    tdetail = []
    for name, v in d["variants"].items():
        for i, s in enumerate(v["sentences"]):
            k, txt = s["fact"], s["text"]
            # 锚点在 P2 里常带单位（"4.26bn GBP"），NUM.fullmatch 取不到，要先 search
            # 出数字部分再比，否则篡改句会被判成"数字仍全对"。
            need = [NUM.search(a).group(0) for a in A.get(k, []) if NUM.search(a)]
            ok = all(a in set(NUM.findall(txt)) for a in need) if need else True
            if s.get("tampered"):
                # 声明被篡改：必须确实对不上声明值
                if ok:
                    tstat["tamper_bad"] += 1
                    tdetail.append(f"{name} #{i} 标了 tampered 但数字仍全对")
                else:
                    tstat["tamper_ok"] += 1
            if s.get("forged"):
                # 错引来源随事件不同（P1 是 Ofgem，P2 从各事件自己的专名里挑）
                src = (s.get("forge") or {}).get("source") or "Ofgem"
                if f"{src} said." in txt:
                    tstat["forge_ok"] += 1
                else:
                    tstat["forge_bad"] += 1
                    tdetail.append(f"{name} #{i} 标了 forged 但没有错误来源")
            if s.get("hop") == 2:
                # 二跳：数字要么与声明一致，要么继承父文档的篡改（已标 tampered）
                if ok or s.get("tampered"):
                    tstat["inherit_ok"] += 1
                else:
                    tstat["inherit_bad"] += 1
                    tdetail.append(f"{name} #{i} 二跳数字对不上但没标 tampered")
    if any(tstat.values()):
        print(f"   篡改 {tstat['tamper_ok']} 对 / {tstat['tamper_bad']} 错；"
              f"错引 {tstat['forge_ok']} 对 / {tstat['forge_bad']} 错；"
              f"二跳继承 {tstat['inherit_ok']} 对 / {tstat['inherit_bad']} 错")
        for x in tdetail[:6]:
            print(f"      {x}")
    print(f"   事实键异常 {len(keybad)} 句；同一事实多句承载 {len(dup)} 处；"
          f"篡改后锚点不符 {tamper_ok_n} 句（预期）")
    for x in keybad[:6]:
        print(f"      {x[0]} #{x[1]} [{x[2]}] {x[3]}  {x[4][:60]}")
    for x in dup[:6]:
        print(f"      {x[0]} [{x[1]}] 出现在 {x[2]}  {x[3][:60]}")
    return chk - hard, chk, len(viol), len(keybad), len(dup)


if __name__ == "__main__":
    agg = [0, 0, 0, 0, 0]
    for p in sys.argv[1:]:
        r = audit(p)
        agg = [a + b for a, b in zip(agg, r)]
    if len(sys.argv) > 2:
        print(f"\n== 合计：一致 {agg[0]}/{agg[1]} = {agg[0] / max(1, agg[1]):.1%}"
              f"；越界 {agg[2]}；事实键异常 {agg[3]}；重复承载 {agg[4]}")
