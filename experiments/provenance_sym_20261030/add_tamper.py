"""在 P1/P2 产出上做篡改嵌入与二跳继承（计划 §3.4）。

做成独立的**后处理步骤**，不改生成器：生成器已经通过结构与锚点验收，动它有回归风险；
篡改本身是纯程序操作，不需要模型。

三条链：
  1. 值篡改：把 WD_1 里一句"复制"句的一个数字改错 → 该句相对父文档（通稿）tampered。
  2. 错引来源：把 WD_1 里另一句加上一个没说过该事的来源 → forged（事实真、来源假）。
  3. 二跳：一个聚合站照抄 WD_1 的三句（含上面两句）→ 相对父文档（WD_1）verified，
     相对通稿 tampered / forged（继承错误）。

标签只记"相对父文档"的判定（tampered / forged 布尔位），"相对通稿"由数值与通稿事实
逐项比对得出，不靠声明——声明会掩盖转抄过程中的二次改写。

两个选择规则都改成了对任意事件可用（P1 时是写死 7%→9% 和 Ofgem）：
  - 错值按原数字推算（保量纲、保位数），并避开本事件已有的数字；
  - 错引来源从**本事件自己的专名锚点**里挑，取出现条数最多的那个——机构名会在多条
    事实里反复出现，人名与地名不会。这样错引是"同一个故事里的另一个机构"，不是
    "从天而降的 Ofgem"，否则光靠世界知识就能看出来，不需要判来源。

用法：
  python3 add_tamper.py p1_g11.json [--out p1_g11_t.json]
  python3 add_tamper.py p2_e01.json --out p2_e01_t.json
"""
import json
import re
import sys

NUM = re.compile(r"\d(?:[\d,]*\d)?(?:\.\d+)?(?:%|\s?(?:bn|m|p)\b)?")
# 已知映射优先（P1 的产出已按这张表写进报告，保持逐字一致）
WRONG = {
    "850": "1,150", "6%": "11%", "12.4p": "15.6p", "4.26bn": "4.02bn",
    "310m": "284m", "7%": "9%", "12%": "8%", "4.2%": "2.8%",
}
FORGE_SRC = "Ofgem"      # 找不到合适来源时的兜底
MONTHS = ("January", "February", "March", "April", "May", "June", "July",
          "August", "September", "October", "November", "December")


def norm(s):
    return re.sub(r"\s+", "", s)


def wrong_value(lit):
    """把一个数字字面改错：保量纲、保位数、保千分位。"""
    m = re.match(r"([\d,]+(?:\.\d+)?)(.*)$", lit)
    if not m:
        return None
    num, unit = m.group(1), m.group(2)
    plain = num.replace(",", "")
    try:
        if "." in plain:
            dec = len(plain.split(".")[1])
            v = float(plain) * 1.15
            s = f"{v:.{dec}f}"
        else:
            v = int(plain)
            nv = int(round(v * 1.13))
            if nv == v:
                nv = v + 7
            s = str(nv)
    except ValueError:
        return None
    if "," in num:                      # 保留千分位
        s = f"{int(s):,}" if "." not in s else s
    if s == plain:
        return None
    return s + unit


def tamper_candidates(text, used):
    """返回可安全改写的 (原值, 错值)。排除日期里的日号、裸年份与已有数字。"""
    out = []
    for m in NUM.finditer(text):
        lit = m.group(0)
        after = text[m.end():m.end() + 12].strip()
        if any(after.startswith(mo) for mo in MONTHS):
            continue                    # "30 September" 的 30 改了会变成 "34 September"
        if re.fullmatch(r"\d{4}", norm(lit)):
            # 裸年份：改成相邻两年，而不是 2027→2291 那种一眼假的错
            y = int(norm(lit))
            for ny in (y + 2, y - 2, y + 1, y - 1):
                if str(ny) not in used:
                    out.append((lit, str(ny)))
                    break
            continue
        new = WRONG.get(norm(lit)) or wrong_value(lit)
        if not new:
            continue
        if norm(new) in used:
            continue
        out.append((lit, new))
    return out


def pick_tamper(sents, used):
    """挑一句句子把它的一个数字改错。返回 (下标, 原文, 新文, 原值, 新值)。

    优先复制句；没有可改的就退到改写句——`tampered` 记的是"相对父文档这个值错了"，
    与这句是不是照抄无关。P2 实测有 4 个事件的复制句全是裸年份/月份日期，第一轮落空。
    """
    for want in ("copied", "rewritten"):
        for i, s in enumerate(sents):
            if s["origin"] != want:
                continue
            cand = tamper_candidates(s["text"], used)
            if not cand:
                continue
            lit, new = cand[0]
            return i, s["text"], s["text"].replace(lit, new, 1), lit, new
    return None


def forge_source(anchors, sents, skip, prefer=None, exclude=()):
    """从本事件的专名锚点里挑一个"没说过这句"的来源。

    优先级：机构名 > 多词专名（人名）> 其他专名，且排除公司自己与已经出现在句中的。
    只按出现频次挑过一版，P2 实测挑到了地名（", Deeside said."），读起来是病句，
    而且地名不是"背书人"——错引必须是"一个本来会说话的机构/人说了没说过的话"。
    也不能挑公司自己（实测挑到 Northbridge Press Group 给它自己的公告背书，等于没造假）。
    """
    if prefer:
        return prefer
    cnt = {}
    for k, anc in anchors.items():
        for a in anc:
            if NUM.search(a) or len(a) < 3:
                continue
            if any(a in x or x in a for x in exclude if x):
                continue
            cnt[a] = cnt.get(a, 0) + 1
    if not cnt:
        return FORGE_SRC
    org = re.compile(r"\b(Union|Unite|Capital|Estates|Securities|Fitch|Moody|Group|"
                     r"Council|Trust|Authority|Agency|Association|Committee|Commission|"
                     r"Bank|Fund|Board|Services|Solutions|Holdings|Partners|Institute|"
                     r"Federation|Office|Ministry|Department|University|Hospital|Force|"
                     r"Party|Press|Media|Networks|Systems|Ltd|plc|Co)\b")
    buckets = ([], [], [])
    for a in cnt:
        if org.search(a):
            buckets[0].append(a)
        elif len(a.split()) >= 2:
            buckets[1].append(a)
        else:
            buckets[2].append(a)
    for b in buckets:
        for a in sorted(b, key=lambda x: (-cnt[x], x)):
            if any(a in s["text"] for s in sents):
                continue            # 已经出现在 WD_1 里，加上去等于没造假
            return a
    return FORGE_SRC


def pick_forge(sents, skip, src):
    """挑一句没有归因的句子，在句尾追加一个错误来源。事实不变，只换背书人。

    不用前缀是因为原句常以从句开头，前缀会拼出病句（实测：
    "According to Ofgem, that the pressure ... was stated by CEO Marian Holt."）。
    句尾追加 ", X said." 语法安全，且正是真实报道里错引的样子。

    优先复制句（"照抄却换了来源"是最干净的错引），没有就退到改写句：错引标签说的是
    "这句给了一个没说过它的来源"，与这句是不是照抄无关。
    """
    ATTR = ("said", "stated", "According to", "noted", "added", "told")

    def free(i, s):
        return (i != skip
                and src not in s["text"]
                and not any(a in s["text"] for a in ATTR))

    for want in ("copied", "rewritten", "own"):
        for i, s in enumerate(sents):
            if s["origin"] == want and free(i, s):
                return i, s["text"], s["text"].rstrip(".") + f", {src} said.", want
    return None


def main():
    src_path = sys.argv[1]
    out = (sys.argv[sys.argv.index("--out") + 1] if "--out" in sys.argv
           else src_path.replace(".json", "_t.json"))
    d = json.load(open(src_path))
    wd = next(n for n in d["variants"] if d["variants"][n]["role"] == "WD")
    sents = d["variants"][wd]["sentences"]
    anchors = d.get("anchors") or {}

    used = {norm(x) for v in d["variants"].values() for s in v["sentences"]
            for x in NUM.findall(s["text"])}
    t = pick_tamper(sents, used)
    if not t:
        raise SystemExit("没有可篡改的复制句")
    ti, t_old, t_new, t_from, t_to = t
    sents[ti]["text"] = t_new
    sents[ti]["tampered"] = True
    sents[ti]["tamper"] = {"from": t_from, "to": t_to}
    print(f"== 值篡改 {wd} #{ti} [{sents[ti]['fact']}] {t_from} → {t_to}")
    print(f"   后: {t_new}")

    src = forge_source(anchors, sents, ti,
                       prefer=(sys.argv[sys.argv.index("--forge-src") + 1]
                               if "--forge-src" in sys.argv else None),
                       exclude=(d.get("company"),))
    f = pick_forge(sents, ti, src)
    forged_idx = None
    if f:
        fi, f_old, f_new, f_org = f
        sents[fi]["text"] = f_new
        sents[fi]["forged"] = True
        sents[fi]["forge"] = {"source": src, "was": f_org}
        forged_idx = fi
        print(f"== 错引来源 {wd} #{fi} [{sents[fi]['fact']}]（原为 {f_org}）加 ', {src} said.'")
        print(f"   后: {f_new}")
    else:
        print(f"== 错引来源 {wd} 无可用句（全部已带归因），本事件缺错引")

    # 二跳：照抄 WD_1 的三句，含被篡改与被错引的两句
    order = [ti] + ([forged_idx] if forged_idx is not None else [])
    order += [i for i in range(len(sents)) if i not in order][:3 - len(order)]
    hop = [dict(sents[i]) for i in order]
    for s in hop:
        s["origin"] = "copied"
        s["hop"] = 2
    d["variants"]["HOP2"] = {"role": "HOP2", "parent": wd, "sentences": hop}
    print(f"== 二跳：照抄 {wd} 的第 {order} 句")

    # 只打"相对父文档"的判定；"相对通稿"由 audit_p1.py 用数值逐项比对得出，
    # 不在这里声明——声明会掩盖转抄过程中的二次改写。
    for name, v in d["variants"].items():
        for s in v["sentences"]:
            s.setdefault("tampered", False)
            s.setdefault("forged", False)
    json.dump(d, open(out, "w"), ensure_ascii=False, indent=1)
    print(f"写出 {out}")


if __name__ == "__main__":
    main()
