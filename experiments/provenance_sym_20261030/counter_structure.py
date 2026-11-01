"""§6.2 第一步：从 COUNTER 抽取真实传播的结构特征，按 WD/PD/ND 分组统计。

只做测量，不生成任何东西。目标是给"照抄比例、句序保持、删尾、新增比例、归因密度"
一组真实分布，后面生成 20 个事件时用来对比。

COUNTER 的组织：每对 (0001.xml, 0001p.xml) 是同一事件的两篇报道，前者是通讯社通稿
（多为 APP），后者是一家报纸的版本；`classification` 是这一对的属性，取 WD（整篇基于
通稿）、PD（部分内容来自通稿）、ND（无派生，同一事件的独立报道）。

两个必须先说清的事实，都是本轮实测：

1. WD 不等于逐字照抄。一个 WD 对的通稿 2-gram 只有 56% 出现在报纸版本里，因为报纸会
   改写：`9` 写成 `نو`、`عالمی بلائینڈ کرکٹ کپ` 换成 `بلائنڈ کرکٹ ورلڈکپ`、电头换成本地
   分社。所以"照抄比例"在真实数据里是个中等值，不能按 1.0 设计。
2. 文档 §6.1 引用的"2-gram 区分 WD+PD 与 ND 是 0.810"，本脚本复算为 0.813
   （文档级词 2-gram Jaccard），分词与标点口径一致。这是文档级度量，不是句子级。

语言是乌尔都语。句子按 。 ؟ ! 切分；乌尔都语句号后面经常不带空格（`سکا۔آخری`），
不能按"标点+空格"切。切出的句数比文件属性 totalnoofsentences 系统性少 1（1132/1200
对差 -1，电头被并进第一句），相对特征不受影响。

Usage:
  python3 counter_structure.py ~/Downloads/COUNTER [out.jsonl]
"""
import collections
import glob
import json
import os
import re
import statistics as st
import sys
import unicodedata
import xml.etree.ElementTree as ET

from sklearn.metrics import roc_auc_score

ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/Downloads/COUNTER")
OUT = sys.argv[2] if len(sys.argv) > 2 else "counter_structure.jsonl"
SPLIT = re.compile(r"(?<=[۔؟!])")
PAIR = re.compile(r"^(\d+)([pP]?)\.xml$")

# 归因标记。乌尔都语归因表达多，这里只收能人工核对的几种：
#   اے پی پی  = APP（通讯社电头）  کے مطابق = according to  نے کہا = said
#   کا کہنا ہے = says  بتایا = told  ذرائع = sources  کے حوالے سے = quoting
ATTR = ("کے مطابق", "نے کہا", "کا کہنا ہے", "بتایا", "ذرائع", "کے حوالے سے")
WIRE = "اے پی پی"
QUOTE = ('"', "“", "”", "‘", "’")

TAUS = (0.30, 0.50, 0.70)
TAU_MATCH = 0.50     # 结构特征（句序、删尾、导语）统一用这个阈值判定"这句来自原稿"
LEAD_MIN_WORDS = 5


def sent_split(body):
    return [s.strip() for s in SPLIT.split(body or "") if s.strip()]


def toks(s):
    out = []
    for w in s.split():
        w = "".join(ch for ch in w if not unicodedata.category(ch).startswith("P"))
        if w:
            out.append(w)
    return out


def grams(ts, n=3):
    if len(ts) < n:
        return {tuple(ts)} if ts else set()
    return {tuple(ts[i:i + n]) for i in range(len(ts) - n + 1)}


def cgrams(s, n=4):
    s = " ".join(toks(s))
    return {s[i:i + n] for i in range(max(0, len(s) - n + 1))}


def jac(a, b):
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def cont(a, b):
    return len(a & b) / max(1, len(b))


def lis_len(seq):
    """最长递增子序列：匹配句在原文中的顺序保持程度。"""
    if not seq:
        return 0
    best = [1] * len(seq)
    for i in range(len(seq)):
        for j in range(i):
            if seq[j] < seq[i]:
                best[i] = max(best[i], best[j] + 1)
    return max(best)


def read(path):
    r = ET.parse(path).getroot()
    return {"attr": dict(r.attrib), "headline": (r.findtext("headline") or "").strip(),
            "body": (r.findtext("body") or "").strip()}


def main():
    files = sorted(glob.glob(os.path.join(ROOT, "*.xml")))
    ids = collections.defaultdict(dict)
    for f in files:
        m = PAIR.match(os.path.basename(f))
        if m:
            ids[m.group(1)]["p" if m.group(2) else "b"] = f
    pairs = [i for i in sorted(ids) if len(ids[i]) == 2]
    print(f"语料 {ROOT}：{len(files)} 个文件，{len(ids)} 个编号，{len(pairs)} 个完整对")

    rows = []
    for i in pairs:
        b, p = read(ids[i]["b"]), read(ids[i]["p"])
        cls = b["attr"].get("classification", "?")
        sb, sp = sent_split(b["body"]), sent_split(p["body"])
        if not sb or not sp:
            continue
        tb, tp = toks(b["body"]), toks(p["body"])
        gb = [grams(toks(s)) for s in sb]
        gp = [grams(toks(s)) for s in sp]
        cb = [cgrams(s) for s in sb]
        cp = [cgrams(s) for s in sp]

        # 每个 p 句在 base 里最像的一句；词级 3-gram 与字符级 4-gram 取较大值，
        # 因为乌尔都语词形变化多，纯词级会把改写误判成新增
        best = []
        for j in range(len(sp)):
            sims = [max(jac(gp[j], gb[k]), jac(cp[j], cb[k])) for k in range(len(sb))]
            k = max(range(len(sims)), key=lambda x: sims[x])
            best.append((sims[k], k))

        rec = {"id": i, "class": cls, "domain": b["attr"].get("domain"),
               "wire": b["attr"].get("newspaper"), "paper": p["attr"].get("newspaper"),
               "n_sent_base": len(sb), "n_sent_p": len(sp),
               "n_sent_base_attr": int(b["attr"].get("totalnoofsentences", 0)),
               "sent_ratio": len(sp) / len(sb),
               "word_ratio": len(tp) / max(1, len(tb)),
               "doc_jac2": jac(grams(tb, 2), grams(tp, 2)),
               "doc_cont2": cont(grams(tb, 2), grams(tp, 2)),
               "doc_jac3": jac(grams(tb, 3), grams(tp, 3)),
               "doc_cont3": cont(grams(tb, 3), grams(tp, 3)),
               "headline_jac": jac(grams(toks(b["headline"])), grams(toks(p["headline"]))),
               "attr_density": sum(1 for s in sp if any(a in s for a in ATTR)) / len(sp),
               "quote_density": sum(1 for s in sp if any(q in s for q in QUOTE)) / len(sp),
               "wire_mark": int(WIRE in p["body"]),
               "max_sim_mean": st.mean(s for s, _ in best),
               "max_sim_median": st.median(s for s, _ in best)}

        # 导语：base 中第一个够长的句子（第一句常是两三个词的电头）
        lead = next((k for k, s in enumerate(sb) if len(toks(s)) >= LEAD_MIN_WORDS), 0)
        matched = [(s, k) for s, k in best if s >= TAU_MATCH]
        seq = [k for _, k in matched]
        rec["lead_kept"] = int(bool(seq) and lead in seq)
        rec["order_score"] = (lis_len(seq) / len(seq)) if seq else 0.0
        rec["tail_drop"] = ((len(sb) - 1 - max(seq)) / len(sb)) if seq else 1.0
        for t in TAUS:
            rec[f"copy@{t}"] = sum(1 for s, _ in best if s >= t) / len(sp)
        rec["novel"] = sum(1 for s, _ in best if s < 0.20) / len(sp)
        rec["dateline"] = int(bool(re.match(r"^\S+\s*\([^)]*\)", sp[0])))
        rows.append(rec)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"写出 {OUT}：{len(rows)} 对\n")

    FEATS = ("copy@0.3", "copy@0.5", "copy@0.7", "novel", "order_score", "tail_drop",
             "lead_kept", "headline_jac", "dateline", "attr_density", "quote_density",
             "wire_mark",
             "sent_ratio", "word_ratio", "doc_jac2", "doc_cont2", "doc_jac3",
             "max_sim_mean")
    print("== 1 三类结构特征（均值 | p25 p50 p75）")
    print(f"   {'特征':14s} " + " ".join(f"{c:>21s}" for c in ("WD", "PD", "ND")))
    for ft in FEATS:
        cells = []
        for c in ("WD", "PD", "ND"):
            v = [r[ft] for r in rows if r["class"] == c]
            if not v:
                cells.append(f"{'--':>21s}")
                continue
            q = st.quantiles(v, n=4) if len(v) > 3 else [min(v), st.median(v), max(v)]
            cells.append(f"{st.mean(v):5.2f} | {q[0]:4.2f} {q[1]:4.2f} {q[2]:4.2f}")
        print(f"   {ft:14s} " + " ".join(cells))

    print("\n== 2 每类规模")
    for c in ("WD", "PD", "ND"):
        v = [r for r in rows if r["class"] == c]
        print(f"   {c}: {len(v)} 对，报纸版平均 {st.mean([r['n_sent_p'] for r in v]):.1f} 句，"
              f"句数比 {st.mean([r['sent_ratio'] for r in v]):.2f}，"
              f"词数比 {st.mean([r['word_ratio'] for r in v]):.2f}")

    print("\n== 3 文档级 n-gram 的分离度（复核文档 §6.1 的 0.810）")
    y = [int(r["class"] in ("WD", "PD")) for r in rows]
    for ft in ("doc_jac2", "doc_cont2", "doc_jac3", "doc_cont3"):
        print(f"   {ft:10s} WD+PD vs ND AUROC {roc_auc_score(y, [r[ft] for r in rows]):.3f}")
    for c in ("WD", "PD", "ND"):
        v = [r for r in rows if r["class"] == c]
        print(f"   {c}: doc_jac2 {st.mean([r['doc_jac2'] for r in v]):.3f}  "
              f"doc_cont2 {st.mean([r['doc_cont2'] for r in v]):.3f}  "
              f"headline_jac {st.mean([r['headline_jac'] for r in v]):.3f}")

    print("\n== 4 句子级阈值的敏感性（照抄句占比）")
    for t in TAUS:
        print(f"   tau={t:.2f}  " + "  ".join(
            f"{c} {st.mean([r[f'copy@{t}'] for r in rows if r['class'] == c]):.2f}"
            for c in ("WD", "PD", "ND")))

    print("\n== 5 归因标记抽检（WD 的前 2 对）")
    shown = 0
    for i in pairs:
        if shown >= 2 or rows[[r["id"] for r in rows].index(i)]["class"] != "WD":
            continue
        for s in sent_split(read(ids[i]["p"])["body"])[:2]:
            hit = [a for a in ATTR if a in s] + (["WIRE"] if WIRE in s else [])
            print(f"   [{','.join(hit) if hit else '-'}] {s[:64]}")
        shown += 1


if __name__ == "__main__":
    main()
