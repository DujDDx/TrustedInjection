"""按 COUNTER 的口径测量一份 P1/P2 产出文档的结构特征。

为什么单独一个脚本：生成器里按"来源标签"算照抄比例是错的口径。COUNTER 的
0.47 是**相似度判定**——句级相似度 ≥0.50 记为照抄，<0.20 记为新增，中间为改写。
标签说的是"程序怎么做的"，相似度说的是"文本看起来怎样"，验收要比后者。

用法：
  python3 p1_measure.py p1_event_v2.json [more.json ...]
"""
import json
import sys
import unicodedata

TAU_COPY = (0.50, 0.30)   # 主阈值 + 敏感性阈值
TAU_NEW = 0.20

# COUNTER 实测（乌尔都语 600 对，计划 §2.2）
COUNTER = {
    "WD": {"sent_ratio": 0.62, "word_ratio": 0.76, "copy": 0.47, "copy30": 0.64,
           "new": 0.24, "order": 0.64, "lead": 0.53, "jac2": 0.35, "cont2": 0.57},
    "PD": {"sent_ratio": 0.92, "word_ratio": 1.08, "copy": 0.05, "copy30": 0.12,
           "new": 0.71, "order": 0.18, "lead": 0.09, "jac2": 0.11, "cont2": 0.19},
    "ND": {"sent_ratio": 1.85, "word_ratio": 2.12, "copy": 0.02, "copy30": 0.04,
           "new": 0.88, "order": 0.08, "lead": 0.04, "jac2": 0.06, "cont2": 0.10},
}


def toks(s):
    return [w for w in ("".join(c for c in t if not unicodedata.category(c).startswith("P")).lower()
                        for t in s.split()) if w]


def grams(ts, n):
    if len(ts) < n:
        return {tuple(ts)} if ts else set()
    return {tuple(ts[i:i + n]) for i in range(len(ts) - n + 1)}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def cont(a, b):
    return len(a & b) / max(1, len(b))


def lis(seq):
    if not seq:
        return 0
    d = []
    for x in seq:
        lo, hi = 0, len(d)
        while lo < hi:
            m = (lo + hi) // 2
            if d[m] < x:
                lo = m + 1
            else:
                hi = m
        if lo == len(d):
            d.append(x)
        else:
            d[lo] = x
    return len(d)


def profile(doc_texts, base_texts):
    """COUNTER 口径的句级画像 + 文档级重合。"""
    b3 = [grams(toks(b), 3) for b in base_texts]
    sims, match = [], []
    for i, d in enumerate(doc_texts):
        g = grams(toks(d), 3)
        best, bi = 0.0, -1
        for j, bg in enumerate(b3):
            s = jac(g, bg)
            if s > best:
                best, bi = s, j
        sims.append(best)
        match.append(bi if best >= TAU_COPY[0] else -1)
    n = max(1, len(doc_texts))
    matched = [m for m in match if m >= 0]
    tb = grams(toks(" ".join(base_texts)), 2)
    tp = grams(toks(" ".join(doc_texts)), 2)
    # 导语保留：通稿第 1 句是否被本文任一句以 ≥0.50 相似度保留（不一定在首位）
    lead = 1.0 if base_texts and any(
        jac(grams(toks(d), 3), grams(toks(base_texts[0]), 3)) >= TAU_COPY[0]
        for d in doc_texts) else 0.0
    return {
        "sent": len(doc_texts),
        "words": sum(len(toks(d)) for d in doc_texts),
        "copy": round(sum(s >= TAU_COPY[0] for s in sims) / n, 3),
        "copy30": round(sum(s >= TAU_COPY[1] for s in sims) / n, 3),
        "new": round(sum(s < TAU_NEW for s in sims) / n, 3),
        "order": round(lis(matched) / len(matched), 3) if len(matched) >= 2 else None,
        "lead": round(lead, 3),
        "jac2": round(jac(tb, tp), 3),
        # 包含率与 P0 的 counter_structure.py 保持一致：cont(b,p)=|b∩p|/|p|，
        # 即"本文的 2-gram 有多少来自通稿"（精确率）。计划 §2.3 的文字写成
        # "通稿 2-gram 有多少出现在报纸版里"（召回），与代码口径相反，以代码为准。
        "cont2": round(cont(tb, tp), 3),
        "sims": [round(s, 3) for s in sims],
    }


def row(name, role, p, base_sent, base_words):
    c = COUNTER[role]
    p = dict(p)
    p["sent_ratio"] = round(p["sent"] / base_sent, 3)
    p["word_ratio"] = round(p["words"] / base_words, 3)
    keys = ["sent_ratio", "word_ratio", "copy", "copy30", "new", "order", "lead",
            "jac2", "cont2"]
    cells, worst = [], 0.0
    for k in keys:
        v, t = p.get(k), c.get(k)
        if v is None or t is None:
            cells.append(f"{k}=—")
            continue
        d = abs(v - t)
        worst = max(worst, d)
        flag = " " if d <= 0.10 else "!"
        cells.append(f"{k}={v:.3f}({v - t:+.3f}){flag}")
    print(f"  {name:14s} [{role}] " + "  ".join(cells))
    return worst


def main():
    for path in sys.argv[1:]:
        r = json.load(open(path))
        w = [t for _, t in r["wire"]["sentences"]]
        bw = sum(len(toks(t)) for t in w)
        print(f"\n=== {path}   W={len(w)} 句 / {bw} 词  prompt_sha={r.get('prompt_sha')}")
        for name, v in r["variants"].items():
            p = profile([s["text"] for s in v["sentences"]], w)
            worst = row(name, v["role"], p, len(w), bw)
            print(f"  {'':14s} 句级相似度 {p['sims']}  最差偏离 {worst:.3f}")
        if "sibling_jac2" in r:
            print(f"  同源兄弟 jac2 = {r['sibling_jac2']}（难负例，应显著高于 0.06）")


if __name__ == "__main__":
    main()
