"""P3: pilot diagnostics D1-D3, D6-D8 (plan v1.2 §5). Report only, no gates.

D4 (single-value plausibility) and D5 (manual read-through) are not run here: D4 needs a
model judge on a sample of values, D5 is a human read. Both are marked as not run in the
report rather than approximated by a proxy that would look like evidence.

    python3 diagnose.py [--dir .]
"""
import collections
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def load(d, name):
    return [json.loads(l) for l in open(os.path.join(d, name)) if l.strip()]


def toks(s):
    return [w.strip(".,;:!?()\"'").lower() for w in s.split() if w.strip(".,;:!?()\"'")]


def grams(ts, n):
    if len(ts) < n:
        return {tuple(ts)} if ts else set()
    return {tuple(ts[i:i + n]) for i in range(len(ts) - n + 1)}


def jac(a, b):
    return len(a & b) / max(1, len(a | b))


def main():
    d = HERE
    if "--dir" in sys.argv:
        d = sys.argv[sys.argv.index("--dir") + 1]
    docs = {x["doc_id"]: x for x in load(d, "docs.jsonl")}
    pairs = load(d, "pairs.jsonl")
    events = load(d, "events.jsonl")

    # ---------------- D1: structural rule ----------------
    # "a value that shows up in two of the three documents" is all the structure can say;
    # it cannot say which of the two is the source. Accuracy is therefore capped near
    # 50% on any pair where a copy relation exists, which is the point of the design.
    print("== D1 结构规则（只看值出现在哪几篇）")
    d1 = collections.defaultdict(lambda: [0, 0, 0, 0])   # tp, fp, fn, tn
    base = collections.defaultdict(lambda: [0, 0])       # positives, spans
    for pr in pairs:
        trio = [docs[x] for x in pr["docs"]]
        where = collections.defaultdict(set)
        for i, doc in enumerate(trio):
            for sp in doc["spans"]:
                where[sp["value"]].add(i)
        for i, doc in enumerate(trio):
            for sp in doc["spans"]:
                pred = len(where[sp["value"]]) >= 2
                # truth per §3.3, for all three sets (not just Set A)
                if pr["set"] == "A":
                    truth = doc["role"] == "B" and sp["fact"] in doc["a_ids"]
                elif pr["set"] == "B":
                    truth = doc["role"] == "Bf"
                else:
                    truth = False
                c = d1[(pr["branch"], pr["set"])]
                c[0 if (pred and truth) else 1 if pred else 2 if truth else 3] += 1
                base[(pr["branch"], pr["set"])][0] += truth
                base[(pr["branch"], pr["set"])][1] += 1
    for k in sorted(d1):
        tp, fp, fn, tn = d1[k]
        acc = (tp + tn) / max(1, tp + fp + fn + tn)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        pos = base[k][0] / max(1, base[k][1])
        print(f"  {k[0]:6s} Set {k[1]}  准确率 {acc:.3f}  精确率 {prec:.3f} "
              f" 召回率 {rec:.3f}  正例占比 {pos:.3f}  (tp {tp} fp {fp} fn {fn} tn {tn})")

    # ---------------- D2: wording inheritance ----------------
    print("\n== D2 措辞继承（1/3-gram Jaccard + MPNet 余弦）")
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    emb = {k: st.encode(v["text"], normalize_embeddings=True)
           for k, v in docs.items()}

    def pair_sims(role_a, role_b):
        g1, g3, cos = [], [], []
        for ev in events:
            eid, br = ev["event_id"], ev["branch"]
            da, db = docs.get(f"{eid}-{br}-{role_a}"), docs.get(f"{eid}-{br}-{role_b}")
            if not da or not db:
                continue
            ta, tb = toks(da["text"]), toks(db["text"])
            g1.append(jac(grams(ta, 1), grams(tb, 1)))
            g3.append(jac(grams(ta, 3), grams(tb, 3)))
            cos.append(float(emb[da["doc_id"]] @ emb[db["doc_id"]]))
        f = lambda v: f"{np.mean(v):.3f}" if v else "—"
        return f(g1), f(g3), f(cos)

    for a, b in (("A", "B"), ("A", "C"), ("A", "Bf"), ("B", "Bf"), ("C", "Bf")):
        j1, j3, co = pair_sims(a, b)
        print(f"  {a}–{b:2s}  1-gram {j1}  3-gram {j3}  MPNet {co}")

    # can wording alone tell B from C in Set A? And how far is the X slot in Set B?
    def sep(role_x, role_y, role_ref="A"):
        right = tot = 0
        for ev in events:
            eid, br = ev["event_id"], ev["branch"]
            dx, dy, dr = (docs.get(f"{eid}-{br}-{r}") for r in (role_x, role_y, role_ref))
            if not (dx and dy and dr):
                continue
            sx = jac(grams(toks(dr["text"]), 3), grams(toks(dx["text"]), 3))
            sy = jac(grams(toks(dr["text"]), 3), grams(toks(dy["text"]), 3))
            right += sx > sy
            tot += 1
        return right, tot

    r, t = sep("B", "C")
    print(f"  Set A 用 3-gram 相似度分出 B 与 C：{r}/{t} = {r / max(1, t):.1%}（随机 50%）")
    r, t = sep("Bf", "C")
    print(f"  Set A 分出 Bf 与 C（同参照 A）：{r}/{t} = {r / max(1, t):.1%}")

    # Set B: X comes from another event, so this is not a hard test — report the gap
    xs, xsim = 0, []
    for pr in pairs:
        if pr["set"] != "B":
            continue
        a_id = [x for x in pr["docs"] if x.endswith("-A")][0]
        x_id = [x for x in pr["docs"] if x not in
                [d for d in pr["docs"] if d.startswith(pr["event_id"])]][0]
        if a_id in docs and x_id in docs:
            xsim.append(jac(grams(toks(docs[a_id]["text"]), 3),
                            grams(toks(docs[x_id]["text"]), 3)))
            xs += 1
    print(f"  Set B 中 A 与 X（别的事件）的 3-gram：{np.mean(xsim):.3f}"
          f"  （A–Bf 0.414，差 {0.414 - np.mean(xsim):.3f}）")

    # ---------------- D3: length / sentence probes ----------------
    print("\n== D3 长度与句数")
    byrole = collections.defaultdict(list)
    for x in docs.values():
        byrole[x["role"]].append((len(x["text"].split()),
                                  len([s for s in x["text"].replace("!", ".").replace("?", ".")
                                       .split(".") if s.strip()])))
    for r in ("A", "C", "B", "Bf"):
        v = byrole.get(r, [])
        if v:
            print(f"  {r:2s}  词数 {np.mean([x[0] for x in v]):5.1f} "
                  f"（{min(x[0] for x in v)}–{max(x[0] for x in v)}）  "
                  f"句数 {np.mean([x[1] for x in v]):4.1f}")

    # ---------------- D6: n=6, m=2 ----------------
    print("\n== D6 n=6, m=2（A、Bf 只有 2 条事实）")
    ev_meta = {e["event_id"]: e for e in map(json.loads, open(os.path.join(d, "events_pilot.jsonl")))}
    hard = [e for e in ev_meta.values() if e["n"] == 6 and e["m"] == 2]
    if not hard:
        print("  该档已按 2026-11-01 决定取消（E0004、E0012 的 m 由 2 改为 1），无事件可测")
    else:
        ok = [e for e in hard if all((e["event_id"], b, r) in
                                     {(x["event_id"], x["branch"], x["role"]) for x in docs.values()}
                                     for b in ("dup", "rumor") for r in ("A", "B", "Bf"))]
        print(f"  事件数 {len(hard)}；两分支四篇齐全的 {len(ok)}")
        for e in hard:
            ids = [x for x in docs.values() if x["event_id"] == e["event_id"]]
            print(f"    {e['event_id']}  {len(ids)} 篇  " +
                  ", ".join(f"{x['branch']}/{x['role']}={len(x['text'].split())}w"
                            for x in ids if x["role"] in ("A", "Bf")))

    # ---------------- D7: single-document probe, A vs Bf ----------------
    print("\n== D7 单篇方向探针（只给一篇，分 A 与 Bf）")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    rows = [(x["text"], 0 if x["role"] == "A" else 1)
            for x in docs.values() if x["role"] in ("A", "Bf")]
    X = [r[0] for r in rows]
    y = np.array([r[1] for r in rows])
    if len(set(y)) == 2 and len(y) >= 20:
        clf = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), min_df=2),
                            LogisticRegression(max_iter=1000))
        acc = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        print(f"  TF-IDF + LR 5 折：{acc.mean():.3f}（随机 0.500，n={len(y)}）")
        Xe = np.stack([emb[x["doc_id"]] for x in docs.values()
                       if x["role"] in ("A", "Bf")])
        clf2 = LogisticRegression(max_iter=2000)
        acc2 = cross_val_score(clf2, Xe, y, cv=5, scoring="accuracy")
        print(f"  MPNet + LR 5 折：{acc2.mean():.3f}")

    # ---------------- D8: mirror check ----------------
    print("\n== D8 镜像检查")
    evs = {(e["event_id"], e["branch"]): e for e in events}
    hit = miss = 0
    for (eid, br), e in evs.items():
        if br != "rumor":
            continue
        b, c = docs.get(f"{eid}-{br}-B"), docs.get(f"{eid}-{br}-C")
        if not b or not c:
            continue
        # compare value *sets* per fact: a fact may carry two tags, and taking the last
        # value alone loses one of them (that is what made this read 12/14 at first)
        bv, cv = collections.defaultdict(set), collections.defaultdict(set)
        for sp in b["spans"]:
            if sp["version"] == "rumor":
                bv[sp["fact"]].add(sp["value"])
        for sp in c["spans"]:
            cv[sp["fact"]].add(sp["value"])
        for f, v in bv.items():
            if cv.get(f) == v:
                hit += 1
            else:
                miss += 1
    print(f"  C 与 B 在 Rumor 事实上取值一致 {hit} 处，不一致 {miss} 处"
          f"（一致 = 镜像生效，比例 {hit / max(1, hit + miss):.1%}；"
          f"目标 50%±5pp，实际比例以 build_pairs 的 mirroring 行为准）")


if __name__ == "__main__":
    main()
