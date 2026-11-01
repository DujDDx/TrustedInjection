"""P2：批量生成事件（每个事件 5 篇文档），复用 P1 的提示词与布局。

与 `struct_p1.py` 的差别只有三处，其余（提示词、改写强度、复制/改写比例、导语与句序
随机化）逐字复用同一个模块，避免两套实现漂移：

  1. 事实包来自 `events_p2.jsonl`（`gen_events.py` 起草），不是写死的单事件；
  2. 通稿句数按事件取 6/7/8，让整数取整噪声在 20 个事件上摊开；
  3. 句→事实的归属改成**分值匹配**（`assign_facts`），不再要求"全部锚点命中"。

第 3 点是必需的：P1 手写事实时做到了"每条事实的锚点在全包内唯一"，模型起草的事件包
做不到——议员选区、工会、公司名本来就会在好几条事实里出现。硬要求唯一会把自然写法
挡掉（试过三轮，每次都卡在"锚点也全在别处"）。改成按命中比例取最高分，并把"最高分
不唯一 / 没有锚点命中"的句子报出来，可靠性是可测量的，不是假定的。

用法（服务器或本机，TI_BASE 指向写作者）：
  python3 struct_p2.py --events events_p2.jsonl --out-dir . --start 0 --end 20
"""
import json
import os
import random
import sys

import struct_p1 as S
from llm_client import complete_json

# 加固改写：与 S.R_PROMPT 是同一个操作，只是把"保留较多原措辞"这句去掉。文档提示词
# 必须逐字相同，改写提示词不必——它不参与"所有文档提示词一致"的判定。
STRONG_PROMPT = """Rewrite each sentence below so that it shares as little wording as
possible with the original while keeping exactly the same information.

Rules:
- keep every number, name, place and date exactly as written;
- do not merge or split sentences; return one rewrite per input sentence, same order;
- do not add or drop any fact;
- output only JSON: {"rewrites": ["...", "..."]}

Sentences:
__SENTS__"""


def hard_rewrite(pairs, cap=0.48, rounds=2):
    """把仍然 ≥cap 的改写再抽一轮更激进的版本。

    实测（P2 前 4 个事件）24 句改写里有 5 句 ≥0.50，最高 0.765——模型直接回了原句。
    这些句子既把 copy 比例顶高（0.571 vs 0.47），又把导语保留顶高（0.750 vs 0.53），
    因为测的是相似度，不是标签。只对不达标的句子重抽，达标的句子不动。
    """
    out = [c for _, c in pairs]
    for _ in range(rounds):
        bad = [i for i, (o, c) in enumerate(zip([p[0] for p in pairs], out))
               if S.sim(c, o) >= cap]
        if not bad:
            break
        src = [pairs[i][0] for i in bad]
        try:
            r = complete_json(STRONG_PROMPT.replace("__SENTS__", "\n".join(
                f"{j + 1}. {s}" for j, s in enumerate(src))), max_tokens=900)["rewrites"]
        except Exception:                   # noqa: BLE001
            break
        for k, i in enumerate(bad):
            if k >= len(r):
                continue
            cand = str(r[k]).strip()
            if S.sim(cand, pairs[i][0]) < S.sim(out[i], pairs[i][0]):
                out[i] = cand
    return out


# ---------------------------------------------------------------- 归属匹配
def assign_facts(texts, facts, anchors):
    """给每个句子找事实键。返回 (keys, stats)。

    分数 = 该事实的锚点在这句里出现的比例。要求：
      - 最高分 ≥ 0.5 且**严格高于**次高（差 ≥ 0.01），否则记为未定；
      - 同一条事实只能被一句认领（分值高的那句留下，另一句记为未定）。
    未定的句子不丢，键写 "?"，在 rec["assign"] 里计数——这些句子的事实标签不可信，
    训练时按不可用处理。
    """
    score = []
    for t in texts:
        row = {}
        for k in facts:
            anc = anchors.get(k) or []
            if not anc:
                continue
            hit = sum(1 for a in anc if S._norm(a) in S._norm(t))
            row[k] = hit / len(anc)
        score.append(row)

    keys, claimed = [], {}
    for i, row in enumerate(score):
        if not row:
            keys.append(None)
            continue
        top = sorted(row.items(), key=lambda kv: -kv[1])
        k, v = top[0]
        second = top[1][1] if len(top) > 1 else 0.0
        if v < 0.5 or (v - second) < 0.01:
            keys.append(None)
            continue
        keys.append(k)

    # 同一事实被多句认领：留分值最高的一句
    for i, k in enumerate(keys):
        if k is None:
            continue
        if k in claimed:
            j = claimed[k]
            if score[i][k] > score[j][k]:
                keys[j] = None
                claimed[k] = i
            else:
                keys[i] = None
        else:
            claimed[k] = i

    stats = {"total": len(texts), "unassigned": sum(1 for k in keys if k is None)}
    return keys, stats


# ---------------------------------------------------------------- 一个事件
def build_event(ev, seed, outlets):
    n = ev["n_wire"]
    groups = ("wire", "nd", "pd", "wd")
    facts = {g: {k: v["note"] for k, v in ev[g].items()} for g in groups}
    anchors = {k: v["anchors"] for g in groups for k, v in ev[g].items()}
    S.ANCHORS = anchors
    S.ALL_FACTS = {k: v for g in groups for k, v in facts[g].items()}

    rng = random.Random(seed)
    rec = {"event": ev["id"], "seed": seed, "domain": ev.get("domain"),
           "company": ev.get("company"), "n_wire": n,
           "prompt_sha": S.hashlib.sha256(S.PROMPT.encode()).hexdigest()[:16],
           "anchors": anchors, "wire": {}, "variants": {}, "assign": {}}

    def note_assign(name, keys, texts):
        st = {"total": len(texts), "unassigned": sum(1 for k in keys if k is None)}
        rec["assign"][name] = st
        if st["unassigned"]:
            print(f"   {name}: {st['unassigned']}/{st['total']} 句未能归属事实")
        return st

    # 1. 通稿
    hl, w = S.gen_doc(outlets["wire"], n, facts["wire"], want=n, tol=0)
    if len(w) != n:
        raise RuntimeError(f"{ev['id']} 通稿写了 {len(w)} 句，要求 {n}")
    keys, _ = assign_facts([t for _, t in w], facts["wire"], anchors)
    w = [(keys[i] or f, t) for i, (f, t) in enumerate(w)]
    note_assign("W", keys, w)
    rec["wire"] = {"headline": hl, "sentences": w}
    print(f"== {ev['id']} W {n} 句  {hl[:60]}")

    # 2. WD ×3
    #    导语保留与句序不再用 P1 的固定三模式（(T,F),(F,T),(T,T)）：那样每个事件的
    #    lead 恒为 2/3 = 0.667，而 COUNTER 是 0.53（P1 三事件实测 0.667，偏 +0.137）。
    #    改成按 COUNTER 比例逐篇随机抽，20 事件 × 3 篇 = 60 篇上自然收敛。
    own_ids = list(facts["wd"])
    for j in range(3):
        keep_lead = rng.random() < 0.53      # COUNTER WD lead_kept
        # 打乱句序的概率不能照抄 COUNTER 的 order=0.64：我们的打乱是"把最后一个复制
        # 句提到最前"，非打乱时 LIS=1.0、打乱后约 0.5，所以均值 = 1-p/2。要落到
        # 0.64 需要 p≈0.72。P1 的 1/3 打乱给出 0.685，正是这个式子。
        scramble = rng.random() < 0.72
        copy_idx, rewrite_idx, order = S.wd_layout(n, keep_lead, scramble, rng,
                                                   exclude_lead=True)
        cp = S.rewrite([w[i][1] for i in copy_idx], draws=2, target=0.80,
                       cap=0.95, floor=0.55)
        cmap = {i: cp[k] for k, i in enumerate(copy_idx)}
        for i in copy_idx:                  # 编辑过头就不算照抄了，退回原文
            if S.sim(cmap[i], w[i][1]) < 0.50:
                cmap[i] = w[i][1]
        rw = S.rewrite([w[i][1] for i in rewrite_idx], draws=3,
                       target=0.35, cap=0.48, floor=0.22)
        rw = hard_rewrite([(w[i][1], rw[k]) for k, i in enumerate(rewrite_idx)])
        rmap = {i: rw[k] for k, i in enumerate(rewrite_idx)}
        # 加固后仍 ≥0.50 的，按"照抄"记：标签必须与表象一致（审计的硬约束就是这个）。
        # 硬说它是改写会让"说是改写、看着像照抄"变成不可判定的样本。
        rorigin = {i: ("rewritten" if S.sim(rmap[i], w[i][1]) < 0.50 else "copied")
                   for i in rewrite_idx}
        own_id = own_ids[j]
        _, own_sents = S.gen_doc(outlets[f"wd{j + 1}"], 1, {own_id: facts["wd"][own_id]},
                                 want=1)
        own_txt = own_sents[0][1] if own_sents else facts["wd"][own_id]
        out = [(w[i][0], cmap[i], "copied") if i in copy_idx
               else (w[i][0], rmap[i], rorigin[i]) for i in order]
        out.append((own_id, own_txt, "own"))
        name = f"WD_{j + 1}"
        rec["variants"][name] = {"role": "WD", "outlet": outlets[f"wd{j + 1}"],
                                 "sentences": [{"fact": f, "text": t, "origin": o}
                                               for f, t, o in out]}

    # 3. PD：借 2 句改写，其余自采
    pd_total = round(S.R_PD * n)
    borrow = [0, min(2, n - 1)]
    rws = S.rewrite([w[i][1] for i in borrow], draws=3, target=0.30,
                    cap=0.45, floor=0.22)
    rws = hard_rewrite([(w[i][1], rws[k]) for k, i in enumerate(borrow)])
    porigin = {i: ("rewritten" if S.sim(rws[k], w[i][1]) < 0.50 else "copied")
               for k, i in enumerate(borrow)}
    want_own = max(1, pd_total - len(borrow))
    hl_pd, pd_own = S.gen_doc(outlets["pd"], want_own, facts["pd"], want=want_own)
    own_keys, _ = assign_facts([t for _, t in pd_own], facts["pd"], anchors)
    pd = [(w[borrow[0]][0], rws[0], porigin[borrow[0]])]
    half = len(pd_own) // 2
    pd += [(own_keys[i] or f, t, "own") for i, (f, t) in enumerate(pd_own[:half])]
    pd.append((w[borrow[1]][0], rws[1], porigin[borrow[1]]))
    pd += [(own_keys[i + half] or f, t, "own")
           for i, (f, t) in enumerate(pd_own[half:])]
    note_assign("PD_own", own_keys, pd_own)
    rec["variants"]["PD"] = {"role": "PD", "headline": hl_pd,
                             "sentences": [{"fact": f, "text": t, "origin": o}
                                           for f, t, o in pd]}

    # 4. ND：两条共同事实 + 全部自采，不看 W 正文
    nd_total = round(S.R_ND * n)
    wkeys = list(facts["wire"])
    nd_pool = {**{k: facts["wire"][k] for k in (wkeys[1], wkeys[2])}, **facts["nd"]}
    hl_nd, nd = S.gen_doc(outlets["nd"], nd_total, nd_pool, want=nd_total, tol=0)
    nd_keys, _ = assign_facts([t for _, t in nd], nd_pool, anchors)
    note_assign("ND", nd_keys, nd)
    rec["variants"]["ND"] = {"role": "ND", "headline": hl_nd,
                             "sentences": [{"fact": nd_keys[i] or f, "text": t,
                                            "origin": "own"}
                                           for i, (f, t) in enumerate(nd)]}

    # 5. 去重（自采句不得与通稿 ≥0.50）+ 锚点自检（缺失 / 多余数字）
    wt = [t for _, t in w]
    for name, v in rec["variants"].items():
        if v["role"] == "HOP2":
            continue
        fx, lf = decontaminate(v, wt)
        if fx or lf:
            print(f"   {name}: 去重修 {fx} 句，未修 {lf} 句")
    bad = tot = 0
    for name, v in rec["variants"].items():
        for s in v["sentences"]:
            k = s["fact"]
            if k not in S.ALL_FACTS:
                continue
            tot += 1
            miss, extra = S.anchor_report(k, s["text"])
            if miss or extra:
                bad += 1
                print(f"   {name} [{k}] 缺失 {miss} 多余 {extra}")
    rec["anchor_missing"], rec["anchor_total"] = bad, tot

    # 6. 一句一事实
    issues = S.fact_integrity(rec["variants"])
    rec["fact_integrity_issues"] = [{"variant": n_, "fact": k, "why": w_}
                                    for n_, k, w_, _ in issues]
    print(f"   锚点 {bad}/{tot}；一句一事实 {len(issues)} 处")
    return rec


OUTLETS = [
    ("Pennine Wire Service", "The Northern Echo", "The Yorkshire Post", "Western Mail",
     "The Herald", "East Anglian Daily Times"),
    ("Meridian News Agency", "The Press and Journal", "Western Morning News",
     "Belfast Telegraph", "The Courier", "Evening Standard"),
    ("County News Wire", "Manchester Evening News", "The Journal", "South Wales Argus",
     "Yorkshire Evening Post", "The Argus"),
]


def best_of(sent, wire, rounds=2):
    """重抽 rounds 次，取与通稿最大相似度最低的版本。"""
    cur, cur_sim = sent, max(S.sim(sent, t) for t in wire)
    for _ in range(rounds):
        try:
            r = complete_json(STRONG_PROMPT.replace("__SENTS__", f"1. {cur}"),
                              max_tokens=200)["rewrites"]
        except Exception:                   # noqa: BLE001
            break
        cand = str(r[0]).strip() if r else ""
        if not cand:
            continue
        s = max(S.sim(cand, t) for t in wire)
        if s < cur_sim:
            cur, cur_sim = cand, s
    return cur, cur_sim


def decontaminate(variant, wire, cap=0.50):
    """把"自采"句里与通稿 ≥cap 的那些改掉。

    ND 要复述通稿的两条共同事实（cont2 目标 0.10 就靠这两条），模型常把它们写得与
    通稿几乎同句——P2 前 20 个事件实测 16 句"自采"相似度 ≥0.50，最高 0.53。标签说
    自采、文本看着照抄，正是审计硬约束要挡的。只改文本，不动标签与句序。
    """
    fixed = left = 0
    for s in variant["sentences"]:
        if s["origin"] == "copied":
            continue
        if max(S.sim(s["text"], t) for t in wire) < cap:
            continue
        new, sim = best_of(s["text"], wire)
        if sim < cap:
            s["text"] = new
            fixed += 1
        else:
            left += 1
    return fixed, left


def stable_seed(s):
    """事件 id → 稳定种子。不能用内置 hash()：字符串哈希每次进程都变，同一事件
    重跑会得到不同的复制下标，复现不了。"""
    h = 0
    for c in s:
        h = (h * 131 + ord(c)) % (2 ** 31 - 1)
    return h


def main():
    a = sys.argv
    ev_path = a[a.index("--events") + 1] if "--events" in a else "events_p2.jsonl"
    out_dir = a[a.index("--out-dir") + 1] if "--out-dir" in a else "."
    start = int(a[a.index("--start") + 1]) if "--start" in a else 0
    end = int(a[a.index("--end") + 1]) if "--end" in a else 10 ** 6
    events = [json.loads(l) for l in open(ev_path) if l.strip()][start:end]
    print(f"共 {len(events)} 个事件（{start}..{start + len(events)}）")

    for i, ev in enumerate(events):
        wire, wd1, wd2, wd3, pd, nd = OUTLETS[i % len(OUTLETS)]
        outlets = {"wire": wire, "wd1": wd1, "wd2": wd2, "wd3": wd3, "pd": pd, "nd": nd}
        path = os.path.join(out_dir, f"p2_{ev['id']}.json")
        if os.path.exists(path):
            print(f"-- {ev['id']} 已存在，跳过")
            continue
        try:
            rec = build_event(ev, seed=stable_seed(ev["id"]), outlets=outlets)
        except Exception as e:              # noqa: BLE001
            print(f"!! {ev['id']} 失败：{type(e).__name__}: {str(e)[:140]}")
            continue
        with open(path, "w") as f:
            f.write(json.dumps(rec, ensure_ascii=False, indent=1) + "\n")
        print(f"   写出 {path}")


if __name__ == "__main__":
    main()
