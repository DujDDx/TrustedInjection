"""汇总 P2 各事件的结构指标，按 §5.2 的验收口径出判定。

口径（用户 2026-10-31 决定 B 之后收窄，见 P1 实测 §7）：
  硬指标：句数比、照抄比例、新增比例、文档级 jac2、文档级 cont2，以及 order/lead
          （order 只在匹配句 ≥2 时有定义，按有定义的事件计）
  只记录：词数比（本数据集句长系统性偏短，不能与真实语料直接比）
          ND 新增比例（句长偏短的下游效应）

用法：
  python3 p2_summarize.py p2_e*.json
"""
import json
import statistics as st
import sys

from p1_measure import COUNTER, profile, toks

GATE = ["sent_ratio", "copy", "new", "order", "lead", "jac2", "cont2"]
RECORD = ["word_ratio", "copy30"]
TOL = 0.10


def main():
    paths = sys.argv[1:]
    if not paths:
        print("用法：python3 p2_summarize.py p2_e*.json")
        return
    per_role = {}
    events = []
    for p in paths:
        d = json.load(open(p))
        w = [t for _, t in d["wire"]["sentences"]]
        bw = sum(len(toks(t)) for t in w)
        rows = {}
        for name, v in d["variants"].items():
            if v["role"] == "HOP2":
                continue
            pr = profile([s["text"] for s in v["sentences"]], w)
            pr["sent_ratio"] = round(pr["sent"] / len(w), 3)
            pr["word_ratio"] = round(pr["words"] / bw, 3)
            rows[v["role"]] = pr
            per_role.setdefault(v["role"], []).append(pr)
        events.append((d.get("event"), len(w), rows))

    print(f"事件数 {len(events)}；通稿句数 {sorted(set(e[1] for e in events))}")
    print(f"\n{'角色':6s} {'指标':12s} {'均值':>7s} {'标准差':>7s} {'最小':>7s} "
          f"{'最大':>7s} {'目标':>7s} {'差':>7s}  判定")
    summary = {}
    for role in ("WD", "PD", "ND"):
        rows = per_role.get(role, [])
        if not rows:
            continue
        c = COUNTER[role]
        for k in GATE + RECORD:
            vals = [r[k] for r in rows if r.get(k) is not None]
            if not vals:
                continue
            m = st.mean(vals)
            d = m - c[k]
            if k in GATE:
                verdict = "通过" if abs(d) <= TOL else "**超标**"
            else:
                verdict = "（只记录）"
            sd = st.pstdev(vals) if len(vals) > 1 else 0.0
            print(f"{role:6s} {k:12s} {m:7.3f} {sd:7.3f} {min(vals):7.3f} "
                  f"{max(vals):7.3f} {c[k]:7.3f} {d:+7.3f}  {verdict}"
                  + (f"  仅 {len(vals)} 个事件有定义" if len(vals) < len(rows) else ""))
            summary[(role, k)] = (m, d, verdict)

    # 逐事件最差偏离，用于定位个别事件是否把均值拖偏。
    #    只查文档级可比的指标（句数比、jac2、cont2、词数比）。copy/new/lead/order 是
    #    单篇的 0/1 或序数量，单篇取 0 或 1 本来就会离均值很远——20 个事件会全部被标
    #    成"超标"，而语料均值其实是达标的。这类指标只能按语料均值判。
    EV_KEYS = ["sent_ratio", "jac2", "cont2"]
    print("\n逐事件最差硬指标偏离（只看文档级指标；>0.10 的事件）")
    bad = 0
    for ev, n, rows in events:
        worst = (0.0, None, None)
        for role, pr in rows.items():
            for k in EV_KEYS:
                v = pr.get(k)
                if v is None:
                    continue
                d = abs(v - COUNTER[role][k])
                if d > worst[0]:
                    worst = (d, f"{role}.{k}", v)
        if worst[0] > TOL:
            bad += 1
            print(f"  {ev}  W={n}  {worst[1]}={worst[2]:.3f}  偏离 {worst[0]:.3f}")
    print(f"  超标事件 {bad}/{len(events)}")

    gate_fail = [f"{r}.{k}" for (r, k), (_, d, vd) in summary.items()
                 if k in GATE and vd != "通过"]
    print(f"\n硬指标超标项：{', '.join(gate_fail) if gate_fail else '无'}")
    print(f"提示：词数比只作记录（本数据集句长偏短，见 P1 实测 §7）")


if __name__ == "__main__":
    main()
