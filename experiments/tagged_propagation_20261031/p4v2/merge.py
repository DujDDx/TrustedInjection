"""Merge the eight P4 batches into events_p4v2.jsonl and check the whole table.

    python3 merge.py            # writes ../events_p4v2.jsonl, then runs the validator on it

Each batch must already pass validate_events.py on its own. Across batches this adds the
checks a single batch cannot see: ID ranges, the 260:140 n split, and (through the
validator's file-level checks) entities and subjects reused between domains.
"""
import collections
import glob
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "events_p4v2.jsonl")
EXPECT = {1: (33, 17), 2: (33, 17), 3: (33, 17), 4: (33, 17),
          5: (32, 18), 6: (32, 18), 7: (32, 18), 8: (32, 18)}


def main():
    files = sorted(glob.glob(os.path.join(HERE, "batch_*.jsonl")))
    rows, bad = [], 0
    for f in files:
        k = int(os.path.basename(f).split("_")[1])
        part = [json.loads(l) for l in open(f) if l.strip()]
        lo = 2001 + (k - 1) * 50
        want = {f"E{i}" for i in range(lo, lo + 50)}
        got = {r["event_id"] for r in part}
        n = collections.Counter(r["n"] for r in part)
        msg = []
        if got != want:
            msg.append(f"ids off by {len(got ^ want)}")
        if (n[5], n[6]) != EXPECT[k]:
            msg.append(f"n split {n[5]}:{n[6]}, want {EXPECT[k][0]}:{EXPECT[k][1]}")
        print(f"{os.path.basename(f)}: {len(part)} events {'OK' if not msg else '; '.join(msg)}")
        bad += bool(msg)
        rows += part
    if len(files) != 8:
        print(f"only {len(files)} of 8 batches present")
        bad += 1
    rows.sort(key=lambda r: r["event_id"])
    with open(OUT, "w") as fo:
        for r in rows:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} events to {os.path.relpath(OUT)}\n")
    rc = subprocess.call([sys.executable, os.path.join(HERE, "..", "validate_events.py"), OUT])
    sys.exit(1 if bad or rc else 0)


if __name__ == "__main__":
    main()
