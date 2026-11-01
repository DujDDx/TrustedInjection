"""把"自采"句里与通稿过于相似的那些改掉（P2 验收发现的问题，一次性修数据用）。

真正的修法已经并进 `struct_p2.decontaminate`，新生成的事件不会再带这个问题；本脚本
只用来修已经生成的 20 个事件，避免为 16 句重跑 40 分钟的生成。

用法：
  python3 fix_copy_leak.py p2_e01.json [more.json ...]
"""
import json
import sys

from struct_p2 import decontaminate


def main():
    for path in sys.argv[1:]:
        d = json.load(open(path))
        wire = [t for _, t in d["wire"]["sentences"]]
        fixed = left = 0
        for name, v in d["variants"].items():
            if v["role"] == "HOP2":
                continue
            fx, lf = decontaminate(v, wire)
            fixed += fx
            left += lf
            if lf:
                print(f"   {path} {name} 有 {lf} 句改写后仍 ≥0.50")
        if fixed or left:
            json.dump(d, open(path, "w"), ensure_ascii=False, indent=1)
        print(f"== {path}  修 {fixed} 句，未修 {left} 句")


if __name__ == "__main__":
    main()
