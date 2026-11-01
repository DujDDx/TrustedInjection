"""统一的模型调用客户端（本地 llama.cpp /v1/chat/completions）。

两个后端，靠 TI_BACKEND 选，默认 local：
  local      本地 llama-server（OpenAI 兼容），默认 http://127.0.0.1:8818
  anthropic  ~/.config/ti/ds.json 里的 opencode.ai/zen/go（额度受限，仅备用）

**思考一律关闭**：Qwen3 系默认开思考，实测一次 JSON 任务 700 token 全进了
reasoning_content，content 为空、JSON 解析失败。所以请求里固定带
chat_template_kwargs.enable_thinking=false，并在解析前再剥一次  thinking 块。
"""
import json
import os
import re
import time
import urllib.error
import urllib.request
import uuid

BASE = os.environ.get("TI_BASE", "http://127.0.0.1:8818").rstrip("/")
MODEL = os.environ.get("TI_MODEL", "local-model")
BACKEND = os.environ.get("TI_BACKEND", "local")
CFG = os.path.expanduser("~/.config/ti/ds.json")
UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")

_THINK = re.compile(r"<think\b[^>]*>.*?</think\s*>", re.S | re.I)


def _local(prompt, max_tokens, temperature, timeout, seed=None):
    body = {"model": MODEL, "max_tokens": max_tokens, "temperature": temperature,
            "chat_template_kwargs": {"enable_thinking": False},
            "messages": [{"role": "user", "content": prompt}]}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    ch = d["choices"][0]
    msg = ch.get("message", {}) or {}
    txt = msg.get("content") or ""
    if not txt.strip() and (msg.get("reasoning_content") or ""):
        raise RuntimeError("thinking_not_disabled: reasoning only, content empty")
    if ch.get("finish_reason") == "length":
        raise RuntimeError(f"truncated at max_tokens={max_tokens}")
    return txt


def _anthropic(prompt, max_tokens, temperature, timeout):
    with open(CFG) as f:
        c = json.load(f)
    body = {"model": os.environ.get("TI_MODEL", "deepseek-v4.1-flash"),
            "max_tokens": max_tokens, "temperature": temperature,
            "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": prompt}]}
    h = {"x-api-key": os.environ.get("TI_KEY", c["key"]),
         "anthropic-version": "2023-06-01", "content-type": "application/json",
         "x-opencode-session": str(uuid.uuid4()), "User-Agent": UA}
    req = urllib.request.Request(os.environ.get("TI_BASE", c["base"]).rstrip("/")
                                 + "/v1/messages", data=json.dumps(body).encode(),
                                 headers=h, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return "".join(b.get("text", "") for b in d.get("content", [])
                   if b.get("type") == "text")


def complete(prompt, max_tokens=900, temperature=0.85, retries=3, timeout=600, seed=None):
    """返回纯文本。空输出、被截断、只剩思考，都会重试。"""
    fn = _local if BACKEND == "local" else _anthropic
    last = None
    for a in range(1, retries + 1):
        try:
            t = (fn(prompt, max_tokens, temperature, timeout, seed)
                 if BACKEND == "local" else fn(prompt, max_tokens, temperature, timeout))
            t = _THINK.sub("", t).strip()
            if t:
                return t
            last = "empty_text"
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {str(e)[:160]}"
            if "thinking_not_disabled" in last or "truncated" in last:
                time.sleep(1)
            else:
                time.sleep(min(10, 2 * a))
    raise RuntimeError(last)


def complete_json(prompt, **kw):
    """要求模型只吐 JSON 的场景：抽出最外层对象并解析。"""
    t = complete(prompt, **kw)
    i, j = t.find("{"), t.rfind("}")
    if i < 0 or j < 0:
        raise ValueError(f"no json in output: {t[:200]!r}")
    return json.loads(t[i:j + 1])


if __name__ == "__main__":
    import sys
    print(f"backend={BACKEND} base={BASE} model={MODEL}")
    p = sys.argv[1] if len(sys.argv) > 1 else "Reply with the single word: ready"
    t0 = time.time()
    print(repr(complete(p, max_tokens=64, temperature=0.0)), f"{time.time() - t0:.1f}s")
