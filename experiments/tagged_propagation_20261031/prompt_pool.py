"""Writing-prompt pool for the documents (user 11-02).

A and C stand for two reporters who observed the same facts and wrote independently, so
they should not share one house style. Every document (A, C, B, Bf) draws its own prompt,
the four documents of one (event, branch) never share one, and prompt choice is independent
of role, so "different style" is not a role cue.

Only the opening style instruction varies. The rest of gen_docs.PROMPT - tag rules,
"write in your own words", MATERIAL block, "Return only the report" - is kept verbatim in
every prompt, and the material is exactly what gen_docs.py gives each role.

TRAIN prompts are used for training data; HELDOUT prompts appear only in test documents,
to show the model does not rely on the styles it was trained on.
"""
import hashlib
import random

import gen_docs as G

_HEAD = "You are a news reporter. Write a short news report based only on the material below."
assert G.PROMPT.startswith(_HEAD)
_RULES = G.PROMPT[len(_HEAD):]          # everything after the opening line, unchanged

STYLES = {
    # ---- train
    "p00": "You are a news reporter. Write a short news report based only on the material below.",
    "p01": "You are a wire-service reporter. Write a brief, factual news item for the wire based only on the material below. Put the most important fact first.",
    "p02": "You are a reporter for a regional daily newspaper. Write a short news story for local readers based only on the material below, in a plain, readable style.",
    "p03": "You are a business journalist. Write a short news report based only on the material below, in the concise, matter-of-fact tone of a financial news desk.",
    "p04": "You write news briefs for a morning newsletter. Write a compact news item based only on the material below; keep each sentence short and direct.",
    "p05": "You are a radio newsreader preparing a script. Write a short news report based only on the material below, in clear sentences that are easy to read aloud.",
    "p06": "You are a reporter for a national broadsheet. Write a short news report based only on the material below, in a measured, formal register.",
    "p07": "You are an online news editor. Write a short web news story based only on the material below, opening with the development readers will care about most.",
    "p08": "You are a correspondent filing a short dispatch. Write a news report based only on the material below, telling what happened in the order it happened.",
    "p09": "You are a reporter for a trade publication. Write a short news report based only on the material below for readers who follow this sector closely.",
    "p10": "You are a local television news writer. Write a short news report based only on the material below, in a conversational but factual tone.",
    "p11": "You are a news agency reporter. Write a short, neutral news report based only on the material below, attributing claims to their sources where the material does.",
    # ---- held out: test documents only
    "p12": "You are a reporter for a weekly community paper. Write a short news story based only on the material below, in a warm but factual style.",
    "p13": "You are an explanatory journalist. Write a short news report based only on the material below that makes clear why the development matters.",
    "p14": "You are a sub-editor condensing a story for a print column. Write a tight news report based only on the material below, with no wasted words.",
    "p15": "You are a foreign desk reporter writing for an international audience. Write a short news report based only on the material below, giving the location and context early.",
}
TRAIN = [f"p{i:02d}" for i in range(12)]
HELDOUT = ["p12", "p13", "p14", "p15"]
ROLES = ("A", "C", "B", "Bf")


def prompt(pid):
    return STYLES[pid] + _RULES


def assign(event_id, branch, pool):
    """Four distinct prompts for (A, C, B, Bf), drawn from pool by a fixed seed."""
    seed = int(hashlib.md5(f"{event_id}-{branch}-prompts".encode()).hexdigest(), 16)
    return dict(zip(ROLES, random.Random(seed).sample(pool, 4)))


assert prompt("p00") == G.PROMPT
