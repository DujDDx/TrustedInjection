"""Semantic review of an event table with DeepSeek: what validate_events.py cannot judge.

    python3 review_ds.py ../events_p4v2.jsonl review_p4v2.jsonl

One call per event, temperature 0. Findings are leads for a human check, not verdicts.
"""
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_events_ds import chat  # noqa: E402

PROMPT = """You review one event of a dataset of invented news stories. Each fact is a sentence with <Tag> placeholders; each tag has a truth value and a rumor value. Facts f2..fn are shuffled and any of them may be left out of a document; only f1 is always present.

Report ONLY real problems of these kinds:
- depends: a fact among f2..fn only makes sense if another fact other than f1 is present (it refers to something introduced only in that other fact). Referring back to f1's subject is fine.
- rumor_type: a rumor value is not the same kind of thing as its truth (a town replaced by a country, a person by an organisation, a month by a season, a different unit).
- grammar: substituting the truth or the rumor value into the fact gives an ungrammatical or nonsensical sentence (doubled words, wrong number agreement, wrong preposition).
- seed: a fact states something the seed_text does not say, or contradicts it.
- category: a tag's cat is wrong (date, place, agent = person or organisation, quantity, name = named thing).

Return JSON {"issues": [{"kind": ..., "where": "f3 or <Tag>", "detail": short}]}; return {"issues": []} if there is nothing.

EVENT:
"""


def review(e):
    out, _ = chat([{"role": "user", "content": PROMPT + json.dumps(e, ensure_ascii=False)}],
                  temperature=0.0, max_tokens=800)
    try:
        return {"event_id": e["event_id"], "issues": json.loads(out).get("issues", [])}
    except json.JSONDecodeError:
        return {"event_id": e["event_id"], "issues": [{"kind": "unparsed", "detail": out[:200]}]}


def main():
    rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
    with ThreadPoolExecutor(8) as ex, open(sys.argv[2], "w") as fo:
        for r in ex.map(review, rows):
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
