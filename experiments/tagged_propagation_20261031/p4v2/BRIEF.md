# Brief: writing P4 events (tagged propagation dataset, plan v1.2)

You write one batch of **50 events** for a training dataset about news copying.
Your batch parameters (domain, ID range, n split) are in the message that sent you here.

## What an event is and how it is used (read this, the rules follow from it)

An event is a short invented news story broken into n facts (n = 5 or 6). Each fact is one
sentence in which every name, place, date and number is replaced by a tag such as `<City>`.
Every tag has a **truth** value and a **rumor** (wrong) value.

A local model later writes four short reports per event from different subsets of the facts.
It sees only the tagged facts, never the values. The facts are **shuffled**, and any of
f2..fn may be **left out** of a given report. f1 is always included. Values are filled in
afterwards. In one variant one report copies another but swaps one fact's truth values for
its rumor values. A classifier must then tell **from wording alone** which report copied
which. So:

- Nothing about a value may reveal whether it is the truth or the rumor. If rumors are
  usually smaller, rounder, earlier, longer or more exotic than truths, the dataset is broken.
  **For numbers, years, months and dates, make the rumor larger than the truth about half the
  time and smaller about half the time**, varying the gap (roughly 15-60%). The validator
  fails a batch outside 40-60%.
- Truth and rumor have exactly the same form: same unit, same notation, same kind of thing.
  `14m pounds` / `21m pounds`, never `1.2m pounds` / `900,000 pounds`. `12 May` / `3 June`.
  A town for a town, a person for a person, a hospital for a hospital.
- f1 names the subject of the story (the company, team, body, event…). Other facts may refer
  back to it ("the company", "the club", "it"), but **never to anything introduced in f2..fn**,
  and they must not use linking words (also, this, these, then, however, another…). Each of
  f2..fn must read correctly with only f1 present.

## Hard rules (the validator enforces all of these)

Grid and tags
- `n` = 5 or 6 facts, `m` = 1 always. Your n split is given in your message.
- Fact ids `f1`..`fn`. Each fact: one sentence, 7-24 words, ends with a full stop, starts
  with a capital letter or a tag, contains **1-3 tags**, no tag repeated, **no digits** outside
  tags. At least two facts carry 2+ tags. **No proper name outside a tag**: after the first
  word, no capitalised word may appear in a fact ("in Japan" must be "in <Country>", or
  dropped).
- 6-11 tags per event; each tag appears in exactly one fact; every defined tag is used.
- Tag names: CamelCase/underscore, no spaces, e.g. `<Person_A>`, `<Attendees>`, `<Month_B>`.
  A year value goes in a tag whose name contains `Year`; a month value in a tag whose
  name contains `Month`.
- Categories `cat`: `date`, `place`, `agent` (person or organisation), `quantity`, `name`
  (named thing: a product, event, scheme, prize, ship…). Every event has at least one
  date, place, agent and quantity tag. f1 has at least one agent/name/place tag.

Values
- Proper nouns or numbers with units, e.g. `4,200`, `3.5 percent`, `18 days`, `14m pounds`,
  `2,400 kronur`, `October`, `12 May`, `2027`. Never start with an article (`the`/`a`/`an`).
  No full stop except a decimal point; no `? ! ; | < > "`; at most 40 characters.
- Within an event no two values are equal (case-insensitive), across all truths and rumors.
- Avoid a value being part of another value in the same event ("Leeds" + "Leeds Council").

Seed text
- `seed_text` is the source story written as **a real short news report** (35-130 words,
  newspaper register, varied sentences). It contains **every truth value verbatim** and
  **no rumor value**. It is archived for traceability and is not fed to any model.
  Write the seed first, then extract the facts from it; that keeps the facts natural.

Tense and time
- `tense` is `future` (the story is about something announced or planned) or `past` (it has
  happened). The story is set in the present, 2026. Truth dates must make sense for the
  tense ("will open in 2027"; "closed in 1998"). Rumor dates follow the direction-balance
  rule and need not be plausible.
- Your batch should have 40-60% future events.

Fields, one JSON object per line:
`event_id, domain, topic, tense, n, m, seed_text, facts[{id, text}], tags{<Tag>: {cat, truth, rumor}}`
`domain` is your batch's domain key; `topic` is a short snake_case subtopic of your choice.

Across your batch (validator file-level checks)
- No two facts in your batch with the same sentence frame (the fact text with tags removed,
  lowercased). Vary the structure, not just the nouns.
- No agent/name truth value in more than 2 events; no place truth value more than 4 times.
- No two events whose f1 subject is the same entity.

## Quality the validator cannot see (I will read samples and send failures back)

- Each event is a distinct, believable news story. Spread across **at least 30 different
  subtopics**, none used more than 3 times. Use many countries and regions: no single country
  in more than 30% of your events. Avoid the UK water/rail cluster that ruined the last table.
- People: invented names, diverse in origin, no real living persons. Organisations and
  products: mostly invented. Real public bodies, universities and cities are fine as values.
- Facts are concrete and newsworthy (who / what / where / when / how much). No filler like
  "The event is important."
- Values fit their sentence once substituted, for both truth and rumor (grammar, number
  agreement: "<Patients> patients" with `2,300`, not with `one`). An "a"/"an" right before
  a tag is corrected automatically at fill time.
- Quantities carry their unit inside the value when the sentence needs one
  (`14m pounds`, not `14m` followed by "pounds" in the fact); counts can be bare (`2,300`).

## Example (passes the validator)

See `example_event.json` in this directory.

## Workflow

1. Work in `/mnt/hdd4/TrustedInjection/experiments/tagged_propagation_20261031/p4v2/`.
2. Write your events to **your batch file only** (name in your message), one JSON per line.
   Writing with a small Python script that builds the dicts and dumps them is fine and
   avoids JSON escaping mistakes. Do not edit any other file, do not touch the validator, do
   not run anything that calls the local model server, do not commit.
3. Run `python3 ../validate_events.py <your_batch_file>` and fix everything until it prints
   `failing 0  file-level problems 0` and exits 0. If you believe a check is wrong, say so
   in your report rather than working around it.
4. Re-read your batch once against "Quality the validator cannot see" and fix what you find.
5. Report: validator summary lines, subtopic and country counts, the direction-balance lines,
   and anything you were unsure about. Keep it short.
