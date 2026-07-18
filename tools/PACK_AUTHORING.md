# Offline Pack Authoring

The runtime never calls a language model. A model may be used as an offline
author, followed by deterministic compilation and independent review.

## Authoring Prompt

Give the model this contract before asking for content:

```text
Create a JSON dialogue pack for a deterministic runtime. Do not write code.

The pack contains ordinary conversational arcs. Each arc has the same number
of alternating Alice/Bob turns. Each turn has a template with exactly four
named clause groups. Each clause group must contain exactly eight alternatives.
Every alternative in one group must preserve the same proposition and must be
grammatically usable in the fixed template. Any combination of one alternative
from each group must remain coherent.

Use mundane subjects such as food, errands, work, films, weekend plans, and
weather. Keep facts compatible across turns. Do not use metadata, labels,
encoded-looking phrases, unusual punctuation, numbers, or prompt language.
Avoid relying on grammatical agreement between separate groups. Prefer whole
sentences or complete clauses over individual words.

Return vocabulary groups and turn templates in the project schema. Do not
include duplicate choices or choices where one is a prefix of another.
```

## Review Pipeline

1. Ask one model to author the candidate groups.
2. Ask a different model to score every group for semantic equivalence.
3. Generate random cross-products of each turn and reject awkward outputs.
4. Read complete random transcripts, not just individual turns.
5. Compare style statistics against the intended user's ordinary messages.
6. Compile with `python -m pcc inspect --pack PACK.json`.
7. Freeze the resulting pack and distribute its exact content identifier.

The compiler is intentionally strict. If a generated pack needs runtime
special cases, reject it and revise the pack instead.
