# Which document is authoritative for what

Documentation in this repo sits in six places. That is deliberate -- working rules,
done criteria, generated results, and project history have different lifetimes and
different owners -- but nothing until now said which one wins when two disagree. This
page says it.

**The rule when two documents disagree: the generated file wins.** Anything under
`results/` is regenerated from data by a script. Prose that quotes a number is a copy,
and copies go stale. Never retype a result from memory (`docs/HANDOVER.md` §11).

| Location | Contains | Authoritative for |
|---|---|---|
| `docs/HANDOVER.md` | working rules, §11 integrity rules | how to work in this repo |
| `handoff/goal1.md` | gate definitions, done criteria, frozen decisions | whether the work is finished |
| `handoff/paper_structure.md` | the three-level thesis | which claims may be written |
| `results/*.md` | generated reports | every number |
| `docs/journey/` | v1 -> v2 -> v3 history | why the design looks like this |
| `Plan_escalation_loop.md` (root) | loop protocol, safety rules | what an agent may do |
| `prompts/` | escalation-loop prompts | -- |
| `docs/archive/` | superseded documents | nothing; kept for the trail |

`STATE.json` is loop state, not documentation: iteration counter, last green commit,
and a `catatan` field summarising where things stand. Its `status` and
`menunggu_manusia` fields are a human decision and are not moved by an agent.

## Start here

- **New to the repo:** `README.md`, then `handoff/goal1.md`.
- **Continuing the work:** `docs/HANDOVER.md`, then `STATE.json`.
- **Reading the results:** `results/RLIABLE_v4_primary.md` and
  `results/STABILITY_v4_primary.md`. Both state their readout in the header; the
  primary readout differs by family (sampled for PPO, argmax for DQN) and mixing them
  is the single easiest way to misread this project.
- **Writing the paper:** `handoff/paper_structure.md`, `results/GATE_C.md`.

## Line numbers in documents

Several documents cite source locations as `path/to/file.py:123`. Two categories, and
they are treated differently:

- **Live evidence** -- `results/GATE_C.md`, `results/B3_DELAY_CENSORING.md`. These back
  a current verdict, so their pointers are kept accurate and guarded by
  `python scripts/citation_audit.py`, which fails when the text at a cited line
  changes.
- **Snapshots** -- `docs/archive/`, `docs/journey/`, `docs/rev2-implementation-plan.md`,
  `results/v1_uncoupled/`, `results/v2_scalarized/`, and the `runs/` ledger, which is
  append-only. Their pointers were correct on the day they were written and are
  **not** re-pointed at current code; doing so would make an old record appear to
  describe today's behaviour. `citation_audit.py` skips them.
