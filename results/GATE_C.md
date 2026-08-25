# Gate C — validity checklist (wave v4)

Evaluated 2026-08-15 against `handoff/goal1.md` §C, updated 2026-08-18 after the
PPO seed extension. Gate C is stated there as non-negotiable, so each row below
records what was actually verified, not what was intended. Two rows still do not
pass in full: **C2 partially**, and **C4 per family** — PASS for the PPO family
at 20 seeds, FAIL for the DQN family at 5.

> **Line pointers corrected 2026-08-23, no verdict moved.** Six `path:line`
> citations in the Evidence column had drifted 1-5 lines as the files were edited
> after this table was written: `train_baselines.py:81,154` -> `:84,159,163`,
> `train_proposed.py:77,143` -> `:75,142`, `run_wave.py:59` -> `:60`,
> `evaluate_checkpoints.py:33` -> `:40`. Every fact they support was re-read at the
> corrected line and still holds, so C2 stays PARTIAL and C5 stays PASS. Run
> `python scripts/citation_audit.py` to re-check; it exits non-zero on drift.

| # | Criterion | Verdict | Evidence |
|---|---|---|---|
| C1 | Treatment-identity test PASS, >=3 seeds x all floor modes | **PASS** | `scripts/test_treatment_identity.py`, 9/9 runs (seeds 42/43/44 x floor `none`/`static`/`dynamic`, 200 steps each): `(f_min, lam, delta, violation_rate, reward)` bit-identical between two independent env instances fed the same action sequence |
| C2 | Zero per-algorithm hyperparameters; all from one `configs/experiment_config.yaml` | **PARTIAL** | Binding half holds: agents are constructed with no keyword overrides (`training/train_baselines.py:84,159,163`, `training/train_proposed.py:75,142`), and `scripts/run_wave.py:63` passes an identical `common` argument list to every job. Only two things differ by algorithm, both structural rather than tuned: the backbone flag (`gat`/`sage` — that *is* the treatment) and the per-family step budget (`DQN_STEPS=200_000`, `PPO_STEPS=1_000_000`, which C3 then keeps separate). **Deviation (as the wave ran):** optimiser hyperparameters (`lr`, `gamma`, `clip_eps`) were Python defaults in `agents/*.py`, not entries in the YAML, so the single-source clause was met in substance but not literally. **Remediated 2026-08-16**, after the wave: `configs/experiment_config.yaml` gained an `agent:` block and the constructors now resolve through `agents/hparams.py`. Every value is bit-identical to the default it replaced (`tests/test_hparams_identity.py`), and one held-out eval episode reproduces the pre-change row bit-for-bit, so the v4 checkpoints stay reproducible from the committed config. The verdict for the wave as executed stays PARTIAL — the binaries that produced these checkpoints did not read the YAML — and upgrading it is a human call, not this file's |
| C3 | DQN (200K) and PPO (1M) never pooled in any statistical claim | **PASS, with one flagged case** | `scripts/rliable_report.py:34` pairs each proposed model only with baselines of its own family; no script computes a cross-family aggregate. Flagged: Gate B's own ranges (B1-B4) are defined in `goal1.md` as spanning all 8 algorithms, so the pre-registered gate numbers do cross families. They are a property of the task, not a claim that one algorithm beats another, and the split below shows the verdicts do not depend on the pooling |
| C4 | Seed >= 20 for the bimodal cell-edge KPI; collapse rate as binomial proportion + Wilson CI | **PASS (PPO family) / FAIL (DQN family)** | Second clause met throughout: `results/STABILITY_v4_primary.md` reports collapse rate per algorithm with Wilson 95% CI. First clause met **only for the PPO family**, extended to 20 seeds on 2026-08-18 (42-61) under a rule declared on family cost before running. The DQN family still has 5 seeds, so C4 fails there and no characterisation claim is made for it. Reported per family rather than as a single global verdict, because that is what the data supports |
| C5 | Held-out evaluation (seed >= 10000) fully disjoint from training seeds | **PASS** | `EVAL_SEED_BASE = 10_000` (`scripts/evaluate_checkpoints.py:40`); training seeds are 42-46 |
| C6 | Operating point frozen and committed **before** the full wave | **PASS** | The operating-point keys were last committed in `aad4198`, 2026-08-06 12:54 +0700; first v4 checkpoint written 2026-08-09 07:57, and none of those keys has changed since. Operating point: `delta=0.085`, `lambda_arrival=60000`, `buffer.urllc_max_bits=307200`, `dual_update_every=12500`, `floor.mode=none`. Evidence was originally stated as "`git diff HEAD -- configs/experiment_config.yaml` is empty"; that diff is no longer empty because the C2 remediation of 2026-08-16 **added** an `agent:` block to the same file, months after the wave. The check is therefore per-key, not per-file: `git diff aad4198 HEAD -- configs/experiment_config.yaml` touches only the added `agent:` block |

## C3 supplement — Gate B ranges within each budget family

Not a re-run of the gate. The pre-registered Gate B verdict stands exactly as
computed across all 8 algorithms (B1 16.33%, B2 11.98 pp, B3 1.01 ms, B4 0/5).
This split exists only to answer the C3 question "does any B verdict depend on
mixing the two budgets?" — it does not.

| set | B1 relative range | B2 range | B3 range |
|---|---|---|---|
| all 8 (pre-registered, **this is the gate**) | 16.33% | 11.98 pp | 1.01 ms |
| DQN family (4, 200K steps) | 8.95% | 6.30 pp | 0.19 ms |
| PPO family (4, 1M steps) | 7.27% | 5.98 pp | 1.01 ms |

B1 (>= 5%) and B2 (>= 5 pp) pass in every set. B3 (>= 2 ms) fails in every set.
No verdict flips under the split, so the cross-family definition of the B ranges
is not doing any work in the outcome.

> **Updated 2026-08-17.** These numbers previously read B1 11.43%, B2 8.70 pp,
> B4 1/5 — computed while the DQN family's non-greedy readout was epsilon=1.0
> random actions. They are now the per-family primary readout, copied from
> `results/GATE_B_v4_primary.md`, which generates this split itself so a
> hand-maintained copy cannot go stale again. No B verdict changed. B1 and B2
> strengthened, and B4 improved from 1/5 to 0/5 because `embb_p5_mbps` is no
> longer saturated at 0.0000 across five algorithms — that saturation was itself
> partly an artefact of the random actions.

## What C4 blocks

C4 lands on exactly the metric the scoping decision of 2026-08-15 promoted into
the claim set: `collapse_rate`. It is now satisfied for one family and not the
other, so it is reported per family.

Measured (`results/STABILITY_v4_primary.md`, per-family primary readout):

| family | algorithm | collapsed | Wilson 95% |
|---|---|---|---|
| PPO, n=20 | `central-ppo` | 3/20 | [0.05, 0.36] |
| PPO, n=20 | `gnn-mappo_gat` | 14/20 | [0.48, 0.85] |
| PPO, n=20 | `gnn-mappo_sage` | 19/20 | [0.76, 0.99] |
| PPO, n=20 | `ippo` | 20/20 | [0.84, 1.00] |
| PPO, n=5 (post-hoc) | `mlp-knn-ppo` | 5/5 | [0.57, 1.00] |
| DQN, n=5 | `central-dqn` | 0/5 | [0.00, 0.43] |
| DQN, n=5 | `gnn-madqn_gat` | 2/5 | [0.12, 0.77] |
| DQN, n=5 | `gnn-madqn_sage` | 0/5 | [0.00, 0.43] |
| DQN, n=5 | `idqn` | 2/5 | [0.12, 0.77] |

**In the PPO family** `central-ppo` is Wilson-disjoint from all three per-agent
variants, so the comparative claim holds — and at n=20 it rests on a sample that
can carry it. **In the DQN family** the intervals overlap (`[0.00, 0.43]` against
`[0.12, 0.77]`), so no separation survives inside that family, and
`gnn-madqn_sage` at 0/5 is a direct counter-example to any universal reading.

> **Two corrections, 2026-08-18, both against the tidier story.** (1) This section
> previously reported `central-ppo` at **0/5** and called it the one architecture
> that never collapses. At n=20 it collapses in **3 of 20 seeds**: the zero was a
> small-sample artefact, and the correct statement is "collapses far less often",
> not "never". (2) The DQN rows previously came from the epsilon=1.0 readout and
> read 5/5 across the family (corrected 2026-08-17).

C4's >= 20 requirement was pre-registered to resolve *within*-algorithm
bimodality. The PPO family now meets it, and the first thing it resolved was one
of our own numbers. The DQN family does not meet it, so the gate stays FAILED
there and no characterisation claim is made for it.

## Human decision 2026-08-16 — claim split, threshold untouched

The threshold was never amended. What the human decision changed was claim scope,
recorded in `handoff/goal1.md` §"Keputusan scoping C4 2026-08-16"; the seed
extension of 2026-08-18 then satisfied the threshold itself for one family.

- **Comparative claim — kept, PPO family only.** "Proposed models collapse more often
  than centralised baselines." Needs separation, not a precise rate. The Wilson
  intervals are disjoint inside the PPO family at n=20; in the DQN family every
  interval overlaps at n=5, so the claim is not made there at all. Note also that
  within the PPO family the line runs between centralised and per-agent, not between
  GNN and baseline: `ippo` collapses in 20 of 20 seeds, more often than either GNN
  variant.
- **Characterisation claim — now permitted for the PPO family only.** With n=20 the
  exact rates may be stated (`central-ppo` 3 of 20, `gnn-mappo_gat` 14 of 20,
  `gnn-mappo_sage` 19 of 20, `ippo` 20 of 20). For the DQN family, and for the
  post-hoc `mlp-knn-ppo` baseline, no characterisation claim is made — they are still
  at 5 seeds.
- **Wording rule, binding on every report and on the paper:** write "collapsed in k of
  N seeds", never "collapses X% of the time". At n=5 only six rates are attainable
  (0, 0.2, 0.4, 0.6, 0.8, 1.0); a percentage implies a continuous precision the data
  does not have. This still applies at n=20, where the attainable grid is finer but
  no less discrete.

A seed extension was costed against the wave's own `elapsed_sec` before being
considered, under a rule fixed on family cost rather than on results: extend the
cheaper family. Measured, that is **PPO** (8.32 job-hours per seed for its four
algorithms, against 77.57 for DQN) — the opposite of what the 200K-vs-1M step budgets
suggest, because DQN learns on every step while PPO updates once per 512. 15 extra
seeds would cost 124.8 job-hours (PPO) or 1163.6 (DQN) against 20.5 remaining under
the +-450 job-hour budget, so under that reading the extension does not fit.

**Budget accounting fixed 2026-08-16** (goal1.md, "Kebijakan akuntansi anggaran"): the
+-450 GPU-hours are read as **device-hours**, because summing job-hours double-counts
one GPU running six jobs in parallel. Under that policy the wave cost 147.22 of 450 and
the PPO extension (~43 device-hours) **fits**. It is therefore deferred rather than
ruled out — the human ordered zero-shot first on value per GPU-hour. C4 stays FAILED
either way; nothing about the gate depends on the accounting.
