# D2c -- gradient-norm ratio into the GNN

Seeds: [42, 43, 44, 45, 46]. Updates recorded per checkpoint: 15 (PPO), 200 (DQN). Total updates: 2160.

**What this measures, and why it is not a repeat of D2a/D2b.** D2a and D2b test the consequence -- the policy's output barely depends on neighbour messages. This tests the mechanism PLAN-01 D2 actually states: whether gradient reaches the GNN at all. It is what separates PLAN-03 section 5 (residual/JK, aimed at over-smoothing) from PLAN-04 (auxiliary loss, aimed at an untrained GNN path); PLAN-04 Larangan 4 forbids running both at once.

**How it was measured.** The real training loop runs, resumed from the v4 checkpoint onto a scratch copy. `PPOAgent.learn` / `DQNAgent.learn` are wrapped at class level; the original runs first and `p.grad` is read after it returns. `optimizer.step()` does not clear `.grad` and `zero_grad()` only fires at the start of the next update, so these are the gradients that were actually applied. The loss path is untouched -- PLAN-01's correction block requires exactly this and rules out a rollout with a re-written loss.

**Two ratios, because one would be misread.** `ratio_l2` is what PLAN-01 D2c asks for. `ratio_rms` divides each group's norm by the square root of its parameter count first. The groups are not the same size -- `gnn-*_sage` has 9,344 backbone parameters against 18,188 in the head (`results/DIAG_EQUIVARIANCE.md`) -- so a `ratio_l2` below 1 can mean *fewer parameters* rather than *smaller gradients*. A verdict of "the GNN path is untrained" is only safe when both agree.

`clip_grad_norm_` rescales every gradient by one common factor, which cancels in both ratios, so clipping does not affect these numbers.

**Limitation, DQN only.** `_maybe_resume` restores model, optimizer, RNG and CMDP state but **not** the `ReplayBuffer`, so the DQN runs refill `replay_start` steps with the converged policy before learning resumes. The DQN ratio therefore describes gradient flow at convergence on near-on-policy data, not the historical training mixture. PPO is unaffected: it is on-policy, so a fresh `RolloutBuffer` is faithful.

Median and IQR across updates, not a single number -- one update can be a fluke.

| algo | backbone | seed | n updates | ratio_l2 median [IQR] | ratio_rms median [IQR] |
|---|---|---|---|---|---|
| `gnn-madqn_gat` | gat | 42 | 201 | 2.2020 [1.6538, 2.8963] | 2.1325 [1.6016, 2.8050] |
| `gnn-madqn_gat` | gat | 43 | 201 | 2.3167 [1.6684, 2.8241] | 2.2437 [1.6158, 2.7351] |
| `gnn-madqn_gat` | gat | 44 | 201 | 1.9086 [1.3820, 2.5403] | 1.8484 [1.3384, 2.4602] |
| `gnn-madqn_gat` | gat | 45 | 201 | 4.5217 [3.5436, 5.8966] | 4.3790 [3.4318, 5.7106] |
| `gnn-madqn_gat` | gat | 46 | 201 | 1.4401 [1.1178, 2.1993] | 1.3947 [1.0826, 2.1299] |
| `gnn-madqn_sage` | sage | 42 | 201 | 0.4880 [0.3328, 0.7415] | 0.6808 [0.4643, 1.0345] |
| `gnn-madqn_sage` | sage | 43 | 201 | 0.4177 [0.2930, 0.5824] | 0.5828 [0.4087, 0.8126] |
| `gnn-madqn_sage` | sage | 44 | 201 | 0.4817 [0.3353, 0.6224] | 0.6720 [0.4679, 0.8684] |
| `gnn-madqn_sage` | sage | 45 | 201 | 0.4602 [0.3028, 0.7824] | 0.6421 [0.4225, 1.0916] |
| `gnn-madqn_sage` | sage | 46 | 201 | 0.0704 [0.0622, 0.0757] | 0.0982 [0.0868, 0.1056] |
| `gnn-mappo_gat` | gat | 42 | 15 | 3.9814 [1.7619, 15.2728] | 3.8558 [1.7063, 14.7911] |
| `gnn-mappo_gat` | gat | 43 | 15 | 6.0355 [4.3636, 10.8761] | 5.8451 [4.2260, 10.5331] |
| `gnn-mappo_gat` | gat | 44 | 15 | 0.0076 [0.0064, 0.0096] | 0.0074 [0.0062, 0.0093] |
| `gnn-mappo_gat` | gat | 45 | 15 | 0.0024 [0.0018, 0.0028] | 0.0023 [0.0018, 0.0027] |
| `gnn-mappo_gat` | gat | 46 | 15 | 5.9034 [2.9864, 11.0819] | 5.7172 [2.8922, 10.7324] |
| `gnn-mappo_sage` | sage | 42 | 15 | 1.9243 [1.0030, 4.0309] | 2.6847 [1.3994, 5.6237] |
| `gnn-mappo_sage` | sage | 43 | 15 | 4.8913 [1.9536, 8.0538] | 6.8242 [2.7256, 11.2364] |
| `gnn-mappo_sage` | sage | 44 | 15 | 8.5672 [5.5583, 14.4443] | 11.9527 [7.7547, 20.1522] |
| `gnn-mappo_sage` | sage | 45 | 15 | 1.4502 [0.1673, 2.4381] | 2.0233 [0.2334, 3.4016] |
| `gnn-mappo_sage` | sage | 46 | 15 | 5.5967 [2.9471, 6.0900] | 7.8083 [4.1117, 8.4966] |

## Per variant

| variant | ratio_l2 median | ratio_rms median | n updates |
|---|---|---|---|
| `gnn-madqn_gat` | 2.3055 | 2.2328 | 1005 |
| `gnn-madqn_sage` | 0.3782 | 0.5277 | 1005 |
| `gnn-mappo_gat` | 2.0793 | 2.0137 | 75 |
| `gnn-mappo_sage` | 3.5003 | 4.8835 | 75 |
