# D5 -- collision-storm hypothesis

Checkpoints: `results/logs/gnn-mappo_gat_v4_seed*.pt,results/logs/ippo_v4_seed*.pt`. Episodes per arm: 10, seeds from `EVAL_SEED_BASE = 10000`.

**This is a hypothesis under test, not an explanation.** PLAN-01 §Larangan 3 bars writing the collision-storm account into the paper before D5 confirms it, and records a limit already known: policy entropy spans only 0.100 nat while the readout gap is 46.28 Mbps, so whatever this is, it is not a simple function of entropy. If D5 does not confirm, the *operational validity limit* framing (PLAN-07 §4) still stands without it.

**Both readouts appear here on purpose, and neither is the primary one.** D5 is about the difference between them: the greedy-vs-sampled contrast is the measurement. These are not KPI readings and must not be compared against `results/RLIABLE_*.md`.

**What each column means.** `mode_share` is the mean fraction of the 5 gNB picking the same tier at a step (1.0 = unanimous); the hypothesis predicts it rising sharply under argmax for the collapsing model. `action_spearman` is the mean pairwise rank correlation between the five action series. `sinr_corr` is the mean pairwise Pearson correlation of the per-gNB log-SINR series -- a collision storm drops every gNB together, so it predicts high positive correlation, whereas independent per-cell degradation does not. Constant series are skipped rather than scored 0: a gNB pinned at one action carries no evidence either way.

**`action_spearman` / `sinr_corr` read `--` when every gNB held one action for the whole episode.** Rank correlation is undefined on constant series, and that case is the *extreme* of the hypothesis rather than missing data -- a bare number there would be invented. `all const` marks it, and `mode_share` / `unanimous frac` carry the full answer for those rows.

`embb_p5_mbps` is carried along only to show which arm actually collapsed.

| algo | seed | arm | mode_share | unanimous frac | all const | action_spearman | sinr_corr | embb_p5 |
|---|---|---|---|---|---|---|---|---|
| `gnn-mappo_gat` | 42 | greedy | 0.9993 | 0.9980 | 0.00 | 0.9976 | 0.9999 | 0.558158 |
| `gnn-mappo_gat` | 42 | sampled | 0.3547 | 0.0000 | 0.00 | -0.0014 | 0.3165 | 0.000005 |
| `gnn-mappo_gat` | 43 | greedy | 0.9983 | 0.9955 | 0.00 | 0.9814 | 0.9514 | 1.912687 |
| `gnn-mappo_gat` | 43 | sampled | 0.3574 | 0.0000 | 0.00 | -0.0062 | 0.3152 | 0.000006 |
| `gnn-mappo_gat` | 44 | greedy | 1.0000 | 1.0000 | 1.00 | -- | -- | 1.912687 |
| `gnn-mappo_gat` | 44 | sampled | 0.4377 | 0.0020 | 0.00 | -0.0038 | 0.3050 | 1.573219 |
| `gnn-mappo_gat` | 45 | greedy | 1.0000 | 1.0000 | 1.00 | -- | -- | 1.912687 |
| `gnn-mappo_gat` | 45 | sampled | 0.4488 | 0.0035 | 0.00 | 0.0054 | 0.3058 | 0.000004 |
| `gnn-mappo_gat` | 46 | greedy | 0.9999 | 0.9995 | 0.00 | 0.9994 | 0.9999 | 1.716552 |
| `gnn-mappo_gat` | 46 | sampled | 0.3616 | 0.0005 | 0.00 | -0.0090 | 0.2976 | 0.000006 |
| `gnn-mappo_gat` | 47 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 1.885919 |
| `gnn-mappo_gat` | 47 | sampled | 0.4378 | 0.0045 | 0.00 | -0.0117 | 0.3196 | 0.000007 |
| `gnn-mappo_gat` | 48 | greedy | 1.0000 | 1.0000 | 1.00 | -- | -- | 1.912687 |
| `gnn-mappo_gat` | 48 | sampled | 0.3989 | 0.0010 | 0.00 | -0.0048 | 0.3244 | 0.000005 |
| `gnn-mappo_gat` | 49 | greedy | 0.9987 | 0.9965 | 0.30 | 0.9949 | 0.9994 | 1.896799 |
| `gnn-mappo_gat` | 49 | sampled | 0.3610 | 0.0000 | 0.00 | -0.0135 | 0.3067 | 0.000006 |
| `gnn-mappo_gat` | 50 | greedy | 1.0000 | 1.0000 | 1.00 | -- | -- | 1.792554 |
| `gnn-mappo_gat` | 50 | sampled | 0.4670 | 0.0045 | 0.00 | -0.0040 | 0.3293 | 1.448006 |
| `gnn-mappo_gat` | 51 | greedy | 1.0000 | 1.0000 | 0.20 | 1.0000 | 1.0000 | 1.898589 |
| `gnn-mappo_gat` | 51 | sampled | 0.3993 | 0.0020 | 0.00 | -0.0030 | 0.3260 | 0.000004 |
| `gnn-mappo_gat` | 52 | greedy | 1.0000 | 1.0000 | 1.00 | -- | -- | 1.912687 |
| `gnn-mappo_gat` | 52 | sampled | 0.3768 | 0.0005 | 0.00 | -0.0038 | 0.3264 | 0.000005 |
| `gnn-mappo_gat` | 53 | greedy | 0.9997 | 0.9990 | 0.00 | 0.9982 | 0.9973 | 1.788329 |
| `gnn-mappo_gat` | 53 | sampled | 0.3679 | 0.0005 | 0.00 | -0.0035 | 0.3226 | 0.000007 |
| `gnn-mappo_gat` | 54 | greedy | 1.0000 | 1.0000 | 1.00 | -- | -- | 1.912687 |
| `gnn-mappo_gat` | 54 | sampled | 0.4230 | 0.0010 | 0.00 | -0.0079 | 0.3101 | 1.040919 |
| `gnn-mappo_gat` | 55 | greedy | 0.9980 | 0.9940 | 0.00 | 0.9875 | 0.9994 | 0.966099 |
| `gnn-mappo_gat` | 55 | sampled | 0.3574 | 0.0000 | 0.00 | -0.0191 | 0.3056 | 0.571680 |
| `gnn-mappo_gat` | 56 | greedy | 0.9977 | 0.9935 | 0.00 | 0.9873 | 0.9997 | 0.733328 |
| `gnn-mappo_gat` | 56 | sampled | 0.3549 | 0.0000 | 0.00 | -0.0107 | 0.3182 | 0.000005 |
| `gnn-mappo_gat` | 57 | greedy | 0.9989 | 0.9970 | 0.00 | 0.9938 | 0.9970 | 1.788029 |
| `gnn-mappo_gat` | 57 | sampled | 0.3607 | 0.0000 | 0.00 | -0.0016 | 0.3165 | 0.189401 |
| `gnn-mappo_gat` | 58 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 0.000038 |
| `gnn-mappo_gat` | 58 | sampled | 0.4221 | 0.0020 | 0.00 | -0.0127 | 0.4097 | 0.000003 |
| `gnn-mappo_gat` | 59 | greedy | 1.0000 | 1.0000 | 1.00 | -- | -- | 1.910268 |
| `gnn-mappo_gat` | 59 | sampled | 0.4095 | 0.0015 | 0.00 | -0.0098 | 0.3032 | 0.000007 |
| `gnn-mappo_gat` | 60 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 1.894399 |
| `gnn-mappo_gat` | 60 | sampled | 0.3915 | 0.0005 | 0.00 | -0.0016 | 0.3165 | 1.608690 |
| `gnn-mappo_gat` | 61 | greedy | 0.9978 | 0.9940 | 0.00 | 0.9893 | 0.9950 | 1.788029 |
| `gnn-mappo_gat` | 61 | sampled | 0.3630 | 0.0000 | 0.00 | -0.0073 | 0.3146 | 0.000006 |
| `ippo` | 42 | greedy | 0.6443 | 0.0515 | 0.00 | -0.0783 | 0.3140 | 0.000002 |
| `ippo` | 42 | sampled | 0.3657 | 0.0000 | 0.00 | -0.0086 | 0.3157 | 0.000003 |
| `ippo` | 43 | greedy | 0.8149 | 0.5005 | 0.00 | 0.6922 | 0.9994 | 0.000004 |
| `ippo` | 43 | sampled | 0.3725 | 0.0005 | 0.00 | -0.0140 | 0.3767 | 0.000003 |
| `ippo` | 44 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 0.000038 |
| `ippo` | 44 | sampled | 0.3677 | 0.0005 | 0.00 | -0.0099 | 0.3670 | 0.000003 |
| `ippo` | 45 | greedy | 0.9330 | 0.7420 | 0.00 | 0.7810 | 0.9974 | 0.000014 |
| `ippo` | 45 | sampled | 0.3638 | 0.0000 | 0.00 | -0.0124 | 0.3256 | 0.000004 |
| `ippo` | 46 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 0.000038 |
| `ippo` | 46 | sampled | 0.3719 | 0.0000 | 0.00 | -0.0115 | 0.3284 | 0.000003 |
| `ippo` | 47 | greedy | 0.7024 | 0.2065 | 0.00 | 0.0609 | 0.8859 | 0.192897 |
| `ippo` | 47 | sampled | 0.3683 | 0.0000 | 0.00 | 0.0004 | 0.3285 | 0.000003 |
| `ippo` | 48 | greedy | 0.9983 | 0.9945 | 0.00 | 0.8602 | 0.9981 | 0.000038 |
| `ippo` | 48 | sampled | 0.3743 | 0.0005 | 0.00 | -0.0105 | 0.3294 | 0.000003 |
| `ippo` | 49 | greedy | 0.7292 | 0.1955 | 0.00 | 0.0213 | 0.7463 | 0.000002 |
| `ippo` | 49 | sampled | 0.3777 | 0.0000 | 0.00 | -0.0166 | 0.3249 | 0.000003 |
| `ippo` | 50 | greedy | 0.9312 | 0.7495 | 0.00 | 0.9059 | 0.9994 | 0.000020 |
| `ippo` | 50 | sampled | 0.3739 | 0.0005 | 0.00 | -0.0144 | 0.3626 | 0.000003 |
| `ippo` | 51 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 0.000038 |
| `ippo` | 51 | sampled | 0.3903 | 0.0015 | 0.00 | -0.0162 | 0.4392 | 0.000003 |
| `ippo` | 52 | greedy | 0.5615 | 0.0355 | 0.00 | 0.0246 | 0.7909 | 0.000002 |
| `ippo` | 52 | sampled | 0.3609 | 0.0000 | 0.00 | -0.0080 | 0.3277 | 0.000004 |
| `ippo` | 53 | greedy | 0.6469 | 0.0505 | 0.00 | -0.0655 | 0.7117 | 0.000002 |
| `ippo` | 53 | sampled | 0.3686 | 0.0000 | 0.00 | -0.0108 | 0.3204 | 0.000004 |
| `ippo` | 54 | greedy | 0.9884 | 0.9505 | 0.00 | 0.9628 | 0.9996 | 0.000035 |
| `ippo` | 54 | sampled | 0.3845 | 0.0005 | 0.00 | -0.0126 | 0.3765 | 0.000003 |
| `ippo` | 55 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 0.000038 |
| `ippo` | 55 | sampled | 0.3692 | 0.0000 | 0.00 | -0.0113 | 0.3320 | 0.000003 |
| `ippo` | 56 | greedy | 0.7025 | 0.0430 | 0.00 | -0.0602 | 0.5730 | 0.000003 |
| `ippo` | 56 | sampled | 0.3646 | 0.0000 | 0.00 | -0.0119 | 0.3294 | 0.000003 |
| `ippo` | 57 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 0.000038 |
| `ippo` | 57 | sampled | 0.3874 | 0.0015 | 0.00 | -0.0145 | 0.4471 | 0.000003 |
| `ippo` | 58 | greedy | 0.9360 | 0.7680 | 0.00 | 0.8789 | 0.8816 | 0.000035 |
| `ippo` | 58 | sampled | 0.3766 | 0.0000 | 0.00 | -0.0085 | 0.3330 | 0.000003 |
| `ippo` | 59 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 0.000038 |
| `ippo` | 59 | sampled | 0.3719 | 0.0005 | 0.00 | -0.0138 | 0.3490 | 0.000003 |
| `ippo` | 60 | greedy | 1.0000 | 1.0000 | 0.00 | 1.0000 | 1.0000 | 0.000038 |
| `ippo` | 60 | sampled | 0.3677 | 0.0000 | 0.00 | -0.0106 | 0.3301 | 0.000004 |
| `ippo` | 61 | greedy | 0.7439 | 0.2670 | 0.00 | 0.1404 | 0.8245 | 0.000011 |
| `ippo` | 61 | sampled | 0.3751 | 0.0000 | 0.00 | -0.0108 | 0.3306 | 0.000003 |

## Greedy minus sampled, per algorithm

`locked` counts the seeds whose greedy `mode_share` reached 0.998 or above -- the hypothesis's own prediction, stated as a count rather than hidden inside a mean. The Delta columns average only the seeds where the statistic is defined; `n=` says how many that was, so a mean taken over a subset is never mistaken for a mean over all seeds.

| algo | seeds | locked (greedy) | Delta mode_share | Delta action_spearman | Delta sinr_corr | Delta embb_p5 |
|---|---|---|---|---|---|---|
| `gnn-mappo_gat` | 20 | 18/20 | +0.6069 (n=20) | +1.0023 (n=13) | +0.6734 (n=13) | +1.283061 (n=20) |
| `ippo` | 20 | 8/20 | +0.4940 (n=20) | +0.6175 (n=20) | +0.5374 (n=20) | +0.009663 (n=20) |
