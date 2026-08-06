# Attention vs Interference Mechanism

n edges pooled across all episodes/seeds: 400000

Pearson r = -0.0107 (p=1.097e-11), Spearman r = 0.0040 (p=0.01204) -- attention weight (layer 2, single head) vs raw path-loss dB (env-native scale, not the /100 scaled value fed to the model). Effect size ~0 in both cases; the tiny p-values are not evidence of a real effect here -- the 400,000 pooled edge-observations come from only 10 checkpoints x 10 episodes each (~4,000 steps), so within-episode/within-seed autocorrelation inflates apparent significance. Read as: no detectable linear or rank association between learned attention and physical interference strength.

**Causal ablation** (mandatory): attention forced uniform over neighbors (zero the learned `att` parameter of both GATv2Conv layers -> softmax degenerates to 1/degree) during greedy eval. Correlation alone would be decoration without this.

**Result:** 9/10 checkpoints show zero difference between normal and uniform attention -- the trained greedy policy's action choices are insensitive to the attention reweighting entirely. The one exception (`gnn-madqn_gat` seed 44) is not a case *for* attention: uniform attention *rescues* a collapsed seed (0.000039 -> 0.771623 Mbps), i.e. the learned attention pattern was actively harmful there, not helpful. Combined with the near-zero correlation above, this data does not support a "GAT learns to attend to strong interferers" mechanism story for these checkpoints.

| algo | seed | embb_p5 normal | embb_p5 uniform-attn | degradation |
|---|---|---|---|---|
| `gnn-madqn_gat` | 42 | 1.876635 | 1.894174 | -0.017540 |
| `gnn-madqn_gat` | 43 | 0.000038 | 0.000038 | +0.000000 |
| `gnn-madqn_gat` | 44 | 0.000039 | 0.771623 | -0.771584 |
| `gnn-madqn_gat` | 45 | 1.848713 | 1.848713 | +0.000000 |
| `gnn-madqn_gat` | 46 | 1.848713 | 1.848713 | +0.000000 |
| `gnn-mappo_gat` | 42 | 0.000038 | 0.000038 | +0.000000 |
| `gnn-mappo_gat` | 43 | 1.906528 | 1.906528 | +0.000000 |
| `gnn-mappo_gat` | 44 | 1.863268 | 1.863268 | +0.000000 |
| `gnn-mappo_gat` | 45 | 1.884705 | 1.884705 | +0.000000 |
| `gnn-mappo_gat` | 46 | 1.906528 | 1.906528 | +0.000000 |