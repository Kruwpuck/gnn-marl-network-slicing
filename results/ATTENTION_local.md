# Attention vs Interference Mechanism

n edges pooled across all episodes/seeds: 400000

Pearson r = -0.0107 (p=1.097e-11), Spearman r = 0.0040 (p=0.01204) -- attention weight (layer 2, single head) vs raw path-loss dB (env-native scale, not the /100 scaled value fed to the model).

**Causal ablation** (mandatory): attention forced uniform over neighbors (zero the learned `att` parameter of both GATv2Conv layers -> softmax degenerates to 1/degree) during greedy eval. Correlation alone would be decoration without this.

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