# Attention vs Interference Mechanism

n edges pooled across all episodes/seeds: 8000

Pearson r = 0.2584 (p=3.56e-122), Spearman r = 0.2283 (p=3.893e-95) -- attention weight (layer 2, single head) vs raw path-loss dB (env-native scale, not the /100 scaled value fed to the model).

**Causal ablation** (mandatory): attention forced uniform over neighbors (zero the learned `att` parameter of both GATv2Conv layers -> softmax degenerates to 1/degree) during greedy eval. Correlation alone would be decoration without this.

| algo | seed | embb_p5 normal | embb_p5 uniform-attn | degradation |
|---|---|---|---|---|
| `gnn-madqn_gat` | 42 | 1.724542 | 1.731546 | -0.007004 |