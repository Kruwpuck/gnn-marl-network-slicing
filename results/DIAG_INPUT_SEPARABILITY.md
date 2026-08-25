# D6 -- where the node representation collapses

Checkpoints: `results/logs/gnn-*_v4_seed*.pt` (50 matched). State read after 50 policy steps, because `reset()` zeroes 7 of the 8 observation columns and every model would otherwise look identical at t=0.

**Why not cosine.** The eight observation columns are all non-negative (`envs/network_slicing_env.py:537-539`), so every node vector sits in the positive orthant and cosine is lifted by construction; `conv2` has no activation (`gnn/gat_backbone.py:52`), so the embeddings are free to be signed. Comparing 0.910 against 1.0000 compares two different scales. Mean-centring does not fix it either: it forces `sum_i v_i = 0`, hence a negative mean off-diagonal inner product with a floor near `-1/(n-1) = -0.25` at n=5, so a value near 1 cannot occur. Two scale-free statistics are used instead. Raw cosine is kept alongside so the numbers stay continuous with the committed D3 table.

- `rel_spread` = `||X - mean_over_nodes(X)||_F / ||X||_F`. **0 means the nodes are identical.**
- `eff_rank` = `exp(entropy of the normalised singular values)` of the centred matrix. **0.0 means exactly identical nodes** -- the centred matrix is all zeros and there is no varying direction. It cannot be confused with a real value; the smallest of those is 1.0.

**Stages come from the real forward pass**, captured with a forward hook on `conv1` (pre-activation), a forward *pre*-hook on `conv2` (the post-activation `conv1` actually handed on), and the return value. The two backbones differ -- ELU and `edge_attr` for `gat`, ReLU and none for `sage` -- and re-implementing the forward here would be free to drift from `gnn/`.

| backbone | stage | rel_spread median | min | max | eff_rank median | cos_raw median | n |
|---|---|---|---|---|---|---|---|
| `gat` | `input` | 0.2831 | 0.1123 | 0.4287 | 2.72 | 0.9098 | n=25 |
| `gat` | `conv1_pre` | 0.0040 | 0.0002 | 0.0263 | 2.93 | 1.0000 | n=25 |
| `gat` | `conv1_act` | 0.0039 | 0.0002 | 0.0734 | 2.94 | 1.0000 | n=25 |
| `gat` | `conv2` | 0.0000 | 0.0000 | 0.0020 | 1.96 | 1.0000 | n=25 |
| `sage` | `input` | 0.1403 | 0.1123 | 0.4261 | 2.74 | 0.9757 | n=25 |
| `sage` | `conv1_pre` | 0.0926 | 0.0604 | 0.2987 | 2.68 | 0.9921 | n=25 |
| `sage` | `conv1_act` | 0.0841 | 0.0556 | 0.3518 | 2.60 | 0.9928 | n=25 |
| `sage` | `conv2` | 0.0348 | 0.0121 | 0.2556 | 2.26 | 0.9987 | n=25 |

## Which observation columns tell the gNB apart

Across-node standard deviation of each column, relative to that column's own mean (median over 50 checkpoints). A column counts as *varying* at `0.01` -- 1% of its own scale.

| column | relative std, median | varying in |
|---|---|---|
| `ch_gain` | 0.0566 | 50/50 checkpoints |
| `sinr_embb` | 0.8874 | 50/50 checkpoints |
| `sinr_urllc` | 0.9619 | 50/50 checkpoints |
| `q_embb` | 1.2291 | 50/50 checkpoints |
| `urllc_backlog` | 0.8196 | 34/50 checkpoints |
| `viol_ewma` | 1.6920 | 39/50 checkpoints |
| `prev_alloc_lag2` | 0.0000 | 1/50 checkpoints |
| `prev_alloc` | 0.0000 | 1/50 checkpoints |

Columns varying per checkpoint: median 6 of 8, range 4.0-7.0. Non-constant (std > 1e-6): median 6 of 8.

## Verdict

Decision rule, written before the run: the input is degenerate if `rel_spread < 0.05` **and** at most one column varies -- in which case PLAN-03 sections 2 and 7 come first and section 5 is deferred. If the input spread is alive and collapses at `conv1`/`conv2`, the GNN is what collapses it and the current order (section 5 first) stands.

Checkpoints with a degenerate input at `0.05`: **0/50**. At `0.02`: 0/50. At `0.1`: 0/50.

The threshold is reported at three values because a verdict that moves between them is a verdict driven by the threshold, and that would itself be the finding.

### Which layer does it

Ratio of `rel_spread` between consecutive stages, per checkpoint, median. A ratio far below 1 means that stage is where the separation was lost.

| backbone | input -> conv1_pre | conv1_pre -> conv1_act | conv1_act -> conv2 |
|---|---|---|---|
| `gat` | 0.0145 | 0.9861 | 0.0079 |
| `sage` | 0.6343 | 1.0030 | 0.3464 |

### Observation columns the policy itself flattened

`prev_alloc_lag2`, `prev_alloc` vary in at most 5 of 50 checkpoints. These are the previous allocation and its lag (`envs/network_slicing_env.py:538`), so the reading is that every gNB picked the SAME action -- the same lockstep D5 measured as `mode_share` 1.0000, now visible in the observation itself. The policy's own degenerate output feeds back as node features that carry no node identity, which is a loop no document anticipated: it is both a symptom of the collapse and an input to it.

No checkpoint was skipped: every backbone matched exposed `conv1`/`conv2`.

