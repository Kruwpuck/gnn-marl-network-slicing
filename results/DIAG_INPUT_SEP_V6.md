# D6 -- where the node representation collapses

Checkpoints: `results/logs/gnn-*_v6_seed*.pt` (75 matched). State read after 50 policy steps, because `reset()` zeroes 7 of the 8 observation columns and every model would otherwise look identical at t=0.

**Why not cosine.** The eight observation columns are all non-negative (`envs/network_slicing_env.py:552-554`), so every node vector sits in the positive orthant and cosine is lifted by construction; `conv2` has no activation (`gnn/gat_backbone.py:75`), so the embeddings are free to be signed. Comparing 0.910 against 1.0000 compares two different scales. Mean-centring does not fix it either: it forces `sum_i v_i = 0`, hence a negative mean off-diagonal inner product with a floor near `-1/(n-1) = -0.25` at n=5, so a value near 1 cannot occur. Two scale-free statistics are used instead. Raw cosine is kept alongside so the numbers stay continuous with the committed D3 table.

- `rel_spread` = `||X - mean_over_nodes(X)||_F / ||X||_F`. **0 means the nodes are identical.**
- `eff_rank` = `exp(entropy of the normalised singular values)` of the centred matrix. **0.0 means exactly identical nodes** -- the centred matrix is all zeros and there is no varying direction. It cannot be confused with a real value; the smallest of those is 1.0.

**Stages come from the real forward pass**, captured with a forward hook on `conv1` (pre-activation), a forward *pre*-hook on `conv2` (the post-activation `conv1` actually handed on), and the return value. The two backbones differ -- ELU and `edge_attr` for `gat`, ReLU and none for `sage` -- and re-implementing the forward here would be free to drift from `gnn/`.

| backbone | stage | rel_spread median | min | max | eff_rank median | cos_raw median | n |
|---|---|---|---|---|---|---|---|
| `gatedge` | `input` | 0.1590 | 0.1123 | 0.4410 | 2.84 | 0.9689 | n=25 |
| `gatedge` | `conv1_pre` | 0.0092 | 0.0004 | 0.0952 | 2.53 | 0.9999 | n=25 |
| `gatedge` | `conv1_act` | 0.0120 | 0.0004 | 0.1384 | 2.45 | 0.9998 | n=25 |
| `gatedge` | `conv2` | 0.0003 | 0.0000 | 0.0072 | 1.61 | 1.0000 | n=25 |
| `gatres` | `input` | 0.3705 | 0.1123 | 0.4490 | 2.76 | 0.8445 | n=25 |
| `gatres` | `conv1_pre` | 0.0014 | 0.0002 | 0.0730 | 2.73 | 1.0000 | n=25 |
| `gatres` | `conv1_act` | 0.0338 | 0.0141 | 0.1771 | 2.69 | 0.9986 | n=25 |
| `gatres` | `conv2` | 0.0059 | 0.0033 | 0.0341 | 2.58 | 1.0000 | n=25 |
| `gatres-edge` | `input` | 0.3305 | 0.1123 | 0.4471 | 2.70 | 0.8777 | n=25 |
| `gatres-edge` | `conv1_pre` | 0.0021 | 0.0006 | 0.0550 | 2.58 | 1.0000 | n=25 |
| `gatres-edge` | `conv1_act` | 0.0310 | 0.0140 | 0.2609 | 2.68 | 0.9988 | n=25 |
| `gatres-edge` | `conv2` | 0.0070 | 0.0031 | 0.0326 | 2.50 | 0.9999 | n=25 |

## Which observation columns tell the gNB apart

Across-node standard deviation of each column, relative to that column's own mean (median over 75 checkpoints). A column counts as *varying* at `0.01` -- 1% of its own scale.

| column | relative std, median | varying in |
|---|---|---|
| `ch_gain` | 0.0566 | 75/75 checkpoints |
| `sinr_embb` | 0.9984 | 75/75 checkpoints |
| `sinr_urllc` | 0.8494 | 75/75 checkpoints |
| `q_embb` | 1.2329 | 75/75 checkpoints |
| `urllc_backlog` | 0.8169 | 51/75 checkpoints |
| `viol_ewma` | 1.4694 | 60/75 checkpoints |
| `prev_alloc_lag2` | 0.0000 | 2/75 checkpoints |
| `prev_alloc` | 0.0000 | 3/75 checkpoints |

Columns varying per checkpoint: median 6 of 8, range 4.0-8.0. Non-constant (std > 1e-6): median 6 of 8.

## Verdict

Decision rule, written before the run: the input is degenerate if `rel_spread < 0.05` **and** at most one column varies -- in which case PLAN-03 sections 2 and 7 come first and section 5 is deferred. If the input spread is alive and collapses at `conv1`/`conv2`, the GNN is what collapses it and the current order (section 5 first) stands.

Checkpoints with a degenerate input at `0.05`: **0/75**. At `0.02`: 0/75. At `0.1`: 0/75.

The threshold is reported at three values because a verdict that moves between them is a verdict driven by the threshold, and that would itself be the finding.

### Which layer does it

Ratio of `rel_spread` between consecutive stages, per checkpoint, median. A ratio far below 1 means that stage is where the separation was lost.

| backbone | input -> conv1_pre | conv1_pre -> conv1_act | conv1_act -> conv2 |
|---|---|---|---|
| `gatedge` | 0.0514 | 0.9996 | 0.0091 |
| `gatres` | 0.0039 | 23.0156 | 0.1952 |
| `gatres-edge` | 0.0070 | 17.0021 | 0.2111 |

### Observation columns the policy itself flattened

`prev_alloc_lag2`, `prev_alloc` vary in at most 7 of 75 checkpoints. These are the previous allocation and its lag (`envs/network_slicing_env.py:553`), so the reading is that every gNB picked the SAME action -- the same lockstep D5 measured as `mode_share` 1.0000, now visible in the observation itself. The policy's own degenerate output feeds back as node features that carry no node identity, which is a loop no document anticipated: it is both a symptom of the collapse and an input to it.

No checkpoint was skipped: every backbone matched exposed `conv1`/`conv2`.

