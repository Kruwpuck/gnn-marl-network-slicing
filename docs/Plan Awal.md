# Technical Research Planning Blueprint
## Benchmarking PPO and DQN Algorithms in a GNN-MARL Architecture for Dynamic Resource Allocation in 5G/6G Network Slicing

## TL;DR
- This blueprint gives you the complete, code-ready scaffolding: a Gymnasium/PettingZoo multi-agent environment where each gNB is an agent, a GNN (use **GAT** as the primary message-passing layer) for spatial feature extraction and inter-agent message passing, and a benchmark of **discrete DQN vs. continuous PPO** over PRB/bandwidth allocation for eMBB and URLLC slices — with full mathematical formulation, 3GPP TR 38.901 channel/traffic models, named public repositories, pinned library versions, and a 3-phase experiment plan culminating in zero-shot generalization (train on 5 gNB, test on 20 gNB).
- The literature strongly validates your Category-3 design: Shao et al. formulate dense-cellular slicing as a MARL problem "in which each BS represents an agent" and "leverage graph attention network (GAT) to strengthen the temporal and spatial cooperation between agents" (stacked on both DQN and A2C) — a direct precedent; the GraphSAGE-MAPPO mesh work is the second. Consensus across wireless surveys is that **GAT** (anisotropic, learns interference-weighted attention) is the best default for interference-coupled wireless graphs, while **GraphSAGE** is the strongest choice if zero-shot size-generalization is your headline contribution.
- For the toolchain, use **Ray RLlib 2.55.x** (native multi-agent, CTDE, parameter sharing) over Stable-Baselines3 for the MARL training loop, **PyTorch Geometric 2.6.x** for the GNN, **Gymnasium 1.x** + **PettingZoo 1.25.x** for the environment API, and generate synthetic data from 3GPP TR 38.901 path-loss formulas plus Poisson (URLLC) / MMPP (eMBB) traffic — seeding from public datasets (Colosseum O-RAN, DeepMIMO, Sionna, DeepVerse 6G) for realism.

---

## Key Findings

1. **Your "each gNB = one agent" MARL formulation with a GNN cooperation layer is well-precedented.** Shao et al. (arXiv:2108.05063, IEEE 2021) state they "formulate this challenge as a multi-agent reinforcement learning (MARL) problem in which each BS represents an agent" and "leverage graph attention network (GAT) to strengthen the temporal and spatial cooperation between agents," with the GAT stacked on top of both DQN (value-based) and A2C (policy+value), and service demands modeled with reference to 3GPP TR 36.814 and TS 22.261. This is the closest published analog to your proposed GNN-MADQN / GNN-MAPPO comparison.

2. **GAT is the recommended primary GNN; GraphSAGE is the recommended generalization-robustness variant.** Wireless resource-allocation surveys consistently find GAT's learned attention weights model interference relationships among nodes better than GCN's fixed degree-normalized aggregation, while GraphSAGE's inductive, fixed-size aggregation is what enables zero-shot transfer to larger topologies.

3. **The discrete-vs-continuous action split maps cleanly onto DQN vs PPO.** DQN requires a discrete/tiered PRB action set; PPO handles continuous bandwidth fractions natively and avoids the quantization error and action-space explosion that discretization causes (confirmed by the 6G O-RAN URLLC PPO paper, which notes PPO is chosen because "the conventional Deep Q-Networks (DQN) method cannot be directly used... as it yields discrete outputs" and discretization "introduces quantisation errors").

4. **3GPP TR 38.901 gives you exact, citable path-loss formulas** for UMa and UMi (LOS/NLOS), and Shannon capacity C = B·log₂(1+SINR) is the standard rate model used across the network-slicing DRL literature. Traffic is modeled with Poisson processes (URLLC sporadic) and MMPP (eMBB bursty/self-similar).

5. **Ray RLlib has first-class multi-agent support** — its docs state it "natively supports multi-agent reinforcement learning (MARL)" with independent learning, collaborative shared-policy/CTDE training, and adversarial training — whereas Stable-Baselines3 is single-agent and only handles multi-agent via SuperSuit wrappers. For a genuine MARL benchmark, RLlib (or MARLlib on top of it) is the better backbone.

---

## Details

### A. MATHEMATICAL FORMULATION

**System model.** Consider $N$ base stations (gNBs) indexed $i \in \mathcal{N}=\{1,\dots,N\}$, each an independent agent. Each gNB serves users belonging to $S$ network slices; here $\mathcal{S}=\{\text{eMBB}, \text{URLLC}\}$. Total system bandwidth $W$ is partitioned into PRBs; let $B$ be the number of allocatable PRBs per gNB. Time is slotted, $t=0,1,2,\dots$. Model the network as a graph $\mathcal{G}_t=(\mathcal{V},\mathcal{E}_t)$ where vertices are gNBs and edges encode interference/proximity coupling.

**A.1 State / Observation Space (per agent).**
Each agent $i$ observes a local node-feature vector. Define the per-agent observation:

$$o_i^{(t)} = \Big[ \underbrace{\bar{\gamma}_i, \;\text{SINR}_i^{\text{eMBB}}, \text{SINR}_i^{\text{URLLC}}}_{\text{channel}},\; \underbrace{q_i^{\text{eMBB}}, q_i^{\text{URLLC}}}_{\text{queue length}},\; \underbrace{d_i^{\text{eMBB}}, d_i^{\text{URLLC}}}_{\text{traffic demand}},\; \underbrace{b_i^{\text{prev}}}_{\text{last allocation}} \Big]$$

- **Node features (per gNB $i$):** mean CSI / channel gain $\bar{\gamma}_i$; per-slice SINR; per-slice queue backlog $q_i^s$ (bits or packets); per-slice arriving traffic demand $d_i^s$; previous PRB split $b_i^{\text{prev}}$; optionally number of associated users per slice.
- **Edge features (per link $(i,j)\in\mathcal{E}_t$):** inter-gNB distance $\rho_{ij}$; path loss $\text{PL}_{ij}$; mutual interference coefficient $I_{ij}$ (from the interference matrix). The adjacency/edge-weight is typically $a_{ij}=\exp(-\rho_{ij}/\rho_0)$ or a thresholded interference value.
- **Global features (shared / appended):** SLA thresholds. Per 3GPP TR 38.913, the URLLC user-plane latency target is **0.5 ms each for UL and DL** (commonly cited as a 1 ms combined budget) at **>99.999% reliability**; 3GPP TS 22.261 specifies that real-time control services "require the E2E latency of 1 ms or less... and reliability of 99.9999%." Also include the eMBB minimum throughput target $T_{\min}^{\text{eMBB}}$ and total available PRBs.

The GNN consumes the full graph: node feature matrix $\mathbf{X}\in\mathbb{R}^{N\times F}$, edge index, and edge weights, producing per-node embeddings $\mathbf{h}_i$ that condition each agent's policy.

**A.2 Action Space — DQN (discrete) vs PPO (continuous).**

*Discrete (DQN).* Each agent selects from a finite set of tiered PRB allocations to the slices. With $B$ PRBs and a quantization step, define discrete allocation levels $\mathcal{A}^{\text{disc}}_i = \{(b^{\text{eMBB}}, b^{\text{URLLC}}) : b^{\text{eMBB}}+b^{\text{URLLC}}\le B\}$, or a tiered split such as $a_i \in \{0\%, 10\%, 20\%,\dots,100\%\}$ of PRBs to URLLC (remainder to eMBB). A common compact design uses $L$ tiers, giving $|\mathcal{A}^{\text{disc}}_i|=L$. DQN learns $Q_i(o_i,a_i;\theta)$ and acts greedily/$\epsilon$-greedily. (Dueling/Double DQN are recommended stabilizers — the Dueling DQN slicing work for eMBB+URLLC found it outperforms vanilla DQN and Double DQN on QoE, spectral efficiency, and stability.)

*Continuous (PPO).* Each agent outputs a continuous allocation fraction $\mathbf{a}_i\in[0,1]^{S}$ with simplex constraint $\sum_s a_i^s = 1$, where $a_i^s$ is the fraction of $W_i$ (or $B$ PRBs) assigned to slice $s$. PPO parameterizes a stochastic policy $\pi_\theta(\mathbf{a}_i|o_i)$ (Gaussian with softmax/Dirichlet projection onto the simplex), optimizing the clipped surrogate objective

$$L^{\text{CLIP}}(\theta)=\mathbb{E}_t\Big[\min\big(r_t(\theta)\hat{A}_t,\; \text{clip}(r_t(\theta),1-\epsilon,1+\epsilon)\hat{A}_t\big)\Big],\quad r_t(\theta)=\frac{\pi_\theta(a_t|o_t)}{\pi_{\theta_{\text{old}}}(a_t|o_t)}.$$

This avoids the quantization error and dimensionality blow-up of discretizing a continuous resource.

**A.3 Multi-objective Reward Function.**
The achievable rate uses Shannon capacity. For UE $k$ on slice $s$ at gNB $i$ with allocated bandwidth $B_{i}^{s}$:

$$R_{i,k}^{s}(t)=B_{i}^{s}\log_2\!\Big(1+\text{SINR}_{i,k}\Big),\qquad \text{SINR}_{i,k}=\frac{P_{i,k}\,g_{i,k}}{N_0 B_{i}^{s} + \sum_{j\ne i} P_{j}\,g_{j,k}}.$$

Spectral efficiency $\text{SE}=\sum_{i,k} R_{i,k}/W$. A weighted multi-objective reward (maximize eMBB throughput/SE, minimize URLLC delay and SLA-violation rate) following the network-slicing DRL literature:

$$r_i^{(t)} = w_1\cdot \underbrace{\frac{T_i^{\text{eMBB}}}{T_{\text{ref}}}}_{\text{eMBB throughput}} \;-\; w_2\cdot \underbrace{\mathbb{1}[\bar{D}_i^{\text{URLLC}}>D_{\max}^{\text{URLLC}}]}_{\text{URLLC SLA violation}} \;-\; w_3\cdot \underbrace{\frac{\bar{D}_i^{\text{URLLC}}}{D_{\max}^{\text{URLLC}}}}_{\text{normalized delay}} \;+\; w_4\cdot \text{SE}_i.$$

Two concrete published reward templates to anchor your design:
- **DORA (Dynamic O-RAN Resource Allocation):** $R = w_U R_U + w_E R_E + w_M R_M$ with weights $(0.5, 0.4, 0.1)$, where URLLC reward $R_U=\max(-1,\min(0,(\bar{L}-L_{\text{tar}})/L_{\text{tar}}))$ penalizes latency above target, and eMBB reward $R_E=\max(-1,\min(0,(T_{\text{tar}}-\bar{T})/T_{\text{tar}}))$ rewards meeting throughput target (in DORA, $L_{\text{tar}}=400$ ms and $T_{\text{tar}}=7$ Mbps for their specific time granularity).
- **Smooth sigmoid QoS reward:** $r_0^l = 1/(1+\exp(-\alpha\cdot(Q^l-\text{thr}^l)/\text{thr}^l))$ summed over slices, with an exponential penalty for severe violations — gives a differentiable, normalized signal across heterogeneous metrics (latency vs throughput).

For MARL credit assignment, use a team reward (sum of per-agent rewards) for cooperative CTDE training, optionally with a per-agent shaping term.

### B. GNN ARCHITECTURE

**Graph construction.** Nodes = gNBs (agents). Edges = inter-gNB coupling, defined either by (a) geographic proximity (Euclidean distance threshold, as in Shao et al.), or (b) interference strength (complete graph weighted by $I_{ij}$, as in subnetwork/V2X works). For interference-limited dense deployments, an interference-weighted graph is more physically meaningful.

- **Node features:** the per-gNB state vector in A.1 (CSI, per-slice SINR, queue, demand, previous allocation). If you model users as nodes too (heterogeneous/bipartite graph), UE node features = channel gain, slice membership, buffer occupancy.
- **Edge features/weights:** distance $\rho_{ij}$, path loss $\text{PL}_{ij}$, interference coefficient $I_{ij}$. GAT can ingest edge features via edge-conditioned attention.

**Message passing scheme — recommendation.** The three candidates and their trade-offs:
- **GCN:** fixed, degree-normalized isotropic aggregation; cheapest; treats all neighbors equally — poor at distinguishing strong vs weak interferers; prone to over-smoothing beyond 2-3 layers.
- **GraphSAGE:** inductive, samples and aggregates a fixed-size neighborhood with learnable aggregators (mean/pool/LSTM); **the parameter count is independent of node count**, which is precisely what enables zero-shot transfer to larger topologies. Best choice to headline size-generalization.
- **GAT:** anisotropic; learns attention weights $\alpha_{ij}$ over neighbors, so it can learn to weight high-interference neighbors more — best expressiveness for interference-coupled allocation; multi-head attention stabilizes training.

**Recommendation:** Use **GAT** as your primary architecture (best precedent in slicing-MARL via Shao et al.; best models interference). Use **GraphSAGE** as the architecture for your zero-shot generalization phase, since its inductive nature is the cleanest way to argue topology-size robustness. Keep layers shallow (2-3) to avoid over-smoothing; the Jumping-Knowledge GAT (JGAT) variant mitigates over-smoothing at greater depth if needed (the Nature *Scientific Reports* JGAT work shows GAT/GCN performance "sharply declines after reaching the optimum at layers 3 and 2, respectively, due to the over-smoothing problem," while LSTM jumping connections alleviate it). Implement all three in PyTorch Geometric (`GCNConv`, `SAGEConv`, `GATv2Conv`) so the GNN backbone is a swappable module behind the same MARL policy head — this gives you a clean ablation.

### C. 3GPP-BASED SYNTHETIC DATA GENERATION

**C.1 Traffic models.**
- **URLLC (sporadic):** homogeneous **Poisson process** with rate $\lambda_{\text{URLLC}}$; inter-arrival times $\sim \text{Exp}(\lambda)$; small fixed payloads. Generate per slot: `n_arrivals = np.random.poisson(lam * dt)`. For calibration, the ColO-RAN dataset (Bonati et al., arXiv:2112.09559) uses "4 Mbit/s constant bitrate traffic to eMBB users, and 44.6 kbit/s and 89.3 kbit/s Poisson traffic to MTC and URLLC, respectively" over a 10 MHz / 50-PRB channel.
- **eMBB (bursty/self-similar):** **Markov-Modulated Poisson Process (MMPP)** — a two-state (bursty/idle) modulated Poisson process whose arrival rate switches according to a continuous-time Markov chain. MMPP is the standard model for capturing the burstiness and (via superposition of 2-state MMPPs, which can be fit to match the variance-time curve over multiple time-scales) the self-similarity of multimedia/data traffic. Parameters: rate matrix $\Lambda=\text{diag}(\lambda_1,\lambda_2)$ and generator $Q=\begin{psmallmatrix}-r_1 & r_1\\ r_2 & -r_2\end{psmallmatrix}$. For heavier long-range dependence, use a Poisson-Pareto Burst Process (PPBP) or a multifractal model.

**C.2 Channel models — 3GPP TR 38.901 (verbatim formulas).**
$f_c$ in GHz, distances in meters; $d_{3D}$ is 3D Tx-Rx distance, $d_{2D}$ ground distance, breakpoint $d'_{BP}=4\,h'_{BS}h'_{UT}f_c\cdot 10^9/c$.

*Urban Macro (UMa), $h_{BS}=25$ m, $1.5\le h_{UT}\le 22.5$ m:*
$$\text{PL}_{\text{UMa-LOS}}=\begin{cases}28.0+22\log_{10}(d_{3D})+20\log_{10}(f_c), & 10\le d_{2D}\le d'_{BP}\\ 28.0+40\log_{10}(d_{3D})+20\log_{10}(f_c)-9\log_{10}((d'_{BP})^2+(h_{BS}-h_{UT})^2), & d'_{BP}\le d_{2D}\le 5\text{km}\end{cases}$$
$$\text{PL}'_{\text{UMa-NLOS}}=13.54+39.08\log_{10}(d_{3D})+20\log_{10}(f_c)-0.6(h_{UT}-1.5);\quad \text{PL}_{\text{UMa-NLOS}}=\max(\text{PL}_{\text{UMa-LOS}},\text{PL}'_{\text{UMa-NLOS}}).$$
$\sigma_{SF}=4$ dB (LOS), $6$ dB (NLOS).

*Urban Micro (UMi) - Street Canyon, $h_{BS}=10$ m:*
$$\text{PL}_{\text{UMi-LOS}}=\begin{cases}32.4+21\log_{10}(d_{3D})+20\log_{10}(f_c), & 10\le d_{2D}\le d'_{BP}\\ 32.4+40\log_{10}(d_{3D})+20\log_{10}(f_c)-9.5\log_{10}((d'_{BP})^2+(h_{BS}-h_{UT})^2), & d'_{BP}\le d_{2D}\le 5\text{km}\end{cases}$$
$$\text{PL}'_{\text{UMi-NLOS}}=35.3\log_{10}(d_{3D})+22.4+21.3\log_{10}(f_c)-0.3(h_{UT}-1.5);\quad \text{PL}_{\text{UMi-NLOS}}=\max(\text{PL}_{\text{UMi-LOS}},\text{PL}'_{\text{UMi-NLOS}}).$$
$\sigma_{SF}=4$ dB (LOS), $7.82$ dB (NLOS).

*Fading:* add log-normal shadow fading $\mathcal{N}(0,\sigma_{SF}^2)$ in dB; small-scale fading via Rayleigh (NLOS) or Rician (LOS) coefficients, or use a TR 38.901 clustered delay-line model for richer CSI.

*SINR / noise:* received power $P_r=P_t+G_t+G_r-\text{PL}-\text{shadowing}$ (dB); thermal noise $N=-174\text{ dBm/Hz}+10\log_{10}(B)+\text{NF}$; $\text{SINR}=P_r/(N+I)$ where $I$ aggregates inter-cell interference from co-channel gNBs.

**C.3 Public repositories and integration.**
- **Sionna (NVIDIA), `NVlabs/sionna`** (current v2.x): GPU-accelerated, differentiable link-level + ray-tracing (Sionna RT) + system-level (Sionna SYS); TensorFlow-based. Use Sionna RT to pre-generate site-specific CIRs / path-gain matrices, export to `.npy`, and load them as the channel oracle inside `step()`. (See also `tkn-tub/ns3sionna` and `wineslab/sionna-channel-generator` for export pipelines.)
- **DeepMIMO (`DeepMIMO/DeepMIMO-python`, v3/v4):** parameterized ray-tracing channels (Remcom Wireless InSite) for outdoor scenarios (e.g., the O1 street scenario with 18 BSs). Pre-generate channel matrices for fixed gNB/UE layouts; sample per-episode.
- **DeepVerse 6G (`wireless-intelligence-lab/DeepVerse6G-python`, `pip install deepverse`):** multi-modal ray-traced channel + sensing datasets; the O1 urban-street scenario has 4 BSs. Good for digital-twin-style realism.
- **Colosseum / OpenRAN Gym O-RAN datasets:** `wineslab/colosseum-oran-commag-dataset` (KPMs, 7 BS / 42 UE, RBG allocations + RR/WF/PF scheduling policies for eMBB/URLLC/mMTC) and `genesys-neu/oran-resource-opt` (PRB-allocation RL on Colosseum traces). Also `wineslab/ns-o-ran-gym` provides a Gymnasium base environment for O-RAN. Use these CSV KPM traces to validate/calibrate your synthetic generator or as a realistic replay source.
- **5G traffic generators:** `0913ktg/5G-Traffic-Generator` (Python), `5G-Toolkit` (NumPy-based, 3GPP-compliant link/system simulation), and `rohan-chandrashekar/5G-Network-Slicing` (Python eMBB/mMTC/URLLC slicing sim) as starting points for traffic-trace scripts.
- **3GPP path-loss reference code:** `jakthra/PseudoRayTracingOSM/pathloss_38901.py` provides a verified NumPy implementation of the UMa LOS/NLOS formulas above; `YIJ18/MATLAB-3GPP-LOS-Path-Loss-Models` for MATLAB.

**Integration pattern into `step()`:** Precompute/cache the static path-loss matrix $\text{PL}_{ij}$ and per-UE channel gains at `reset()` (sampling UE positions + LOS/NLOS per TR 38.901 LOS probability). Each `step()`: (1) advance traffic generators (Poisson/MMPP) to get per-slice arrivals, update queues; (2) apply the agents' PRB/bandwidth actions; (3) compute per-UE SINR and Shannon rate; (4) update delays/throughput, compute SLA violations; (5) assemble the new graph (node features + edge weights) and per-agent rewards; (6) return PyG-compatible observations. Keep channel sampling vectorized (NumPy) for speed; only call Sionna/DeepMIMO offline to build the channel database.

### D. TOOLS & LIBRARIES (pinned versions, June 2026)

| Component | Recommended version | Notes |
|---|---|---|
| Python | 3.11 or 3.12 | 3.12 supported across the stack; avoid 3.13/3.14 until all deps catch up |
| PyTorch | 2.4 (or 2.8) | pick to match PyG |
| PyTorch Geometric (PyG) | **2.6.x** (2.6.1) | "fully compatible with PyTorch 2.4"; PyG 2.7 needs torch 2.8. Use 2.6.x for the most stable RLlib/torch combo |
| Gymnasium | **1.0+ / 1.1** | v1.0 finalized the core API (Env, Space, VectorEnv); single-agent env standard |
| PettingZoo | **1.25.x** | multi-agent API; requires gymnasium ≥1.0.0; Parallel API for simultaneous gNB actions |
| Ray RLlib | **2.55.x** | native multi-agent (independent / shared-policy CTDE / adversarial); Gymnasium + PettingZoo support |
| Stable-Baselines3 | 2.x | single-agent only; multi-agent only via SuperSuit wrappers; use as a sanity-check baseline, not the MARL backbone |
| SuperSuit | latest matching PettingZoo | env preprocessing wrappers |
| (optional) MARLlib | on Ray/RLlib | unified MARL algorithm pipeline if you want MADDPG/MAPPO/QMIX out of the box |
| NumPy | 1.26 / 2.x | check PyG/torch pin |

**Library-conflict guidance:** (1) PyG↔PyTorch coupling is the most brittle link — install PyG wheels matching your exact torch+CUDA build; PyG 2.6 ↔ torch 2.4, PyG 2.7 ↔ torch 2.8. (2) PettingZoo 1.25 requires gymnasium ≥1.0; older RLlib examples pinned Ray 2.7 but current PettingZoo tutorials bumped the Ray dependency to 2.55 — align Ray and PettingZoo. (3) RLlib's new API stack (RLModule) is the path for custom GNN policies; you'll wrap your PyG GNN as a custom `RLModule` and feed the graph through the observation dict. (4) Sionna is TensorFlow-based — keep it in a separate offline data-generation environment to avoid TF/PyTorch dependency clashes; never co-import in the training process.

### E. RECENT LITERATURE (2021-2025)

1. **Shao, Li, Hu, Wu, Zhao, Zhang — "Graph Attention Network-based Multi-agent Reinforcement Learning for Slicing Resource Management in Dense Cellular Network"** (2021, arXiv:2108.05063; full IEEE journal version, with a shorter conference version at IEEE WCNC 2021). Each BS is a MARL agent; a GAT captures spatial/temporal cooperation among BSs (graph defined by Euclidean distance), stacked on **both DQN and A2C**. Introduces a compact, interpretable reward and references 3GPP TR 36.814 / TS 22.261 for service demands. **The single closest precedent to your GNN-MADQN/GNN-MAPPO benchmark.**

2. **Xu, Han, Fu, Zhu, Wu, Zhu — "GNN-DRL-Based Intelligent Routing and Resource Allocation Algorithms for Multi-Layer Wireless Mesh Network"** (2026, *Sensors* 26(4):1170, DOI 10.3390/s26041170). Proposes **GraphSAGE-MAPPO**: inductive GraphSAGE extracts node resource/link-state features into fixed-size vectors feeding a distributed MAPPO (CTDE). Mixes discrete routing + continuous resource actions. The authors report it "can flexibly adjust routing strategies to better meet the QoS requirements of various services and has good generalization performance for network topology and resource changes" — directly relevant to your Phase-3 zero-shot goal.

3. **Orhan, Swamy, Tetzlaff, Nassar, Nikopour, Talwar — "Connection Management xApp for O-RAN RIC: A Graph Neural Network and Reinforcement Learning Approach"** (2021, IEEE ICMLA 2021, pp. 936-941, DOI 10.1109/ICMLA52953.2021.00154; arXiv:2110.07525). GNN+DRL for user-cell association/load balancing in O-RAN; nodes = cells/users. Reports "up to 10% gain in throughput, 45-140% gain [in] cell coverage, 20-45% gains in load balancing, compared to baseline greedy techniques," plus ~54% improvement in 5th-percentile user data-rate coverage.

4. **Wang, Bennis, Zhou — "Graph Attention-Based MADRL for Access Control and Resource Allocation in Wireless Networked Control Systems"** (2024, IEEE Transactions on Wireless Communications). GAT within a multi-agent DRL framework for joint access-control + resource allocation under spectral/energy constraints; graph captures sensor/controller/actuator relationships.

5. **He, Wang, Ye, Li, Juang — "Resource Allocation based on Graph Neural Networks in Vehicular Communications"** (2020, IEEE GLOBECOM 2020, pp. 1-5, DOI 10.1109/GLOBECOM42002.2020.9322537). V2V network as a graph (nodes = V2V pairs, edges = interference-link channel gains); GNN learns low-dimensional features, each link is a DQN-style agent doing distributed sub-band/power selection. Template for interference-graph + multi-agent RL.

6. **Ji, Wu, Fan, Cheng, Chen, Wang, Letaief — "Graph Neural Networks and Deep Reinforcement Learning Based Resource Allocation for V2X Communications"** (2025, IEEE Internet of Things Journal; arXiv:2407.06518). Integrates **GraphSAGE with DRL**; dynamic graph with links as nodes; supports distributed deployment and scalability to varying link counts. Code released.

7. **(Survey/framing) "Graph Neural Network Meets Multi-Agent Reinforcement Learning: Fundamentals, Applications, and Future Directions"** (2024, arXiv:2404.04898). Systematic treatment of GNNComm-MARL for resource allocation and mobility management; finds GNNComm-MARL "can achieve better performance with lower communication overhead compared to conventional communication schemes" — a strong framing reference for your introduction.

8. **(Algorithm-comparison context) UAV-based 5G slicing MADRL** (2025, arXiv:2512.03835): benchmarks **MAPPO vs MADDPG vs MADQN** under CTDE for slicing — finds "MAPPO achieves the best overall QoS–energy tradeoff," MADDPG offers precise continuous control, and "MADQN provides a computationally efficient baseline for discretized action spaces," concluding "no single MARL algorithm is universally dominant." Useful to motivate your discrete-vs-continuous comparison and set expectations on which algorithm wins where.

Also worth tracking: the GAT-based survey *A Survey of Graph-Based Resource Management in Wireless Networks* (Dai et al., 2025, U. Waterloo) for the GCN/GraphSAGE/GAT design taxonomy; the *Dueling DQN for eMBB and URLLC Hybrid Services* paper (Chen, Shao, Shen, Zeng, *Sensors* 2023, DOI 10.3390/s23052518) for a discrete-action slicing reward template; and the curated bibliography `jwwthu/GNN-Communication-Networks` for an up-to-date GNN+communications paper list.

### F. EXPERIMENT ROADMAP / MILESTONES

**Phase 0 — Environment & data engineering (weeks 1-4).** Build the Gymnasium single-agent prototype, then lift to a PettingZoo `ParallelEnv` / RLlib `MultiAgentEnv`. Implement the TR 38.901 path-loss + Poisson/MMPP traffic generators; validate generated traffic statistics (mean rate, burstiness/Hurst parameter) and SINR distributions against Colosseum/DeepMIMO references. Deliverable: a seeded, reproducible NS-Gym environment emitting PyG graphs.

**Phase 1 — Baselines (weeks 5-8).** Implement and tune, in increasing architectural complexity:
- (B1) **Centralized DQN / PPO with MLP** — single agent sees the global state, flat MLP policy. Upper-bound on coordination, worst on scalability.
- (B2) **Independent DQN / PPO (IQL/IPPO) without GNN** — one agent per gNB, no message passing. Tests the value of the GNN.
- (P) **Proposed GNN-MADQN and GNN-MAPPO** — per-gNB agents with shared GAT backbone + DQN or PPO heads, CTDE via RLlib.

**Phase 2 — Three evaluation phases.**
1. **Training/convergence:** plot cumulative-reward learning curves vs environment steps, sample efficiency, and wall-clock training time per algorithm. Expect PPO to be more stable in the continuous action space and DQN faster per-step but sensitive to discretization granularity.
2. **Testing/network performance:** on held-out traffic/channel seeds, measure eMBB throughput and system spectral efficiency, URLLC mean/99.9th-percentile delay, and **SLA violation rate** (fraction of slots with URLLC delay > target). Report PRB utilization and fairness.
3. **Zero-shot generalization:** train on a **5-gNB** topology, then evaluate without retraining on **20-gNB** (and intermediate 10-gNB) topologies. This is where the GNN — especially GraphSAGE's inductive, node-count-invariant parameters — should dramatically outperform MLP baselines, whose fixed input dimension cannot even accept the larger observation without retraining. Metric: performance retention ratio (large-topology reward / small-topology reward) and SLA-violation degradation.

**Phase 3 — Ablations & analysis (weeks 13-16).** Swap GNN backbone (GCN vs GraphSAGE vs GAT) holding the RL algorithm fixed; vary graph construction (distance vs interference edges); vary reward weights $(w_1,\dots,w_4)$; test sensitivity to traffic burstiness. Deliverable: thesis methodology + results chapters.

**Decision thresholds / benchmarks that change the plan:**
- If GAT shows over-smoothing/instability beyond ~3 layers → switch to JGAT or reduce depth.
- If DQN's discretization caps URLLC SLA performance → increase tier resolution or concede the discrete-action limitation as a finding.
- If zero-shot retention < ~70% for GAT but GraphSAGE holds up → make GraphSAGE the headline architecture for the generalization claim.
- If synthetic-only results are questioned → calibrate against the Colosseum O-RAN dataset as an external-validity check.

---

## Recommendations

1. **Start now with Phase 0 using GAT + Ray RLlib 2.55.x + PyG 2.6.x on a 5-gNB topology.** Get the IPPO and GNN-MAPPO pipeline running end-to-end on synthetic data before optimizing realism. Lock library versions in a `requirements.txt`/conda env immediately to avoid the PyG↔torch and PettingZoo↔Ray version traps.
2. **Treat the GNN backbone as a swappable module** (GCN/GraphSAGE/GAT behind one interface) so your ablation and your zero-shot-generalization argument both fall out of the same codebase. Use GAT as default, GraphSAGE for the generalization headline.
3. **Anchor your reward on the DORA weighted template** $(w_U,w_E,w_M)$ and the smooth-sigmoid QoS form, then ablate weights. Define SLA violation crisply (URLLC delay > your chosen latency budget) as your primary URLLC metric.
4. **Generate channels offline with Sionna RT or DeepMIMO into a cached database; keep TensorFlow (Sionna) isolated from the PyTorch training env.** Use Colosseum O-RAN CSV traces as an external calibration/validation set to defend the synthetic approach.
5. **Sequence the algorithms:** prove IPPO/IDQN baselines first, then add the GNN; report convergence, network KPIs, and zero-shot generalization as three separate result tables. Expect MAPPO+GNN to win on QoS tradeoff and generalization, DQN+GNN to be the efficient discrete baseline — but treat that as a hypothesis to test, per the UAV-slicing finding that no single MARL algorithm is universally dominant.

## Caveats
- Several of the most relevant precedents (Shao et al. full journal version; Wang-Bennis-Zhou TWC 2024; Ji et al. IoT-J 2025) had exact volume/issue/page numbers that I could not fully verify from secondary sources — confirm the canonical citation on IEEE Xplore before formal submission.
- SLA latency targets are use-case dependent. The 3GPP TR 38.913 URLLC user-plane target is 0.5 ms per direction at >99.999% reliability, and TS 22.261 cites 1 ms / 99.9999% for real-time control while tolerating 10-20 ms for remote services; published DRL slicing papers use looser numeric targets (e.g., DORA uses 400 ms) because simulated delay scales differ. **Pick and justify your SLA threshold explicitly for your environment's time granularity.**
- Synthetic 3GPP-based data establishes algorithmic validity but not deployment realism; zero-shot results on synthetic topologies are evidence of architectural robustness, not a guarantee of field performance.
- The "no single MARL algorithm is universally dominant" / "PPO beats DQN" expectations are forward-looking claims from recent preprints; treat them as hypotheses to test, not settled results.
- Library versions move fast; the pinned versions reflect the stable state as of June 2026 — re-check PyG/PyTorch/Ray compatibility at implementation time, and prefer installing PyG from wheels matched to your exact torch+CUDA build.