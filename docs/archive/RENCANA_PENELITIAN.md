# Rencana Penelitian: GNN-MARL untuk Alokasi Sumber Daya Dinamis pada Network Slicing 5G/6G

> **Topik**: Benchmarking PPO dan DQN dalam Arsitektur GNN-MARL untuk Dynamic Resource Allocation pada 5G/6G Network Slicing
> **Durasi**: 16 Minggu (4 Fase)
> **Tujuan**: Membandingkan GNN-MADQN vs GNN-MAPPO dalam efisiensi alokasi PRB untuk slice eMBB dan URLLC, dengan uji generalisasi zero-shot pada topologi berbeda.

---

## Ringkasan Eksekutif

Setiap base station (gNB) dimodelkan sebagai agen independen dalam kerangka Multi-Agent Reinforcement Learning (MARL). Graph Neural Network (GNN) bertindak sebagai lapisan kooperasi antar-agen yang mengekstrak fitur spasial dan memodelkan interferensi antar-sel. Dua algoritma dibandingkan:

- **DQN diskrit** — aksi alokasi PRB berlevel/tiered
- **PPO kontinu** — fraksi bandwidth kontinu tanpa error kuantisasi

---

## Stack Teknologi (Versi Terkunci)

| Komponen | Versi | Catatan |
|---|---|---|
| Python | 3.11 / 3.12 | Hindari 3.13+ |
| PyTorch | 2.4 | Harus match PyG |
| PyTorch Geometric (PyG) | 2.6.x | `GCNConv`, `SAGEConv`, `GATv2Conv` |
| Gymnasium | 1.0+ | API environment single-agent |
| PettingZoo | 1.25.x | API multi-agent (Parallel API); butuh gymnasium ≥1.0 |
| Ray RLlib | 2.55.x | MARL backbone (CTDE, parameter sharing) |
| NumPy | 1.26 / 2.x | Cek kompatibilitas dengan PyG/torch |

**Peringatan konflik library:**
- PyG ↔ PyTorch paling rawan — install PyG wheel yang match exact torch+CUDA build
- Sionna berbasis TensorFlow — pisahkan di environment tersendiri untuk pre-generate channel database

---

## Formulasi Matematis Kunci

### Ruang Observasi per-Agen

$$o_i^{(t)} = [\bar{\gamma}_i,\; \text{SINR}_i^{eMBB},\; \text{SINR}_i^{URLLC},\; q_i^{eMBB},\; q_i^{URLLC},\; d_i^{eMBB},\; d_i^{URLLC},\; b_i^{prev}]$$

### Reward Multi-Objektif

$$r_i^{(t)} = w_1\cdot\frac{T_i^{eMBB}}{T_{ref}} - w_2\cdot\mathbb{1}[\bar{D}_i^{URLLC}>D_{max}] - w_3\cdot\frac{\bar{D}_i^{URLLC}}{D_{max}} + w_4\cdot SE_i$$

Template DORA (anchor awal): weights $(w_U, w_E, w_M) = (0.5, 0.4, 0.1)$

### Model Kanal (3GPP TR 38.901)

UMa LOS: $\text{PL} = 28.0 + 22\log_{10}(d_{3D}) + 20\log_{10}(f_c)$ dB

Shannon Capacity: $R = B\log_2(1 + \text{SINR})$ bps

---

## Alur Penelitian (Research Flow)

```
FASE 0: Persiapan Lingkungan & Rekayasa Data  (Minggu 1-4)
   │
   ├─ Buat NetworkSlicingEnv (Gymnasium single-agent, 5 gNB)
   ├─ Lift ke PettingZoo ParallelEnv → RLlib MultiAgentEnv
   ├─ Implementasi path-loss TR 38.901 (UMa/UMi LOS+NLOS)
   ├─ Implementasi traffic: Poisson (URLLC) + MMPP (eMBB)
   ├─ Validasi statistik SINR & traffic vs Colosseum O-RAN
   └─ Output: env emit PyG graph (node features + edge_index + edge_attr)
         │
         ▼
FASE 1: Implementasi Baseline  (Minggu 5-8)
   │
   ├─ B1: Centralized DQN/PPO + MLP  [upper bound koordinasi]
   ├─ B2: Independent DQN/PPO tanpa GNN (IQL/IPPO)  [nilai GNN]
   └─ P:  GNN-MADQN & GNN-MAPPO (GAT backbone, CTDE via RLlib)
         │
         ▼
FASE 2a: Evaluasi Training & Konvergensi  (Minggu 9-10)
   │
   ├─ Plot reward kumulatif vs environment steps
   ├─ Sample efficiency (reward per 100K steps)
   └─ Wall-clock training time
         │
         ▼
FASE 2b: Evaluasi Performa Jaringan  (Minggu 10-11)
   │
   ├─ eMBB: throughput (Mbps), spectral efficiency (bps/Hz)
   ├─ URLLC: mean delay, 99.9th-percentile delay, SLA violation rate
   └─ PRB utilization, Jain's Fairness Index
         │
         ▼
FASE 2c: Generalisasi Zero-Shot  (Minggu 11-12)
   │
   ├─ Train: 5-gNB topology
   ├─ Test tanpa retraining: 10-gNB → 20-gNB
   └─ Metrik: Performance Retention Ratio = reward(N-gNB) / reward(5-gNB)
         │
         ▼
FASE 3: Ablasi & Analisis  (Minggu 13-16)
   │
   ├─ Swap GNN backbone: GCN vs GraphSAGE vs GAT (fix RL = MAPPO)
   ├─ Variasi graph: distance-threshold vs interference-weighted
   ├─ Variasi reward weights (w1..w4)
   └─ Sensitivitas terhadap burstiness traffic MMPP
         │
         ▼
OUTPUT: Bab Metodologi + Bab Hasil Thesis
```

---

## Fase 0 — Persiapan Lingkungan & Rekayasa Data (Minggu 1–4)

### Tujuan
Environment simulasi reproducible yang emit PyG graph, tervalidasi vs data referensi nyata.

### Tugas

**Minggu 1–2: Setup Environment**
- [ ] Buat `NetworkSlicingEnv(gymnasium.Env)` — obs/action space, reward function
- [ ] Definisikan observation: 8 fitur per gNB (lihat formula di atas)
- [ ] Action space diskrit (11 level, 0-100% step 10%) dan kontinu ([0,1]^2 simplex)
- [ ] Lift ke `pettingzoo.ParallelEnv` — satu env untuk N gNB simultan
- [ ] Wrap ke `ray.rllib.MultiAgentEnv`

**Minggu 2–3: Channel & Traffic Model**
- [ ] Implementasi `ChannelModel` — TR 38.901 UMa/UMi LOS/NLOS path-loss
- [ ] Tambah shadow fading log-normal ($\sigma_{SF}$ = 4–7.82 dB sesuai skenario)
- [ ] Implementasi `PoissonTraffic` — URLLC sporadic, `n = np.random.poisson(λ·dt)`
- [ ] Implementasi `MMPPTraffic` — eMBB bursty, 2-state Markov-modulated
- [ ] Pre-compute path-loss matrix di `reset()`, sampling per `step()`

**Minggu 3–4: Validasi & Integrasi PyG**
- [ ] Validasi distribusi SINR vs Colosseum O-RAN CSV traces
- [ ] Pastikan `step()` return dict: `{x: tensor(N,F), edge_index: tensor(2,E), edge_attr: tensor(E,D)}`
- [ ] Unit test `reset()` + `step()` deterministik dengan fixed seed
- [ ] Benchmark kecepatan: target < 10 ms per `step()`

### Deliverable
`NetworkSlicingEnv` — seeded, reproducible, emit PyG-compatible graph.

### Keputusan Kritis
- Graph **interference-weighted** (lebih fisik dari distance-only)
- Sionna/DeepMIMO untuk pre-generate channel database — **jangan co-import** TF+PyTorch

---

## Fase 1 — Implementasi Baseline (Minggu 5–8)

### Arsitektur Agent

```
B1: global_state → MLP → DQN/PPO head   (1 agent, centralized)
B2: local_obs   → MLP → DQN/PPO head   (N agents, no message passing)
P:  graph(x,E)  → GAT → embedding_i → DQN/PPO head  (N agents + GNN)
```

### Tugas

**Minggu 5–6: Baseline B1 & B2**
- [ ] Centralized DQN — flat MLP, global state, single agent
- [ ] IPPO — Independent PPO, MLP, per-gNB, RLlib
- [ ] IDQN — Independent DQN, MLP, per-gNB, RLlib
- [ ] Training 1M steps, plot reward curve, pastikan tidak diverge

**Minggu 6–8: Proposed Method**
- [ ] Implementasi `GATBackbone` dengan `GATv2Conv` (PyG, 2-3 layer)
- [ ] Implementasi `SAGEBackbone` dan `GCNBackbone` — antarmuka sama (swappable)
- [ ] Wrap GNN sebagai custom `RLModule` di RLlib (new API stack)
- [ ] GNN-MADQN — GAT backbone + Dueling DQN head
- [ ] GNN-MAPPO — GAT backbone + PPO head, CTDE (shared critic)
- [ ] Training 1M steps, validasi convergence

### Deliverable
4 algoritma berjalan: Centralized-MLP, IPPO, GNN-MADQN, GNN-MAPPO.

### Keputusan Kritis
- GNN backbone = **modul swappable** — satu interface `forward(x, edge_index, edge_attr) → embedding`
- Gunakan **Dueling DQN** bukan vanilla DQN (lebih stabil, terbukti di literatur slicing)
- **Parameter sharing** antar-agen (bobot sama semua gNB) untuk efisiensi sample

---

## Fase 2 — Evaluasi (Minggu 9–12)

### Fase 2a: Training & Konvergensi (Minggu 9–10)

| Metrik | Cara Ukur |
|---|---|
| Cumulative reward | Moving average (window 100 episode) |
| Sample efficiency | Reward pada 100K, 500K, 1M steps |
| Stabilitas training | Std dev reward moving average |
| Wall-clock time | Jam total training per algoritma |

**Hipotesis**: PPO lebih stabil (continuous action); DQN lebih cepat per-step tapi sensitif granularitas diskretisasi.

### Fase 2b: Performa Jaringan (Minggu 10–11)

| Metrik | Slice | Target |
|---|---|---|
| Throughput (Mbps) | eMBB | Maksimum |
| Spectral Efficiency (bps/Hz) | System | Maksimum |
| Mean Delay (ms) | URLLC | < SLA threshold |
| 99.9th-percentile Delay (ms) | URLLC | < SLA threshold |
| SLA Violation Rate (%) | URLLC | Minimum |
| PRB Utilization (%) | System | ~90–95% |
| Jain's Fairness Index | System | Mendekati 1.0 |

> **Catatan SLA**: Tetapkan threshold URLLC secara eksplisit sesuai time granularity simulasi (3GPP TR 38.913: 0.5 ms per arah; DRL papers sering pakai 400 ms — justifikasi pilihanmu).

### Fase 2c: Generalisasi Zero-Shot (Minggu 11–12)

**Protokol**:
1. Train semua algoritma pada topologi **5 gNB** (1M steps, seeds 1-5)
2. Evaluate langsung **tanpa retraining** pada **10-gNB** dan **20-gNB**
3. Ukur **Performance Retention Ratio** = reward(N) / reward(5) × 100%

**Hipotesis**: GNN (terutama GraphSAGE) retain > 70%; MLP tidak bisa handle ukuran input berbeda (fixed input dimension).

---

## Fase 3 — Ablasi & Analisis (Minggu 13–16)

### Ablasi GNN Backbone (Fix: MAPPO)

| Backbone | Karakteristik | Ekspektasi |
|---|---|---|
| GCN | Fixed degree-normalized | Paling murah, rentan over-smoothing |
| GraphSAGE | Inductive, fixed-size aggregation | Zero-shot generalization terbaik |
| GAT | Anisotropic attention | Interference modeling terbaik |

### Ablasi Lainnya

| Variabel | Nilai yang Dicoba |
|---|---|
| Graph construction | Distance-threshold vs interference-weighted |
| Reward weights | (0.5,0.2,0.2,0.1), (0.4,0.3,0.2,0.1), (0.3,0.3,0.3,0.1) |
| GNN depth | 1, 2, 3 layer |
| Traffic burstiness | MMPP: low/medium/high burst parameter |

### Deliverable
Bab Metodologi + Bab Hasil Thesis dengan tabel komparasi lengkap.

---

## Threshold Keputusan (Decision Gates)

| Kondisi | Tindakan |
|---|---|
| GAT over-smoothing/tidak stabil > 3 layer | Switch ke JGAT atau kurangi depth |
| DQN discretization cap URLLC SLA | Naikkan resolusi tier atau dokumentasikan sebagai finding |
| Zero-shot retention GAT < 70%, GraphSAGE oke | Jadikan GraphSAGE headline architecture |
| Hasil synthetic dipertanyakan reviewer | Kalibrasi vs Colosseum O-RAN dataset sebagai validasi eksternal |

---

## Dataset & Repositori Publik

| Sumber | Kegunaan | Repo |
|---|---|---|
| Colosseum O-RAN | Validasi/kalibrasi synthetic data | `wineslab/colosseum-oran-commag-dataset` |
| DeepMIMO | Channel pre-generation (ray-traced, 18 BS) | `DeepMIMO/DeepMIMO-python` |
| Sionna NVIDIA | GPU channel simulation (offline) | `NVlabs/sionna` |
| DeepVerse 6G | Multi-modal channel dataset | `wireless-intelligence-lab/DeepVerse6G-python` |
| OpenRAN Gym | Gymnasium base env O-RAN | `wineslab/ns-o-ran-gym` |
| Path-loss reference | Implementasi NumPy TR 38.901 | `jakthra/PseudoRayTracingOSM/pathloss_38901.py` |

---

## Literatur Kunci

| # | Paper | Relevansi |
|---|---|---|
| 1 | Shao et al. (2021) arXiv:2108.05063 | Precedent terdekat: GAT+MARL untuk slicing, DQN+A2C |
| 2 | Xu et al. (2026) Sensors 26(4):1170 | GraphSAGE-MAPPO, generalisasi topologi |
| 3 | Orhan et al. (2021) arXiv:2110.07525 | GNN+DRL untuk O-RAN RIC |
| 4 | UAV-5G MARL (2025) arXiv:2512.03835 | Benchmark MAPPO vs MADDPG vs MADQN |
| 5 | Survey GNN-MARL (2024) arXiv:2404.04898 | Framing: GNN+MARL untuk resource allocation |

---

## Target Struktur Direktori Proyek

```
gnn-marl-network-slicing/
├── envs/
│   ├── network_slicing_env.py    # Gymnasium single-agent env
│   ├── multi_agent_env.py        # PettingZoo ParallelEnv
│   └── channel_model.py          # TR 38.901 path-loss + fading
├── traffic/
│   ├── poisson_traffic.py        # URLLC traffic generator
│   └── mmpp_traffic.py           # eMBB bursty traffic generator
├── gnn/
│   ├── gat_backbone.py           # GAT (GATv2Conv, 2-3 layer)
│   ├── sage_backbone.py          # GraphSAGE (inductive)
│   └── gcn_backbone.py           # GCN (baseline)
├── agents/
│   ├── dqn_agent.py              # Dueling DQN
│   ├── ppo_agent.py              # PPO
│   └── rllib_module.py           # Custom RLModule untuk RLlib
├── training/
│   ├── train_baselines.py        # Training B1 & B2
│   └── train_proposed.py         # Training GNN-MARL
├── evaluation/
│   ├── convergence_eval.py       # Fase 2a
│   ├── network_perf_eval.py      # Fase 2b
│   └── zero_shot_eval.py         # Fase 2c
├── ablation/
│   └── backbone_ablation.py      # Fase 3
├── data/
│   ├── channel_db/               # Pre-generated channel matrices (.npy)
│   └── colosseum_traces/         # Colosseum O-RAN reference CSV
├── results/
│   ├── figures/                  # Plot reward, performa, zero-shot
│   └── tables/                   # Tabel komparasi LaTeX-ready
├── configs/
│   └── experiment_config.yaml    # Hyperparameter & random seed
├── requirements.txt
└── README.md
```

---

## Catatan Penting

- **Lock seed**: numpy, torch, ray — semua di config untuk reproducibility
- **SLA threshold**: tetapkan eksplisit, justifikasi sesuai time granularity simulasi
- **Sionna isolation**: jangan co-import TensorFlow + PyTorch — gunakan env Python terpisah untuk pre-generate saja
- **Synthetic validity**: hasil synthetic = bukti algoritmik, bukan jaminan performa deployment nyata
- **Library update**: cek ulang kompatibilitas PyG/PyTorch/Ray saat implementasi — versi di atas = stable per Juni 2026
