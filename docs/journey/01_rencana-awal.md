[← Index](00_INDEX.md) | [Run pertama (v1) →](02_run1-v1-uncoupled.md)

# 01 — Rencana Penelitian Awal

## Problem statement

Network slicing 5G/6G membagi satu jaringan fisik jadi beberapa "slice" virtual dengan kebutuhan berbeda — dalam riset ini dua slice: **eMBB** (enhanced Mobile Broadband, butuh throughput tinggi) dan **URLLC** (Ultra-Reliable Low-Latency Communication, butuh delay sangat rendah dan reliabilitas sangat tinggi). Setiap base station (gNB) harus membagi Physical Resource Block (PRB) miliknya antara dua slice ini, setiap slot waktu, sambil mempertimbangkan interferensi dari gNB tetangga (frequency reuse-1). Ini adalah masalah **dynamic resource allocation** multi-agen: 5 gNB, masing-masing memutuskan alokasinya sendiri, tapi keputusan satu gNB memengaruhi kualitas sinyal gNB lain lewat interferensi co-channel.

## Hipotesis

Graph Neural Network (GNN) sebagai lapisan ekstraksi fitur di atas policy Multi-Agent RL (MARL) bisa menangkap struktur spasial/interferensi antar-gNB dan menghasilkan alokasi yang lebih baik dibanding agen yang independen sepenuhnya (tidak melihat kondisi tetangga) atau agen tersentralisasi (melihat semua state tapi tanpa struktur graph). Hipotesis ini hanya bisa diuji di environment di mana interferensi antar-gNB benar-benar dependen pada alokasi tetangga — jika tidak, GNN tidak punya nilai tambah teoritis (lihat 02 untuk bagaimana ini gagal duluan di v1).

## 8 algoritma yang dibandingkan

| Kategori | Algoritma | Backbone | Aksi |
|---|---|---|---|
| Proposed | `gnn-madqn_gat` | GATv2 | Diskrit (tiered) |
| Proposed | `gnn-madqn_sage` | GraphSAGE | Diskrit (tiered) |
| Proposed | `gnn-mappo_gat` | GATv2 | Diskrit (tiered) |
| Proposed | `gnn-mappo_sage` | GraphSAGE | Diskrit (tiered) |
| Baseline | `idqn` | MLP (independent, no graph) | Diskrit |
| Baseline | `central-dqn` | MLP (satu agen melihat seluruh state) | Diskrit |
| Baseline | `ippo` | MLP (independent, no graph) | Diskrit |
| Baseline | `central-ppo` | MLP (satu agen melihat seluruh state) | Diskrit |

Dua keluarga budget berbeda diperbandingkan **secara terpisah** (bukan lintas keluarga): DQN dengan replay buffer off-policy vs PPO on-policy punya sample efficiency yang secara struktural berbeda, jadi perbandingan yang adil adalah gnn-madqn vs idqn/central-dqn, dan gnn-mappo vs ippo/central-ppo.

## Formulasi

### Observasi per-agen (shape `(n_gnb, 8)`)

$$o_i^{(t)} = [\bar{\gamma}_i,\; \text{SINR}_i^{eMBB},\; \text{SINR}_i^{URLLC},\; q_i^{eMBB},\; q_i^{URLLC},\; d_i^{eMBB},\; d_i^{URLLC},\; b_i^{prev}]$$

(kolom persis berubah antar revisi — lihat file 03/04 untuk detail per versi.)

### Reward multi-objektif (bentuk awal, scalarized)

$$r_i^{(t)} = w_1\cdot\frac{T_i^{eMBB}}{T_{ref}} - w_2\cdot\mathbb{1}[\bar{D}_i^{URLLC}>D_{max}] - w_3\cdot\frac{\bar{D}_i^{URLLC}}{D_{max}} + w_4\cdot SE_i$$

Anchor bobot awal (referensi DORA): $(w_U, w_E, w_M) = (0.5, 0.4, 0.1)$ — bentuk ini yang dipakai di v1/v2 sebelum diganti jadi formulasi CMDP di v3 (lihat 04, alasan kenapa scalarized-weight rapuh terhadap starvation).

### Model kanal (3GPP TR 38.901, UMa LOS)

$$\text{PL} = 28.0 + 22\log_{10}(d_{3D}) + 20\log_{10}(f_c) \text{ dB}$$

$$R = B\log_2(1+\text{SINR}) \text{ bps (Shannon capacity)}$$

### Model traffic

- **URLLC**: proses Poisson (arrival rate tetap, paket berukuran tetap)
- **eMBB**: Markov-Modulated Poisson Process (MMPP) 2-state (burst/idle), lebih realistis untuk traffic data yang bursty dibanding Poisson murni

## Justifikasi metode dari literatur

- **CTDE (Centralized Training, Decentralized Execution) dengan parameter sharing**: setiap gNB memakai bobot jaringan yang sama (satu policy dibagi ke semua agen), standar di MARL untuk masalah homogen (semua gNB punya action space & observation space sama) — mengurangi jumlah parameter dan mempercepat konvergensi dibanding policy terpisah per-agen.
- **GATv2 vs GraphSAGE sebagai dua backbone berbeda**: GAT belajar bobot atensi per-edge (menangkap "gNB tetangga mana yang paling relevan"), sementara SAGE melakukan aggregation degree-agnostic tanpa bobot belajar per-edge — dua titik desain berbeda untuk melihat apakah atensi eksplisit membantu di masalah ini atau cukup aggregation sederhana.
- **DQN diskrit vs PPO kontinu-diskretisasi**: DQN aksi ber-level (tiered) rawan error kuantisasi tapi off-policy (sample-efficient); PPO on-policy lebih stabil tapi butuh lebih banyak step — dua budget training berbeda (lihat tabel di bawah) mencerminkan perbedaan struktural ini, bukan tuning yang tidak adil.
- **DORA sebagai anchor bobot reward awal**: dipakai sebagai titik mula multi-objective scalarization sebelum ditemukan rapuh terhadap starvation slice (lihat 02 dan 04).

## Hyperparameter

### DQN (dueling architecture, parameter sharing lintas gNB)

| Hyperparameter | Nilai |
|---|---|
| Arsitektur | Dueling DQN (value stream + advantage stream terpisah) |
| n_actions (tier alokasi PRB URLLC) | 11 |
| hidden dim | 128 |
| epsilon (awal → minimum) | 1.0 → 0.05 |
| epsilon decay | 0.9995 per step |
| learning rate | 1e-3 |
| gamma (discount) | 0.99 |
| loss | Huber loss |
| target network | **tidak dipakai** — bootstrap dari online net yang sama (di-`detach()`) |
| replay buffer size | 50,000 transisi |
| replay warm-up (`REPLAY_START`) | 1,000 step sebelum mulai belajar |
| batch size | 64 |
| budget training | 200,000 step |

### PPO (actor-critic dengan GAE, parameter sharing lintas gNB)

| Hyperparameter | Nilai |
|---|---|
| n_actions | 11 |
| hidden dim | 128 (Tanh activation) |
| learning rate | 3e-4 |
| clip epsilon | 0.2 |
| entropy coefficient | 0.01 |
| value loss coefficient | 0.5 |
| max grad norm | 0.5 |
| gamma (discount) | 0.99 |
| GAE lambda | 0.95 |
| rollout length | 512 step |
| batch size (per update) | 64 |
| budget training | 1,000,000 step |

### GNN backbone

| Backbone | Arsitektur | Input dim | Hidden | Output dim |
|---|---|---|---|---|
| GATv2 | 2 layer (`GATv2Conv`), layer 1 concat heads, layer 2 no-concat | 8 | 32, heads=4 (→128 setelah concat) | 64 |
| GraphSAGE | 2 layer (`SAGEConv`), inductive, edge_attr tidak dipakai | 8 | 64 | 64 |

## Kenapa budget DQN dan PPO berbeda (200K vs 1M step)

DQN off-policy dengan replay buffer secara empiris jauh lebih sample-efficient per-step (satu transisi dipakai berulang lewat sampling), sementara PPO on-policy hanya memakai tiap transisi sekali per update — butuh jauh lebih banyak step untuk konvergen ke performa setara. Budget berbeda ini bukan bias eksperimen; perbandingan yang metodologis sah selalu dalam keluarga budget yang sama (DQN vs DQN, PPO vs PPO), tidak pernah lintas keluarga.

## Stack teknologi

| Komponen | Versi terpasang |
|---|---|
| Python | 3.11.9 |
| PyTorch | 2.4.1+cu124 |
| PyTorch Geometric | 2.8.0 |
| Gymnasium | 1.1.0 |
| NumPy | 1.26.4 |

---

[← Index](00_INDEX.md) | [Run pertama (v1) →](02_run1-v1-uncoupled.md)
