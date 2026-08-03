# Hasil Training GNN-MARL Network Slicing (env v3: CMDP + packet-level URLLC)

*Generated otomatis oleh `scripts/analyze_results.py` dari 72 run di `results/logs/`. Untuk perbandingan lintas-seed yang benar (IQM + bootstrap CI, held-out eval episodes) lihat `results/RLIABLE.md`, dihasilkan dari `scripts/evaluate_checkpoints.py` + `scripts/rliable_report.py` — dokumen ini memakai training-time CSV logs (mixed exploration/lambda-drift), cocok untuk diagnosis konvergensi tapi bukan untuk klaim signifikansi.*

## 1. Ringkasan Eksekutif

- **Timely throughput tertinggi keseluruhan:** `gnn-mappo_gat_floornone_seed44` (31.041 Mbps, SLA 97.6%)
- **SLA satisfaction terbaik:** `gnn-mappo_gat_floornone_seed44` (97.6%)
- **Cell-edge (P5) eMBB throughput terbaik:** `gnn-madqn_sage_floornone_seed45` (1.725 Mbps)
- **Terbaik di keluarga DQN (budget setara, 200,000 step):** `gnn-madqn_sage_seed45`
- **Terbaik di keluarga PPO (budget setara, 1,000,000 step):** `gnn-mappo_gat_floornone_seed44`

## 2. Setup Eksperimen

- Jumlah gNB: **5**, bandwidth: **10 MHz**, skenario: UMa, slot: **1.0 ms**
- Episode length: **200 slot** (200 ms per episode)
- URLLC max delay (SLA deadline): **10.0 ms** — paket per-gNB di FIFO, di-drop kalau nunggu > deadline (deadline-drop) atau backlog > `buffer.urllc_max_bits` (overflow-drop). Delay yang dilaporkan (`urllc_delay_ms_mean/p95/p99`) HANYA dari paket yang **terkirim** — jadi selalu <= deadline by construction; pelanggaran SLA muncul sebagai `sla_violation_pct` (drop), bukan delay yang meledak.
- eMBB min throughput target (t_ref reward): **4.0 Mbps**
- **Objective/constraint (CMDP):** reward = w1=0.4 (eMBB throughput terkirim) + w4=0.1 (spectral efficiency) `- lambda * violation_rate`. `lambda` adalah dual variable yang **dipelajari** (bukan bobot tetap), diupdate tiap 2000 step menuju target `delta=0.12` (lr=0.01), **persist lintas episode** dalam satu run. Reward per-step di-clip ke `[-10, +10]`. **PENTING:** karena skala reward bergeser seiring lambda, `reward/step` TIDAK sebanding antar run dengan trajektori lambda berbeda — lihat §5 (dipakai sebagai diagnostik konvergensi saja, bukan §1/§4/§6).
- **PRB floor URLLC:** mode=`dynamic` (base=0.1, k=0.3, range=[0.1,0.4]) — diterapkan via action projection di `env.step()`, identik untuk 8 algoritma.
- **Coupled interference (reuse-1):** AKTIF — interferensi antar-gNB terikat pada fraksi alokasi slice tetangga.
- **UE-association bug-fix (v2->v3):** `ue_radius_m=100` — UE sekarang di-drop dalam radius ini dari gNB pemiliknya sendiri. Sebelumnya (v1/v2) posisi UE dan gNB ditarik independen dari `uniform(0, area_size)` yang sama, jadi UE index milik gNB i tidak berkorelasi jarak dengan gNB i sama sekali — bukan skenario 3GPP UMa yang valid (UE seharusnya di-drop di sel yang melayaninya). Diagnostik (`scripts/diag_topology_sweep.py`) menunjukkan ~72% gNB dulunya interference-limited (SIR negatif) akibat bug ini, tidak tergantung `area_size` (scaling area seragam menggeser path-loss own-link dan interference-link dengan konstanta yang sama, saling meniadakan di rasio SIR) dan cuma lemah tergantung `n_gnb`. Setelah fix: ~92-98% gNB punya SIR positif. **Konsekuensi metodologis: seluruh hasil v1 (`results/v1_uncoupled/`) dan v2 (`results/v2_scalarized/`) TIDAK representatif dan tidak lagi jadi baseline yang valid** — diarsip untuk referensi historis/ablasi saja. Hanya hasil v3 (dokumen ini) yang sah dilaporkan di paper.
- Budget training: DQN family = 200,000 step, PPO family = 1,000,000 step
- Run dianalisis: central-dqn_seed43, central-dqn_seed44, central-dqn_seed45, central-dqn_seed46, central-ppo_floornone_seed42, central-ppo_floornone_seed43, central-ppo_floornone_seed44, central-ppo_floornone_seed45, central-ppo_floornone_seed46, central-ppo_floorstatic_seed42, central-ppo_floorstatic_seed43, central-ppo_floorstatic_seed44, central-ppo_floorstatic_seed45, central-ppo_floorstatic_seed46, central-ppo_seed43, central-ppo_seed44, central-ppo_seed45, central-ppo_seed46, gnn-madqn_gat_seed43, gnn-madqn_gat_seed44, gnn-madqn_gat_seed45, gnn-madqn_gat_seed46, gnn-madqn_sage_floornone_seed42, gnn-madqn_sage_floornone_seed43, gnn-madqn_sage_floornone_seed44, gnn-madqn_sage_floornone_seed45, gnn-madqn_sage_floornone_seed46, gnn-madqn_sage_floorstatic_seed42, gnn-madqn_sage_floorstatic_seed43, gnn-madqn_sage_floorstatic_seed44, gnn-madqn_sage_floorstatic_seed45, gnn-madqn_sage_floorstatic_seed46, gnn-madqn_sage_seed43, gnn-madqn_sage_seed44, gnn-madqn_sage_seed45, gnn-madqn_sage_seed46, gnn-mappo_gat_floornone_seed42, gnn-mappo_gat_floornone_seed43, gnn-mappo_gat_floornone_seed44, gnn-mappo_gat_floornone_seed45, gnn-mappo_gat_floornone_seed46, gnn-mappo_gat_floorstatic_seed42, gnn-mappo_gat_floorstatic_seed43, gnn-mappo_gat_floorstatic_seed44, gnn-mappo_gat_floorstatic_seed45, gnn-mappo_gat_floorstatic_seed46, gnn-mappo_gat_seed43, gnn-mappo_gat_seed44, gnn-mappo_gat_seed45, gnn-mappo_gat_seed46, gnn-mappo_sage_seed43, gnn-mappo_sage_seed44, gnn-mappo_sage_seed45, gnn-mappo_sage_seed46, idqn_floornone_seed42, idqn_floornone_seed43, idqn_floornone_seed44, idqn_floornone_seed45, idqn_floornone_seed46, idqn_floorstatic_seed42, idqn_floorstatic_seed43, idqn_floorstatic_seed44, idqn_floorstatic_seed45, idqn_floorstatic_seed46, idqn_seed43, idqn_seed44, idqn_seed45, idqn_seed46, ippo_seed43, ippo_seed44, ippo_seed45, ippo_seed46

## 3. ⚠️ Catatan Metodologis (baca sebelum menafsirkan angka)

1. **`reward/step` bukan KPI perbandingan yang sah di bawah CMDP** (§2) — dipertahankan di §5 hanya untuk melihat apakah training konvergen, bukan untuk ranking algoritma. Ranking utama (§1, §4, §6) memakai KPI operator: `timely_throughput_mbps`, `sla_satisfaction_pct`, `embb_p5_mbps` (cell-edge), `jains_fairness`.
2. **Budget training timpang** (DQN 200K vs PPO 1M step). Perbandingan yang metodologis sah adalah **di dalam keluarga yang sama** (§6), bukan lintas keluarga.
3. **Angka di dokumen ini berasal dari training-time CSV** (rata-rata jendela 10% baris terakhir dari `MetricsLogger`), bercampur eksplorasi (epsilon/policy stokastik) dan drift lambda. Untuk klaim di paper, pakai `results/RLIABLE.md` (held-out greedy eval, multi-seed, bootstrap CI).
4. **Kalau hanya 1 seed dianalisis**, selisih kecil antar algoritma **belum bisa diklaim signifikan secara statistik** — lihat §10 dan `results/RLIABLE.md` untuk hasil multi-seed.

## 4. Tabel Hasil Utama (KPI Operator)

| algo | family | final_step | timely thr. Mbps | SLA sat. % | eMBB Mbps (P5) | delay ms (mean/p95/p99, delivered-only) | Jain's fairness | lambda | anomali |
|---|---|---|---|---|---|---|---|---|---|
| `gnn-mappo_gat_floornone_seed44` | proposed | 999,935 | 31.041 | 97.6 | 8.15 (1.36) | 0.97 / 4.12 / 7.17 | 0.570 | 0.58 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floornone_seed45` | proposed | 999,935 | 30.990 | 97.3 | 8.19 (0.00) | 0.76 / 3.79 / 7.05 | 0.547 | 0.58 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floornone_seed42` | proposed | 999,935 | 30.855 | 96.9 | 8.28 (0.00) | 0.83 / 4.07 / 7.36 | 0.531 | 0.61 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floornone_seed43` | proposed | 999,935 | 30.845 | 96.8 | 8.06 (0.23) | 0.69 / 3.84 / 7.32 | 0.553 | 0.60 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floornone_seed46` | proposed | 999,935 | 30.692 | 96.3 | 8.18 (0.00) | 0.63 / 3.88 / 7.51 | 0.541 | 0.62 | degradasi dari peak (reward) |
| `gnn-mappo_sage_seed45` | proposed | 999,935 | 30.541 | 95.8 | 8.20 (0.00) | 0.59 / 4.15 / 7.16 | 0.507 | 0.65 | - |
| `gnn-mappo_gat_seed44` | proposed | 999,935 | 30.519 | 95.8 | 8.27 (0.00) | 0.66 / 4.55 / 7.39 | 0.531 | 0.65 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floorstatic_seed43` | proposed | 999,935 | 30.505 | 95.7 | 8.08 (0.00) | 0.56 / 3.81 / 7.74 | 0.537 | 0.66 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floorstatic_seed45` | proposed | 999,935 | 30.504 | 95.7 | 8.26 (0.00) | 0.57 / 3.94 / 7.41 | 0.497 | 0.65 | degradasi dari peak (reward) |
| `ippo_seed44` | baseline | 999,935 | 30.473 | 95.6 | 8.24 (0.00) | 0.58 / 4.27 / 7.39 | 0.518 | 0.64 | degradasi dari peak (reward) |
| `ippo_seed45` | baseline | 999,935 | 30.465 | 95.5 | 8.16 (0.00) | 0.56 / 4.11 / 7.26 | 0.520 | 0.64 | degradasi dari peak (reward) |
| `gnn-mappo_gat_seed46` | proposed | 999,935 | 30.465 | 95.6 | 8.07 (0.00) | 0.59 / 4.21 / 7.44 | 0.545 | 0.63 | degradasi dari peak (reward) |
| `ippo_seed46` | baseline | 999,935 | 30.453 | 95.5 | 8.27 (0.00) | 0.57 / 4.16 / 7.50 | 0.493 | 0.64 | degradasi dari peak (reward) |
| `gnn-mappo_gat_seed45` | proposed | 999,935 | 30.420 | 95.4 | 8.05 (0.21) | 0.60 / 4.28 / 7.42 | 0.556 | 0.64 | degradasi dari peak (reward) |
| `gnn-mappo_sage_seed46` | proposed | 999,935 | 30.395 | 95.3 | 8.08 (0.62) | 0.63 / 4.51 / 7.56 | 0.560 | 0.65 | degradasi dari peak (reward) |
| `gnn-mappo_sage_seed44` | proposed | 999,935 | 30.389 | 95.3 | 8.03 (0.00) | 0.56 / 4.15 / 7.58 | 0.524 | 0.64 | degradasi dari peak (reward) |
| `ippo_seed43` | baseline | 999,935 | 30.385 | 95.3 | 8.22 (0.00) | 0.57 / 4.07 / 7.75 | 0.501 | 0.65 | degradasi dari peak (reward) |
| `gnn-mappo_sage_seed43` | proposed | 999,935 | 30.374 | 95.3 | 7.98 (1.53) | 0.63 / 4.36 / 7.76 | 0.578 | 0.64 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floorstatic_seed44` | proposed | 999,935 | 30.347 | 95.2 | 8.28 (0.00) | 0.66 / 4.42 / 7.99 | 0.534 | 0.65 | degradasi dari peak (reward) |
| `gnn-mappo_gat_seed43` | proposed | 999,935 | 30.339 | 95.2 | 8.01 (0.71) | 0.60 / 4.30 / 7.83 | 0.564 | 0.64 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floorstatic_seed46` | proposed | 999,935 | 30.310 | 95.1 | 8.35 (0.02) | 0.73 / 4.59 / 8.14 | 0.523 | 0.65 | degradasi dari peak (reward) |
| `central-ppo_seed43` | baseline | 999,935 | 30.302 | 95.1 | 7.92 (1.56) | 0.72 / 4.86 / 7.84 | 0.599 | 0.66 | degradasi dari peak (reward) |
| `central-ppo_seed45` | baseline | 999,935 | 30.294 | 95.1 | 7.87 (1.56) | 0.70 / 4.82 / 7.42 | 0.604 | 0.67 | degradasi dari peak (reward) |
| `central-ppo_seed46` | baseline | 999,935 | 30.248 | 94.9 | 7.89 (1.52) | 0.69 / 4.98 / 7.58 | 0.603 | 0.67 | degradasi dari peak (reward) |
| `central-ppo_seed44` | baseline | 999,935 | 30.241 | 94.9 | 7.87 (1.51) | 0.68 / 5.07 / 7.69 | 0.605 | 0.68 | degradasi dari peak (reward) |
| `gnn-mappo_gat_floorstatic_seed42` | proposed | 999,935 | 30.239 | 94.9 | 8.36 (0.07) | 0.75 / 4.81 / 7.94 | 0.516 | 0.65 | degradasi dari peak (reward) |
| `gnn-madqn_sage_seed45` | proposed | 199,999 | 29.741 | 93.5 | 7.39 (1.57) | 1.01 / 4.57 / 5.31 | 0.611 | 0.93 | degradasi dari peak (reward) |
| `gnn-madqn_gat_seed46` | proposed | 199,999 | 29.704 | 93.4 | 7.32 (1.43) | 1.02 / 4.66 / 5.36 | 0.608 | 0.93 | degradasi dari peak (reward) |
| `central-dqn_seed43` | baseline | 199,999 | 29.697 | 93.4 | 7.58 (1.69) | 1.10 / 4.56 / 5.20 | 0.601 | 0.93 | degradasi dari peak (reward) |
| `gnn-madqn_sage_seed43` | proposed | 199,999 | 29.682 | 93.4 | 7.58 (1.61) | 1.06 / 4.58 / 5.17 | 0.600 | 0.93 | degradasi dari peak (reward) |
| `central-dqn_seed46` | baseline | 199,999 | 29.682 | 93.3 | 7.09 (0.06) | 0.83 / 4.52 / 5.37 | 0.693 | 0.93 | degradasi dari peak (reward) |
| `gnn-madqn_sage_seed44` | proposed | 199,999 | 29.675 | 93.3 | 7.29 (1.54) | 1.03 / 4.64 / 5.21 | 0.619 | 0.93 | degradasi dari peak (reward) |
| `idqn_seed44` | baseline | 199,999 | 29.641 | 93.2 | 7.35 (1.65) | 0.95 / 4.42 / 5.24 | 0.610 | 0.93 | degradasi dari peak (reward) |
| `idqn_seed45` | baseline | 199,999 | 29.637 | 93.1 | 7.48 (0.03) | 0.69 / 4.20 / 5.57 | 0.602 | 0.93 | degradasi dari peak (reward) |
| `central-ppo_floorstatic_seed43` | baseline | 999,935 | 29.634 | 93.0 | 7.79 (1.60) | 0.70 / 5.33 / 8.63 | 0.608 | 0.76 | degradasi dari peak (reward) |
| `gnn-madqn_gat_seed45` | proposed | 199,999 | 29.628 | 93.2 | 7.42 (1.69) | 0.96 / 4.69 / 5.40 | 0.604 | 0.93 | degradasi dari peak (reward) |
| `idqn_seed46` | baseline | 199,999 | 29.607 | 93.1 | 7.28 (1.60) | 0.95 / 4.62 / 5.45 | 0.612 | 0.94 | degradasi dari peak (reward) |
| `central-dqn_seed45` | baseline | 199,999 | 29.588 | 93.0 | 6.74 (0.10) | 0.73 / 4.41 / 5.34 | 0.737 | 0.93 | degradasi dari peak (reward) |
| `gnn-madqn_sage_seed46` | proposed | 199,999 | 29.581 | 93.0 | 7.41 (1.69) | 0.98 / 4.88 / 5.47 | 0.602 | 0.93 | degradasi dari peak (reward) |
| `gnn-madqn_gat_seed44` | proposed | 199,999 | 29.570 | 93.0 | 7.10 (0.08) | 0.83 / 4.45 / 5.43 | 0.683 | 0.93 | degradasi dari peak (reward) |
| `central-ppo_floornone_seed46` | baseline | 999,935 | 29.556 | 92.8 | 7.81 (1.46) | 0.77 / 5.58 / 8.54 | 0.612 | 0.79 | degradasi dari peak (reward) |
| `idqn_seed43` | baseline | 199,999 | 29.543 | 92.8 | 7.27 (0.00) | 0.67 / 4.09 / 5.51 | 0.588 | 0.94 | degradasi dari peak (reward) |
| `gnn-madqn_gat_seed43` | proposed | 199,999 | 29.543 | 92.8 | 6.40 (0.00) | 0.71 / 4.12 / 5.28 | 0.759 | 0.94 | degradasi dari peak (reward) |
| `central-dqn_seed44` | baseline | 199,999 | 29.487 | 92.7 | 7.36 (1.56) | 0.95 / 4.84 / 5.40 | 0.610 | 0.93 | degradasi dari peak (reward) |
| `central-ppo_floorstatic_seed45` | baseline | 999,935 | 29.481 | 92.6 | 7.89 (1.63) | 0.78 / 5.73 / 8.51 | 0.604 | 0.77 | degradasi dari peak (reward) |
| `central-ppo_floorstatic_seed42` | baseline | 999,935 | 29.478 | 92.5 | 7.89 (1.63) | 0.77 / 5.82 / 8.36 | 0.603 | 0.78 | degradasi dari peak (reward) |
| `central-ppo_floorstatic_seed46` | baseline | 999,935 | 29.470 | 92.5 | 7.86 (1.61) | 0.78 / 5.67 / 8.59 | 0.609 | 0.78 | degradasi dari peak (reward) |
| `central-ppo_floorstatic_seed44` | baseline | 999,935 | 29.463 | 92.5 | 7.89 (1.42) | 0.78 / 5.82 / 8.77 | 0.609 | 0.77 | degradasi dari peak (reward) |
| `idqn_floornone_seed43` | baseline | 199,999 | 29.462 | 92.7 | 7.63 (0.00) | 0.88 / 5.01 / 7.11 | 0.624 | 0.96 | degradasi dari peak (reward) |
| `central-ppo_floornone_seed43` | baseline | 999,935 | 29.355 | 92.2 | 7.87 (1.63) | 0.89 / 5.82 / 8.95 | 0.602 | 0.81 | degradasi dari peak (reward) |
| `central-ppo_floornone_seed42` | baseline | 999,935 | 29.281 | 92.0 | 7.85 (1.50) | 0.92 / 6.09 / 8.58 | 0.613 | 0.80 | degradasi dari peak (reward) |
| `idqn_floorstatic_seed42` | baseline | 199,999 | 29.263 | 92.0 | 7.60 (0.00) | 0.80 / 4.86 / 6.37 | 0.599 | 0.95 | degradasi dari peak (reward) |
| `idqn_floorstatic_seed44` | baseline | 199,999 | 29.152 | 91.6 | 7.21 (1.63) | 0.80 / 4.65 / 5.80 | 0.614 | 0.96 | degradasi dari peak (reward) |
| `central-ppo_floornone_seed45` | baseline | 999,935 | 29.125 | 91.7 | 7.82 (1.62) | 1.09 / 6.31 / 8.94 | 0.608 | 0.88 | degradasi dari peak (reward) |
| `central-ppo_floornone_seed44` | baseline | 999,935 | 29.117 | 91.6 | 7.89 (1.57) | 1.03 / 6.32 / 9.08 | 0.607 | 0.82 | degradasi dari peak (reward) |
| `gnn-madqn_sage_floornone_seed45` | proposed | 199,999 | 29.092 | 91.4 | 7.36 (1.72) | 0.88 / 4.91 / 6.27 | 0.610 | 1.00 | degradasi dari peak (reward) |
| `idqn_floornone_seed46` | baseline | 199,999 | 28.945 | 91.0 | 7.11 (1.68) | 0.81 / 4.87 / 6.00 | 0.615 | 0.97 | degradasi dari peak (reward) |
| `idqn_floorstatic_seed45` | baseline | 199,999 | 28.918 | 90.9 | 7.23 (1.59) | 0.95 / 5.20 / 6.44 | 0.610 | 0.96 | degradasi dari peak (reward) |
| `gnn-madqn_sage_floorstatic_seed42` | proposed | 199,999 | 28.827 | 90.7 | 7.52 (1.51) | 0.93 / 5.01 / 6.10 | 0.610 | 0.97 | degradasi dari peak (reward) |
| `idqn_floornone_seed42` | baseline | 199,999 | 28.719 | 90.4 | 7.42 (1.66) | 1.04 / 5.15 / 6.18 | 0.608 | 0.96 | degradasi dari peak (reward) |
| `gnn-madqn_sage_floorstatic_seed45` | proposed | 199,999 | 28.524 | 89.8 | 7.52 (1.72) | 1.15 / 5.72 / 6.74 | 0.602 | 0.96 | SLA rendah, degradasi dari peak (reward) |
| `idqn_floorstatic_seed46` | baseline | 199,999 | 28.490 | 89.6 | 7.45 (1.61) | 1.15 / 5.71 / 6.80 | 0.607 | 0.96 | SLA rendah, degradasi dari peak (reward) |
| `idqn_floornone_seed45` | baseline | 199,999 | 28.460 | 89.6 | 7.44 (1.62) | 1.17 / 5.86 / 6.97 | 0.606 | 0.97 | SLA rendah, degradasi dari peak (reward) |
| `idqn_floornone_seed44` | baseline | 199,999 | 28.412 | 89.5 | 7.54 (1.59) | 1.18 / 5.83 / 6.94 | 0.603 | 0.96 | SLA rendah, degradasi dari peak (reward) |
| `gnn-madqn_sage_floornone_seed42` | proposed | 199,999 | 28.221 | 88.8 | 7.38 (1.60) | 0.99 / 5.19 / 6.37 | 0.616 | 1.02 | SLA rendah, degradasi dari peak (reward) |
| `gnn-madqn_sage_floorstatic_seed46` | proposed | 199,999 | 28.104 | 88.6 | 7.31 (1.66) | 1.34 / 6.09 / 7.36 | 0.610 | 0.97 | SLA rendah, degradasi dari peak (reward) |
| `gnn-madqn_sage_floornone_seed43` | proposed | 199,999 | 28.049 | 88.3 | 7.27 (1.71) | 1.13 / 5.73 / 6.92 | 0.609 | 0.97 | SLA rendah, degradasi dari peak (reward) |
| `idqn_floorstatic_seed43` | baseline | 199,999 | 27.994 | 88.1 | 7.76 (0.01) | 1.00 / 5.61 / 7.62 | 0.564 | 0.99 | SLA rendah, degradasi dari peak (reward) |
| `gnn-madqn_sage_floornone_seed44` | proposed | 199,999 | 27.788 | 87.6 | 7.56 (1.68) | 1.53 / 6.47 / 7.43 | 0.603 | 0.99 | SLA rendah, degradasi dari peak (reward) |
| `gnn-madqn_sage_floorstatic_seed44` | proposed | 199,999 | 27.639 | 87.1 | 7.57 (1.66) | 1.57 / 6.31 / 7.33 | 0.600 | 0.96 | SLA rendah, degradasi dari peak (reward) |
| `gnn-madqn_sage_floorstatic_seed43` | proposed | 199,999 | 27.483 | 86.7 | 7.67 (1.70) | 1.61 / 6.68 / 7.68 | 0.598 | 0.96 | SLA rendah, degradasi dari peak (reward) |
| `gnn-madqn_sage_floornone_seed46` | proposed | 199,999 | 27.078 | 85.3 | 7.18 (1.60) | 1.29 / 5.57 / 7.19 | 0.613 | 0.98 | SLA rendah, degradasi dari peak (reward) |

## 5. Analisis Tren per Algoritma (diagnostik konvergensi — reward/step, bukan KPI perbandingan)

Tren dihitung dari kemiringan (slope) `reward_per_step` (MA-100 ternormalisasi) pada 25% baris terakhir training. **Reward di sini tidak sebanding antar algoritma** (§2, §3) — pakai kolom ini hanya untuk cek training sudah konvergen/stabil sebelum ambil metrik final-window.

| algo | trend | peak reward/step (step) | final reward/step | convergence step |
|---|---|---|---|---|
| `central-dqn_seed43` | improving | 1.2408 (step 77,399) | 1.1478 | 29,999 |
| `central-dqn_seed44` | improving | 1.1929 (step 77,199) | 1.0512 | 19,599 |
| `central-dqn_seed45` | plateau | 1.1749 (step 76,999) | 1.0058 | 1,599 |
| `central-dqn_seed46` | improving | 1.2083 (step 76,799) | 1.0748 | 1,399 |
| `central-ppo_floornone_seed42` | plateau | 1.1866 (step 436,223) | 1.0953 | 17,919 |
| `central-ppo_floornone_seed43` | plateau | 1.1811 (step 436,735) | 1.1020 | 19,967 |
| `central-ppo_floornone_seed44` | plateau | 1.1880 (step 435,711) | 1.0985 | 19,455 |
| `central-ppo_floornone_seed45` | plateau | 1.1664 (step 436,223) | 1.0814 | 8,703 |
| `central-ppo_floornone_seed46` | plateau | 1.1932 (step 435,199) | 1.0985 | 3,071 |
| `central-ppo_floorstatic_seed42` | plateau | 1.1982 (step 436,223) | 1.1089 | 16,895 |
| `central-ppo_floorstatic_seed43` | plateau | 1.1905 (step 436,735) | 1.1000 | 19,967 |
| `central-ppo_floorstatic_seed44` | plateau | 1.1896 (step 435,711) | 1.1122 | 19,455 |
| `central-ppo_floorstatic_seed45` | plateau | 1.1942 (step 436,223) | 1.1127 | 17,407 |
| `central-ppo_floorstatic_seed46` | plateau | 1.1857 (step 435,199) | 1.1053 | 3,071 |
| `central-ppo_seed43` | plateau | 1.2154 (step 436,735) | 1.1415 | 19,967 |
| `central-ppo_seed44` | plateau | 1.2042 (step 435,711) | 1.1294 | 3,583 |
| `central-ppo_seed45` | plateau | 1.2201 (step 436,223) | 1.1326 | 19,455 |
| `central-ppo_seed46` | plateau | 1.2123 (step 435,199) | 1.1331 | 1,535 |
| `gnn-madqn_gat_seed43` | degrading | 1.2286 (step 77,399) | 0.9859 | 1,799 |
| `gnn-madqn_gat_seed44` | improving | 1.1737 (step 3,599) | 1.0713 | 1,799 |
| `gnn-madqn_gat_seed45` | plateau | 1.3018 (step 76,999) | 1.1372 | 1,599 |
| `gnn-madqn_gat_seed46` | improving | 1.3062 (step 76,799) | 1.1367 | 1,199 |
| `gnn-madqn_sage_floornone_seed42` | improving | 1.2756 (step 199) | 1.1014 | 399 |
| `gnn-madqn_sage_floornone_seed43` | improving | 1.2558 (step 77,399) | 1.0777 | 20,399 |
| `gnn-madqn_sage_floornone_seed44` | improving | 1.2144 (step 72,999) | 1.0943 | 1,799 |
| `gnn-madqn_sage_floornone_seed45` | improving | 1.1886 (step 63,999) | 1.0774 | 1,399 |
| `gnn-madqn_sage_floornone_seed46` | plateau | 1.1871 (step 1,399) | 1.0199 | 199 |
| `gnn-madqn_sage_floorstatic_seed42` | improving | 1.2746 (step 77,599) | 1.1144 | 399 |
| `gnn-madqn_sage_floorstatic_seed43` | improving | 1.2695 (step 77,399) | 1.1185 | 19,999 |
| `gnn-madqn_sage_floorstatic_seed44` | plateau | 1.2305 (step 31,599) | 1.0484 | 1,799 |
| `gnn-madqn_sage_floorstatic_seed45` | improving | 1.3101 (step 76,999) | 1.1168 | 1,599 |
| `gnn-madqn_sage_floorstatic_seed46` | improving | 1.2816 (step 76,799) | 1.0914 | 999 |
| `gnn-madqn_sage_seed43` | plateau | 1.3114 (step 77,399) | 1.1178 | 19,999 |
| `gnn-madqn_sage_seed44` | plateau | 1.3046 (step 77,199) | 1.1446 | 29,799 |
| `gnn-madqn_sage_seed45` | improving | 1.3262 (step 76,999) | 1.1484 | 19,999 |
| `gnn-madqn_sage_seed46` | improving | 1.2885 (step 76,799) | 1.1460 | 1,199 |
| `gnn-mappo_gat_floornone_seed42` | plateau | 1.2872 (step 436,223) | 1.2018 | 3,583 |
| `gnn-mappo_gat_floornone_seed43` | degrading | 1.2865 (step 436,735) | 1.1782 | 3,583 |
| `gnn-mappo_gat_floornone_seed44` | plateau | 1.2600 (step 435,711) | 1.1867 | 3,583 |
| `gnn-mappo_gat_floornone_seed45` | plateau | 1.3235 (step 436,223) | 1.1836 | 3,071 |
| `gnn-mappo_gat_floornone_seed46` | plateau | 1.2694 (step 435,199) | 1.1906 | 1,535 |
| `gnn-mappo_gat_floorstatic_seed42` | plateau | 1.2565 (step 436,223) | 1.1911 | 4,095 |
| `gnn-mappo_gat_floorstatic_seed43` | plateau | 1.2761 (step 436,735) | 1.1592 | 3,583 |
| `gnn-mappo_gat_floorstatic_seed44` | plateau | 1.2819 (step 435,711) | 1.1911 | 3,583 |
| `gnn-mappo_gat_floorstatic_seed45` | degrading | 1.2585 (step 436,223) | 1.1768 | 3,071 |
| `gnn-mappo_gat_floorstatic_seed46` | plateau | 1.2596 (step 95,743) | 1.1903 | 1,535 |
| `gnn-mappo_gat_seed43` | plateau | 1.2632 (step 436,735) | 1.1538 | 3,583 |
| `gnn-mappo_gat_seed44` | plateau | 1.2599 (step 96,255) | 1.1944 | 3,583 |
| `gnn-mappo_gat_seed45` | plateau | 1.2475 (step 436,223) | 1.1646 | 3,071 |
| `gnn-mappo_gat_seed46` | plateau | 1.2673 (step 435,199) | 1.1544 | 1,535 |
| `gnn-mappo_sage_seed43` | plateau | 1.2809 (step 436,735) | 1.1510 | 3,583 |
| `gnn-mappo_sage_seed44` | degrading | 1.2465 (step 435,711) | 1.1557 | 3,583 |
| `gnn-mappo_sage_seed45` | plateau | 1.2396 (step 942,591) | 1.1798 | 3,071 |
| `gnn-mappo_sage_seed46` | plateau | 1.2515 (step 435,199) | 1.1634 | 1,535 |
| `idqn_floornone_seed42` | improving | 1.3277 (step 199) | 1.0981 | 399 |
| `idqn_floornone_seed43` | improving | 1.2047 (step 77,399) | 1.1010 | 24,799 |
| `idqn_floornone_seed44` | improving | 1.2845 (step 77,199) | 1.1176 | 24,599 |
| `idqn_floornone_seed45` | improving | 1.2884 (step 76,999) | 1.0905 | 1,599 |
| `idqn_floornone_seed46` | plateau | 1.2802 (step 76,799) | 1.0883 | 799 |
| `idqn_floorstatic_seed42` | improving | 1.3139 (step 199) | 1.1209 | 399 |
| `idqn_floorstatic_seed43` | improving | 1.2292 (step 77,399) | 1.0962 | 25,399 |
| `idqn_floorstatic_seed44` | plateau | 1.2647 (step 77,199) | 1.0849 | 1,999 |
| `idqn_floorstatic_seed45` | plateau | 1.2564 (step 76,999) | 1.0943 | 1,599 |
| `idqn_floorstatic_seed46` | improving | 1.2762 (step 76,799) | 1.1099 | 999 |
| `idqn_seed43` | plateau | 1.1796 (step 175,399) | 1.0777 | 19,799 |
| `idqn_seed44` | plateau | 1.2957 (step 77,199) | 1.1192 | 19,399 |
| `idqn_seed45` | improving | 1.2057 (step 65,599) | 1.0662 | 1,399 |
| `idqn_seed46` | improving | 1.3283 (step 76,799) | 1.1241 | 1,199 |
| `ippo_seed43` | plateau | 1.2674 (step 436,735) | 1.1748 | 3,583 |
| `ippo_seed44` | plateau | 1.2648 (step 435,711) | 1.1832 | 3,583 |
| `ippo_seed45` | plateau | 1.2636 (step 436,223) | 1.1717 | 3,071 |
| `ippo_seed46` | plateau | 1.2639 (step 435,199) | 1.1829 | 1,535 |

## 6. Proposed vs Baseline per Keluarga (budget setara, KPI operator)

### Keluarga DQN (@ 200,000 step)

| algo | family | timely thr. Mbps | SLA % | eMBB P5 Mbps |
|---|---|---|---|---|
| `gnn-madqn_sage_seed45` | proposed | 29.741 | 93.5 | 1.573 |
| `gnn-madqn_gat_seed46` | proposed | 29.704 | 93.4 | 1.427 |
| `central-dqn_seed43` | baseline | 29.697 | 93.4 | 1.688 |
| `gnn-madqn_sage_seed43` | proposed | 29.682 | 93.4 | 1.613 |
| `central-dqn_seed46` | baseline | 29.682 | 93.3 | 0.058 |
| `gnn-madqn_sage_seed44` | proposed | 29.675 | 93.3 | 1.536 |
| `idqn_seed44` | baseline | 29.641 | 93.2 | 1.650 |
| `idqn_seed45` | baseline | 29.637 | 93.1 | 0.025 |
| `gnn-madqn_gat_seed45` | proposed | 29.628 | 93.2 | 1.689 |
| `idqn_seed46` | baseline | 29.607 | 93.1 | 1.603 |
| `central-dqn_seed45` | baseline | 29.588 | 93.0 | 0.096 |
| `gnn-madqn_sage_seed46` | proposed | 29.581 | 93.0 | 1.689 |
| `gnn-madqn_gat_seed44` | proposed | 29.570 | 93.0 | 0.076 |
| `idqn_seed43` | baseline | 29.543 | 92.8 | 0.002 |
| `gnn-madqn_gat_seed43` | proposed | 29.543 | 92.8 | 0.000 |
| `central-dqn_seed44` | baseline | 29.487 | 92.7 | 1.560 |
| `idqn_floornone_seed43` | baseline | 29.462 | 92.7 | 0.000 |
| `idqn_floorstatic_seed42` | baseline | 29.263 | 92.0 | 0.000 |
| `idqn_floorstatic_seed44` | baseline | 29.152 | 91.6 | 1.631 |
| `gnn-madqn_sage_floornone_seed45` | proposed | 29.092 | 91.4 | 1.725 |
| `idqn_floornone_seed46` | baseline | 28.945 | 91.0 | 1.677 |
| `idqn_floorstatic_seed45` | baseline | 28.918 | 90.9 | 1.591 |
| `gnn-madqn_sage_floorstatic_seed42` | proposed | 28.827 | 90.7 | 1.511 |
| `idqn_floornone_seed42` | baseline | 28.719 | 90.4 | 1.656 |
| `gnn-madqn_sage_floorstatic_seed45` | proposed | 28.524 | 89.8 | 1.721 |
| `idqn_floorstatic_seed46` | baseline | 28.490 | 89.6 | 1.609 |
| `idqn_floornone_seed45` | baseline | 28.460 | 89.6 | 1.620 |
| `idqn_floornone_seed44` | baseline | 28.412 | 89.5 | 1.594 |
| `gnn-madqn_sage_floornone_seed42` | proposed | 28.221 | 88.8 | 1.597 |
| `gnn-madqn_sage_floorstatic_seed46` | proposed | 28.104 | 88.6 | 1.656 |
| `gnn-madqn_sage_floornone_seed43` | proposed | 28.049 | 88.3 | 1.709 |
| `idqn_floorstatic_seed43` | baseline | 27.994 | 88.1 | 0.010 |
| `gnn-madqn_sage_floornone_seed44` | proposed | 27.788 | 87.6 | 1.679 |
| `gnn-madqn_sage_floorstatic_seed44` | proposed | 27.639 | 87.1 | 1.660 |
| `gnn-madqn_sage_floorstatic_seed43` | proposed | 27.483 | 86.7 | 1.705 |
| `gnn-madqn_sage_floornone_seed46` | proposed | 27.078 | 85.3 | 1.603 |

**Pemenang keluarga DQN (timely throughput): `gnn-madqn_sage_seed45` (PROPOSED MENANG)** — verifikasi klaim ini di `results/RLIABLE.md` (CI overlap = 'comparable', bukan 'menang').

### Keluarga PPO (@ 1,000,000 step)

| algo | family | timely thr. Mbps | SLA % | eMBB P5 Mbps |
|---|---|---|---|---|
| `gnn-mappo_gat_floornone_seed44` | proposed | 31.041 | 97.6 | 1.361 |
| `gnn-mappo_gat_floornone_seed45` | proposed | 30.990 | 97.3 | 0.001 |
| `gnn-mappo_gat_floornone_seed42` | proposed | 30.855 | 96.9 | 0.000 |
| `gnn-mappo_gat_floornone_seed43` | proposed | 30.845 | 96.8 | 0.226 |
| `gnn-mappo_gat_floornone_seed46` | proposed | 30.692 | 96.3 | 0.000 |
| `gnn-mappo_sage_seed45` | proposed | 30.541 | 95.8 | 0.000 |
| `gnn-mappo_gat_seed44` | proposed | 30.519 | 95.8 | 0.000 |
| `gnn-mappo_gat_floorstatic_seed43` | proposed | 30.505 | 95.7 | 0.000 |
| `gnn-mappo_gat_floorstatic_seed45` | proposed | 30.504 | 95.7 | 0.000 |
| `ippo_seed44` | baseline | 30.473 | 95.6 | 0.000 |
| `ippo_seed45` | baseline | 30.465 | 95.5 | 0.000 |
| `gnn-mappo_gat_seed46` | proposed | 30.465 | 95.6 | 0.000 |
| `ippo_seed46` | baseline | 30.453 | 95.5 | 0.000 |
| `gnn-mappo_gat_seed45` | proposed | 30.420 | 95.4 | 0.209 |
| `gnn-mappo_sage_seed46` | proposed | 30.395 | 95.3 | 0.619 |
| `gnn-mappo_sage_seed44` | proposed | 30.389 | 95.3 | 0.000 |
| `ippo_seed43` | baseline | 30.385 | 95.3 | 0.000 |
| `gnn-mappo_sage_seed43` | proposed | 30.374 | 95.3 | 1.526 |
| `gnn-mappo_gat_floorstatic_seed44` | proposed | 30.347 | 95.2 | 0.000 |
| `gnn-mappo_gat_seed43` | proposed | 30.339 | 95.2 | 0.713 |
| `gnn-mappo_gat_floorstatic_seed46` | proposed | 30.310 | 95.1 | 0.017 |
| `central-ppo_seed43` | baseline | 30.302 | 95.1 | 1.559 |
| `central-ppo_seed45` | baseline | 30.294 | 95.1 | 1.556 |
| `central-ppo_seed46` | baseline | 30.248 | 94.9 | 1.515 |
| `central-ppo_seed44` | baseline | 30.241 | 94.9 | 1.506 |
| `gnn-mappo_gat_floorstatic_seed42` | proposed | 30.239 | 94.9 | 0.067 |
| `central-ppo_floorstatic_seed43` | baseline | 29.634 | 93.0 | 1.603 |
| `central-ppo_floornone_seed46` | baseline | 29.556 | 92.8 | 1.457 |
| `central-ppo_floorstatic_seed45` | baseline | 29.481 | 92.6 | 1.628 |
| `central-ppo_floorstatic_seed42` | baseline | 29.478 | 92.5 | 1.631 |
| `central-ppo_floorstatic_seed46` | baseline | 29.470 | 92.5 | 1.610 |
| `central-ppo_floorstatic_seed44` | baseline | 29.463 | 92.5 | 1.424 |
| `central-ppo_floornone_seed43` | baseline | 29.355 | 92.2 | 1.628 |
| `central-ppo_floornone_seed42` | baseline | 29.281 | 92.0 | 1.497 |
| `central-ppo_floornone_seed45` | baseline | 29.125 | 91.7 | 1.623 |
| `central-ppo_floornone_seed44` | baseline | 29.117 | 91.6 | 1.568 |

**Pemenang keluarga PPO (timely throughput): `gnn-mappo_gat_floornone_seed44` (PROPOSED MENANG)** — verifikasi klaim ini di `results/RLIABLE.md` (CI overlap = 'comparable', bukan 'menang').

## 7. Analisis KPI Jaringan

- Timely throughput tertinggi: `gnn-mappo_gat_floornone_seed44` (31.041 Mbps)
- Fairness (Jain's index) tertinggi: `gnn-madqn_gat_seed43` (0.759)
- Cell-edge (P5) eMBB tertinggi: `gnn-madqn_sage_floornone_seed45` (1.725 Mbps)
- **Delay yang dilaporkan (§4) hanya untuk paket yang terkirim** (deadline-drop menjamin <= deadline). Kualitas URLLC nyata dibaca dari `sla_satisfaction_pct` (drop rate), bukan delay.

## 8. Temuan & Anomali (Diagnosis)

**Drop didominasi overflow (buffer penuh), bukan deadline** pada `central-dqn_seed43`, `central-dqn_seed44`, `central-dqn_seed45`, `central-dqn_seed46`, `central-ppo_floornone_seed42`, `central-ppo_floornone_seed43`, `central-ppo_floornone_seed44`, `central-ppo_floornone_seed45`, `central-ppo_floornone_seed46`, `central-ppo_floorstatic_seed42`, `central-ppo_floorstatic_seed43`, `central-ppo_floorstatic_seed44`, `central-ppo_floorstatic_seed45`, `central-ppo_floorstatic_seed46`, `central-ppo_seed43`, `central-ppo_seed44`, `central-ppo_seed45`, `central-ppo_seed46`, `gnn-madqn_gat_seed43`, `gnn-madqn_gat_seed44`, `gnn-madqn_gat_seed45`, `gnn-madqn_gat_seed46`, `gnn-madqn_sage_floornone_seed42`, `gnn-madqn_sage_floornone_seed43`, `gnn-madqn_sage_floornone_seed44`, `gnn-madqn_sage_floornone_seed45`, `gnn-madqn_sage_floornone_seed46`, `gnn-madqn_sage_floorstatic_seed42`, `gnn-madqn_sage_floorstatic_seed43`, `gnn-madqn_sage_floorstatic_seed44`, `gnn-madqn_sage_floorstatic_seed45`, `gnn-madqn_sage_floorstatic_seed46`, `gnn-madqn_sage_seed43`, `gnn-madqn_sage_seed44`, `gnn-madqn_sage_seed45`, `gnn-madqn_sage_seed46`, `gnn-mappo_gat_floornone_seed42`, `gnn-mappo_gat_floornone_seed43`, `gnn-mappo_gat_floornone_seed44`, `gnn-mappo_gat_floornone_seed45`, `gnn-mappo_gat_floornone_seed46`, `gnn-mappo_gat_floorstatic_seed42`, `gnn-mappo_gat_floorstatic_seed43`, `gnn-mappo_gat_floorstatic_seed44`, `gnn-mappo_gat_floorstatic_seed45`, `gnn-mappo_gat_floorstatic_seed46`, `gnn-mappo_gat_seed43`, `gnn-mappo_gat_seed44`, `gnn-mappo_gat_seed45`, `gnn-mappo_gat_seed46`, `gnn-mappo_sage_seed43`, `gnn-mappo_sage_seed44`, `gnn-mappo_sage_seed45`, `gnn-mappo_sage_seed46`, `idqn_floornone_seed42`, `idqn_floornone_seed43`, `idqn_floornone_seed44`, `idqn_floornone_seed45`, `idqn_floornone_seed46`, `idqn_floorstatic_seed42`, `idqn_floorstatic_seed43`, `idqn_floorstatic_seed44`, `idqn_floorstatic_seed45`, `idqn_floorstatic_seed46`, `idqn_seed43`, `idqn_seed44`, `idqn_seed45`, `idqn_seed46`, `ippo_seed43`, `ippo_seed44`, `ippo_seed45`, `ippo_seed46` — pertimbangkan menaikkan `buffer.urllc_max_bits` kalau overflow tidak diinginkan sebagai mekanisme utama (deadline-drop lebih representatif untuk SLA URLLC).

## 9. Kesimpulan: Hasil Terbaik

- **Timely throughput tertinggi lintas semua run:** `gnn-mappo_gat_floornone_seed44` (31.041 Mbps)
- **Terbaik keluarga DQN (adil, budget sama):** `gnn-madqn_sage_seed45`
- **Terbaik keluarga PPO (adil, budget sama):** `gnn-mappo_gat_floornone_seed44`
- **SLA satisfaction terbaik:** `gnn-mappo_gat_floornone_seed44` (97.6%)

**Catatan kehati-hatian:** angka di atas dari training-time CSV (1 seed per run kalau cuma 1 log per algo ada di `results/logs/`). Klaim "algoritma X signifikan lebih baik dari Y" harus dirujuk ke `results/RLIABLE.md` (multi-seed, held-out eval, bootstrap CI) — lihat §10.

## 10. Rekomendasi Sebelum Paper

1. ~~Ganti reward linear-clip dengan log-scale~~ — **superseded**: v3 memindahkan delay/violation dari reward term ke CMDP constraint (§2), jadi isu saturasi tidak relevan lagi.
2. ~~Tambah PRB floor URLLC~~ — **sudah diterapkan** (§2, action projection di `env.step()`).
3. **Verifikasi hasil di `results/RLIABLE.md`** — IQM + bootstrap CI per keluarga budget, bandingkan tiap varian GNN vs baseline sepadan sebelum klaim menang di paper.
4. **Kalau CI proposed vs baseline overlap**, laporkan sebagai "comparable", bukan "lebih baik" — jangan cherry-pick titik tunggal dari tabel §4/§6.
5. **Kalau semua run masih menunjukkan SLA di bawah target CMDP** (§8), pertimbangkan menaikkan `cmdp.lambda_lr` atau menurunkan `cmdp.dual_update_every` supaya lambda konvergen lebih cepat relatif ke budget training yang tersedia.

## 11. Indeks File

- Figures: `results/figures/*.png` (generated oleh `scripts/make_paper_figures.py`)
- `results/figures/paper_metrics_long.csv` — data tidy per algo/metric/step
- `results/results_summary.csv` — tabel training-time dari dokumen ini (machine-readable)
- `results/eval/*_eval.csv` — held-out greedy evaluation per episode (`evaluate_checkpoints.py`)
- `results/RLIABLE.md` — IQM + bootstrap CI, klaim statistik yang sah untuk paper
- `results/v2_scalarized/` — arsip hasil v2 (scalarized reward, fluid queue, delay-clip tanpa deadline). **INVALIDATED**: dijalankan sebelum UE-association bug-fix (§2) — bukan baseline yang sah.
- `results/v1_uncoupled/` — arsip hasil v1 (uncoupled interference, reward clip `[-2,+2]`). **INVALIDATED**: sama, dijalankan sebelum bug-fix (§2).
- `results/checkpoints/*_last.pt`, `*_best.pt` — checkpoint tiap run (`_best.pt` TIDAK dipakai untuk eval di bawah CMDP — reward turun seiring lambda naik, jadi 'best' menyesatkan; eval pakai `_last.pt`/`.pt` final)
