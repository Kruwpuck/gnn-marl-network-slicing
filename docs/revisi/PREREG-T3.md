# PREREG-T3 — PPO lawan DQN pada anggaran setara (200K langkah)

**Status:** dikunci sebelum satu pun run T3 dilatih.
**Ditulis:** 2026-10-02, atas perintah user hari itu ("jalanin untukku eksperimennya 1 per 1",
lalu memilih T3 lebih dulu). Pembekuan 2026-09-25 untuk T3 dicabut oleh perintah itu; T8 dan
PLAN-02 tetap beku.

## 1. Pertanyaan

Pertanyaan headline paper (`docs/revisi/PAPER-CLAIM-AUDIT.md` T3): apakah agen value-based
dengan aksi PRB diskret menyamai agen policy-gradient dengan fraksi bandwidth kontinu. Gate C3
melarang menggabungkan DQN (200K) dan PPO (1M) dalam klaim statistik apa pun karena anggarannya
beda. Eksperimen ini menyamakan anggaran: PPO dihentikan di 200K.

Tidak ada prediksi arah. Ini studi benchmarking; hasilnya dilaporkan apa adanya.

## 2. Rancangan — mengikat (ditetapkan manusia 2026-09-25)

- **Tidak** melatih dengan `--steps 200000`. Config 1M persis, lalu berhenti di 200K:
  `training/train_proposed.py --steps 1000000 --stop-at 200192`. `--stop-at` mengakhiri loop
  tanpa menyentuh `--steps`; `.pt` mencatat `steps=200192, budget_steps=1000000`.
- **200192, bukan 200000:** batas rollout PPO (512) pertama yang ≥ 200K, sama dengan baris
  `step=200191` di CSV v6. Selisih 192 langkah (0,1%) dari DQN 200K dinyatakan, tidak dikoreksi.
- Arm `gnn-mappo_{gatres, gatedge, gatres-edge}`, seed 42–46 (= seed DQN v6), n=15.
- Config `configs/generated/floor_none.yaml` (md5 `a0a0b99c…`), file yang sama dengan wave v6.
  Kode training/agents/gnn/envs tidak berubah sejak 2026-08-26 (`git log`).
- `--resume --ckpt-interval 25000`, seperti `scripts/run_wave.py:71`. Tag `_v6at200k`.
- Dijalankan **satu per satu** di laptop (RTX 4060 Laptop) lewat `scripts/run_t3.ps1`.

## 3. Verifikasi melekat — hasilnya sudah diketahui sebelum run

Rancangan mewajibkan metrik di step 200K cocok dengan CSV v6 di `step=200191`, dan mismatch
dilaporkan sebagai temuan, bukan ditambal.

**Temuan, 2026-10-02, sebelum T3 jalan:** training tidak pernah memanggil `torch.manual_seed`
(grep seluruh `training/ agents/ gnn/ envs/ traffic/`; hanya skrip eval dan diagnostik yang
memanggilnya). `--seed` mengunci numpy dan reset env, **tidak** mengunci bobot awal maupun
sampling aksi. Run uji `--stop-at 1024` dengan config dan seed v6 persis sudah berbeda dari
`gnn-mappo_gatres_v6_seed42.csv` di `step=511` (15 kolom, mis. `ep_reward` 489,9016 lawan
487,5994) — sebelum update gradien pertama. Jadi verifikasi ini **gagal secara konstruksi**, di
mesin apa pun, dan berlaku juga untuk seluruh wave v3/v4/v6.

Konsekuensi, diputuskan user 2026-10-02: **tidak ditambal**. Run T3 adalah sampel independen
dari distribusi yang sama dengan v6 (config, kode, seed env sama), bukan potongan lintasan v6.
Itu tetap sah untuk perbandingan anggaran-setara, karena DQN v6 juga tidak di-seed di torch.
Cek baris 200191 tetap dijalankan dan selisihnya dilaporkan.

## 4. Pembacaan dan analisis

- `scripts/evaluate_checkpoints.py` atas `results/logs/gnn-mappo_*_v6at200k_seed*.pt`,
  `--episodes 30`, pembacaan primer per keluarga (`results/READOUT_PROVENANCE.md`): PPO sampled
  (`--stochastic`), DQN argmax — DQN memakai eval v6 yang sudah ada.
- Metrik: empat metrik primer non-zero-shot `PREREG-V6.md` §3 — `cell_edge_collapse_rate`
  (Wilson), `embb_p5_mbps`, `timely_throughput_mbps`, `sla_satisfaction_pct`. Zero-shot tidak
  termasuk.
- Perbandingan per arm, PPO@200K lawan DQN@200K, n=5 lawan n=5, IQM + bootstrap CI
  (`scripts/rliable_report.py`). Verdict hanya dari CI yang terpisah.
- Hanya angka dari file yang di-generate.

## 5. Yang tidak dilakukan

- Tidak ada penambahan seed torch, tidak ada ubahan config, tidak ada seed tambahan.
- Tidak ada edit `paper/main.tex` (T8 masih beku).
