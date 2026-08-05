# GOAL

Rekalibrasi environment v3 (`gnn-marl-network-slicing`) sehingga constraint CMDP benar-benar mengikat pada policy **terlatih**, lalu jalankan wave v4 (8 algoritma × N seed, perlakuan identik) supaya eksperimen punya daya diskriminasi untuk menguji apakah arsitektur GNN berpengaruh — **bukan** untuk membuat GNN menang.

Wave v3 menghasilkan 35/40 `COMPARABLE` karena saturasi KPI: kalibrasi beban dilakukan terhadap policy statis, sementara policy terlatih menemukan kebijakan jauh di dalam batas `delta=0.12` (violation 3.5–4.5%, λ konvergen 0.58–1.02 dari plafon 100). Task terlalu longgar → arsitektur tidak punya ruang berbeda. Itu cacat desain eksperimen, dan itu yang diperbaiki di sini.

---

## Kriteria selesai

Wave v4 dinyatakan **selesai** hanya jika **semua** gate di bawah lolos. Kriteria ini dikunci **sebelum** wave dijalankan (pre-registrasi) dan **tidak boleh diubah setelah melihat hasil**.

### A. Gate kalibrasi (harus lolos SEBELUM wave penuh)

| # | Kriteria | Ambang | Cara ukur |
|---|---|---|---|
| A1 | Constraint mengikat pada policy terlatih | violation held-out baseline referensi ∈ `[0.7·δ, 1.0·δ]` | latih 1 baseline referensi (`ippo`) sampai konvergen, ukur held-out |
| A2a | Constraint feasible | lantai violation ruang aksi baseline referensi ≤ `δ − 1 std window` | sapuan statis per-gNB (`scripts/probe_action_floor.py`), tanpa training |
| A2b | Dual punya efek pada perilaku | violation held-out bergeser ≥ 1 std window antara run λ-beku dan run λ steady-state | dua run identik, satu dengan `lambda_lr = 0` |
| A3 | Mekanisme drop benar | `deadline_drop : overflow_drop` ≥ 3 : 1 | agregat seluruh run kalibrasi |
| A4 | Variance window stabil | std violation level-window < 2.0 pp | window = `dual_update_every` step |

> Baseline referensi dipilih **sebelum** kalibrasi dan wajib baseline (non-GNN), supaya titik operasi tidak pernah ditentukan oleh perilaku model proposed.

> **Mode pengukuran (eksplisit).** A1, A2a, A2b diukur pada policy **deterministik held-out**. A3 dan A4 diukur pada log **training (stokastik)**, karena keduanya properti dinamika training. Dual variable dikendalikan violation stokastik saat training, jadi selisih training-vs-held-out wajib dilaporkan sebagai artefak kalibrasi, bukan dibiarkan implisit.

> **Amandemen 2026-08-06** (dicatat di `runs/2026-08-05-run01/ledger.md`, sebelum wave dijalankan). A2 lama — "`λ` steady-state ≥ 5.0" — salah spesifikasi: ia mengukur **besaran** dual, bukan apakah constraint mengikat. Ronde kalibrasi 3 lolos A2 secara hampa (λ 1.03 → 18.59, perilaku policy tidak bergerak) justru karena `δ` berada **di bawah** lantai yang bisa dicapai ruang aksi `central-ppo`; pada constraint infeasible, λ → ∞ adalah perilaku Lagrangian yang benar, bukan dual yang malas. A2 dipecah jadi A2a (feasibility, syarat perlu) + A2b (sensitivitas, yang sebenarnya dimaksud). Baseline referensi A1 dipindah `central-ppo` → `ippo`: `central-ppo` menyiarkan satu tier PRB ke 5 gNB, jadi titik operasi ikut ditentukan ruang aksi tersempit di wave. `ippo` tetap non-GNN sehingga prinsip A1 utuh. Kedua amandemen dipilih atas dasar properti *task* (feasibility, daya sensitivitas), bukan properti *hasil* — tidak ada hasil per-algoritma v4 yang dilihat saat amandemen ini dibuat.

### B. Gate diskriminasi (dievaluasi SETELAH wave)

| # | Kriteria | Ambang v4 | Nilai v3 |
|---|---|---|---|
| B1 | `timely_throughput_mbps` rentang relatif antar-algoritma | ≥ 5% | 1.0% |
| B2 | `sla_satisfaction_pct` rentang antar-algoritma | ≥ 5 pp | 1.1 pp |
| B3 | `urllc_delay_p99` rentang antar-algoritma | ≥ 2 ms | 0.4 ms |
| B4 | Jumlah KPI tersaturasi | ≤ 1 dari 5 | 4 dari 5 |

### C. Gate validitas (wajib, tidak bisa ditawar)

- [ ] **C1** — Uji identitas perlakuan PASS: aksi identik → `(floor_applied, lam, delta, violation_rate, reward)` identik bit-per-bit, diuji di ≥ 3 seed × seluruh floor-mode
- [ ] **C2** — Nol hyperparameter di-set per-algoritma; seluruhnya dari satu `configs/experiment_config.yaml`
- [ ] **C3** — Keluarga DQN (200K) dan PPO (1M) tidak pernah digabung dalam klaim statistik apa pun
- [ ] **C4** — Seed ≥ 20 untuk KPI cell-edge bimodal; collapse rate dilaporkan sebagai proporsi binomial + CI Wilson/Clopper–Pearson
- [ ] **C5** — Evaluasi held-out (seed ≥ 10000) terpisah penuh dari seed training
- [ ] **C6** — Titik operasi (`lambda_arrival`, `delta`, `buffer.urllc_max_bits`) dibekukan dan di-commit **sebelum** wave penuh dijalankan

### D. Kriteria pelaporan

- [ ] **D1** — Wave v3 dan v4 **dua-duanya** dilaporkan. v3 dibingkai sebagai studi kalibrasi/pilot, bukan dihapus
- [ ] **D2** — Verdict CI dipatuhi apa adanya: CI tumpang tindih → `COMPARABLE`, bukan "superior"
- [ ] **D3** — Kekalahan proposed dilaporkan dengan aturan yang sama dengan kemenangan
- [ ] **D4** — Kalau setelah v4 masih `COMPARABLE`, itu **hasil sah**. Fallback: paper metodologi + zero-shot + stabilitas (lihat §Fallback)

> **BUKAN kriteria selesai:** "GNN-MARL mengungguli baseline." Klaim itu adalah *hipotesis yang diuji*, bukan target yang dikejar. Loop tidak boleh melanjutkan iterasi hanya karena hasilnya belum menguntungkan proposed.

---

## Batasan

- Dilarang menyentuh: `.venv\`, dan **file yang sudah ada** di `results\checkpoints\`, `results\logs\`, `results\eval\` (12 hari wall-clock GPU, tidak bisa dibuat ulang). Boleh dibaca. Dilarang menimpa/menghapus. Run baru wajib pakai tag berbeda sehingga tidak bertabrakan nama
- Interpreter: `.venv\Scripts\python.exe`. Jangan pakai `python` sistem
- Training berat: jalankan di `localhost` (mesin ini, RTX 3060 12 GB). Tidak ada VPS di jalur kerja ini — cek `nvidia-smi` dulu, jangan tabrakan dengan run yang sedang jalan
- Maks iterasi: `15`
- Setiap perubahan environment/objective **wajib** diterapkan identik ke 8 algoritma (`gnn-madqn_gat`, `gnn-madqn_sage`, `gnn-mappo_gat`, `gnn-mappo_sage`, `idqn`, `central-dqn`, `ippo`, `central-ppo`)
- Budget compute: wave penuh ± 450 GPU-jam. Jangan mulai wave sebelum Gate A lolos
- Kalibrasi wajib pakai policy **terlatih**, bukan policy statis (ini akar kegagalan v3)

---

## Larangan keras

### Operasional
- Dilarang mengedit/menghapus file test
- Dilarang mengubah definisi metrik atau ambang kriteria selesai di dokumen ini
- Dilarang `--force`, `reset --hard`, `rm -rf`, `git push`

### Integritas riset — sama mengikatnya dengan larangan operasional

1. **Dilarang menyetel parameter ke hasil.** `lambda_arrival`, `delta`, `buffer.urllc_max_bits`, `n_gnb`, `ue_radius_m` hanya boleh dipilih dengan justifikasi properti *task* (constraint mengikat, kontensi nyata, deadline jadi mekanisme dominan). **Tidak pernah** dengan justifikasi properti *hasil* (algoritma mana yang unggul).
2. **Dilarang perlakuan asimetris.** Tidak ada tambahan kapasitas, step training, kanal observasi, atau floor-mode berbeda untuk model proposed.
3. **Dilarang regime-shopping.** Kalau beberapa titik operasi diuji, laporkan seluruh grid — bukan hanya yang menguntungkan.
4. **Dilarang promosi metrik post-hoc.** Metrik primer dikunci di dokumen ini sebelum wave. `embb_p5_mbps` sudah dipromosikan jadi kelas satu **berdasarkan alasan a priori** (proksi cell-edge yang justru dilindungi mekanisme koordinasi), bukan karena hasil v3.
5. **Dilarang membuang seed yang kolaps** atau mengganti seed sampai stabil. Collapse adalah data.
6. **Dilarang mengklaim gain latensi URLLC** selama drop masih didominasi overflow (lihat Gate A3).
7. **Dilarang menaikkan derajat verdict.** `COMPARABLE` tetap `COMPARABLE`.

---

## Metrik primer (dikunci sebelum wave)

1. `timely_throughput_mbps`
2. `sla_satisfaction_pct`
3. `urllc_delay_p99`
4. `embb_p5_mbps` — cell-edge, kelas satu
5. `jains_fairness`
6. `cell_edge_collapse_rate` — **baru**, outcome diskret (Bernoulli per seed), CI Wilson

Statistik: rliable IQM + stratified bootstrap 95% CI + probability of improvement, **per keluarga budget**. Collapse rate pakai CI binomial, bukan mean kontinu.

---

## Fallback (kalau v4 tetap COMPARABLE)

Bukan kegagalan. Paper bergeser ke kontribusi metodologi + temuan negatif:

1. Benchmark slicing Gymnasium/PettingZoo terkalibrasi + **protokol kalibrasi terhadap policy terlatih** (rilis publik)
2. **Zero-shot topology generalization** (latih 5 gNB → uji 20/50) — baseline MLP/central secara struktural tidak bisa ikut; ini klaim terkuat yang jujur
3. **Collapse rate / CVaR** sebagai metrik robustness kelas satu
4. **Bukti mekanisme**: korelasi bobot atensi GAT vs matriks interferensi sebenarnya, divalidasi lewat ablasi kausal (bukan korelasi saja)
5. Batas fase beban di mana koordinasi mulai terpisah — hasil ilmiah positif meski 5 gNB ada di bawah batas itu

Target venue fallback: ICBINB, reproducibility/benchmark track, atau TNSM sebagai empirical study.

---

## Referensi internal

| Dokumen | Isi |
|---|---|
| `docs/journey/07_hasil-evaluasi-v3.md` | Hasil v3 lengkap + akar masalah saturasi |
| `research/2026-08-05-run01-gnn-marl-fair-chance.md` | Riset redesign: rezim beban, zero-shot, stabilitas, batas etika |
| `configs/experiment_config.yaml` | Sumber tunggal seluruh hyperparameter bersama |
| `scripts/test_treatment_identity.py` | Uji Gate C1 |

---

**Dibekukan pada:** `2026-08-05T22:26:37`
**Run:** `2026-08-05-run01`
**Ditandatangani manusia:** `Habb` — perubahan pada dokumen ini setelah wave dimulai wajib dicatat di ledger dengan alasan eksplisit.