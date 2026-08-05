# LEDGER — run 2026-08-05-run01

Append-only. Satu baris per aksi. Format:

```
## <ISO8601> | lvl <n> | <AKSI> | <HASIL> | <sha atau ->
```

`AKSI` ∈ `EXEC | SEARCH | BRIEF | RESEARCH | HUMAN-RESEARCH | BLOCKED | RESUME | COMPACT | GATE`

---

## 2026-08-05T22:26:37 | lvl 1 | GATE | G0 RECON lolos, manusia ketik LANJUT. GPU luang (836/12288 MiB), venv cuda True, claude 2.1.222 | 04f4773
## 2026-08-05T22:26:37 | lvl 1 | EXEC | FASE 1 scaffold: runs/, research/, prompts/, STATE.json, ledger.md | -
## 2026-08-05T22:45:00 | lvl 1 | EXEC | iterasi 1: calibrate_trained.py baseline referensi ippo->central-ppo (Gate A1), ambang A1 -> [0.7d,1.0d], tambah A2/A3/A4. pytest 80 passed | 4eb07ad
## 2026-08-05T22:45:00 | lvl 1 | EXEC | verifikasi gate_a234 pada data v3 (central-ppo_seed43): lam_ss=0.680, rasio drop 0.44:1, window std 2.63pp -> v3 gagal A2/A3/A4 | 4eb07ad
## 2026-08-05T22:47:00 | lvl 1 | EXEC | TEMUAN: seluruh CSV training seed42 main-wave (8 algo) header-only 335 B -- kurva training seed42 v3 hilang. Eval CSV + .pt utuh, verdict v3 tidak terpengaruh | 4eb07ad
## 2026-08-05T22:50:00 | lvl 1 | EXEC | probe statis calibrate_load.py @ config sekarang (delta=0.05, urllc_max_bits=131072): overflow 0.00% di semua frac, viol minimum 6.04% @ frac 0.80 > delta. Static tidak capai delta; keputusan: tetap uji policy terlatih (Gate A1 didefinisikan pada policy terlatih) | 4eb07ad
## 2026-08-05T23:25:00 | lvl 1 | EXEC | Gate A ronde 1 (central-ppo_calib_seed42, 1M step, 28 menit): A1 PASS 4.60% in [3.50,5.00], A3 PASS inf:1 (overflow 0.00%), A2 FAIL lam_ss=1.033, A4 FAIL 2.74pp. Log: transcripts/1-gateA-calib1-central-ppo.log | 4eb07ad
## 2026-08-05T23:30:00 | lvl 1 | EXEC | diagnosis A2: lambda_lr=0.01 * gap~0.01 = 1e-4/update, 500 update = +0.03. Dual tidak bisa mengikat by construction, berapa pun bebannya. Akar kegagalan v3 yang sebenarnya, bukan beban traffic | 4eb07ad
## 2026-08-05T23:32:00 | lvl 1 | EXEC | probe A4 vs ukuran window pada CSV ronde 1: 2.74pp@2000, 1.87@4000, 1.56@6000, 1.17@10000 (~1/sqrt(n)) | 4eb07ad
## 2026-08-05T23:35:00 | lvl 1 | EXEC | ronde 2: lambda_lr 0.01->2.0 (aturan a priori: gap satu-delta harus bawa lambda ke ambang A2=5.0 dalam 40 update budget DQN), dual_update_every 2000->5000 (SNR A4). delta/lambda_arrival/urllc_max_bits TIDAK disentuh. pytest 80 passed | 93d1dde
## 2026-08-06T00:10:00 | lvl 1 | EXEC | Gate A ronde 2 (central-ppo_calib2_seed42): A1 PASS 4.09%, A3 PASS inf:1, A4 PASS 1.61pp, A2 FAIL lam_ss=3.192 | 93d1dde
## 2026-08-06T00:15:00 | lvl 1 | EXEC | trajektori lambda ronde 2 per 100K step: 1.000 1.275 1.640 1.753 1.873 2.140 2.610 2.850 3.040 3.342 -- monoton, belum plateau. Violation datar 5.5-6.2%, gap realized ~0.0075 (bukan 0.05 yang diasumsikan aturan pertama) | 93d1dde
## 2026-08-06T00:20:00 | lvl 1 | EXEC | ronde 3: lambda_lr 2.0->14.0. Aturan sama, gap-nya sekarang terukur: lr >= (5.0-1.0)/(0.0075*40 update DQN) = 13.3. Budget DQN yang mengikat karena C2 melarang hyperparameter per-algoritma | 93d1dde
## 2026-08-06T01:00:00 | lvl 1 | EXEC | Gate A ronde 3 (central-ppo_calib3_seed42, lambda_lr=14.0): A1 4.14% PASS, A2 18.594 PASS, A3 inf:1 PASS, A4 1.62pp PASS -- 4/4 secara angka | 93d1dde
## 2026-08-06T01:05:00 | lvl 5 | BLOCKED | A2 lolos HAMPA. Perbandingan 20% step terakhir ronde 1/2/3: lam 1.03->3.19->18.59 (18x), viol 6.15->5.98->5.91 (turun 0.24pp, di bawah std window 1.61pp), eMBB 7.82/7.82/7.85, timely 29.82/29.87/29.87, p99 8.81/8.82/8.98. Dual naik 18x tanpa mengubah perilaku. Tulis handoff/STUCK.md, BERHENTI, tunggu keputusan manusia | 93d1dde
