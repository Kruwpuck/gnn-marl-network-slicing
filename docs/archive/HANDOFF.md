# Handoff — GNN-MARL Network Slicing (Env v2 "Coupled Interference")

Context dump buat lanjut kerjaan ini di CLI lain (Ubuntu). Baca semua sebelum lanjut.

## Setup

- **Remote training machine:** `Adaptive Network@100.98.9.120` (Tailscale IP), Windows, project di
  `C:\Users\Adaptive Network\Documents\Lung Cancer\gnn-marl-network-slicing`
- **GitHub repo:** `Kruwpuck/gnn-marl-network-slicing`, branch `main`
- **Akses dari sesi kerja (bukan mesin training) ke remote:** paramiko SSH, karena harness gak punya interactive shell/password prompt. Butuh bikin ulang helper di Ubuntu:
  - `ssh_run.py` — exec command tunggal, return code/stdout/stderr, timeout param.
  - `ssh_upload_exec.py` — upload file lalu exec.
  - Kredensial SSH ke remote: user harus kasih tahu ulang (host, user, password/key) — belum ada di file ini karena sensitif, tanya user langsung.
- **Push ke GitHub:** git identity di remote sudah di-set repo-local: `user.name "Habb"`, `user.email "ihabhasanainakmal0409@gmail.com"`. Push pakai PAT inline di URL (`git push https://<TOKEN>@github.com/...`), TIDAK pernah disimpan ke git config. User yang kasih PAT tiap kali push (token gak persist).

## Masalah awal (v1) yang mau diperbaiki

Hasil v1 (commit `2363711`): semua baseline (idqn dkk) ngalahin proposed (gnn-madqn/gnn-mappo). Root cause (dari baca kode, bukan tebakan):

1. **Gak ada action coupling antar-gNB** di `envs/network_slicing_env.py` `step()` — interferensi dihitung dari tx power penuh, gak tergantung alokasi slice tetangga. Problem jadi trivially per-gNB independent, GNN message-passing gak ada gunanya.
2. **Observation nyaris statis** — `_get_obs()` cuma fungsi path-loss konstan, SINR asli gak pernah masuk observation, ada kolom duplikat.
3. **Reward clip `[-2,+2]`** bikin jenuh gradien penalti delay pas delay katastrofik, `gnn-madqn` kejebak URLLC starvation.

## Solusi (plan sudah di-approve user, sudah diimplementasi)

Redesign environment jadi frequency reuse-1 dengan co-slice interference coupling — bikin koordinasi antar-gNB perlu secara nyata (bukan manipulasi angka). Arsitektur agent/GNN/training script SAMA SEKALI TIDAK diubah — supaya perbandingan tetap adil.

**Keputusan user (sudah final, jangan tanya ulang):**
- Scope: full (coupling + obs + reward, bukan cuma sebagian)
- Training budget re-run: sama seperti v1 — DQN 200K steps, PPO 1M steps, seed 42
- Hasil v1 diarsipkan ke `results/v1_uncoupled/` (git mv, sudah dilakukan, commit `98bc371`), hasil baru jadi v2

**Kriteria sukses riset (jujur, WAJIB dipegang):** proposed harus unggul di ≥1 keluarga metrik (reward/step ternormalisasi) ATAU unggul jelas di KPI (SLA/delay/fairness) SECARA JUJUR dari hasil real training. Kalau tetap kalah setelah env diperbaiki: laporkan apa adanya + diagnosis lanjutan. TIDAK ADA pemalsuan angka, tidak ada cherry-picking, tidak ada override hasil training manual.

## Perubahan kode yang SUDAH dilakukan (commit belum di-push untuk yang ini — cek status git remote dulu)

Semua sudah diupload ke remote dan lolos test (`pytest tests/` 72 passed):

### `envs/network_slicing_env.py`
- Constructor: `self.slice_coupled_interference: bool = bool(cfg["env"].get("slice_coupled_interference", True))`
- `reset()`: init `self._last_sinr_embb`, `self._last_sinr_urllc`, `self._last_delay_ms` (zeros shape `(n_gnb,)`)
- `step()`: interferensi tetangga diskalakan pakai fraksi alokasi slice tetangga (`embb_fracs`/`urllc_fracs`) kalau `slice_coupled_interference=True`; kalau False, balik ke perilaku v1 (full interference, tidak coupled) — buat ablation.
- `_compute_reward()`: pakai `np.log1p(delay_ratio)` (log-scale) ganti linear delay penalty, clip diperlonggar `[-10,10]` (safety bound saja, bukan design constraint).
- `_get_obs()`: shape tetap `(n_gnb, 8)` tapi isi diganti jadi dinamis — `[ch_gain, sinr_embb_norm, sinr_urllc_norm, q_e, q_u, last_delay_norm, neighbor_mean, prev_alloc]`.

### `configs/experiment_config.yaml`
- Tambah `env.slice_coupled_interference: true`

### `tests/test_env.py`
- Tambah `test_neighbor_allocation_couples_reward` (bukti coupling: aksi tetangga beda → reward gNB lain beda)
- Tambah `test_reward_not_saturated_by_extreme_delay` (bukti gradien delay gak jenuh: reward makin turun monoton walau delay makin ekstrem)
- Update bound reward test lama dari `[-2,2]` ke `[-10,10]`

### Tidak diubah
`agents/*`, `gnn/*`, `training/train_*.py`, `scripts/train_batch_*.py`, `training/metrics_logger.py`, `scripts/make_paper_figures.py`, `scripts/analyze_results.py`, `envs/channel_model.py`, `envs/multi_agent_env.py`.

## Status eksekusi saat handoff ini dibuat

**Smoke test (2000 steps, seed 999, sebelum commit ke run panjang):**
- `idqn_seed999.csv` — LOLOS. Reward finite 44.85-164.53/episode, SLA 77.3-95.8%. Catatan: kolom `loss` naik tajam antar episode (0.037 → 35882.45) — perlu diawasi pas full run, bukan blocker smoke test.
- `ippo_seed999.csv` — LOLOS bersih. Reward +244.771, +184.066.
- `gnn-mappo_gat_seed999.csv` — LOLOS bersih. Reward +236.45, +200.27.
- `gnn-madqn_gat_seed999.csv` — **BELUM DIPASTIKAN SELESAI saat handoff ini ditulis.** Terakhir cek: stuck di step 1199 (episode 6) beberapa menit, tapi dikonfirmasi ALIVE (bukan hang) via `tasklist` (CPU time naik 3:55→4:35) dan `nvidia-smi` (GPU util 26%, proses aktif). Ini expected-slow, bukan bug: `DQNAgent.learn()` di `agents/dqn_agent.py` loop per-item Python (bukan batched) buat forward pass GNN — udah jadi penyebab run DQN v1 makan 5-12 jam buat 200K steps, ditambah overhead perhitungan interferensi O(n²) baru dari env v2.
- **`gnn-madqn_sage` dan DQN baseline (idqn/central-dqn?) smoke test lain BELUM tentu sudah dijalankan** — cek dulu file CSV di remote `results/logs/*_seed999.csv` yang ada, jangan asumsi.

## Yang HARUS dilakukan lanjut (urutan)

1. **Cek ulang status smoke test dari awal** — jangan percaya catatan di atas mentah-mentah, sudah lewat waktu. SSH ke remote, baca `results/logs/*_seed999.csv`, pastikan semua 4 keluarga (DQN proposed, DQN baseline, PPO proposed, PPO baseline) selesai dengan reward finite, tidak NaN, delay tidak meledak instan.
2. **Kalau smoke test semua lolos:** hapus/abaikan file `*_seed999.csv` (bukan bagian dari hasil resmi), lalu **WAJIB minta konfirmasi eksplisit user dulu** sebelum mulai full run (run panjang ~2-3 hari GPU). Ini syarat yang sudah dijanjikan ke user di plan — jangan skip.
3. **Full run** (setelah user acc): urutan disarankan — PPO dulu (`scripts/train_batch_ppo.py`, ~30 menit × 4 job, PARALEL-safe vs 2 script DQN karena nama file log/checkpoint beda), lalu 2 script DQN (`train_batch_dqn_baseline.py`, `train_batch_dqn_proposed.py`, 5-12 jam × 2 job masing-masing) — PPO dan kedua DQN batch BISA jalan paralel satu sama lain (proses terpisah), tapi di dalam masing-masing script jobnya jalan sequential (isolasi crash, bukan buat speed).
4. **Setelah full run selesai:** jalankan `python scripts/make_paper_figures.py` lalu `python scripts/analyze_results.py` → hasilkan `results/RESULTS.md` v2. WAJIB masukkan tabel perbandingan v1-vs-v2 per algoritma di laporan.
5. **Commit + push v2**: pakai identity `Habb <ihabhasanainakmal0409@gmail.com>` yang udah ke-config di remote repo-local. Push butuh PAT baru dari user (token gak disimpan).

## File plan lengkap (kalau butuh detail teknis persis)

Plan asli tersimpan di `C:\Users\ihabh\.claude\plans\bikinin-script-yg-bisa-sorted-eclipse.md` di mesin Windows ini — kalau CLI Ubuntu butuh isi lengkapnya (spek kode detail, rumus interferensi, dll), minta user copy-paste isi file itu juga, karena ini cuma ringkasan.

## Catatan teknis penting (biar gak ulang kesalahan)

- SSH via paramiko: `recv()` bisa timeout (`PipeTimeout`/`TimeoutError`) padahal proses remote masih jalan normal — jangan simpulkan gagal dari exception ini doang, selalu cross-check baca file output/CSV langsung.
- `start /B ... > log.txt 2>&1` di Windows `cmd.exe` via SSH `exec_command` — child process KEMATIAN begitu channel SSH ditutup (job-object/console attach behavior Windows). Jangan pakai cara ini buat detach proses training. Solusi yang jalan: launch tiap training run sebagai SSH call foreground yang dibungkus `run_in_background: true` di level harness (bukan level Windows).
