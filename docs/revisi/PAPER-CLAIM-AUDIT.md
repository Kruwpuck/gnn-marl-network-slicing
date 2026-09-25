# Audit klaim `paper/main.tex` terhadap temuan v4 dan v6

**Dibuat 2026-09-25.** Audit ini diminta sebagai *daftar tabrakan*, bukan perbaikan. Ia hidup di
file ini — bukan di file rencana yang bisa ditimpa — supaya sepuluh temuan di bawah bisa dirujuk,
diberi status, dan ditutup satu per satu.

Yang diaudit: Abstract, Introduction, dan daftar kontribusi, terhadap apa yang benar-benar diukur
di wave v3/v4/v6 dan didiagnostik D1–D6.

---

## 0. Larangan yang mengikat sampai T8 diputuskan

> **Selama T8 belum diputuskan, tidak ada edit klaim ke `paper/main.tex`.**

T8 menentukan paper ini jenis apa — dokumen protokol atau *empirical benchmarking study*. Setiap
tabrakan lain diselesaikan dengan cara yang berbeda tergantung jawaban itu, jadi menyunting T1
atau T3 sekarang berarti menulisnya dua kali. Yang boleh dikerjakan hanyalah perbaikan **mekanis**:
mengisi nilai yang sudah beku, dan membuang sisa template IEEEtran.

Arah sementara (belum final, dibawa ke pembimbing): *empirical benchmarking study*, sejalan dengan
judul tesis "Benchmarking PPO dan DQN...".

---

## 1. Status

| # | Tabrakan | Status |
|---|---|---|
| T1 | Premis motivasi utama salah (batasan milik encoder terpusat, bukan MLP) | **DISETUJUI — BEKU** menunggu T8 |
| T2 | Equivariance disajikan sebagai keunggulan khas GNN | TERBUKA |
| T3 | Pertanyaan riset inti dilarang gerbang validitas sendiri | **DIJAWAB sebagian** — lihat §3: checkpoint @200K tidak ada, biaya 22,44 jam; seleksi `_best.pt` di hasil v4/v6 diperiksa dan **nihil** |
| T4 | "identical ... seed protocol" tidak lagi benar | TERBUKA |
| T5 | Kontribusi 2 menjanjikan pemilihan tetangga yang tidak ada | TERBUKA |
| T6 | Kontribusi 4 menjanjikan protokol yang membedakan, datanya tidak membedakan | TERBUKA |
| T7 | Kontribusi 1 mengandaikan atribusi yang akan menemukan sesuatu | TERBUKA |
| T8 | Scope menyatakan kampanye belum dijalankan | **MENUNGGU PEMBIMBING** — memblokir T1–T7 |
| T9 | Tabel reproducibility semua TBD | **SELESAI 2026-09-25** |
| T10 | Sisa template dan placeholder | **SELESAI 2026-09-25** (kecuali empat figure placeholder, sengaja) |

---

## 2. Sepuluh tabrakan

### T1 — Premis motivasi utama salah (Abstract + Intro baris 120–130)

Paper: *"The observation of each agent is encoded by a multi-layer perceptron (MLP), whose input
dimension is fixed at design time. A policy trained on a five-cell deployment simply cannot be
evaluated on a ten-cell deployment: the tensor shapes do not match."*

Data: D1 (journey 10) mengukur **`ippo`, `idqn`, `mlp-knn-ppo` equivariant sempurna** — varians
aksi 0,000e+00 lintas 120 permutasi. `ZEROSHOT*` menunjukkan keduanya menghasilkan angka di
`n_gnb=10` dan 20 tanpa retraining. Yang benar-benar tidak bisa hanya `central-*`.

Batasannya milik encoder **terpusat**, bukan MLP per-agen. Ini premis yang menyangga seluruh
motivasi paper.

**Keputusan 2026-09-25:** disetujui, "MLP" → "encoder terpusat". Penerapannya ditahan sampai T8.

### T2 — Equivariance disajikan sebagai keunggulan khas GNN (Intro baris 131–135)

Paper: *"Graph neural networks (GNNs) address exactly this gap ... a GNN encoder is
permutation-equivariant and parameter-shared across nodes, so the same weights apply to a graph of
any size."*

Data: ketujuh arsitektur per-agen equivariant, MLP termasuk. Sumbernya **parameter sharing**,
bukan message passing (journey 10). Equivariance tidak membedakan GNN dari baseline per-agen.

### T3 — Pertanyaan riset inti dilarang gerbang validitas sendiri (Intro baris 145–147)

Paper: *"whether a value-based agent forced to discretise the PRB allocation can match a
policy-gradient agent that allocates a continuous bandwidth fraction."*

Data: Gate C3 menetapkan **DQN (200K) dan PPO (1M) tidak pernah digabung dalam klaim statistik
apa pun**, karena anggaran langkahnya berbeda (`results/GATE_C.md`, verdict **PASS** dengan satu
kasus ditandai). Pertanyaan headline paper persis perbandingan lintas-keluarga itu. Sebagaimana
wave dijalankan, ia tidak bisa dijawab tanpa melanggar gerbang proyek sendiri.

Jalan keluarnya adalah perbandingan anggaran-setara. Kelayakan dan biayanya di §3.

### T4 — "identical ... seed protocol" tidak lagi benar (Abstract baris 62)

Paper: *"a controlled comparison between a value-based agent with discretised PRB tiers
(GNN-MADQN) and a policy-gradient agent with continuous bandwidth fractions (GNN-MAPPO) under an
identical environment, reward and seed protocol."*

Data: keluarga PPO diperluas ke 20 seed (42–61) pada 2026-08-18; DQN tetap 5. Gate C4 **LOLOS
untuk PPO, GAGAL untuk DQN**. Ditambah C2 **PARSIAL** — biner yang menghasilkan checkpoint v4
tidak membaca YAML. Dan pembacaan primer berbeda per keluarga (sampled PPO, argmax DQN).

### T5 — Kontribusi 2 menjanjikan pemilihan tetangga yang tidak ada (baris 176–183)

Paper: *"Rather than connecting cells by a distance threshold, we weight each edge by the
modelled interference coupling derived from the 3GPP TR 38.901 path-loss model, so that the
message-passing neighbourhood reflects the physical structure of the problem instead of a
geometric proxy."*

Data: graf **lengkap** — `build_interference_graph` memancarkan setiap pasangan gNB terurut,
diameter 1 (journey 10, P5). Tidak ada pemilihan tetangga sama sekali; tiap sel bertetangga dengan
semua sel. Lebih jauh, D2b menemukan atribut edge **tidak terpakai di 0/25 checkpoint v4**, dan
D2b v6 juga null. Bobot fisik itu masuk, lalu tidak dipakai.

### T6 — Kontribusi 4 menjanjikan protokol yang membedakan, datanya tidak membedakan (baris 187–193)

Paper: *"A zero-shot topology-generalisation protocol. We define a Performance Retention Ratio and
an experimental procedure that isolates architectural generalisation from retraining."*

Data: ablasi tiga tingkat memberi `ippo` 0,954, `mlp-knn-ppo` 0,949, GNN 0,942–0,955 — **nol
perbedaan**. Protokolnya bekerja; jawabannya null. Paper membingkainya sebagai alat yang akan
memisahkan, tanpa menyiapkan pembaca untuk hasil yang tidak memisahkan.

### T7 — Kontribusi 1 mengandaikan atribusi yang akan menemukan sesuatu (baris 167–176)

Paper: faktorisasi *"is what makes an attribution study --- separating the contribution of the
graph prior from that of the RL objective --- methodologically sound."*

Data: faktorisasinya memang sah. Tapi atribusinya sudah dijalankan dan hasilnya: KPI
`COMPARABLE`, ablasi atensi kausal null di v4 dan v6 (journey 12), informasi edge tidak terpakai.
Kalimatnya tidak salah, tapi ia menyiapkan pembaca untuk kontribusi graph prior yang datanya
tidak dukung.

### T8 — Scope menyatakan kampanye belum dijalankan (baris 200–206)

Paper, §Scope and Non-Goals: *"This paper reports a design and a protocol, not measurements. The
experimental campaign described in Section~\ref{sec:eval} has not been executed at the time of
writing, and no performance figures are claimed."*

Abstract (baris 70–72) versinya lebih lunak: *"The experimental campaign is ongoing; this paper
contributes the system model, the architecture, the benchmarking protocol and the reproducibility
package rather than empirical results."*

Data: v3 (80 run), v4 (40 run + perluasan 20 seed), v6 (75 run) semuanya selesai dan dilaporkan.

Ini bukan tabrakan klaim melainkan keusangan faktual, dan ia harus dibereskan lebih dulu karena
ia menentukan paper ini jenis apa.

### T9 — Tabel reproducibility semua TBD (baris 1010–1038) — SELESAI

Sebelas entri `TBD` padahal seluruh nilainya beku dan terbaca di `configs/experiment_config.yaml`.
Ditambah `Seeds 5` dan `Training steps 10^6` yang salah per T4.

Diisi 2026-09-25 dari config beku. Baris `Training steps` dan `Seeds` dipecah per keluarga. Baris
bobot reward ditulis apa adanya: **tidak ada `w3`** — suku URLLC ditangani sebagai constraint
CMDP, bukan bobot, jadi notasi `(w_1..w_4)` di paper memang tidak pernah cocok dengan kode
(`envs/network_slicing_env.py:97-99,522-525`).

### T10 — Sisa template dan placeholder — SELESAI sebagian

Dibuang 2026-09-25: `Vol.~XX, No.~X, Month~Year` di `\markboth`, heading "References Section" dan
"Biography Section" beserta paragraf instruksi IEEEtran, biografi contoh "Michael Shell" dan
"John Doe".

**Sengaja ditinggal:** empat `\placeholderbox` (arsitektur O-RAN, deployment fisik, agen modular,
transfer zero-shot). Itu bukan sisa template — masing-masing memuat *drawing brief* asli untuk
diagram yang belum dibuat, dan tidak satu pun bisa diambil dari `results/figures/` (isinya kurva
training, bukan diagram). Mengisinya bergantung pada T8.

**Ditandai, bukan tugasku:** `\thanks{Manuscript received ...}` dan `\thanks{Author is with the
Department of ..., email@domain.com}` (baris 26–27). Sisa template juga, tapi isinya data penulis
asli yang harus disediakan manusia.

---

## 3. T3 — kelayakan perbandingan anggaran-setara PPO@200K vs DQN@200K

Diperiksa 2026-09-25, read-only, nol training.

### Checkpoint PPO @200K tidak ada, dan tidak pernah ada

`--ckpt-interval 25000` memang dipasang di tiap job (`scripts/run_wave.py:71`), tapi
`CheckpointManager` cuma menulis dua path tetap:

```
training/metrics_logger.py:216   self.last_path = self.dir / f"{run_name}_last.pt"
training/metrics_logger.py:217   self.best_path = self.dir / f"{run_name}_best.pt"
```

`save()` (`training/metrics_logger.py:225-229`) menimpa `_last.pt` tiap interval. Snapshot 200K
sebuah run PPO 1M ditimpa 32 kali sesudahnya; yang tersisa adalah step terakhir.

`_best.pt` dipilih oleh moving-average reward, bukan oleh anggaran langkah. Memakainya sebagai
"PPO@200K" berarti seleksi bergantung-data pada step sembarang — justru melanggar premis anggaran
setara yang ingin ditegakkan.

### Biaya melatih PPO sampai 200K — diukur, bukan diekstrapolasi

Run PPO v6 yang sudah ada melewati 200K dalam perjalanan ke 1M, jadi biayanya terbaca langsung
dari kolom `elapsed_sec` di `results/logs/gnn-mappo_*_v6_seed4[2-6].csv`, pada baris pertama
dengan `step >= 200000` (step 200191 di semua run).

| arm | jam per run (5 seed) |
|---|---|
| `gatedge` | 1,44 – 1,45 |
| `gatres` | 1,51 – 1,52 |
| `gatres-edge` | 1,51 – 1,53 |

- **3 arm × 5 seed (n=15): mean 1,496 jam, median 1,516 jam, total 22,44 jam.**
- Pembanding, run PPO v6 penuh sampai 1M, 15 run yang sama: mean 7,473 jam, total 112,10 jam.
- Pembanding, DQN v6 (200K, n=15): mean 72,777 jam, total 1091,66 jam.
- Kalau PPO dipakai 20 seed (menyamai wave v6, bukan menyamai DQN): ≈ 89,8 jam.

> `elapsed_sec` adalah wall-clock di bawah konkurensi wave, bukan device-time terisolasi. Angka
> PPO@200K diukur di bawah profil konkurensi yang sama dengan wave v6, jadi valid sebagai estimasi
> biaya mengulang dengan cara yang sama. Bukti kontensinya langsung terlihat: seed 46 tiap arm DQN
> selesai ~42 jam, seed 42–45 ~79–81 jam, untuk anggaran langkah identik.

### Dua tingkat, dan biayanya beda jauh

1. **KPI training-time @200K: biaya nol.** CSV yang ada sudah memuat `embb_mbps_mean`,
   `embb_p5_mbps`, `sla_satisfaction_pct` dan seterusnya di step 200K. Tapi pembacaan primer
   proyek ini adalah eval held-out (seed >= 10000), dan `results/READOUT_PROVENANCE.md` menolak
   laporan yang tidak menyatakan protokolnya. Angka training-time indikatif saja — **tidak boleh
   menggantikan pembacaan primer**.
2. **Pembacaan primer @200K: 22,44 jam** untuk 15 run latih ulang `--steps 200000`, lalu
   `scripts/evaluate_checkpoints.py` seperti biasa.

### Pemeriksaan lanjutan: seluruh hasil v4/v6 **tidak** memakai `_best.pt` — DITUTUP

Pertanyaan yang wajar muncul setelah alasan di atas: kalau seleksi checkpoint bergantung-data
ditolak untuk T3, apakah hasil v4/v6 yang sudah dilaporkan justru mengandungnya? Diperiksa
2026-09-25 dengan grep, nol GPU. **Tidak.**

Empat skrip yang menghasilkan angka laporan semuanya membaca `results/logs/*.pt` — model
**langkah terakhir** yang ditulis `_save_model` setelah loop training selesai, bukan dari
`results/checkpoints/` sama sekali:

| skrip | sumber checkpoint |
|---|---|
| `scripts/evaluate_checkpoints.py:254` | `results/logs/*.pt` |
| `scripts/attention_analysis.py:193` | `results/logs/gnn-*_gat_v4_seed*.pt` |
| `scripts/diag_gnn_reliance.py:212` | `results/logs/gnn-*_v4_seed4*.pt` |
| `scripts/zeroshot_eval.py:133` | `results/logs/*_v4_seed*.pt` |

Satu pengecualian, dan ia tetap bukan `_best.pt`: `scripts/diag_grad_ratio.py:131` membaca
`results/checkpoints/{run}_last.pt`, karena D3 melanjutkan training untuk mengukur gradien dan
hanya file checkpoint yang membawa state optimizer dan nomor langkah.

`_best.pt` muncul tepat sekali di seluruh `scripts/`, dan itu sebuah `unlink`:
`scripts/diag_grad_ratio.py:162` membersihkan artefak scratch supaya run berikutnya tidak bisa
resume dari state yang sudah dimodifikasi D2c.

Yang menjadikan ini tertutup, bukan sekadar kebetulan: alasannya sudah dipikirkan waktu itu dan
ditulis di generator laporan sendiri — `scripts/analyze_results.py:452` menyatakan `_best.pt`
**tidak dipakai untuk eval di bawah CMDP**, karena reward turun seiring lambda naik sehingga
"best" menyesatkan.

### Rekomendasi

Pra-registrasi ditulis **hanya setelah T8 diputuskan**. Kalau papernya jadi benchmarking study,
perbandingan anggaran-setara ini adalah cara sah mencabut batasan C3, dan 22,44 jam murah untuk
itu. Kalau papernya tetap dokumen protokol, pra-registrasinya mubazir.

### Rancangan yang harus dipakai saat pra-registrasi ditulis (ditetapkan 2026-09-25)

Jangan melatih run dengan `total_steps=200_000`. Jadwal apa pun yang bergantung pada total
langkah — peluruhan epsilon, anil learning rate, ritme dual update — akan berbeda dari run 1M,
jadi hasilnya bukan "PPO pada 200K langkah" melainkan "PPO dari kurikulum yang berbeda". Itu
mengganti satu confound dengan confound lain.

Yang setara: **config 1M persis, seed sama, snapshot bertanda di langkah 200K, lalu berhenti.**

Verifikasinya melekat dan murah: cocokkan metrik yang di-log pada step 200K terhadap CSV v6 yang
sudah ada (`results/logs/gnn-mappo_*_v6_seed4[2-6].csv`, baris `step=200191`). Cocok berarti
snapshot itu sah sebagai titik 200K dari run 1M yang sama. Tidak cocok berarti training tidak
deterministik — dan itu sendiri temuan yang wajib dilaporkan, bukan gangguan yang ditambal.

---

## 4. Yang TIDAK bertabrakan

Dicatat supaya tidak ikut dibongkar saat T1–T8 dikerjakan:

- **Terminologi GATv2 sudah benar di seluruh `main.tex`** — nol kemunculan "GAT" telanjang selain
  nama sitasi `shao2021gat` dan satu contoh ilustratif di baris 657.

---

## Data & artefak

| Klaim | Sumber |
|---|---|
| Equivariance MLP per-agen (T1, T2) | `results/DIAG_*`, `docs/journey/10_diagnostik-fase-0.md` |
| Zero-shot lintas topologi tanpa retraining (T1) | `results/ZEROSHOT_v4_primary.md` |
| Gate C2 PARSIAL, C3, C4 per keluarga (T3, T4) | `results/GATE_C.md` |
| Perluasan PPO ke 20 seed (T4) | `docs/journey/09_wave-v4-dan-cacat-pembacaan.md` |
| Graf lengkap, diameter 1 (T5) | `docs/journey/10_diagnostik-fase-0.md` |
| D2b edge attribute tidak terpakai, v4 dan v6 (T5, T7) | `docs/journey/12_atensi-v6-korelasi-tanpa-kausalitas.md` |
| Ablasi atensi null di v4 dan v6 (T7) | `results/ATTENTION_V6.md`, `results/ATTENTION_v4_greedy.md` |
| Retensi zero-shot tiga tingkat (T6) | `results/ZEROSHOT_v4_primary.md` |
| Biaya PPO@200K (T3) | `results/logs/gnn-mappo_*_v6_seed4[2-6].csv`, kolom `elapsed_sec` |
| Nilai tabel reproducibility (T9) | `configs/experiment_config.yaml`, `gnn/gat_backbone.py`, `envs/network_slicing_env.py` |
