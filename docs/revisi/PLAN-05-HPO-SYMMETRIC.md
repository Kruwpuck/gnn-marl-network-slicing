# Fase 3 — HPO Simetris (wave v7)

**Fase:** 3
**Prasyarat:** PLAN-03 selesai (arsitektur final), PLAN-04 selesai jika dijalankan, PLAN-01 D4 (jumlah parameter)
**Keluaran:** config terbaik per algoritma, hasil post-HPO
**Estafet:** PLAN-06 (Fase 4)
**Master:** PLAN-00-MASTER.md

---

> ## KOREKSI 2026-08-24 — batasan konflik K2 tidak lagi mengikat
>
> §3.3 dan §10.6 melarang memperbaiki isu observasi bersamaan dengan HPO, supaya efeknya
> terpisah. **Isu itu sudah tidak ada.** `neighbor_urllc_frac_mean` dibuang dari observasi
> di v3, jauh sebelum dokumen ini ditulis: `envs/network_slicing_env.py:22` mencatat
> penghapusannya beserta penggantinya (`prev_alloc_lag2`), dan `_get_obs()`
> (`envs/network_slicing_env.py:537`) memang tidak memuatnya.
>
> Akibatnya: tidak ada pekerjaan observasi yang bisa bertabrakan dengan HPO, arm
> `obs=strict` di PLAN-03 §7 hilang, dan konflik K2 selesai dengan sendirinya. §3.3 dan
> §10.6 dibiarkan tertulis sebagai jejak, tapi tidak mengikat apa pun.
>
> Detail lengkap kelima premis basi ada di blok koreksi
> `docs/revisi/PLAN-03-EDGE-FEATURES.md`. Tidak ada bagian lain PLAN-05 yang berubah —
> protokol HPO, ruang pencarian, budget trial, dan pemisahan seed tetap berlaku penuh.

---

## 0. Kenapa fase ini terakhir sebelum penulisan

HPO harus dijalankan pada **arsitektur final**. Kalau dijalankan sebelum edge feature dan auxiliary loss selesai, config terbaiknya jadi tidak relevan dan harus diulang.

Ini juga menutup saran pembimbing (tuning LR, hidden layer, optimizer) dengan cara yang defensible.

---

## 1. Justifikasi — kutipan terverifikasi

> "However, note that default hyperparameters were used for all GNN variants. **Hyperparameter tuning would improve performance and may also change the rankings** of the GNN variants considered here."
>
> — *Using Graph Neural Networks in Reinforcement Learning: A Practical Guide*, **ICLR Blogposts 2026**. https://iclr-blogposts.github.io/2026/blog/2026/rl-with-gnns/

Terverifikasi verbatim. Pengakuan eksplisit dari sumber yang direview bahwa default hyperparameter dapat mengubah peringkat antar-arsitektur.

Relevansi langsung: seluruh wave v1–v4 memakai default. Klaim "GNN comparable dengan MLP" punya ancaman validitas yang bisa dinamai — menutupnya adalah kontribusi metodologis.

**Catatan sitasi:** ICLR *Blogpost track*, bukan paper konferensi utama. Tandai eksplisit di daftar pustaka.

---

## 2. Equal parameter budget (dari PLAN-01 D4)

### Kalau D4 menunjukkan jumlah parameter timpang

Perbedaan performa apa pun bisa dijelaskan **kapasitas**, bukan **graph inductive bias**. Berlaku dua arah.

### Tindakan
Tambahkan arm **`ippo-scaled`**: MLP diperlebar (hidden dim dinaikkan) sampai jumlah parameter terlatihnya setara model GNN.

Ini memisahkan dua hipotesis:
- *"graf membantu"* → GNN unggul atas `ippo-scaled`
- *"kapasitas membantu"* → `ippo-scaled` menyusul GNN

Jangan menimpa `ippo`. Arm baru, nama baru.

**Tabel jumlah parameter wajib masuk paper apa pun hasilnya.**

---

## 3. Protokol HPO yang adil

Tiga syarat, semuanya wajib.

### 3.1 Ruang pencarian identik

Parameter bersama — sama untuk 8 algoritma:

| Parameter | Ruang pencarian |
|---|---|
| Learning rate | log-uniform `[1e-4, 5e-3]` |
| Optimizer | `{Adam, AdamW}` |
| Weight decay (jika AdamW) | log-uniform `[1e-4, 1e-1]` |
| Hidden dimension | `{32, 64, 128}` |
| Batch size | `{32, 64, 256}` |
| Discount factor γ | `{0.95, 0.99}` |

Parameter khusus GNN — hanya untuk varian GNN, **tapi jumlah trial tetap sama**:

| Parameter | Ruang pencarian |
|---|---|
| Attention heads | `{1, 2, 4}` |
| Jumlah layer GNN | `{2, 3}` |
| Residual alpha (jika dipakai) | `{0.0, 0.1}` |
| `beta_aux` (jika PLAN-04 dijalankan) | log-uniform `[0.01, 1.0]` |

### 3.2 Budget trial identik
Jumlah trial sama persis per algoritma. Dilaporkan eksplisit di paper.

### 3.3 Observasi dan reward identik
Sudah dipatuhi.

**Konflik K2:** isu `neighbor_urllc_frac_mean` (PLAN-03 §7) **tidak diperbaiki bersamaan dengan HPO** — efeknya jadi tidak terpisah. Itu arm tersendiri di wave v6.

### 3.4 Seed HPO terpisah
HPO memakai seed sendiri (mis. 100–119). Config terbaik dijalankan pada seed evaluasi (42–61). **Jangan** memilih config berdasarkan performa di seed evaluasi — itu kebocoran.

---

## 4. Masalah biaya (harus diputuskan sebelum jalan)

Biaya terukur: PPO 8.32 job-jam/seed, DQN 77.57 job-jam/seed (9.3× lebih mahal).

HPO 20 trial, 1 seed per trial:

| Keluarga | Perhitungan | Job-jam |
|---|---|---|
| PPO (4–5 algo) | 4 × 20 × 8.32 | ~666 |
| DQN (4 algo) | 4 × 20 × 77.57 | ~6206 |

Keluarga DQN **tidak terjangkau** dengan sisa anggaran.

### Opsi

| Opsi | Isi | Catatan |
|---|---|---|
| **A** | Kurangi trial, sama untuk semua (mis. 8) | DQN ~2482 job-jam, masih terlalu mahal |
| **B** | Budget trial berbeda per keluarga, atas dasar biaya | Konsisten dengan preseden aturan biaya seed yang sudah dibekukan. Wajib dideklarasikan sebelum jalan |
| **C** | HPO hanya keluarga PPO | Termurah, tapi klaim HPO hanya berlaku keluarga PPO dan ditulis eksplisit |
| **D** | Vektorisasi `DQNAgent.learn()` dulu | Loop Python per-sampel adalah penyebab 77.57 job-jam. Bisa memangkas biaya drastis |

**Rekomendasi: D dulu** (kalau vektorisasi bisa diverifikasi bit-identik outputnya), **lalu B**. Kalau D tidak layak, ambil B dengan deklarasi eksplisit.

Gunakan akuntansi **device-jam** sesuai kebijakan yang sudah dibekukan.

---

## 5. Uji hipotesis sensitivitas (gratis)

Klaim "GNN-MARL lebih sensitif hyperparameter" tidak punya sitasi primer terverifikasi. Perlakukan sebagai **hipotesis yang diuji HPO itu sendiri**.

Dari hasil HPO, ukur **variance performa antar-trial** per algoritma. Kalau varian GNN jauh lebih besar dari MLP, klaim sensitivitas terdukung oleh datamu sendiri — temuan yang bisa dilaporkan tanpa sitasi eksternal.

---

## 6. Standar statistik — jangan diturunkan

Literatur menyebut 5–10 seed, two-tailed t-test p<0.05. **Standarmu sudah lebih tinggi:**
- 20 seed (keluarga PPO)
- rliable IQM + stratified bootstrap CI + Wilson CI untuk outcome biner
- Held-out evaluation terpisah

Jangan turun ke t-test karena itu yang lazim. Datamu bimodal — t-test menyesatkan di distribusi seperti itu.

Boleh menambahkan t-test sebagai pelengkap, tapi **verdict tetap dari CI**.

---

## 7. Klaim yang sudah terbantah datamu

Argumen "kegagalan MLP disebabkan structural limitation (zero-padding pada topologi dinamis), bukan mis-tuning" hanya berlaku sebagian:

| Arsitektur | Berlaku? |
|---|---|
| `central-*` | **Ya** — `obs_dim` terkunci, CANNOT_RUN |
| MLP per-agen (`ippo`, `mlp-knn-ppo`) | **Tidak** — transfer dengan retensi 0.949–0.954 tanpa zero-padding, karena parameter sharing membuat dimensi input tidak bergantung jumlah gNB |

Kalau dikutip, batasi ke arsitektur terpusat.

---

## 8. Urutan eksekusi

1. Ambil tabel jumlah parameter dari PLAN-01 D4
2. Kalau timpang: siapkan arm `ippo-scaled`
3. Putuskan strategi biaya (§4) — kalau D, vektorisasi `DQNAgent.learn()` dan verifikasi bit-identik
4. **Pra-registrasi protokol HPO:** ruang pencarian, jumlah trial per algoritma, seed HPO, kriteria pemilihan config terbaik — ditulis sebelum HPO dijalankan
5. Jalankan HPO
6. Ukur variance antar-trial per algoritma (§5)
7. Wave v7 dengan config terbaik masing-masing, seed evaluasi standar
8. Laporkan: hasil default (v4/v5/v6) **dan** post-HPO, beserta budget HPO

---

## 9. Kalimat untuk paper

> "Each of the N algorithms received an identical HPO budget of K trials over the same search space, with HPO seeds disjoint from evaluation seeds. Trainable parameter counts are reported for all architectures; where counts differed materially, a capacity-matched baseline was added to separate architectural inductive bias from model capacity."

Kalau hasil post-HPO masih comparable, itu **memperkuat** temuan null. "GNN tidak menang meski sudah dituning setara" jauh lebih sulit dibantah daripada "GNN tidak menang pada default".

---

## 10. Larangan

1. Tuning hanya untuk model proposed — dilarang mutlak
2. Memilih config terbaik berdasarkan performa di seed evaluasi — kebocoran
3. Menurunkan standar statistik ke t-test
4. Mengutip klaim sensitivitas GNN sebagai fakta bersitasi — perlakukan sebagai hipotesis
5. Mengutip klaim structural limitation MLP secara umum — hanya untuk arsitektur terpusat
6. Memperbaiki isu observasi bersamaan dengan HPO (konflik K2)
7. Menyembunyikan hasil default setelah punya hasil post-HPO
8. ICLR Blogpost ditulis sebagai paper ICLR — tandai sebagai blogpost track
