# Fase 0 — Diagnostik

**Fase:** 0 (pertama, kerjakan sekarang)
**Prasyarat:** tidak ada — pakai checkpoint v4 yang sudah ada
**Keluaran:** jawaban atas 5 pertanyaan yang menentukan seluruh fase berikutnya
**Estafet:** PLAN-02 (selalu) → PLAN-03 (selalu) → PLAN-04 (kondisional, tergantung D2)
**Biaya:** tanpa GPU training, perkiraan 1–2 hari
**Master:** PLAN-00-MASTER.md

---

> ## KOREKSI 2026-08-24 — dua premis dokumen ini, plus keputusan D2c
>
> **P5 — D3 §"Cek dulu" salah arah.** Dokumen ini menulis *"dengan 5 gNB, diameter graf
> kecil — 2 layer sudah menjangkau hampir seluruh graf"*, seolah itu alasan menganggap
> over-smoothing tidak mendesak. `envs/channel_model.py:112` menyatakannya sendiri —
> *"Fully-connected inter-gNB interference graph"* — dan loop di bawahnya menerbitkan tiap
> pasangan `(i,j), i≠j` sebagai edge. Diameter bukan "kecil" — diameter
> **1**. Satu layer sudah menjangkau seluruh graf, dan dua layer berarti tiap node
> mengagregasi himpunan tetangga yang **identik** dua kali. Risiko over-smoothing jadi
> **lebih tinggi**, bukan lebih rendah: D3 lebih relevan, bukan kurang.
>
> Ini juga membatasi apa yang bisa diuji D2b. Karena tiap node tujuan punya himpunan
> tetangga yang identik, mengacak label sumber di dalam satu grup tujuan tidak mengubah
> apa pun. Yang bisa dirusak hanyalah pasangan edge-ke-atribut. Pada topologi ini D2b
> menguji sensitivitas terhadap *informasi edge*, bukan terhadap *topologi* — dan untuk
> `sage` tidak bisa dijalankan sama sekali, karena `SAGEConv` (`gnn/sage_backbone.py:19`)
> tidak pernah membaca `edge_attr`.
>
> **P6 — D4 §"Kenapa penting" menebak arah yang salah.** Dokumen ini menulis *"Situasimu
> kemungkinan yang kedua"*, yaitu GNN punya parameter **lebih sedikit** sehingga hasil null
> bisa dijelaskan kapasitas kurang. Diukur: `ippo` 3.852 parameter terlatih melawan
> `gnn-mappo_gat` 37.580 — GNN punya **9,8× lebih banyak**, bukan lebih sedikit
> (`results/DIAG_EQUIVARIANCE.md`). Arahnya terbalik, dan konsekuensinya untuk paper lebih
> kuat daripada yang diantisipasi: baseline dengan sepersepuluh kapasitas menyamai GNN,
> jadi hasil null tidak bisa dijelaskan sebagai kapasitas kurang. Arm `ippo-scaled` di
> PLAN-05 §2 tetap layak, tapi perannya berubah — ia memisahkan "graf membantu" dari
> "kapasitas membantu" dengan menaikkan baseline, bukan menutup kekurangan baseline.
>
> **Keputusan D2c — jangan pakai rollout dengan jalur loss yang ditulis ulang.** Tabel
> gerbang §D2 mengunci keputusan PLAN-04 ke D2a dan D2b, keduanya murni evaluasi
> checkpoint, jadi D2c tidak dijalankan bersama mereka. Kalau dibutuhkan: **gunakan
> training pendek terinstrumentasi yang membaca `p.grad` setelah `agent.learn()`.**
> Menulis ulang sebagian jalur loss untuk menghitung gradien dari rollout berarti mengukur
> jalur yang belum tentu sama dengan jalur training asli. Untuk diagnostik yang menentukan
> gerbang, kesetiaan lebih penting daripada kemurahan — angka yang salah di sini
> menyesatkan seluruh Fase 2.
>
> **D2c kemudian dijalankan (2026-08-24), dan pertanyaannya berbeda dari yang semula
> ditunda.** Gerbang PLAN-04 memang sudah dibuka D2a/D2b. Yang belum terjawab: PLAN-03 §5
> dan PLAN-04 sama-sama lolos gerbang, `PLAN-04 §Larangan 4` melarang dua teknik
> anti-collapse sekaligus, dan keduanya menyasar penyebab berbeda. D2a/D2b menguji
> *akibat* (keluaran tidak bergantung pesan); D2c menguji *mekanisme* yang §D2 tulis
> ("jalur GNN tidak terlatih efektif"). Hasilnya menentukan urutan Fase 2 — lihat
> §Keluaran.
>
> **Implementasi.** D1+D4 di `scripts/diag_equivariance.py`, D2a/D2b/D3 di
> `scripts/diag_gnn_reliance.py`, D2c di `scripts/diag_grad_ratio.py`, D5 di
> `scripts/diag_collision.py`. Hasil di `results/DIAG_EQUIVARIANCE.md`,
> `results/DIAG_GNN_RELIANCE.md`, `results/DIAG_GRAD_RATIO.md`,
> `results/DIAG_COLLISION.md`. Nol file di `envs/` `agents/` `gnn/` `training/` diubah
> (§Larangan 1) — D2c membungkus metode kelas `PPOAgent.learn`/`DQNAgent.learn` dari sisi
> skrip diagnostik dan menjalankan loop training yang asli, di-resume ke salinan scratch
> supaya nol artefak v4 tersentuh (diverifikasi md5, 210 file, nol perbedaan).

---

## Kenapa fase ini dulu

Empat dari lima rencana berikutnya mengubah cara GNN memproses informasi. Kalau ternyata policy head mengabaikan output GNN, seluruh perubahan itu memperbaiki komponen yang tidak dipakai.

Fase ini murah dan menjawabnya lebih dulu.

---

## D1 — Permutation equivariance test

**Prioritas tertinggi dari seluruh diagnostik.**

### Kenapa kuat
Seluruh temuanmu sejauh ini marginal (comparable, kalah 1.2%, beda 1.3pp). Ini kategoris: varians aksi ≈0 atau tidak. Tidak ada CI yang bisa tumpang tindih.

### Prosedur
```python
import itertools

# ambil satu topologi 5 gNB dari seed held-out
results = []
for perm in itertools.permutations(range(5)):
    obs_p, edge_p, attr_p = apply_permutation(obs, edge_index, edge_attr, perm)
    action_p = model(obs_p, edge_p, attr_p)
    results.append(inverse_permutation(action_p, perm))

variance = np.var(np.stack(results), axis=0)
```

Permutasikan node feature, adjacency, dan matriks channel gain secara konsisten. Un-permute output sebelum mengukur varians.

### Antisipasi hasil

| model | ekspektasi |
|---|---|
| GNN (gat, sage) | varians ≈ 0 |
| MLP per-agen (ippo, mlp-knn-ppo) | varians ≈ 0 — parameter sharing juga equivariant |
| central-ppo, central-dqn | varians besar |

**Kalau MLP per-agen juga equivariant, itu bukan kegagalan tes.** Itu justru mengubah tesis mekanistikmu dari observasi empiris jadi properti struktural yang dibuktikan langsung: equivariance (dari parameter sharing), bukan message passing, yang menentukan transferabilitas.

### Untuk paper
> "Kami membuktikan secara langsung bahwa equivariance permutasi — bukan message passing — adalah sumber transferabilitas. Seluruh arsitektur per-agen dengan parameter sharing menunjukkan varians aksi ≈0 di seluruh 120 permutasi indeks, sementara arsitektur terpusat menunjukkan sensitivitas penuh terhadap pelabelan."

---

## D2 — Representation collapse (GERBANG KEPUTUSAN)

Hasil uji ini menentukan apakah PLAN-04 dijalankan.

### Hipotesis
Gradien dari policy head didominasi fitur lokal instan; jalur GNN tidak terlatih efektif dan outputnya diabaikan.

### D2a — Ablasi message passing
```python
# nolkan seluruh pesan tetangga (self-loop saja)
kpi_ablated = evaluate(model, edge_index=self_loops_only)
```
- KPI hampir tidak berubah → **collapse TERKONFIRMASI**
- KPI turun signifikan → GNN dipakai, cari penjelasan lain

### D2b — Randomisasi pesan
```python
# acak permutasi pesan antar-tetangga; magnitudo dipertahankan, struktur dirusak
kpi_random = evaluate(model, messages=shuffled)
```
KPI stabil → policy tidak sensitif struktur graf.

### D2c — Rasio norma gradien
```python
grad_gnn   = norm of gradients into GNN layers
grad_local = norm of gradients into local/policy layers
ratio = grad_gnn / grad_local     # << 1 → jalur GNN tidak terlatih
```

### Gerbang
| Hasil | Konsekuensi |
|---|---|
| Collapse terkonfirmasi (D2a & D2b: KPI stabil) | **Jalankan PLAN-04** setelah PLAN-03 |
| Collapse tidak terkonfirmasi | **Lewati PLAN-04.** Fokus PLAN-03 + PLAN-05 |

Jalankan pada seluruh varian GNN (gat, sage, madqn, mappo) supaya perbandingannya lengkap.

---

## D3 — Over-smoothing

### Cek dulu
Berapa layer GNN yang dipakai sekarang? Dengan 5 gNB, diameter graf kecil — 2 layer sudah menjangkau hampir seluruh graf.

### Uji
```python
sim = cosine_similarity_matrix(h_final)   # embedding layer terakhir
# nilai mendekati 1 di seluruh pasangan → over-smoothing
```

### Konsekuensi
- Terkonfirmasi → PLAN-03 §5 (residual connection)
- Tidak → lewati bagian itu

---

## D4 — Hitung parameter (equal capacity check)

Ancaman validitas yang belum pernah diukur.

```python
for name, model in models.items():
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name}: {n:,}")
```

### Kenapa penting
Kalau jumlah parameter timpang, perbedaan performa apa pun bisa dijelaskan **kapasitas**, bukan **graph inductive bias**. Berlaku dua arah:
- GNN parameter lebih banyak → kemenangan dianggap kapasitas
- GNN parameter lebih sedikit → hasil null bisa jadi kapasitas kurang

Situasimu kemungkinan yang kedua, dan belum diperiksa.

### Konsekuensi
Kalau timpang: tambahkan arm `ippo-scaled` di PLAN-05 (MLP diperlebar sampai parameter setara GNN). Jangan menimpa `ippo`.

**Tabel jumlah parameter wajib masuk paper apa pun hasilnya** — reviewer akan menanyakannya.

---

## D5 — Hipotesis collision storm

### Hipotesis
Saat training, agen belajar berbagi spektrum secara stokastik. Saat dipaksa argmax terdistribusi tanpa koordinasi sinkron, semua agen memilih aksi probabilitas tertinggi serentak → interferensi ekstrem → SINR jatuh → kolaps.

### Bukti yang sudah mendukung
1. sd greedy membengkak (34.95/37.82 pp) vs stokastik (11.04/10.45) → episode bimodal, konsisten dengan collision intermiten
2. `ippo` tidak kolaps di greedy (+0.46) sementara GNN kolaps parah (−51.19)

Poin 2 punya implikasi menarik: `ippo` tidak punya koordinasi halus untuk dirusak. Kalau hipotesis benar, greedy collapse pada GNN justru **bukti tidak langsung bahwa koordinasi ada**.

### Uji
```python
# 1. korelasi aksi antar-agen: greedy vs stokastik
#    collision storm → korelasi naik tajam di greedy
# 2. pola SINR pada episode greedy yang kolaps
#    collision storm → SINR anjlok serentak di seluruh gNB, bukan satu per satu
# 3. bandingkan gnn-mappo (kolaps) vs ippo (tidak kolaps)
```

### Batasan yang sudah diketahui
Entropi kebijakan merentang cuma 0.100 nat sementara celah pembacaan 46.28 Mbps. Entropi **tidak** memprediksi model mana yang dirusak argmax. Jadi collision storm — kalau benar — bukan fungsi sederhana dari entropi.

**Jangan tulis penjelasan ini di paper sebelum diuji.** Kalau tidak terkonfirmasi, framing "batas validitas operasional" (PLAN-07 §4) tetap berdiri tanpanya.

---

## Keluaran fase ini

Checklist yang harus terisi sebelum lanjut:

- [x] D1: varians permutasi per algoritma — tabel lengkap 8 algoritma
- [x] D2: collapse terkonfirmasi ya/tidak — **gerbang untuk PLAN-04**
- [x] D3: over-smoothing ya/tidak — gerbang untuk PLAN-03 §5
- [x] D4: tabel jumlah parameter — gerbang untuk arm `ippo-scaled`
- [x] D5: collision storm terkonfirmasi ya/tidak
- [x] D6: di layer mana representasi node kolaps — **gerbang untuk urutan di dalam Fase 2**
      (ditambahkan 2026-08-25, tidak ada di versi awal dokumen ini)

Catat semuanya di ledger. Ini diagnostik, bukan klaim performa — tidak butuh pra-registrasi, tapi hasilnya dilaporkan apa adanya.

### Hasil — 2026-08-24

Seluruh angka di bawah disalin dari file yang di-generate, bukan diketik ulang dari
ingatan (`docs/HANDOVER.md` §11). File itu yang otoritatif; ringkasan ini salinan.

**D1 — EQUIVARIANT, kategoris** (`results/DIAG_EQUIVARIANCE.md`, 9 algoritma, 120
permutasi). Ketujuh arsitektur per-agen: varians aksi tepat `0.000e+00`. Dua arsitektur
terpusat sensitif penuh terhadap pelabelan: `central-dqn` varians `8.798` dengan deviasi
skor pra-argmax `1.624e+02`, `central-ppo` varians `1.808`. Deviasi skor itu yang bikin
klaimnya aman — `gnn-madqn_sage` punya `4.272e-04`, bukan nol, tapi enam orde di bawah
skala skornya, jadi float noise dan bukan model membaca identitas node.

**Antisipasi §D1 terbukti: MLP per-agen juga equivariant.** `ippo`, `idqn`, dan
`mlp-knn-ppo` ketiganya `0.000e+00`. Sesuai §Larangan 4, dilaporkan apa adanya — dan itu
menguatkan tesis Tingkat 2: equivariance datang dari **parameter sharing**, bukan dari
message passing. Sekarang dibuktikan langsung, bukan disimpulkan dari retensi zero-shot.

**D2 — COLLAPSE TERKONFIRMASI, dengan variasi antar-seed yang harus ikut dilaporkan**
(`results/DIAG_GNN_RELIANCE.md`, 50 checkpoint × 10 episode × 3 arm). Gerbang PLAN-04
**terbuka**.

| Arm | Hasil |
|---|---|
| D2a pesan dinolkan | `timely_throughput_mbps` bergerak >1% di **9/50** checkpoint, `sla_satisfaction_pct` di **10/50**; median perubahan `+0.072` Mbps dari basis ~70 (0,1%) |
| D2b atribut edge diacak | **0/25** di ketiga KPI. Perubahan terbesar 0,404% |

41 dari 50 checkpoint tidak bergerak di atas 1% saat seluruh pesan tetangga dinolkan —
tabel gerbang §D2 menyebut itu "KPI hampir tidak berubah". Tapi 9 checkpoint **bergerak**,
sampai 9,1%, dan tersebar merata di keempat varian, jadi ini variasi antar-seed, bukan
sifat arsitektur. Jangan tulis "GNN tidak dipakai" tanpa kualifikasi itu.

D2b lebih tegas dari D2a: informasi fisik pada edge (path loss) tidak berkontribusi apa
pun pada varian `gat`. Untuk `sage` arm ini **N/A**, bukan 0 — `SAGEConv` tidak pernah
membaca `edge_attr`.

**D2c — HIPOTESIS "JALUR GNN TIDAK TERLATIH" TIDAK DIDUKUNG SEBAGAI PENJELASAN UMUM**
(`results/DIAG_GRAD_RATIO.md`, 20 checkpoint, 2.160 update terekam). §D2c menetapkan
`rasio << 1` sebagai tanda jalur GNN tidak terlatih. Yang memenuhi cuma **3 dari 20** seed.

| Varian | `ratio_l2` median | `ratio_rms` median |
|---|---|---|
| `gnn-madqn_gat` | 2,31 | 2,23 |
| `gnn-madqn_sage` | 0,38 | 0,53 |
| `gnn-mappo_gat` | 2,08 | 2,01 |
| `gnn-mappo_sage` | 3,50 | 4,88 |

Di **13 dari 20** seed backbone justru menerima gradien **lebih besar per parameter**
daripada head. Kedua rasio sepakat arah di **20/20**, jadi kolom RMS mengonfirmasi bukan
membalikkan — itu penting, karena `ratio_l2 < 1` bisa cuma berarti parameter backbone
lebih sedikit (`gnn-*_sage`: 9.344 lawan 18.188).

**Gambaran yang muncul dari D2c + D2 + D3 bersamaan:** GNN **dilatih dengan baik** —
gradien mengalir deras ke sana — tapi keluarannya nyaris tidak dipakai (D2a) dan
embeddingnya kolaps sempurna (D3). Jadi bukan encoder yang diabaikan, melainkan encoder
yang terlatih menuju representasi degenerate.

**Korespondensi lintas-diagnostik, n=5, suggestif bukan mapan.** Pada `gnn-mappo_gat`,
dua seed dengan jalur GNN mati adalah persis dua seed yang policy greedy-nya membeku
total (setiap gNB menahan satu aksi sepanjang episode):

| seed | `ratio_rms` (D2c) | aksi beku total (D5) |
|---|---|---|
| 42 | 3,86 | tidak |
| 43 | 5,85 | tidak |
| 44 | **0,0074** | **ya** |
| 45 | **0,0023** | **ya** |
| 46 | 5,72 | tidak |

D2c mengukur gradien saat training, D5 mengukur perilaku saat evaluasi — dua jalur kode
berbeda, jadi korespondensinya independen. Tapi lima titik. Jangan dinaikkan jadi klaim
kausal di paper tanpa seed lebih banyak.

**Keterbatasan DQN.** `_maybe_resume` tidak memulihkan `ReplayBuffer`, jadi run DQN
mengisi `replay_start` step pertama dengan policy yang sudah konvergen. Rasio DQN
menggambarkan aliran gradien di titik konvergen dengan data mendekati on-policy, bukan
campuran historis. PPO tidak kena — on-policy, jadi `RolloutBuffer` fresh memang setia.

**D3 — OVER-SMOOTHING TERKONFIRMASI, tak terbantahkan untuk `gat`.** Gerbang PLAN-03 §5
**terbuka**. `gat`: cosine similarity embedding = **1.0000 persis di seluruh 25
checkpoint** (min = max), sementara referensi cosine pada observasi mentahnya berkisar
0,7955–0,9844. Input paling terpisah pun menghasilkan embedding kolaps sempurna, jadi ini
perbuatan GNN, bukan warisan input yang memang mirip. `sage` 0,9464–0,9999 (rata-rata
0,991) — tinggi, tapi tidak degenerate.

Konsisten dengan P5: graf **lengkap**, diameter **1**. Dua layer mengagregasi himpunan
tetangga yang identik dua kali.

**D4 — TIMPANG, arah berlawanan dengan dugaan §D4** (`results/DIAG_EQUIVARIANCE.md`).

| algo | total | backbone | head |
|---|---|---|---|
| `ippo` | 3.852 | — | 3.852 |
| `central-ppo`, `mlp-knn-ppo` | 12.044 | — | 12.044 |
| `gnn-*_sage` | 27.532 | 9.344 | 18.188 |
| `idqn` | 35.724 | — | 35.724 |
| `gnn-*_gat` | 37.580 | 19.392 | 18.188 |
| `central-dqn` | 39.820 | — | 39.820 |

`ippo` **9,8× lebih kecil** dari `gnn-mappo_gat` dan tetap menyamainya. Backbone GNN 52%
dari model `gat`, jadi bukan pembulatan. Yang belum pernah disebut dokumen mana pun:
*head* varian GNN sendiri (18.188) sudah **4,7× seluruh `ippo`**, karena embedding 64
dimensi memaksa layer pertama actor jauh lebih lebar. Jadi celah kapasitas bukan cuma
"ada GNN tambahan" — arsitektur GNN melebarkan head-nya juga.

Arm `ippo-scaled` (PLAN-05 §2) tetap layak, tapi perannya berubah: menaikkan baseline
untuk memisahkan "graf membantu" dari "kapasitas membantu", bukan menutup kekurangan
baseline.

**D5 — TIDAK TERKONFIRMASI** (`results/DIAG_COLLISION.md`, 40 checkpoint, 20 seed per
algoritma). §Larangan 3 tetap mengikat: jangan tulis penjelasan collision storm di paper.

| algo | seed lockstep (greedy `mode_share` ≥ 0,998) | median greedy `mode_share` | Δ`embb_p5` greedy − sampled |
|---|---|---|---|
| `gnn-mappo_gat` | **18/20** | 1,000 | +1,283 |
| `ippo` | **8/20** | 0,935 | +0,0097 |

Mekanismenya nyata dan jauh lebih konsisten di GNN. Tapi **`ippo` juga mencapai lockstep
sempurna di 8 dari 20 seed — tanpa kehilangan throughput.** Unanimitas argmax karena itu
**tidak cukup** menyebabkan kolaps, dan hipotesis sebagaimana ditulis §D5 — bahwa
collision storm menjelaskan kenapa GNN kolaps di argmax sementara `ippo` tidak — tidak
didukung datanya sendiri.

Catatan metodologis yang layak masuk paper: uji ini dijalankan lebih dulu pada satu seed
(42), dan di situ kontrasnya bersih — `gnn-mappo_gat` 1,000 melawan `ippo` 0,642.
Berhenti di sana akan menghasilkan konfirmasi yang salah. Yang membalikkannya cuma
menambah seed.

**D5 diuji ulang dengan pengkondisian, 2026-08-24 — verdict tetap, ujinya jauh lebih
kuat.** Versi pertama merata-rata `sinr_corr` dan `mode_share` atas **seluruh** episode.
Itu cacat: §D5 poin bukti 1 menyatakan episodenya bimodal, dan merata-rata populasi
bimodal persis cara menyembunyikan efek intermiten. Versi pertama juga bersandar pada
perbandingan dengan `ippo`, padahal `ippo` bukan pembanding setara — hipotesis collision
storm spesifik pada model yang belajar koordinasi stokastik halus, dan `ippo` tidak punya
itu untuk dirusak.

Uji yang benar ada **di dalam** `gnn-mappo_gat`: bandingkan episode greedy yang kolaps
dengan yang tidak, pada model yang sama.

| kelompok | n | `sinr_corr` median | `mode_share` median | `timely_throughput` median |
|---|---|---|---|---|
| kolaps | 63 | 1,0000 (n=13 terdefinisi) | 1,0000 | 0,0002 Mbps |
| tidak kolaps | 137 | 1,0000 (n=112 terdefinisi) | 1,0000 | 64,66 Mbps |

**Tak terbedakan pada kedua ukuran sinkroni.** Throughput-nya terbelah tajam — 0,0002
lawan 64,66 Mbps, bimodalitas yang §D5 prediksi — tapi sinkroninya tidak. Sinkroni hadir
di **seluruh** episode greedy GNN sementara kolaps cuma terjadi di 63 dari 200: **kondisi
yang selalu ada tidak bisa menjelaskan akibat yang selektif.** Itu argumen yang jauh lebih
kuat daripada versi `ippo`, dan tidak bergantung pada pembanding lintas-arsitektur.

Ambang tidak menyetir apa pun: 61/63/66 episode kolaps di ambang 35%/50%/65%, seluruh
statistik identik sampai empat desimal.

**Dua keterbatasan yang ikut dilaporkan.** Pertama, `sinr_corr` hanya terdefinisi di 13
dari 63 episode kolaps — saat throughput terpaku ~0,0002 Mbps, deret SINR jadi konstan
dan korelasinya tak terdefinisi. Jadi baris kolaps bersandar pada 13 nilai, bukan 63;
`mode_share` terdefinisi di seluruh 200 dan tidak kena. Kedua, ambang collapse ini
didefinisikan pada `timely_throughput_mbps` per **episode**, bukan `embb_p5_mbps` per
**seed** seperti `collapse_rate` yang dipakai Gate (`scripts/stability_report.py`). Unit
yang lebih lemah, dinyatakan bukan ditukar diam-diam — dan memang harus berbeda, karena
memakai aturan cell-edge di sini akan menyeleksi hampir seluruh episode *sampled* dan
hampir nol episode *greedy*, kebalikan dari populasi yang D5 butuhkan.

**Framing "batas validitas operasional" (PLAN-07 §4) tetap berdiri tanpa D5** — memang
sudah dirancang begitu.

**D6 — INPUT TIDAK DEGENERATE; GNN YANG MENGOLAPSKAN, DAN KOLAPSNYA DI LAYER PERTAMA**
(`results/DIAG_INPUT_SEPARABILITY.md`, 50 checkpoint). Diagnostik ini tidak ada di versi
awal PLAN-01; ditambahkan 2026-08-25 karena urutan di dalam Fase 2 bersandar pada premis
yang belum pernah diukur.

D3 melaporkan cosine embedding `gat` 1.0000 lawan cosine observasi 0,7955–0,9844, dan
PLAN-03 §5 diurutkan lebih dulu atas dasar selisih itu. Premis di bawahnya: observasi
8-dimensi **cukup** membedakan gNB. Kalau ternyata tidak, residual/JK menyerang gejala di
tempat yang salah dan §2 (edge feature) serta §7 (observasi) yang duluan.

**Cosine tidak bisa menjawabnya, dan cosine ter-center juga tidak.** Kedelapan kolom
observasi non-negatif (`envs/network_slicing_env.py:552`), jadi tiap vektor node duduk di
ortan positif dan cosine terangkat karena konstruksi; `conv2` tidak punya aktivasi
(`gnn/gat_backbone.py:75`), jadi embedding bebas bertanda. Membandingkan 0,910 dengan
1,0000 membandingkan dua skala berbeda. Center-nya pun tidak menolong: mengurangi mean
antar-node memaksa `Σᵢ vᵢ = 0`, jadi inner product rata-rata off-diagonal dipaksa negatif
dengan lantai sekitar `−1/(n−1) = −0,25` untuk n=5 — nilai "≈1" mustahil muncul. Dipakai
dua statistik bebas skala: `rel_spread = ‖X − x̄‖_F / ‖X‖_F` (0 = node identik) dan
effective rank dari X ter-center. Cosine mentah tetap dilaporkan berdampingan supaya
bersambung dengan tabel D3 yang sudah di-commit.

| backbone | input | conv1 pra-aktivasi | conv1 pasca-aktivasi | conv2 |
|---|---|---|---|---|
| `gat` `rel_spread` median | **0,2831** | 0,0040 | 0,0039 | **0,0000** |
| `sage` `rel_spread` median | **0,1403** | 0,0926 | 0,0841 | **0,0348** |

Aturan keputusan ditulis sebelum dijalankan: input degenerate kalau `rel_spread < 0,05`
**dan** paling banyak satu kolom bervariasi. Yang memenuhi **0 dari 50** — dan 0/50 juga di
0,02 maupun 0,10, jadi ambangnya tidak menyetir apa pun. Kolom yang bervariasi: median
**6 dari 8**, rentang 4–7. **Urutan sekarang bertahan: PLAN-03 §5 lebih dulu.**

Dua hal yang tidak diminta tapi keluar dari data yang sama:

1. **Kolapsnya di `conv1`, bukan menumpuk di dua layer.** Rasio `rel_spread` antar-tahap
   berturut-turut untuk `gat`: input→conv1 **0,0145**, conv1 pra→pasca aktivasi 0,9861,
   conv1→conv2 0,0079. Layer pertama sudah membuang 98,6% separasi; aktivasinya praktis
   tidak bersalah. Konsisten dengan P5 (graf lengkap, diameter 1): satu layer sudah
   merata-ratakan kelima node. Konsekuensi untuk §5 — residual pada layer kedua saja tidak
   akan menolong, sambungannya harus melewati **layer pertama**.
2. **`sage` mempertahankan jauh lebih banyak** (0,6343 dan 0,3464 pada rasio yang sama,
   `rel_spread` akhir 0,0348 lawan 0,0000). Ini penjelasan mekanistik untuk beda `gat`
   lawan `sage` di D3 yang selama ini cuma dicatat sebagai angka: `SAGEConv` menyimpan
   bobot root terpisah, jadi kontribusi diri tidak ikut terata-rata.

**Umpan balik yang tidak diantisipasi dokumen mana pun:** `prev_alloc` dan
`prev_alloc_lag2` (`envs/network_slicing_env.py:553`) punya std antar-node **0,0000** di
49 dari 50 checkpoint. Dua dari delapan kolom observasi tidak membawa identitas node sama
sekali — karena tiap gNB memilih **aksi yang sama**, yaitu lockstep yang D5 ukur sebagai
`mode_share` 1,0000, sekarang terlihat di observasinya sendiri. Keluaran policy yang
degenerate mengumpan balik jadi fitur node yang kosong identitas: sekaligus gejala kolaps
dan masukan bagi kolaps itu.

### Konsekuensi ke fase berikutnya

| Gerbang | Verdict | Akibat |
|---|---|---|
| D2 → PLAN-04 | collapse terkonfirmasi | PLAN-04 **tidak lagi digerbangi ini** — lihat baris D6 dan PLAN-04 §0c |
| D2c → urutan Fase 2 | jalur GNN **terlatih** di mayoritas seed | **PLAN-03 §5 lebih dulu**, baru PLAN-04 (PLAN-04 §0b) |
| D3 → PLAN-03 §5 | over-smoothing terkonfirmasi | **§5 dijalankan.** Pilih satu: residual atau Jumping Knowledge, jangan dua-duanya |
| D4 → PLAN-05 §2 | timpang, GNN lebih besar | arm `ippo-scaled` tetap, peran berubah |
| D5 | tidak terkonfirmasi | tidak ada yang ditulis di paper soal collision storm |
| D6 → urutan **di dalam** Fase 2 | input tidak degenerate (0/50); kolaps di `conv1` | **§5 tetap lebih dulu**, dan sambungannya harus melewati layer pertama |
| D6 → gerbang PLAN-04 | — | gerbang baru: jalankan PLAN-04 hanya kalau §5 memperbaiki representasi **tapi KPI tetap datar** (PLAN-04 §0c) |

**Diagnosis dibingkai ulang.** Sebelum D2c, dua penjelasan bersaing: encoder **diabaikan**
(→ auxiliary loss) atau encoder **degenerate** (→ residual/JK). D2c menyingkirkan yang
pertama untuk mayoritas seed — gradiennya mengalir, bahkan 2–6× lipat head — dan D6
menutupnya dari sisi lain: masukannya memang bisa dibedakan (0/50 degenerate), tapi layer
pertama membuang 98,6% separasinya.

Jadi bukan *"encoder diabaikan"* melainkan **"encoder terlatih menuju representasi
degenerate"**. Sasaran perbaikan bergeser dari **memaksa gradien masuk** — sudah masuk —
ke **memaksa node berbeda satu sama lain**. Itu mengubah bukan cuma urutan, tapi kriteria
memilih target auxiliary kalau PLAN-04 nanti dijalankan, dan gerbang masuk PLAN-04 itu
sendiri (§0c).

D2b (0/25) menambah alasan independen untuk PLAN-03 §2: satu-satunya fitur edge yang ada
sekarang tidak terpakai sama sekali, jadi menambahkan `interference_coupling` harus
disertai uji bahwa fitur baru itu benar-benar dibaca — bukan diasumsikan.

---

## Larangan

1. Jangan mengubah kode training di fase ini — murni evaluasi checkpoint yang ada
2. Jangan lanjut ke PLAN-04 tanpa D2 terkonfirmasi
3. Jangan menulis penjelasan collision storm di paper sebelum D5 terkonfirmasi
4. Kalau D1 menunjukkan MLP per-agen juga equivariant, laporkan apa adanya
