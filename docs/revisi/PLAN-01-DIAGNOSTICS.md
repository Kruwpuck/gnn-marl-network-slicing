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
> checkpoint, jadi D2c tidak dijalankan sekarang. Kalau nanti D2a dan D2b tidak sepakat
> dan D2c benar-benar dibutuhkan: **gunakan training pendek terinstrumentasi yang membaca
> `p.grad` setelah `agent.learn()`.** Menulis ulang sebagian jalur loss untuk menghitung
> gradien dari rollout berarti mengukur jalur yang belum tentu sama dengan jalur training
> asli. Untuk diagnostik yang menentukan gerbang, kesetiaan lebih penting daripada
> kemurahan — angka yang salah di sini menyesatkan seluruh Fase 2. Butuh GPU dan
> dijalankan manusia (`docs/HANDOVER.md` §11).
>
> **Implementasi.** D1+D4 di `scripts/diag_equivariance.py`, D2a/D2b/D3 di
> `scripts/diag_gnn_reliance.py`, D5 di `scripts/diag_collision.py`. Hasil di
> `results/DIAG_EQUIVARIANCE.md`, `results/DIAG_GNN_RELIANCE.md`,
> `results/DIAG_COLLISION.md`. Nol file di `envs/` `agents/` `gnn/` `training/` diubah
> (§Larangan 1).

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

**Framing "batas validitas operasional" (PLAN-07 §4) tetap berdiri tanpa D5** — memang
sudah dirancang begitu.

### Konsekuensi ke fase berikutnya

| Gerbang | Verdict | Akibat |
|---|---|---|
| D2 → PLAN-04 | collapse terkonfirmasi | **PLAN-04 dijalankan** sesudah PLAN-03 |
| D3 → PLAN-03 §5 | over-smoothing terkonfirmasi | **§5 dijalankan.** Pilih satu: residual atau Jumping Knowledge, jangan dua-duanya |
| D4 → PLAN-05 §2 | timpang, GNN lebih besar | arm `ippo-scaled` tetap, peran berubah |
| D5 | tidak terkonfirmasi | tidak ada yang ditulis di paper soal collision storm |

D2b (0/25) menambah alasan independen untuk PLAN-03 §2: satu-satunya fitur edge yang ada
sekarang tidak terpakai sama sekali, jadi menambahkan `interference_coupling` harus
disertai uji bahwa fitur baru itu benar-benar dibaca — bukan diasumsikan.

---

## Larangan

1. Jangan mengubah kode training di fase ini — murni evaluasi checkpoint yang ada
2. Jangan lanjut ke PLAN-04 tanpa D2 terkonfirmasi
3. Jangan menulis penjelasan collision storm di paper sebelum D5 terkonfirmasi
4. Kalau D1 menunjukkan MLP per-agen juga equivariant, laporkan apa adanya
