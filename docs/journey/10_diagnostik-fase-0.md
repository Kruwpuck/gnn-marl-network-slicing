[← Wave v4 & cacat pembacaan](09_wave-v4-dan-cacat-pembacaan.md) | [Index](00_INDEX.md) | [v5 diblokir & wave v6 →](11_v5-diblokir-dan-wave-v6.md)

# 10 — Diagnostik Fase 0: D1–D6

**Periode:** 2026-08-24 → 2026-08-25
**Commit:** `8cf6db9` (D1–D5), `4177cc2` (D2c + perbaikan cacat D5), `2cb78ba` (D6)
**Sifat:** seluruhnya **evaluasi checkpoint v4 tanpa training**. Nol file di `envs/`, `agents/`, `gnn/`, `training/` diubah.

---

## Ringkasan eksekutif

1. **Enam premis yang mendasari rencana perbaikan ternyata basi**, diverifikasi ulang terhadap kode. Tiga di antaranya menyentuh gerbang keputusan, bukan catatan kaki — termasuk fakta bahwa varian yang sepanjang riset ini dinamai `gat` **isinya `GATv2Conv` sejak v1**.
2. **Hasil null tidak bisa dijelaskan sebagai kapasitas kurang.** `ippo` punya 3.852 parameter melawan `gnn-mappo_gat` 37.580 — GNN **9,8× lebih banyak**, kebalikan dari dugaan.
3. **Equivariance bukan keunggulan GNN.** Ketujuh arsitektur per-agen — termasuk MLP tanpa message passing sama sekali — equivariant dengan varians aksi tepat 0. Sumbernya parameter sharing, bukan graf.
4. **Over-smoothing terkonfirmasi tak terbantahkan untuk `gat`:** cosine embedding **1,0000 persis di 25/25 checkpoint**, dan atribut edge tidak terpakai di **0/25**.
5. **D6 menemukan di mana representasinya benar-benar kolaps:** bukan di observasinya (0/50 checkpoint punya input degenerate, diperiksa di tiga ambang), melainkan di **`conv1`, yang membuang 98,6% separasi node**. Itu yang jadi dasar mekanistik seluruh wave v6.

---

## 1. Enam premis basi

Enam dokumen turunan rencana ditulis dari jawaban riset literatur tanpa akses ke kondisi kode terkini. Keenam premisnya dibaca ulang terhadap file dan gugur.

| # | Premis | Keadaan kode sebenarnya | Akibat |
|---|---|---|---|
| P1 | "ganti GAT → GATv2" | `gnn/gat_backbone.py` sudah `GATv2Conv` **sejak v1** | Rencananya gugur. **Varian dinamai `gat` padahal GATv2** — terminologi paper harus dikoreksi |
| P2 | "edge belum bawa fitur fisik" | `envs/channel_model.py` sudah mengirim path loss dB; `edge_dim=1` sudah diset | Rencana fitur edge menyusut ke `interference_coupling` saja |
| P3 | `distance_norm` sebagai fitur ketiga | path loss tanpa shadow fading + `los=True` = fungsi monoton jarak | Fitur dicoret. **Dua fitur edge, bukan tiga** |
| P4 | "buang `neighbor_urllc_frac_mean`" | sudah dibuang di v3 | Satu arm ablasi hilang dari rencana; konflik antar-rencana selesai sendiri |
| P5 | "diameter graf kecil" | graf **lengkap**, diameter **1** | Risiko over-smoothing lebih **tinggi**, bukan lebih rendah |
| P6 | "GNN kemungkinan parameter lebih sedikit" | `ippo` 3.852 lawan `gnn-mappo_gat` 37.580 | **Hasil null tidak bisa dijelaskan kapasitas kurang** |

P6 layak diulang karena ia membalik satu penjelasan yang nyaman. Kalau GNN kalah atau setara, penjelasan "modelnya terlalu kecil" tidak tersedia — ia justru **9,8× lebih besar** dari baseline yang menyainginya.

---

## 2. D1 — equivariance, dan kenapa hasilnya bukan berita baik

Diuji dengan mempermutasi label node dan mengukur varians aksi un-permuted atas seluruh 120 relabelling, sesudah **50 langkah policy** — bukan di t=0, karena `reset()` menolkan 7 dari 8 kolom observasi sehingga setiap model akan lolos karena alasan yang tidak ada hubungannya dengan equivariance.

| Kelompok | `action_var_max` (argmax) | vektor aksi berbeda |
|---|---|---|
| `gnn-madqn_*`, `gnn-mappo_*`, `idqn`, `ippo`, `mlp-knn-ppo` | **0,000e+00** | 1 |
| `central-dqn`, `central-ppo` *(skalar pra-broadcast)* | 8,798e+00 / 1,808e+00 | 2 / 3 |

**MLP per-agen juga equivariant.** Sumber sifat itu adalah **parameter sharing**, bukan message passing — jadi equivariance tidak bisa dijadikan keunggulan arsitektur GNN dalam paper. Baris `central-*` sengaja tidak sebanding: keduanya mengonsumsi observasi global yang di-flatten dan menyiarkan satu aksi ke semua gNB, sehingga vektor aksinya konstan *by construction* dan akan terbaca equivariant sempurna; yang dilaporkan keputusan pra-broadcast-nya.

Satu kehati-hatian instrumen ikut ditulis: urutan reduksi float berubah saat node dipermutasi, jadi argmax atas skor yang nyaris seri bisa berbalik di bit terakhir. `action_var_max` bukan nol dengan `score_dev_max` beberapa ordo di bawah `score_scale` adalah **seri yang dipecah berbeda**, bukan model yang membaca identitas node.

---

## 3. D2 — apakah policy benar-benar memakai tetangganya

Tiga KPI dipakai, bukan `embb_p5_mbps` sendirian. Alasannya struktural: di titik operasi ini sebagian besar checkpoint sudah duduk di lantai cell-edge, dan KPI yang tertahan di ~1e-6 **tidak bisa memburuk** seberapa pun ablasinya mengubah keadaan. Melaporkannya sendirian akan mengubah *tidak ada ruang gerak* jadi *GNN-nya tidak penting* yang keliru.

### D2a — pesan tetangga dinolkan

| KPI | checkpoint bergerak > 1% dari nilainya sendiri |
|---|---|
| `embb_p5_mbps` | 35/50 |
| `timely_throughput_mbps` | **9/50** |
| `sla_satisfaction_pct` | **10/50** |

Pada dua KPI yang masih punya ruang gerak, **41 dari 50 checkpoint tidak bergerak lebih dari 1%** ketika seluruh pesan tetangga dinolkan.

### D2b — atribut edge diacak

| KPI | perubahan absolut terbesar | checkpoint bergerak > 1% |
|---|---|---|
| `embb_p5_mbps` | 0,001405 (0,332%) | **0/25** |
| `timely_throughput_mbps` | 0,280557 (0,404%) | **0/25** |
| `sla_satisfaction_pct` | 0,389270 (0,426%) | **0/25** |

**Nol dari 25 checkpoint** peduli pada atribut edge-nya. Ini yang kemudian jadi pembanding wajib untuk arm v6: arm baru tidak boleh mengulang kondisi ini tanpa ketahuan.

Satu batas dinyatakan di laporannya sendiri: graf antar-gNB **lengkap**, jadi tiap node penerima punya himpunan tetangga identik dan mempermutasi label sumber tidak mengubah apa pun. Yang tersisa untuk dirusak cuma pasangan edge-ke-atribut — jadi di topologi ini D2b menguji sensitivitas terhadap **informasi** edge, bukan terhadap **topologi**.

### D3 — over-smoothing

Cosine antar-embedding akhir: **1,0000 persis di 25/25 checkpoint `gat`**, melawan cosine observasi 0,7955–0,9844. Kategoris, dan itu yang mengurutkan perbaikan residual lebih dulu di rencana v6.

---

## 4. D2c — hipotesis "jalur GNN tidak terlatih" tidak didukung

D2a dan D2b menguji **akibatnya**; D2c menguji **mekanismenya**: apakah gradien sampai ke GNN sama sekali. Itu yang memisahkan perbaikan arsitektur (residual, mengarah ke over-smoothing) dari perbaikan loss tambahan (mengarah ke jalur GNN yang tidak terlatih).

Cara ukurnya dipilih dengan sengaja mahal: **loop training yang sebenarnya dijalankan**, di-resume dari checkpoint v4 ke salinan scratch, `learn()` dibungkus di level kelas, dan `p.grad` dibaca setelah yang asli selesai — jadi yang terbaca adalah gradien yang **benar-benar diterapkan**, bukan rollout dengan jalur loss yang ditulis ulang.

| varian | median `ratio_l2` |
|---|---|
| `gnn-madqn_gat` | 2,3055 |
| `gnn-madqn_sage` | 0,3782 |
| `gnn-mappo_gat` | 2,0793 |
| `gnn-mappo_sage` | 3,5003 |

Rasio di atas 1 berarti backbone menerima gradien **lebih besar per parameter** daripada head. Tiga dari empat varian ada di sana. Hipotesisnya tidak didukung sebagai penjelasan umum, dan konsekuensinya langsung ke urutan kerja: perbaikan arsitektur dijalankan lebih dulu, loss tambahan tidak.

D2c sempat **sengaja ditunda** dengan alasan yang ditulis: tabel gerbang mengunci keputusan ke D2a/D2b yang keduanya evaluasi checkpoint murni, dan untuk diagnostik yang menentukan gerbang, **fidelitas ke jalur training yang sebenarnya mengalahkan kemurahan**. Ia dijalankan kemudian dengan cara yang mahal itu, bukan dengan jalan pintas.

---

## 5. D5 — hipotesis collision storm tidak terkonfirmasi

Diuji sebagai hipotesis, bukan ditulis sebagai penjelasan. Di dalam `gnn-mappo_gat`, episode greedy yang kolaps dan yang tidak **tidak terbedakan pada sinkroni**: `sinr_corr` dan `mode_share` sama-sama 1,0000 pada episode kolaps, dan verdict-nya stabil di ketiga ambang yang diuji (35%, 50%, 65% — 61/200, 63/200, 66/200 episode kolaps).

Konsekuensinya tegas: **nol kalimat tentang collision storm masuk paper.** Batas yang sudah diketahui sebelum uji ini juga dicatat — entropi policy cuma merentang 0,100 nat sementara celah pembacaannya 46,28 Mbps, jadi apa pun fenomenanya, ia bukan fungsi sederhana dari entropi.

Satu cacat di D5 buatan sendiri ditemukan dan diperbaiki di commit yang sama. Ke-800 episode disimpan per baris di `results/diag_collision_episodes.csv`, sehingga pengondisian di atas bisa dihitung ulang pada ambang mana pun langsung dari file itu — bukan dari ingatan laporan.

---

## 6. D6 — di mana representasinya sebenarnya kolaps

D6 lahir karena urutan kerja v6 bersandar pada premis yang belum pernah diukur: kalau kedelapan kolom observasi **sendiri** sudah gagal membedakan kelima gNB, maka memperbaiki residual berarti menyerang gejala di tempat yang salah.

### Kenapa cosine tidak bisa menjawabnya

Dua alasan, keduanya soal instrumen:

1. Kedelapan kolom observasi **non-negatif**, jadi setiap vektor node duduk di ortan positif dan cosine terangkat **oleh konstruksi**. `conv2` tidak punya aktivasi, jadi embedding-nya bebas bertanda. Membandingkan 0,910 dengan 1,0000 membandingkan dua skala berbeda.
2. Mean-centering tidak menyelamatkannya: ia memaksa Σᵢvᵢ = 0, sehingga rata-rata inner product off-diagonal **dipaksa negatif** dengan lantai ≈ −1/(n−1) = −0,25 pada n=5. Nilai mendekati 1 **tidak mungkin muncul**, jadi instrumennya tidak bisa menjawab pertanyaan yang diajukan.

Dua statistik bebas skala dipakai sebagai gantinya — `rel_spread` (0 = node identik) dan `eff_rank` — dihitung identik di tiap tahap, dan diambil dari **forward pass yang sebenarnya** lewat hook, bukan dari implementasi ulang yang bebas melenceng dari `gnn/`.

### Hasilnya

| backbone | tahap | `rel_spread` median | `eff_rank` median | cosine mentah |
|---|---|---|---|---|
| `gat` | `input` | 0,2831 | 2,72 | 0,9098 |
| `gat` | `conv1_pre` | 0,0040 | 2,93 | 1,0000 |
| `gat` | `conv1_act` | 0,0039 | 2,94 | 1,0000 |
| `gat` | `conv2` | **0,0000** | 1,96 | 1,0000 |
| `sage` | `input` | 0,1403 | 2,74 | 0,9757 |
| `sage` | `conv1_pre` | 0,0926 | 2,68 | 0,9921 |
| `sage` | `conv1_act` | 0,0841 | 2,60 | 0,9928 |
| `sage` | `conv2` | **0,0348** | 2,26 | 0,9987 |

**Aturan keputusan ditulis sebelum run:** input disebut degenerate kalau `rel_spread < 0,05` **dan** paling banyak satu kolom bervariasi. Hasilnya **0 dari 50 checkpoint** — dan diperiksa di tiga ambang (0,02, 0,05, 0,1), semuanya 0/50, karena verdict yang berpindah antar-ambang adalah verdict yang disetir ambangnya, dan itu sendiri akan jadi temuan.

Jadi inputnya hidup, dan **yang membuangnya adalah GNN-nya.**

### Lapisan mana yang membuangnya

Rasio `rel_spread` antar-tahap berurutan, median per checkpoint — rasio jauh di bawah 1 berarti di situ separasinya hilang:

| backbone | input → `conv1_pre` | `conv1_pre` → `conv1_act` | `conv1_act` → `conv2` |
|---|---|---|---|
| `gat` | **0,0145** | 0,9861 | 0,0079 |
| `sage` | 0,6343 | 1,0030 | 0,3464 |

`conv1` pada `gat` membuang **98,6% separasi node**, dan aktivasinya (rasio 0,9861) hampir tidak membuang apa pun. **Yang membuang adalah agregasinya, bukan ELU-nya.** `sage` menahan jauh lebih banyak di kedua lapis — konsisten dengan `SAGEConv` yang menyimpan bobot root terpisah.

### Konsekuensi yang langsung jadi rencana v6

Kalau fitur edge ditambahkan tapi `conv1` tetap menghomogenkan keluarannya, informasi barunya hilang **di tempat yang sama**. Dua mekanisme perbaikan karena itu diperkirakan **berurutan, bukan paralel**: residual menyelamatkan separasi, lalu fitur edge punya sesuatu untuk disumbangkan pada separasi yang selamat. Itulah kenapa wave v6 punya arm gabungan, bukan cuma dua arm terisolasi.

Satu prediksi terukur ikut lahir dari tabel kolom D6: `prev_alloc` dan `prev_alloc_lag2` bervariasi hanya di **1 dari 50 checkpoint** — kelima gNB bergerak *lockstep*. Kalau perbaikan arsitektur benar, lockstep itu pecah, dan itu bisa dibaca **tanpa menyentuh KPI apa pun**.

---

## Data & artefak

| Isi | File |
|---|---|
| D1 equivariance + D4 hitungan parameter | `results/DIAG_EQUIVARIANCE.md`, `results/diag_equivariance.csv` |
| D2a, D2b, D3 | `results/DIAG_GNN_RELIANCE.md`, `results/diag_gnn_reliance.csv` |
| D2c rasio norma gradien | `results/DIAG_GRAD_RATIO.md`, `results/diag_grad_ratio.csv` |
| D5 collision storm + 800 episode per baris | `results/DIAG_COLLISION.md`, `results/diag_collision_episodes.csv` |
| D6 separabilitas input per tahap | `results/DIAG_INPUT_SEPARABILITY.md`, `results/diag_input_separability.csv` |
| Enam premis basi + tabel gerbang keputusan | `docs/revisi/PLAN-00-MASTER.md` (blok koreksi 2026-08-24) |
| Definisi tiap diagnostik + larangannya | `docs/revisi/PLAN-01-DIAGNOSTICS.md` |
| Kronologi bertanda waktu | `runs/2026-08-05-run01/ledger.md` (2026-08-24 → 2026-08-25) |

[← Wave v4 & cacat pembacaan](09_wave-v4-dan-cacat-pembacaan.md) | [Index](00_INDEX.md) | [v5 diblokir & wave v6 →](11_v5-diblokir-dan-wave-v6.md)
