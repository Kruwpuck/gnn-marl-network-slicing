# Fase 2b — Auxiliary Loss (KONDISIONAL)

**Fase:** 2b
**Prasyarat:** PLAN-01 D2 **terkonfirmasi** + PLAN-03 selesai
**Keluaran:** GNN yang gradiennya tidak dipotong policy head
**Estafet:** PLAN-05 (Fase 3)
**Master:** PLAN-00-MASTER.md

---

## 0. GERBANG MASUK — jangan lewati

**Dokumen ini hanya dijalankan kalau PLAN-01 D2 mengonfirmasi representation collapse.**

| Hasil D2a & D2b | Tindakan |
|---|---|
| KPI stabil saat message passing dinolkan/diacak | Collapse terkonfirmasi → **lanjutkan** |
| KPI turun signifikan | GNN dipakai → **hentikan, lewati dokumen ini** |

Mengimplementasikan auxiliary loss tanpa bukti collapse menambah kompleksitas untuk masalah yang mungkin tidak ada.

---

## 1. Peringatan sitasi

Blok sumber untuk dokumen ini punya verifikasi paling lemah — hampir seluruh rujukan berakhir dengan placeholder sitasi kosong. Formula terlihat presisi tapi tidak bisa dilacak.

**Belum terverifikasi:** TC-GQN (beserta formula auxiliary loss-nya), CIB, TRR, "Graph-Enhanced Critic Learning for Cooperative Spectrum Access", QMIX-GNN, P-DGN, TD3-D-MA.

**Mungkin nyata, perlu konfirmasi:** MAGNNETO (traffic engineering), Neuro-DCF (distributed congestion control).

**Konsekuensi:** dokumen ini sumber ide teknis, **bukan basis sitasi**. Kalau tekniknya dipakai, cari paper aslinya dan kutip itu.

---

## 2. Auxiliary loss (satu-satunya teknik yang direkomendasikan)

### Prinsip
Paksa encoder GNN mempelajari sesuatu yang hanya bisa didapat dari informasi tetangga, lewat tugas prediksi yang **independen dari reward**. Gradiennya mengalir langsung ke layer GNN tanpa melewati policy head — jadi tidak bisa "dipotong".

### Implementasi

Formula TC-GQN dari sumber terlalu kompleks dan sitasinya kosong. Versi sederhana yang setara secara prinsip:

```python
pred = aux_head(h_gnn_i)           # dari embedding GNN agen i
target = neighbor_state_next       # target di t+1
loss_aux = F.mse_loss(pred, target)

loss_total = loss_rl + beta_aux * loss_aux
```

### Pilihan target

| Target | Kenapa |
|---|---|
| SINR agen sendiri di `t+1` | **Rekomendasi.** SINR-mu bergantung alokasi tetangga lewat coupled interference — memprediksinya menuntut model memahami perilaku tetangga |
| Rata-rata alokasi URLLC tetangga di `t+1` | Langsung, tapi kalau `neighbor_urllc_frac_mean` dibuang dari observasi (PLAN-03 §7), ini jadi target yang lebih bermakna |
| Level interferensi yang diterima di `t+1` | Alternatif |

Pilih **satu**, catat di pra-registrasi.

### Nilai beta_aux
Jangan tetapkan manual. Masukkan sebagai dimensi ruang pencarian HPO di PLAN-05 (`beta_aux ∈ [0.01, 1.0]`, log-uniform), sejajar dengan `heads` dan `layers`.

### Verifikasi
Setelah training dengan auxiliary loss, **ulangi D2a** (ablasi message passing). KPI harus turun signifikan saat pesan dinolkan. Kalau masih stabil, auxiliary loss tidak bekerja.

---

## 3. Teknik lain (tunda)

### CIB (Conditional Information Bottleneck)
Butuh estimator variasional mutual information (MINE/InfoNCE). Kompleksitas implementasi tinggi, sitasi kosong. Tunda kecuali auxiliary loss terbukti tidak cukup.

### TRR (Temporal Relation Regularization)
Meminimalkan divergensi KL distribusi atensi antar langkah waktu. Menargetkan stabilitas temporal, bukan masalah "GNN diabaikan". Sitasi kosong.

Kalau nanti dipakai, kaitkan dengan analisis atensi di PLAN-06.

---

## 4. Penempatan GNN — konflik K4

### Yang harus ditolak
Saran "GNN hanya di centralized critic, actor pakai MLP lokal murni".

**Alasan penolakan:** kalau GNN hanya di critic, yang di-deploy saat eksekusi adalah MLP lokal — **secara arsitektur identik dengan `ippo`**. Seluruh klaim zero-shot transfer dan koordinasi terdistribusi runtuh.

Setupmu sekarang (GNN di actor) sudah benar untuk pertanyaan risetmu.

### Yang layak dipertimbangkan
GNN di **keduanya** (actor + critic) bisa memperbaiki credit assignment — critic melihat graf global saat training, actor pakai ego-graph saat eksekusi.

**Cek dulu:** apakah `gnn-mappo` sudah memakai centralized critic? Kalau critic-nya terdesentralisasi, menambahkan centralized critic mengubah kelas algoritma (IPPO → MAPPO sesungguhnya) — itu wave terpisah dengan pra-registrasi sendiri.

---

## 5. Ego-graph saat eksekusi (cek, bukan ubah)

**Cek implementasimu:** apakah `gnn-mappo` saat inferensi membangun graf global (5 gNB penuh) atau ego-graph per agen?

- Graf global → secara teknis bukan eksekusi terdesentralisasi murni
- Ego-graph dengan K=2 pada 5 gNB → praktis mencakup seluruh graf juga

Dengan 5 gNB perbedaannya memang kabur. Ini bukan bug, tapi **harus dideskripsikan akurat di paper** — dan jadi argumen tambahan untuk menguji di topologi lebih besar.

---

## 6. Non-stationarity (kemungkinan tidak berlaku)

Saran "freeze adjacency selama 2 langkah" menargetkan topologi dinamis.

**Setupmu:** posisi gNB di-resample tiap `reset()`, bukan tiap step. Adjacency sudah stabil dalam episode — masalah ini kemungkinan tidak berlaku.

Kecuali kalau edge weight-mu dinamis (bergantung alokasi tetangga per step). Kalau iya, pertimbangkan; kalau adjacency biner statis, lewati.

---

## 7. Simetri

Auxiliary loss hanya berlaku untuk varian GNN — baseline tidak punya encoder GNN. **Tetap sah** selama:
- Environment, reward, constraint tidak berubah
- Baseline tidak dikurangi kemampuannya
- `beta_aux` dihitung sebagai satu dimensi ruang pencarian GNN di HPO (PLAN-05), budget trial tetap identik
- Dilaporkan eksplisit sebagai perubahan yang hanya menyentuh proposed

Analoginya sama dengan edge feature: perbaikan arsitektur proposed, bukan pelemahan baseline. Hasil tanpa auxiliary loss tetap dilaporkan.

---

## 8. Urutan eksekusi

1. Konfirmasi gerbang §0
2. Cek §4 (centralized vs decentralized critic) dan §5 (ego-graph vs global)
3. Pilih target auxiliary (§2), catat di pra-registrasi
4. Implementasi `aux_head` + loss term
5. Pra-registrasi sebelum wave
6. Wave dengan varian baru, `beta_aux` disweep bersamaan HPO di PLAN-05
7. **Verifikasi:** ulangi D2a — KPI harus turun saat message passing dinolkan
8. Bandingkan dengan varian tanpa auxiliary loss

---

## 9. Larangan

1. Jangan jalankan dokumen ini tanpa D2 terkonfirmasi
2. Jangan kutip TC-GQN, CIB, TRR, QMIX-GNN, P-DGN, TD3-D-MA, "Graph-Enhanced Critic Learning" sebelum sumber aslinya diverifikasi
3. Jangan pindahkan GNN ke critic saja (konflik K4)
4. Jangan implementasikan lebih dari satu teknik anti-collapse sekaligus
5. `beta_aux` disweep di HPO, bukan ditetapkan manual
6. Wajib verifikasi dengan mengulang D2a
7. Hasil tanpa auxiliary loss tetap dilaporkan
8. Kalau auxiliary loss tidak memperbaiki apa pun, itu hasil sah
