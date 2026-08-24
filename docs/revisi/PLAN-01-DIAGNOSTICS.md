# Fase 0 — Diagnostik

**Fase:** 0 (pertama, kerjakan sekarang)
**Prasyarat:** tidak ada — pakai checkpoint v4 yang sudah ada
**Keluaran:** jawaban atas 5 pertanyaan yang menentukan seluruh fase berikutnya
**Estafet:** PLAN-02 (selalu) → PLAN-03 (selalu) → PLAN-04 (kondisional, tergantung D2)
**Biaya:** tanpa GPU training, perkiraan 1–2 hari
**Master:** PLAN-00-MASTER.md

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

- [ ] D1: varians permutasi per algoritma — tabel lengkap 8 algoritma
- [ ] D2: collapse terkonfirmasi ya/tidak — **gerbang untuk PLAN-04**
- [ ] D3: over-smoothing ya/tidak — gerbang untuk PLAN-03 §5
- [ ] D4: tabel jumlah parameter — gerbang untuk arm `ippo-scaled`
- [ ] D5: collision storm terkonfirmasi ya/tidak

Catat semuanya di ledger. Ini diagnostik, bukan klaim performa — tidak butuh pra-registrasi, tapi hasilnya dilaporkan apa adanya.

---

## Larangan

1. Jangan mengubah kode training di fase ini — murni evaluasi checkpoint yang ada
2. Jangan lanjut ke PLAN-04 tanpa D2 terkonfirmasi
3. Jangan menulis penjelasan collision storm di paper sebelum D5 terkonfirmasi
4. Kalau D1 menunjukkan MLP per-agen juga equivariant, laporkan apa adanya
