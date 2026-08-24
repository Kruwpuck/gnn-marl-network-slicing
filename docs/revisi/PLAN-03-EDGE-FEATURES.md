# Fase 2a — Edge Features & Backbone GNN (wave v6)

**Fase:** 2a
**Prasyarat:** PLAN-01 (D2, D3) dan PLAN-02 selesai
**Keluaran:** arsitektur GNN dengan informasi fisik pada edge; prasyarat untuk analisis atensi
**Estafet:** PLAN-04 (kondisional, jika D2 konfirmasi collapse) → PLAN-05 (Fase 3)
**Master:** PLAN-00-MASTER.md

---

## 0. Gerbang masuk

Cek hasil PLAN-01 dulu:

| Hasil D2 | Tindakan |
|---|---|
| Collapse **tidak** terkonfirmasi | Jalankan dokumen ini, lewati PLAN-04 |
| Collapse terkonfirmasi | Jalankan dokumen ini, **lalu** PLAN-04 |

Dokumen ini dijalankan dalam kedua kasus — edge feature berguna terlepas dari status collapse.

| Hasil D3 | Tindakan |
|---|---|
| Over-smoothing terkonfirmasi | Jalankan §5 |
| Tidak | Lewati §5 |

---

## 1. Sitasi

### Terverifikasi
| Sumber | Status |
|---|---|
| Wang, Li, Shi, Wu, "ENGNN: A General Edge-Update Empowered GNN Architecture for Radio Resource Management in Wireless Networks," **IEEE Trans. Wireless Comm., vol. 23, no. 6, pp. 5330–5344, 2023** | Peer-reviewed |
| Brody, Alon, Yahav, "How Attentive are Graph Attention Networks?", **ICLR 2022** | Peer-reviewed |
| Dehghan Tarzjani, Krishnamachari, **arXiv:2510.14137, Okt 2025** | **PREPRINT.** Angka GCN 63.94% dan D-GCN 3.30% terkonfirmasi di abstrak |
| Jumping Knowledge GAT, **Scientific Reports (Nature) 2025** | Peer-reviewed |

### Koreksi
- **D-GCN venue:** NotebookLM menyebut "WiOpt 2026" dan "2026" secara tidak konsisten. Yang terverifikasi hanya arXiv preprint Okt 2025. Jangan tulis venue konferensi.
- **xSlice:** klaim "graf bipartit heterogen" tidak akurat. Draf paper menyebut xSlice beroperasi di level slice dengan ekstraksi fitur berbasis **GCN** yang mengagregasi UE→slice. Dan xSlice memakai GCN — bertentangan dengan rekomendasi "hindari GCN" di jawaban yang sama. **Jangan pakai sebagai bukti heterogeneous graph.**
- **Angka GATv2 1.4–11.5%:** dari task klasifikasi node dan prediksi tautan, bukan wireless RL. Jangan kutip sebagai prediksi hasil kita.

### Daftar hitam
JCPGNN-M, IC-GMRO, APS-GNN, "Dual-Graph MARL", angka UPCommons 24.59%, angka GraphSAGE 23.72% dan GINE 4.70% (belum dikonfirmasi dari isi paper). Lihat PLAN-00 §Daftar hitam.

---

## 2. Edge feature eksplisit (inti dokumen ini)

### Kondisi sekarang
Graf: 5 gNB sebagai node. Edge belum membawa fitur fisik. Ini kehilangan informasi yang paling relevan untuk koordinasi interferensi.

### Perubahan

```python
edge_attr[i,j] = [
    path_loss_db_norm,      # PL_ij, dinormalisasi
    distance_norm,          # d_ij / area_size
    interference_coupling,  # I_ij, koefisien kopling co-slice
]
```

Konsisten dengan dua sumber terverifikasi:
- **ENGNN (TWC 2023):** koefisien kanal instan `|h_ij|²` sebagai edge attribute dinamis
- **Resilient RRM (TSP 2023, PLAN-02):** path loss instan sebagai edge feature, bagian dari mekanisme transferabilitas

Kedua sumber ini juga jadi dasar PLAN-02 — jadi Fase 1 dan 2a saling menguatkan.

### Implementasi

Env sudah menghitung path loss dan matriks interferensi untuk simulasi. Yang perlu: ekspos ke graf.

```python
# env
edge_index, edge_attr = build_graph(gnb_positions, path_loss_matrix, interference_matrix)

# model
conv = GATv2Conv(in_channels, out_channels, heads=H, edge_dim=3)
h = conv(x, edge_index, edge_attr=edge_attr)
```

**`edge_dim` wajib diset.** Tanpa itu `edge_attr` diabaikan diam-diam.

### Uji wajib
```python
out_real = model(x, edge_index, edge_attr=attr_real)
out_zero = model(x, edge_index, edge_attr=torch.zeros_like(attr_real))
assert not torch.allclose(out_real, out_zero), "edge_attr tidak berpengaruh — ada bug"
```

### Interaksi dengan PLAN-02
Kalau PLAN-02 memetakan μ_u sebagai fitur graf (§5), koordinasikan format fitur supaya tidak bentrok. μ per-UE mungkin lebih cocok sebagai node feature (kalau UE jadi node) atau diagregasi ke gNB.

---

## 3. GAT → GATv2

### Dasar
Brody et al. (ICLR 2022): GAT standar hanya mampu **static attention** — peringkat kontribusi tetangga monotonik dan tidak bergantung node pusat. GATv2 membalik urutan operasi linear dan LeakyReLU sehingga menghasilkan **dynamic attention**.

Relevansi: interferensi co-slice bersifat asimetris dan bergantung alokasi instan tetangga. Static attention secara struktural tidak bisa memodelkan "tetangga X penting **ketika** beban saya tinggi".

### Implementasi
```python
from torch_geometric.nn import GATv2Conv   # ganti GATConv
```

Pertahankan varian `sage` — `gnn-madqn_sage` adalah kontra-contoh yang sedang ditelusuri.

### Penamaan
Varian baru diberi nama berbeda (`gnn-mappo_gatv2`), **jangan menimpa** `gat`. Hasil v4/v5 harus tetap bisa dirujuk.

---

## 4. Layer count

Literatur wireless umumnya 2–3 layer. Dengan 5 gNB, diameter graf kecil — 2 layer sudah menjangkau hampir seluruh graf.

Cek konfigurasi sekarang sebelum mengubah apa pun.

---

## 5. Over-smoothing (kondisional pada D3)

Jalankan hanya kalau PLAN-01 D3 mengonfirmasi over-smoothing.

Pilih **satu** (jangan dua-duanya, supaya efek terisolasi):

**Residual connection** — lebih murah, lebih mudah dijelaskan, mulai dari sini:
```python
h_next = h + alpha * conv(h, edge_index, edge_attr)   # alpha = 0.1
```

**Jumping Knowledge:**
```python
from torch_geometric.nn import JumpingKnowledge
jk = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=n_layers)
h_final = jk([h1, h2, h3])
```

---

## 6. D-GCN decoupled aggregation (tunda)

Prinsip: pisahkan kontribusi diri dari kontribusi tetangga.
```
h_i = f_self(x_i) ⊕ f_neighbor(attention_agg({x_j : j ∈ N(i)}))
```

Alasan fisiknya kuat — interferensi elektromagnetik **aditif**, bukan rata-rata. Normalisasi simetris membuat interferensi dominan tenggelam.

**Ditunda karena:**
- Sumbernya preprint, domain berbeda (p-CSMA, prediksi throughput, bukan RL)
- GATv2 + edge feature sudah memberi sebagian manfaat sama (learnable per-neighbor weights, tidak isotropik)

Pertimbangkan ulang **setelah** melihat hasil GATv2 + edge feature.

Jangan kutip angka 63.94% → 3.3% sebagai prediksi hasil kita — itu NMAE prediksi throughput p-CSMA.

---

## 7. Perketat observasi (arm terpisah — konflik K2)

### Temuan
Observasimu memuat `neighbor_urllc_frac_mean` — informasi tetangga yang **sudah teragregasi**, persis yang seharusnya dihasilkan message passing.

Artinya `ippo` di v4 bukan benar-benar "tanpa informasi tetangga". **Ablasi tiga tingkatmu bocor di tingkat pertama.** Ini kemungkinan penjelasan penting untuk hasil null.

### Perubahan
Buang `neighbor_urllc_frac_mean`. Observasi jadi:
```
[ch_gain, sinr_embb, sinr_urllc, q_embb, q_urllc, last_delay, prev_alloc]
```

Informasi tetangga hanya lewat message passing (GNN), lewat k-NN eksplisit (`mlp-knn-ppo`), atau tidak ada (`ippo`, `central-*`). Ablasi tiga tingkat jadi bersih.

### Aturan (konflik K2)
- Mengubah observation space → **wajib diterapkan ke 8 algoritma**
- Hasil tidak sebanding dengan v4/v5 → wave dengan pra-registrasi sendiri
- **Arm terpisah**, jangan digabung dengan arm edge-feature — efeknya harus terisolasi
- **Jangan** dilakukan bersamaan dengan HPO (PLAN-05)

### Rekomendasi urutan
Jalankan sebagai arm kedua di wave v6: `obs=full` (seperti v5) dan `obs=strict`. Dengan begitu efek edge feature dan efek observasi bisa dipisahkan.

---

## 8. Heterogeneous graph (tunda)

Modelkan UE dan gNB sebagai kelas node berbeda.

**Ditunda karena:**
1. Bukti pendukungnya tidak sekuat yang diklaim — xSlice memakai GCN agregasi UE→slice, bukan graf bipartit heterogen
2. APS-GNN belum terverifikasi
3. Perubahan besar: ukuran graf berubah drastis, biaya komputasi naik
4. Jumlah UE bervariasi antar-topologi — perlu dipikirkan ulang agar klaim zero-shot tetap sah

Kalau nanti dikerjakan, wave terpisah dengan pra-registrasi sendiri.

---

## 9. Simetri

Ini perubahan arsitektur GNN, hanya menyentuh proposed. **Tetap sah** selama:
- Environment, reward, constraint tidak berubah (kecuali §7, yang diterapkan ke semua)
- Baseline tidak dikurangi kemampuannya
- Budget HPO tetap identik (PLAN-05)

Wajib dilaporkan eksplisit sebagai perubahan arsitektur. Hasil v4/v5 tetap jadi pembanding.

---

## 10. Urutan eksekusi

1. Cek gerbang §0 (hasil D2, D3)
2. Implementasi edge feature (§2) + uji verifikasi
3. Ganti ke GATv2 (§3), nama varian baru
4. Residual kalau D3 terkonfirmasi (§5)
5. Pra-registrasi wave v6 dengan hipotesis eksplisit
6. Wave v6: arm `edge+gatv2`, dan `obs=strict` sebagai arm terpisah (§7)
7. Evaluasi dengan protokol sama (rliable IQM, Wilson CI, per keluarga)
8. Bandingkan eksplisit: v4 (GAT tanpa edge) vs v5 vs v6

---

## 11. Larangan

1. Jangan kutip xSlice sebagai bukti heterogeneous graph
2. Jangan kutip sumber di daftar hitam PLAN-00
3. D-GCN wajib ditandai preprint; jangan tulis venue konferensi
4. Angka GATv2 1.4–11.5% jangan dipakai sebagai prediksi hasil kita
5. Varian baru diberi nama baru, tidak menimpa varian lama
6. §7 (observasi) adalah arm terpisah, tidak digabung dengan arm edge-feature
7. §7 tidak dilakukan bersamaan dengan HPO
8. Hasil lama tetap dilaporkan sebagai pembanding
