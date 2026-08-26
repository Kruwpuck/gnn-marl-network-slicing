# Fase 2a — Edge Features & Backbone GNN (wave v6)

**Fase:** 2a
**Prasyarat:** PLAN-01 (D2, D3, D6) selesai. **PLAN-02 bukan prasyarat** sejak 2026-08-26 —
dokumen ini tidak memakai keluarannya, dan wave v5 menunggu keputusan titik operasi
(PLAN-00, blok URUTAN EKSEKUSI DITUKAR)
**Keluaran:** arsitektur GNN dengan informasi fisik pada edge; prasyarat untuk analisis atensi
**Estafet:** PLAN-04 (kondisional, jika D2 konfirmasi collapse) → PLAN-05 (Fase 3)
**Master:** PLAN-00-MASTER.md

---

> ## KOREKSI 2026-08-24 — empat premis dokumen ini tidak lagi berlaku
>
> Dokumen ini ditulis dari jawaban NotebookLM tanpa akses ke kondisi kode terkini. Empat
> premisnya diverifikasi ulang terhadap file dan ternyata sudah tidak berlaku. Yang di
> bawah **tidak dihapus** — dibiarkan utuh supaya jejak apa yang diyakini tetap ada — tapi
> jangan dikerjakan.
>
> **P1 — §3 "ganti GAT → GATv2" sudah selesai sebelum dokumen ini ditulis.**
> `gnn/gat_backbone.py:5` mengimpor `GATv2Conv`, dan benar-benar dipakai untuk kedua layer
> (`gnn/gat_backbone.py:29,32`) — bukan sekadar diimpor. Tidak ada
> penggantian, tidak ada varian baru `gnn-mappo_gatv2`, tidak ada penamaan ulang.
> Konsekuensinya lebih besar dari §3 itu sendiri: **varian ini dinamai `gat` padahal
> isinya GATv2 sejak v1.** Setiap kalimat paper yang menulis "GAT" untuk varian ini salah,
> dan sitasi Brody et al. (ICLR 2022) yang §3 siapkan sebagai justifikasi *perubahan*
> sebenarnya adalah sitasi untuk arsitektur yang sudah dipakai. Nama varian **tidak
> diubah** — itu memutus seluruh nama checkpoint, CSV, dan laporan v4 — jadi ini koreksi
> terminologi paper, bukan koreksi kode.
>
> **P2 — §2 "edge belum membawa fitur fisik" salah.** `envs/channel_model.py:113`
> mengembalikan `edge_attr (E,1)` berisi path loss dB; `gnn/base_backbone.py:48`
> menskalakannya `/100.0`; `gnn/gat_backbone.py:30,33` menyetel `edge_dim=1`. Uji wajib §2
> ("`edge_dim` wajib diset, tanpa itu `edge_attr` diabaikan diam-diam") sudah lolos sejak
> awal. Dari tiga fitur usulan, yang benar-benar belum ada **hanya
> `interference_coupling`**.
>
> Tambahan yang membatasi §2: **`SAGEConv` tidak menerima `edge_attr` sama sekali**
> (`gnn/sage_backbone.py:19-20`). Pekerjaan edge feature apa pun hanya menyentuh varian
> `gat`; varian `sage` tidak bisa dibandingkan pada dimensi itu.
>
> **P3 — `distance_norm` sebagai fitur edge ketiga redundan.**
> `envs/channel_model.py:34` menyatakan path loss dihitung **tanpa shadow fading**, dan
> graf inter-gNB selalu memanggilnya `los=True` (`envs/channel_model.py:124`) dengan
> `d3d = d2d` (`envs/channel_model.py:123`).
> Path loss jadi fungsi monoton murni dari jarak — piecewise log, naik ketat. `distance_norm`
> adalah reparametrisasi bijektif dari fitur yang sudah ada: nol informasi baru.
> Menambahkannya menghasilkan "tiga fitur edge" di paper yang sebenarnya dua.
>
> **P4 — §7 "buang `neighbor_urllc_frac_mean`" sudah dikerjakan di v3.**
> `envs/network_slicing_env.py:22` mencatat penghapusannya beserta penggantinya
> (`prev_alloc_lag2`), dan `_get_obs()` (`envs/network_slicing_env.py:537`) memang tidak
> memuatnya. Satu-satunya
> kemunculan nama itu di seluruh repo adalah catatan historis tersebut. Tiga akibat:
> 1. **Arm `obs=strict` di wave v6 hilang seluruhnya.** §7 dan urutan eksekusi §10 poin 6
>    tidak punya pekerjaan tersisa.
> 2. **Konflik K2 selesai dengan sendirinya.** Batasan "jangan perbaiki observasi
>    bersamaan dengan HPO" di PLAN-05 §3.3 dan larangan §10.6 tidak lagi mengikat apa pun.
> 3. Kalimat §7 *"ablasi tiga tingkatmu bocor di tingkat pertama. Ini kemungkinan
>    penjelasan penting untuk hasil null"* **batal** — ablasinya tidak bocor. Hasil null v4
>    (`ippo` 0.954 / `mlp-knn-ppo` 0.949 / GNN 0.942-0.955) kembali tanpa penjelasan dari
>    jalur ini.
>
> **P5 — §4 "dengan 5 gNB, diameter graf kecil" salah arah.**
> `envs/channel_model.py:112` menyatakannya sendiri — *"Fully-connected inter-gNB
> interference graph"* — dan loop di bawahnya menerbitkan tiap pasangan `(i,j), i≠j`
> sebagai edge. Bukan "diameter kecil" — diameter **1**. Satu layer sudah menjangkau
> seluruh graf, dan dua layer berarti tiap node mengagregasi himpunan tetangga yang
> identik dua kali. Risiko over-smoothing **lebih tinggi**, bukan lebih rendah: D3 jadi
> lebih relevan, bukan kurang.
>
> Tidak ada verdict, ambang, atau angka hasil yang bergerak karena koreksi ini. Yang
> bergerak adalah daftar pekerjaan Fase 2.

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

**Terjawab 2026-08-24** (PLAN-01 §Keluaran, dari `results/DIAG_GNN_RELIANCE.md`):

- **D2 collapse terkonfirmasi** → jalankan dokumen ini, **lalu PLAN-04**.
- **D3 over-smoothing terkonfirmasi** → **jalankan §5**. Tak terbantahkan untuk `gat`:
  cosine similarity embedding **1.0000 persis di 25/25 checkpoint**, sementara referensi
  pada observasi mentahnya 0,7955–0,9844. `sage` 0,9464–0,9999, tinggi tapi tidak
  degenerate. Sesuai §5, pilih **satu** teknik saja supaya efeknya terisolasi.

Satu hasil D2 yang mengikat §2 dokumen ini: **atribut edge yang sudah ada tidak terpakai
sama sekali** — D2b mengacak path loss per node tujuan dan menggerakkan 0/25 checkpoint di
ketiga KPI. Menambahkan `interference_coupling` karena itu wajib disertai uji bahwa fitur
baru benar-benar dibaca, bukan diasumsikan. Uji `assert not torch.allclose(...)` di §2
memeriksa `edge_attr` berpengaruh pada keluaran; itu **tidak** membuktikan policy
memakainya. Ulangi D2b sesudah fitur ditambahkan.

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

> ## KOREKSI 2026-08-26 — isi `edge_dim` dinyatakan eksplisit, dan D6 menambah satu argumen
>
> **Argumen tambahan dari D6.** `prev_alloc` dan `prev_alloc_lag2` punya std antar-node
> **0,0000 di 49/50** checkpoint (`results/DIAG_INPUT_SEPARABILITY.md`) — tiap gNB memilih
> aksi yang sama, jadi dua dari delapan kolom observasi tidak membawa identitas node sama
> sekali. Observasi efektifnya **6 dimensi**, dan median "6 dari 8 kolom bervariasi" yang D6
> laporkan **sudah termasuk** keduanya sebagai nol. Sumber informasi pembeda yang baru karena
> itu harus datang dari **struktur graf**, bukan dari kolom node yang sudah jenuh. Itu
> argumen untuk §2 yang tidak ada waktu dokumen ini ditulis.
>
> **Kedua kolom itu tidak dibuang.** Kalau §5 memperbaiki separasi dan lockstep pecah,
> keduanya hidup lagi — nilainya nol karena policy-nya degenerate, bukan karena kolomnya
> tidak berguna. Dicatat sebagai bukti umpan balik lockstep ke observasi, bukan sebagai
> daftar pekerjaan.
>
> **Isi `edge_dim`, supaya arm terbaca sebagai "menambahkan kopling interferensi" dan bukan
> sekadar "`edge_dim` naik":**
>
> | dim | v4 (`gat`, `edge_dim=1`) | v6 (`gatedge`, `edge_dim=2`) | asal |
> |---|---|---|---|
> | 0 | path loss dB gNB→gNB | **sama, tidak berubah** | `envs/channel_model.py:124`, fungsi posisi gNB saja |
> | 1 | — | `interference_coupling` dB | daya interferer relatif terhadap serving link, di UE tujuan |
>
> `distance_norm` **tidak** ditambahkan: P3 di blok koreksi 2026-08-24 sudah mematikannya
> (reparametrisasi bijektif dari path loss, nol informasi baru). Jadi **dua** fitur, bukan
> tiga seperti tertulis di bawah.

### Kondisi sekarang
Graf: 5 gNB sebagai node. Edge belum membawa fitur fisik. Ini kehilangan informasi yang paling relevan untuk koordinasi interferensi.

### Perubahan

```python
edge_attr[i,j] = [
    path_loss_db_norm,      # PL_ij, dinormalisasi
    distance_norm,          # d_ij / area_size    <- DICORET, lihat P3
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

> **D6 (2026-08-25) mempersempit apa yang §5 harus lakukan** —
> `results/DIAG_INPUT_SEPARABILITY.md`, 50 checkpoint.
>
> Premis §5 diverifikasi dan **berlaku**: input tidak degenerate, 0/50 checkpoint di bawah
> ambang (dan 0/50 juga di kedua ambang sensitivitas), median 6 dari 8 kolom observasi
> bervariasi antar-node. Jadi node memang bisa dibedakan sebelum masuk GNN, dan §5 menyasar
> tempat yang benar — bukan §2/§7.
>
> Tapi **kolapsnya di layer pertama, bukan menumpuk di dua layer.** Rasio `rel_spread`
> antar-tahap untuk `gat`: input→`conv1` **0,0145**, `conv1` pra→pasca aktivasi 0,9861,
> `conv1`→`conv2` 0,0079. Layer pertama sudah membuang 98,6% separasi; aktivasinya praktis
> tidak bersalah. Konsisten dengan P5 di blok koreksi atas: graf lengkap, diameter 1, satu
> layer sudah merata-ratakan kelima node.
>
> **Konsekuensi konkret untuk pilihan di bawah:** residual `h + α·conv(h)` yang dipasang
> pada layer kedua saja **tidak akan menolong** — yang perlu diselamatkan adalah `x`, dan
> `x` sudah hilang di `conv1`. Sambungannya harus melewati layer pertama: residual pada
> **kedua** layer dengan proyeksi untuk `x` (dimensinya beda — 8 → hidden·heads → 64), atau
> Jumping Knowledge yang memang menggabungkan `[x, h1, h2]`. Tetap pilih **satu**, dan catat
> mana.
>
> `sage` mempertahankan jauh lebih banyak pada rasio yang sama (0,6343 dan 0,3464,
> `rel_spread` akhir 0,0348 lawan 0,0000 milik `gat`). Itu penjelasan mekanistik untuk beda
> `gat`/`sage` di D3 yang selama ini cuma angka: `SAGEConv` menyimpan bobot root terpisah
> (`gnn/sage_backbone.py:19`), jadi kontribusi diri tidak ikut terata-rata. Residual pada
> `gat` pada dasarnya menambahkan kembali apa yang `sage` sudah punya secara bawaan — dan
> itu prediksi yang bisa diuji, bukan sekadar analogi.

Jalankan hanya kalau PLAN-01 D3 mengonfirmasi over-smoothing.

Pilih **satu** (jangan dua-duanya, supaya efek terisolasi).

**Yang dipilih (2026-08-26): residual yang menjangkau input.** Bukan residual antar-layer —
D6 menutup bentuk itu. `x` sudah hilang di `conv1`, jadi sambungan yang berangkat dari `h`
tidak punya apa pun untuk diselamatkan:

```python
h1 = F.elu(self.conv1(x, edge_index, edge_attr=edge_attr))
h1 = h1 + alpha * self.proj1(x)      # proyeksi: 8 -> hidden*heads
h2 = self.conv2(h1, edge_index, edge_attr=edge_attr)
h2 = h2 + alpha * self.proj2(x)      # proyeksi: 8 -> out_dim
```

`alpha = 0.1` seperti tertulis sejak versi awal §5 — properti dokumen, bukan pilihan dari
hasil. Proyeksi diperlukan karena dimensinya memang beda (8 → hidden·heads → 64); tanpa itu
penjumlahannya tidak terdefinisi.

**Jumping Knowledge** (`[x, h1, h2]`) adalah alternatif yang **tidak** diambil, dicatat
supaya pilihannya bisa ditinjau:

```python
from torch_geometric.nn import JumpingKnowledge
jk = JumpingKnowledge(mode='lstm', channels=hidden_dim, num_layers=n_layers)
h_final = jk([x_proj, h1, h2])
```

Alasan tidak diambil: residual lebih murah (dua `nn.Linear` tanpa bias lawan satu LSTM), dan
ia **uji langsung atas prediksi mekanistik D6** — `SAGEConv` menyimpan bobot root terpisah
(`gnn/sage_backbone.py:19`) dan mempertahankan `rel_spread` 0,0348 lawan 0,0000 milik `gat`;
residual pada `gat` menambahkan kembali persis sifat itu. Kalau `gatres` tidak mendekati
`sage` pada `rel_spread`, prediksinya salah dan itu hasil yang dilaporkan.

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
3. ~~Ganti ke GATv2 (§3), nama varian baru~~ — **tidak ada pekerjaan tersisa** (P1: sudah
   GATv2 sejak v1). Nama varian baru tetap wajib, tapi karena §5/§2, bukan karena §3
4. Residual kalau D3 terkonfirmasi (§5)
5. Pra-registrasi wave v6 dengan hipotesis eksplisit — `docs/revisi/PREREG-V6.md`
6. Wave v6: empat arm `gat` / `gatres` / `gatedge` / `gatres-edge`. ~~`obs=strict`~~ **tidak
   ada** (P4: `neighbor_urllc_frac_mean` sudah dibuang di v3, §7 tidak menyisakan pekerjaan)
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
