# Fase 4 — Bukti Mekanisme & Penulisan

**Fase:** 4 (terakhir)
**Prasyarat:** PLAN-03 selesai (edge feature diperlukan untuk analisis atensi — konflik K3)
**Keluaran:** bab bukti mekanisme, struktur paper final
**Estafet:** penulisan paper
**Master:** PLAN-00-MASTER.md

---

> ## KOREKSI 2026-08-24 — §2 sudah dikerjakan, dan K3 lebih lemah dari yang ditulis
>
> **§2 (korelasi atensi vs matriks interferensi, plus ablasi kausal) sudah ada.**
> Implementasinya `scripts/attention_analysis.py`; hasilnya `results/ATTENTION_v4_greedy.md`
> dan `results/ATTENTION_v4_stoch.md`. Ablasi kausal yang §2 tandai **WAJIB** sudah
> dijalankan di sana — atensi dipaksa seragam dengan menolkan parameter `att` kedua layer
> `GATv2Conv`, kedua arm menarik noise aksi yang sama, dan hasilnya dilaporkan pada tiga
> KPI, bukan `embb_p5_mbps` sendirian, karena KPI yang terpaku di lantai tidak bisa turun
> berapa pun efek ablasinya.
>
> **Peringatan pemakaian:** paruh DQN `results/ATTENTION_v4_stoch.md` **VOID** — dihasilkan
> pembacaan ε=1.0, yaitu aksi acak seragam, bukan policy terlatih
> (`results/quarantine_eps1.0/README.md`). Pengganti yang sah untuk paruh DQN adalah
> `results/ATTENTION_v4_greedy.md`. Paruh PPO tidak terpengaruh.
>
> **Konflik K3 tidak sekuat yang ditulis.** K3 menunda analisis atensi sampai sesudah
> Fase 2 dengan alasan atensi butuh edge feature dan tanpa itu korelasi rendah cuma
> artefak. Tapi edge feature **sudah ada sejak awal**: `envs/channel_model.py:113`
> mengembalikan path loss dB sebagai `edge_attr`, dan `gnn/gat_backbone.py:38` menyetel
> `edge_dim` (default 1, perilaku v4) sehingga benar-benar dibaca. Prasyaratnya sudah terpenuhi, dan analisisnya
> memang sudah jalan. Yang belum ada cuma `interference_coupling` sebagai fitur kedua,
> jadi K3 menyusut jadi: analisis boleh diulang sesudah fitur itu ditambahkan, bukan
> analisis terlarang sebelum Fase 2.
>
> **§0 sudah benar** soal D1 dan D5 dipindah ke PLAN-01 — keduanya kini terimplementasi
> (`scripts/diag_equivariance.py`, `scripts/diag_collision.py`).
>
> Detail lengkap keenam premis basi ada di blok koreksi
> `docs/revisi/PLAN-03-EDGE-FEATURES.md` dan `docs/revisi/PLAN-01-DIAGNOSTICS.md`. Tidak
> ada verdict, ambang, atau angka hasil yang bergerak. §4-§8 (framing readout dua lapis,
> tiga cacat protokol pembacaan, struktur paper, aturan penulisan) tetap berlaku penuh.

---

## 0. Catatan urutan

Sebagian dokumen ini sudah dikerjakan lebih awal:
- **Permutation equivariance test** → dipindah ke PLAN-01 D1 (Fase 0), karena murah dan hasilnya menentukan framing
- **Uji collision storm** → dipindah ke PLAN-01 D5

Yang tersisa di sini butuh edge feature dari Fase 2.

---

## 1. Peringatan sitasi

**Belum terverifikasi:** PC-LLM ("Wireless Power Control Based on Large Language Models"), DRAMA, APS-GNN, GACG.

**Tidak relevan:** klaim "representasi bermakna terkonsentrasi pada shallow layers L ≤ 12" — itu skala kedalaman LLM/transformer. GNN-mu 2–3 layer.

**Rekomendasi yang ditolak (konflik K6):**
- *"Ganti DQN ke ε = 0.0 murni"* — kamu sudah memutuskan ε = 0.05 lewat diagnostik dengan alasan mekanistik
- *"Temperature scaling τ = 0.2 untuk evaluasi PPO"* — mengubah protokol pembacaan yang sudah dipra-registrasi (P3) setelah melihat hasil. Post-hoc

---

## 2. Korelasi atensi vs matriks interferensi

Sudah dideklarasikan di §Fallback goal1.md — sah dilaporkan.

### Prasyarat (konflik K3)
Analisis ini butuh **edge feature** dari PLAN-03. Tanpa itu, atensi tidak punya akses ke informasi fisik — korelasi rendah bukan berarti GNN gagal, cuma berarti tidak ada yang bisa dikorelasikan.

Jalankan setelah Fase 2, atau laporkan keterbatasannya eksplisit.

### Prosedur
```python
alpha = model.gnn.get_attention_weights(obs, edge_index, edge_attr)   # [n_edges]

interference_coupling = env.get_interference_matrix()   # I_ij
path_loss             = env.get_path_loss_matrix()      # PL_ij

rho_interference = spearmanr(alpha.flatten(), interference_coupling.flatten())
rho_pathloss     = spearmanr(alpha.flatten(), path_loss.flatten())
```

Laporkan **distribusi** korelasi lintas episode dan seed, bukan satu angka.

### Validasi kausal (WAJIB)

Korelasi saja bukan bukti — literatur interpretabilitas menunjukkan bobot atensi tidak selalu berkorelasi dengan kepentingan fungsional.

```python
model.gnn.force_uniform_attention = True
kpi_uniform = evaluate(model)

# KPI turun signifikan → atensi dipakai (kausal)
# KPI stabil          → atensi hanya dekorasi
```

**Tanpa ablasi ini, analisis atensi tidak boleh disebut bukti.**

---

## 3. Graph perturbation test (opsional)

Perturbasi ringan (geser koordinat 10%, hapus edge lemah), ukur divergensi Jensen-Shannon representasi laten.

Berguna, tapi lebih kompleks dari permutation test dan metrik `S_aug` yang diberikan tidak punya sitasi terverifikasi. Permutation test (PLAN-01 D1) sudah memberi bukti struktural yang lebih tajam.

Kerjakan hanya kalau ada waktu tersisa.

---

## 4. Framing readout — dua lapis

### Perubahan framing yang disarankan

Saat ini greedy collapse dibingkai sebagai **artefak protokol pembacaan**. Framing yang lebih kuat dan lebih jujur: **batas validitas operasional**.

Bedanya penting:
- *Artefak pembacaan* → masalah metodologi, pakai stokastik saja
- *Batas validitas operasional* → temuan tentang model. Operator butuh kebijakan deterministik; model yang kolaps di argmax **belum siap deploy**, sebagus apa pun performa stokastiknya

### Data pendukung

| model | selisih throughput stokastik − greedy |
|---|---|
| gnn-mappo_gat | −51.19 |
| gnn-mappo_sage | −28.77 |
| ippo | +0.46 |
| central-ppo | −1.01 |

Dibaca sebagai kesiapan operasional: `ippo` dan `central-ppo` menghasilkan kebijakan deterministik yang stabil. Varian GNN-MAPPO tidak. Itu **kekalahan proposed di dimensi yang penting untuk deployment** — D3 mewajibkan melaporkannya setara dengan kemenangan.

### Tindakan
Tulis ulang bagian readout di `paper_structure.md` dengan dua lapis:
1. **Batas validitas operasional** — temuan tentang model
2. **Ancaman validitas metodologis** — kenapa protokol harus di-gate sebelum melihat hasil

Dua-duanya benar dan saling melengkapi.

---

## 5. Kontribusi metodologis: instrumen yang melaporkan sehat padahal salah

Sekarang ada **empat** instansi independen, semuanya menghasilkan **keluaran yang terlihat
valid**. Tiga di jalur pembacaan hasil, satu di jalur verifikasi dokumen:

| # | Cacat | Akibat |
|---|---|---|
| 1 | Greedy/argmax pada PPO | `gnn-mappo_gat` terbaca 16.77 vs `ippo` 68.06 → "proposed WORSE, CI terpisah" |
| 2 | ε=1.0 pada zero-shot `central-dqn` | Model yang secara struktural CANNOT_RUN terbaca sebagai baris OK |
| 3 | ε=1.0 pada collapse rate DQN | Seluruh nilai `embb_p5` = 0 terbaca sebagai kolaps cell-edge, padahal artefak aksi acak |
| 4 | `citation_audit.py --update` mengunci ulang anchor ke baris yang **salah**, lalu exit 0 (2026-08-25) | Enam kutipan menunjuk kode tak berhubungan; dua di antaranya (`results/B3_DELAY_CENSORING.md`, `results/GATE_C.md`) sudah melenceng sebelum sesi itu dan **lolos audit setiap kali dijalankan** |

Itu pola, bukan kebetulan. Beri subsection sendiri dengan kontrafaktualnya. Yang
menyatukannya bukan "protokol pembacaan" melainkan bentuk yang lebih umum: **instrumen
verifikasi yang melaporkan sehat sementara yang diverifikasinya salah.** Cacat #4
memperlihatkan polanya berlaku di luar jalur hasil — `--update` mempercayai apa pun yang
kebetulan ada di nomor baris itu, jadi tiap kali kode digeser ia mengganti kutipan yang benar
dengan kutipan yang salah dan menyatakan audit lolos.

**Kontrafaktual yang paling kuat:** tanpa perbaikan cacat #1, laporan akan menyimpulkan GNN kalah telak (16.77 vs 68.06, CI terpisah) — kesimpulan yang sepenuhnya artefak.

**Kontrafaktual #4:** tanpa memeriksa tiap anchor terhadap baris aslinya, laporan menyatakan
43 kutipan terverifikasi sementara enam menunjuk baris yang keliru — dan pemeriksaan itu
harus manual, karena instrumennya sendiri yang rusak. Perbaikannya bukan menambah audit
kedua di atas audit pertama: `--update` tidak boleh dipercaya untuk **memperbaiki** anchor,
cuma untuk melaporkan bahwa anchor sudah bergeser.

**Kaitkan dengan kritik simulator:** cacat ε=1.0 adalah instansi persis dari "ilusi throughput valid dari alokasi acak" yang dikritik di literatur simulator abstrak. Env v3 kamu sudah memakai packet-level queue dengan finite buffer dan deadline drop — jauh di atas simulator Shannon murni. Yang masih terbuka: HARQ retransmission dan control-plane delay tidak dimodelkan. Tulis sebagai keterbatasan eksplisit.

---

## 6. Penerimaan di venue — penilaian realistis

Klaim bahwa analisis mekanisme "sudah jadi standar baru" di INFOCOM/TWC/ToN/CoNEXT tidak punya sitasi. Penilaian yang lebih hati-hati: interpretabilitas diterima sebagai **kontribusi pendukung** dan makin dihargai, tapi paper murni interpretabilitas tanpa hasil performa tetap sulit di venue kelas satu.

Kekuatan paketmu bukan interpretabilitas sendirian, melainkan kombinasinya:

1. **Struktural** — `central-*` kategoris tidak dapat dievaluasi di luar topologi latihnya
2. **Mekanistik** — parameter sharing (bukan message passing) menentukan transferabilitas; didukung ablasi tiga tingkat + permutation test (PLAN-01 D1)
3. **Metodologis** — tiga cacat protokol pembacaan dengan kontrafaktual
4. **Artefak** — benchmark terkalibrasi + protokol kalibrasi terhadap policy terlatih

Itu paper empirical-study/benchmark yang solid. **Jangan jual sebagai paper "GNN menang".**

---

## 7. Struktur paper (integrasikan seluruh fase)

### Tesis tiga tingkat (dari penetapan 2026-08-17, diperbarui)

**Tingkat 1 — Struktural (utama)**
Baseline terpusat kategoris tidak dapat dievaluasi di luar topologi latihnya (`obs_dim` terkunci). Tidak terbantahkan pada n=5.

**Tingkat 2 — Mekanistik (utama)**
Transferabilitas ditentukan parameter sharing per-agen dengan observasi lokal. Ablasi tiga tingkat: `ippo` 0.954 / `mlp-knn-ppo` 0.949 / GNN 0.942–0.955 — tidak menunjukkan beda. Berdiri di kedua keluarga.

Diperkuat PLAN-01 D1: equivariance permutasi sebagai properti struktural yang dibuktikan langsung.

**Tingkat 3 — Trade-off PPO (turunan)**
Spesifik-keluarga, dengan kontra-contoh DQN eksplisit:
> Di keluarga PPO, kami mengamati trade-off antara proteksi cell-edge dan transferabilitas: `central-ppo` jauh lebih jarang kolaps (3/20) tetapi secara struktural tidak dapat dievaluasi di luar topologi latihnya, sementara seluruh varian per-agen transfer pada retensi ~0.95 dengan collapse rate 14–20/20. Trade-off ini tidak berlaku di keluarga DQN: `gnn-madqn_sage` mencapai 0/5 kolaps dan retensi 0.955 secara bersamaan, meskipun perbedaan collapse rate dalam keluarga DQN tidak terpisah secara statistik pada n=5. Kami melaporkan trade-off sebagai temuan spesifik-keluarga, bukan properti umum arsitektur.

### Bab tambahan dari fase-fase baru
- **Bab kalibrasi:** protokol kalibrasi terhadap policy terlatih (dari Gate A), penjelasan mekanistik saturasi KPI
- **Bab constraint:** per-**cell** resilient constraint (PLAN-02, unit dikoreksi di Q1), hasil
  v5 vs v4 — **dan** temuan bahwa pada titik operasi v4 tidak ada `f_min` yang sekaligus
  feasible dan mengikat bagi baseline referensi non-GNN, karena kedua constraint berebut PRB
  yang sama. Itu hasil negatif yang sah dan dilaporkan penuh (PLAN-00 aturan 9), bukan bagian
  yang dihilangkan karena wave-nya belum jalan. Angkanya di `PREREG-V5.md` §0
- **Bab arsitektur:** edge feature + GATv2 (PLAN-03), hasil v6. Argumen yang dipakai untuk
  fitur edge: SINR pada observasi node adalah **agregat seluruh interferer**, satu penjumlahan
  yang membuang identitas penyumbangnya; fitur edge memberi **dekomposisi per-tetangga** dari
  agregat yang sama — informasi yang secara matematis hilang di dalam penjumlahan itu. Agen
  tahu "SINR saya jelek", tanpa fitur edge tidak tahu "gara-gara tetangga mana"
- **Bab HPO:** protokol simetris + equal parameter budget (PLAN-05), hasil v7
- **Bab bukti mekanisme:** permutation test, atensi + ablasi kausal, tiga cacat readout
- **Keterbatasan:** HARQ dan control-plane delay tidak dimodelkan; C4 keluarga DQN tidak terpenuhi; B3 gagal karena sensor deadline

---

## 8. Aturan penulisan yang mengikat

Warisan keputusan scoping, tetap berlaku:

1. `urllc_delay_p99` dilaporkan tapi **tidak diklaim** (B3 GAGAL pra-registrasi)
2. `embb_p5_mbps` level dilaporkan beserta saturasinya; klaim hanya dalam bentuk diskret (`collapse_rate`)
3. Collapse rate keluarga DQN: "kolaps di k dari 5 seed", **bukan** "kolaps X% waktu" (C4 GAGAL)
4. Klaim komparatif collapse hanya sah di keluarga PPO
5. Kekalahan proposed dilaporkan dengan aturan yang sama dengan kemenangan (D3)
6. Verdict CI dipatuhi apa adanya

---

## 9. Larangan

1. Jangan ubah protokol pembacaan yang sudah dipra-registrasi (konflik K6)
2. Jangan ubah ε DQN dari 0.05
3. Jangan kutip PC-LLM, DRAMA, APS-GNN, GACG sebelum diverifikasi
4. Analisis atensi tanpa ablasi kausal bukan bukti — jangan dilaporkan sebagai bukti
5. Jangan klaim interpretabilitas "sudah jadi standar" tanpa sitasi
6. Jangan jual paper sebagai "GNN menang" kalau datanya tidak mendukung
7. Framing "batas validitas operasional" tidak boleh dipakai menyembunyikan bahwa proposed kalah di dimensi itu
