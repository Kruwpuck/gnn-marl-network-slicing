# Master Plan — Estafet Rencana Perbaikan Riset

**Basis:** wave v4 selesai, B3 & C4 GAGAL tercatat, klaim ter-scoping
**Sumber:** enam blok riset literatur (NotebookLM), sudah diverifikasi ulang
**Status keseluruhan:** NOT DONE menurut goal1.md — dan itu keadaan yang benar

---

## Cara membaca dokumen ini

Ada enam rencana turunan. Masing-masing berdiri sendiri, tapi **urutannya mengikat**: keluaran satu fase jadi prasyarat fase berikutnya. Jangan lompat.

Tiap dokumen punya blok header yang sama:
- **Fase** — posisi di estafet
- **Prasyarat** — apa yang harus selesai dulu
- **Keluaran** — apa yang dihasilkan untuk fase berikutnya
- **Estafet** — dokumen mana yang lanjut

---

## Peta estafet

```
FASE 0  Diagnostik (tanpa training, checkpoint v4)
        ├─ PLAN-01-DIAGNOSTICS.md
        └─ keluaran: tahu apakah GNN dipakai, apakah equivariant,
           berapa parameter tiap model, apakah over-smoothing
                          │
                          ▼
FASE 1  Perbaikan formulasi  →  wave v5
        ├─ PLAN-02-RESILIENT-CELLEDGE.md
        └─ keluaran: per-UE constraint, collapse rate baru
                          │
                          ▼
FASE 2  Perbaikan arsitektur  →  wave v6
        ├─ PLAN-03-EDGE-FEATURES.md      (selalu)
        ├─ PLAN-04-ANTI-COLLAPSE.md      (hanya jika Fase 0 konfirmasi collapse)
        └─ keluaran: arsitektur GNN final
                          │
                          ▼
FASE 3  HPO simetris  →  wave v7
        ├─ PLAN-05-HPO-SYMMETRIC.md
        └─ keluaran: config terbaik per algoritma, hasil post-HPO
                          │
                          ▼
FASE 4  Bukti mekanisme & penulisan
        ├─ PLAN-06-MECHANISM-EVIDENCE.md
        └─ keluaran: analisis atensi, bab XAI, struktur paper final
```

**PLAN-07-CMDP-NOTES.md** bukan fase — itu catatan referensi yang dipakai Fase 1 dan Fase 4.

---

## Ringkasan tiap fase

### FASE 0 — Diagnostik (kerjakan sekarang)

Tanpa GPU training. Seluruhnya pada checkpoint v4 yang sudah ada. Perkiraan: 1–2 hari.

| Uji | Menjawab |
|---|---|
| Permutation equivariance (120 permutasi) | Apakah equivariance sumber transferabilitas? |
| Ablasi message passing | Apakah policy head memakai output GNN? |
| Randomisasi pesan tetangga | Apakah policy sensitif struktur graf? |
| Rasio norma gradien | Apakah jalur GNN terlatih? |
| Cosine similarity embedding | Over-smoothing? |
| Hitung parameter 8 algoritma | Kapasitas timpang? |
| Korelasi aksi greedy vs stokastik | Hipotesis collision storm |

**Gerbang keputusan:** hasil Fase 0 menentukan apakah PLAN-04 dijalankan.

### FASE 1 — Per-UE resilient constraint (wave v5)

Perubahan paling didukung: dua jalur analisis independen menunjuk solusi yang sama.
- Preseden empiris: resilient RRM (TSP 2023)
- Penjelasan mekanistik: constraint agregat secara struktural mengizinkan pengorbanan minoritas

Objective tidak diubah. Constraint kedua ditambahkan di atas infrastruktur primal-dual yang ada.

### FASE 2 — Arsitektur (wave v6)

- **Selalu:** edge feature eksplisit (path loss, jarak, kopling interferensi) + GATv2
- **Kondisional:** auxiliary loss, hanya kalau Fase 0 mengonfirmasi representation collapse
- **Kondisional:** perketat observasi (buang `neighbor_urllc_frac_mean`), lihat catatan konflik di bawah

### FASE 3 — HPO simetris (wave v7)

Budget identik untuk 8 algoritma. Menutup ancaman validitas yang diakui eksplisit di literatur: default hyperparameter dapat mengubah peringkat.

Ada masalah biaya nyata untuk keluarga DQN — lihat PLAN-05 §4.

### FASE 4 — Bukti mekanisme

Analisis atensi + ablasi kausal. Butuh edge feature dari Fase 2, jadi tidak bisa lebih awal.

---

## Konflik yang sudah diselesaikan

Beberapa saran antar-blok bertabrakan. Ini keputusannya, mengikat untuk seluruh dokumen:

### K1 — Status μ (dual per-UE) terhadap transferabilitas

- **Salah (versi awal PLAN-01 lama):** "μ dan z tidak boleh ikut transfer"
- **Benar:** μ sebagai **input** (node/edge attribute) aman dan justru bagus — dimensi node feature tetap, jumlah node yang berubah. Yang merusak transferabilitas adalah μ sebagai **parameter tersimpan di checkpoint**.
- Berlaku di: PLAN-02 §5

### K2 — Kapan observasi diperketat

- Membuang `neighbor_urllc_frac_mean` mengubah observation space untuk 8 algoritma → wave terpisah
- **Tidak boleh** dilakukan bersamaan dengan HPO (efek jadi tidak terpisah)
- **Keputusan:** kerjakan di Fase 2 sebagai arm terpisah, bukan digabung ke arm edge-feature
- Berlaku di: PLAN-03 §7, PLAN-05 §3.3

### K3 — Urutan analisis atensi

- Analisis atensi tanpa edge feature akan menunjukkan korelasi rendah, tapi itu artefak (tidak ada informasi fisik untuk dikorelasikan)
- **Keputusan:** Fase 4 setelah Fase 2, atau laporkan keterbatasan eksplisit
- Berlaku di: PLAN-06 §3

### K4 — Penempatan GNN

- Saran "GNN hanya di critic" **ditolak** — itu membuat actor jadi MLP saat eksekusi, menghapus dasar klaim transferabilitas
- Berlaku di: PLAN-04 §4

### K5 — Parameter dual yang sudah lolos Gate A

- `α_λ`, `λ_init`, `λ_max`, `dual_update_every` **tidak diubah**
- Angka dari literatur tidak konsisten (α_λ = 2.0 vs 10⁻³) dan milikmu sudah terkalibrasi
- Berlaku di: PLAN-07 §3

### K6 — Protokol pembacaan

- P3 sudah dipra-registrasi. Usulan temperature scaling τ=0.2 dan ε=0.0 untuk DQN **ditolak** — post-hoc
- Berlaku di: PLAN-06 §1

---

## Daftar hitam sitasi

Berlaku untuk seluruh dokumen dan paper. Jangan dikutip:

| Sumber | Alasan |
|---|---|
| "Diagnosis Ilmiah Kegagalan GNN-MARL pada Slicing Jaringan 5G..." | Dokumen internal yang di-upload ke NotebookLM. Sitasi sirkular |
| Angka 68.06 vs 68.87 sebagai "temuan literatur" | Itu hasil wave v4 sendiri |
| `[cite: 37]` dan rujukan internal lain | Sama, sirkular |
| "The scalability of coordination policies in swarm robotics" (MDPI) | MDPI tidak menerbitkan jurnal bernama itu |
| JCPGNN-M, APS-GNN, EExApp, IC-GMRO, TELGEN, P-DGN, TC-GQN, QMIX-GNN, TD3-D-MA, PC-LLM, DRAMA, GACG | Belum terverifikasi keberadaannya |
| Angka "GAT unggul 24.59%" (UPCommons) | Repositori tesis, bukan peer-reviewed |
| xSlice sebagai bukti heterogeneous graph | Mekanismenya GCN agregasi UE→slice, bukan graf bipartit heterogen |

**Tindakan wajib:** hapus dokumen internal dari sumber NotebookLM sebelum query berikutnya.

---

## Sitasi terverifikasi — aman dipakai

| Sumber | Dipakai di |
|---|---|
| NaderiAlizadeh, Eisen, Ribeiro, "Learning Resilient Radio Resource Management Policies With Graph Neural Networks," **IEEE Trans. Signal Processing, vol. 71, pp. 995–1009, 2023**. arXiv:2203.11012. Kode: github.com/navid-naderi/Resilient_RRM_GNN | PLAN-02 |
| Wang, Li, Shi, Wu, "ENGNN: A General Edge-Update Empowered GNN Architecture for Radio Resource Management," **IEEE Trans. Wireless Comm., vol. 23, no. 6, pp. 5330–5344, 2023** | PLAN-03 |
| Brody, Alon, Yahav, "How Attentive are Graph Attention Networks?", **ICLR 2022** | PLAN-03 |
| Dehghan Tarzjani, Krishnamachari, "Learning Wireless Interference Patterns: Decoupled GNN...", **arXiv:2510.14137, 2025** — PREPRINT | PLAN-03 |
| "Using Graph Neural Networks in Reinforcement Learning: A Practical Guide", **ICLR Blogposts 2026** — blogpost track | PLAN-05 |
| Shen, Shi, Zhang, Letaief, "Graph Neural Networks for Scalable Radio Resource Management," **IEEE JSAC, vol. 39, no. 1, pp. 101–115, 2021** | PLAN-01, PLAN-03 |
| Jumping Knowledge GAT, **Scientific Reports (Nature) 2025** | PLAN-03 |

---

## Aturan yang berlaku lintas seluruh fase

Warisan dari goal1.md, tetap mengikat:

1. Perubahan environment/reward/constraint diterapkan **identik** ke 8 algoritma
2. Perubahan arsitektur GNN hanya menyentuh proposed — sah, tapi wajib dilaporkan eksplisit dan hasil lama tetap jadi pembanding
3. Parameter dipilih dengan justifikasi properti **task**, tidak pernah properti **hasil**
4. Metrik primer dikunci sebelum wave; tidak ada promosi metrik post-hoc
5. Setiap wave baru wajib pra-registrasi sebelum dijalankan
6. Keluarga DQN dan PPO tidak pernah digabung dalam klaim statistik
7. Hasil lama tetap dilaporkan penuh; wave baru menambah, bukan mengganti
8. Seed yang kolaps tidak dibuang
9. Hasil null tetap dilaporkan sebagai hasil sah
10. Verdict CI dipatuhi apa adanya — CI tumpang tindih berarti COMPARABLE

---

## Yang tidak berubah

Apa pun hasil fase-fase ini, temuan v4 tetap berdiri dan tetap dilaporkan:

1. **Struktural:** `central-*` kategoris tidak dapat dievaluasi di luar topologi latihnya (`obs_dim` terkunci)
2. **Mekanistik:** transferabilitas ditentukan parameter sharing per-agen; ablasi tiga tingkat (`ippo` 0.954 / `mlp-knn-ppo` 0.949 / GNN 0.942–0.955) tidak menunjukkan beda
3. **Metodologis:** tiga cacat protokol pembacaan, semuanya menghasilkan baris yang terlihat valid
4. **Kekalahan proposed** (D3): `gnn-mappo_gat` 68.06 vs `ippo` 68.87 pada throughput, CI terpisah di n=20; collapse rate `central-ppo` 3/20 vs proposed 14–19/20

Kalau seluruh fase selesai dan GNN masih comparable, paper tetap berdiri sebagai empirical study + benchmark + kontribusi metodologis.
