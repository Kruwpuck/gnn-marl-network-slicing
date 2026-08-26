# Master Plan — Estafet Rencana Perbaikan Riset

**Basis:** wave v4 selesai, B3 & C4 GAGAL tercatat, klaim ter-scoping
**Sumber:** enam blok riset literatur (NotebookLM), sudah diverifikasi ulang
**Status keseluruhan:** NOT DONE menurut goal1.md — dan itu keadaan yang benar

---

> ## KOREKSI 2026-08-24 — enam premis basi, estafet Fase 2 menyusut
>
> Enam dokumen turunan ditulis dari jawaban NotebookLM tanpa akses ke kondisi kode
> terkini. Enam premisnya diverifikasi ulang terhadap file dan sudah tidak berlaku. Tiga
> di antaranya menyentuh gerbang, bukan catatan kaki. Ringkas di sini; detail di blok
> koreksi masing-masing dokumen.
>
> | # | Premis | Keadaan kode | Akibat ke estafet |
> |---|---|---|---|
> | P1 | PLAN-03 §3: "ganti GAT → GATv2" | `gnn/gat_backbone.py:5` sudah `GATv2Conv` sejak v1 | §3 gugur. Varian **dinamai `gat` padahal GATv2** — koreksi terminologi paper |
> | P2 | PLAN-03 §2: "edge belum bawa fitur fisik" | `envs/channel_model.py:113` sudah kirim path loss dB; `edge_dim=1` diset | §2 menyusut ke `interference_coupling` saja |
> | P3 | PLAN-03 §2: `distance_norm` fitur ketiga | path loss tanpa shadow fading + `los=True` = fungsi monoton jarak | Fitur dicoret. Dua fitur, bukan tiga |
> | P4 | PLAN-03 §7 / K2: buang `neighbor_urllc_frac_mean` | sudah dibuang di v3, `envs/network_slicing_env.py:22` | **Arm `obs=strict` hilang dari Fase 2. K2 selesai sendiri.** Kalimat "ablasi tiga tingkatmu bocor" batal |
> | P5 | PLAN-01 D3 / PLAN-03 §4: "diameter graf kecil" | graf **lengkap**, diameter **1** | Risiko over-smoothing lebih **tinggi**. D3 lebih relevan, dan D2b terbatas pada informasi edge |
> | P6 | PLAN-01 D4: "GNN kemungkinan parameter lebih sedikit" | `ippo` 3.852 vs `gnn-mappo_gat` 37.580 — GNN **9,8× lebih banyak** | Hasil null tidak bisa dijelaskan kapasitas kurang. Peran `ippo-scaled` berubah |
>
> **Konflik yang bergeser:**
> - **K2 — selesai dengan sendirinya** (P4). Tidak ada pekerjaan observasi yang bisa
>   bertabrakan dengan HPO. Batasan di PLAN-05 §3.3 dan §10.6 tidak lagi mengikat.
> - **K3 — lebih lemah dari yang ditulis** (P1/P2). Prasyarat edge feature sudah terpenuhi
>   sejak awal, dan analisis atensi + ablasi kausal **sudah dikerjakan**
>   (`scripts/attention_analysis.py`, `results/ATTENTION_v4_greedy.md`). K3 menyusut jadi:
>   ulangi analisis sesudah `interference_coupling` ditambahkan.
> - **K1, K4, K5, K6 tidak berubah.**
>
> **Fase 0 sudah dieksekusi.** D1+D4 `scripts/diag_equivariance.py`, D2a/D2b/D3
> `scripts/diag_gnn_reliance.py`, D5 `scripts/diag_collision.py`. Hasil di
> `results/DIAG_*.md`; checklist terisi di PLAN-01 §Keluaran. D2c sengaja ditunda — kalau
> nanti dibutuhkan, pakai training pendek terinstrumentasi, bukan rollout dengan jalur
> loss yang ditulis ulang (PLAN-01 blok koreksi).
>
> **Gerbang keputusan sudah terjawab** — §"Gerbang keputusan" di ringkasan Fase 0 di bawah
> kini punya jawabannya:
>
> | Uji | Verdict | Akibat ke estafet |
> |---|---|---|
> | D1 equivariance | **equivariant**, kategoris — 7 arsitektur per-agen varians tepat 0, dua arsitektur terpusat sensitif penuh | MLP per-agen **juga** equivariant. Tesis Tingkat 2 diperkuat: sumbernya parameter sharing, bukan message passing |
> | D2 collapse | **terkonfirmasi** — 41/50 checkpoint tak bergerak >1% saat pesan tetangga dinolkan; atribut edge 0/25 | ~~PLAN-04 dijalankan sesudah PLAN-03~~ — **tidak lagi menggerbangi PLAN-04** sejak 2026-08-25, lihat baris D6 |
> | D3 over-smoothing | **terkonfirmasi**, tak terbantahkan untuk `gat` — cosine embedding 1.0000 persis di 25/25 checkpoint | **PLAN-03 §5 dijalankan** |
> | D4 kapasitas | **timpang, arah terbalik** — `ippo` 3.852 vs `gnn-*_gat` 37.580 | arm `ippo-scaled` tetap, perannya berubah |
> | D2c aliran gradien | **hipotesis "jalur GNN tidak terlatih" tidak didukung** — `rasio << 1` cuma di 3/20 seed; di 13/20 backbone justru menerima gradien lebih besar per parameter daripada head | **menentukan urutan Fase 2: PLAN-03 §5 lebih dulu** |
> | D5 collision storm | **tidak terkonfirmasi** — di dalam `gnn-mappo_gat`, episode greedy yang kolaps dan yang tidak tak terbedakan pada sinkroni | tidak ada yang ditulis di paper soal collision storm |
> | D6 separabilitas input (2026-08-25) | **input tidak degenerate** — 0/50 checkpoint, dan 0/50 juga di kedua ambang sensitivitas; separasi hilang **di `conv1`** (rasio 0,0145 untuk `gat`, aktivasinya tidak bersalah) | **§5 tetap lebih dulu**, sambungannya harus melewati layer pertama; dan **gerbang PLAN-04 diganti** (PLAN-04 §0c) |
>
> **Peta estafet:** Fase 2 menjalankan **PLAN-03 (termasuk §5) lalu PLAN-04**, arm
> `obs=strict` tidak ada, dan urutan §5 sebelum auxiliary loss kini punya dasar terukur
> bukan sekadar urutan dokumen — over-smoothing hadir di 25/25 checkpoint sementara jalur
> GNN mati cuma di 3/20 seed, jadi §5 menyasar kondisi yang hadir di mana-mana
> (PLAN-04 §0b).
>
> **DIAGNOSIS DIBINGKAI ULANG — dari D2c + D2 + D3 + D6 bersamaan, dan tidak diantisipasi
> dokumen mana pun:** GNN **dilatih dengan baik** — gradien mengalir deras ke sana — dan
> **masukannya memang bisa dibedakan** (D6: 0/50 checkpoint degenerate, median 6 dari 8
> kolom observasi bervariasi antar-node). Yang terjadi: layer pertama membuang 98,6%
> separasi itu, keluarannya nyaris tidak dipakai, dan embeddingnya kolaps sempurna.
>
> Bukan *"encoder diabaikan"* melainkan **"encoder terlatih menuju representasi
> degenerate"**. Sasaran perbaikan bergeser dari **memaksa gradien masuk** — sudah masuk —
> ke **memaksa node berbeda satu sama lain**. Konsekuensinya ke gerbang: PLAN-04 §0c
> menggantikan §0, dan §5 harus menyambung melewati layer pertama, bukan layer kedua saja.
>
> Satu umpan balik yang ikut terlihat: `prev_alloc` dan `prev_alloc_lag2` punya std
> antar-node **0,0000 di 49/50 checkpoint** — dua dari delapan kolom observasi tidak
> membawa identitas node karena tiap gNB memilih aksi yang sama. Keluaran policy yang
> degenerate mengumpan balik jadi fitur node yang kosong identitas: sekaligus gejala kolaps
> dan masukan bagi kolaps itu.
>
> Tidak ada verdict, ambang, definisi metrik, atau angka hasil v4 yang bergerak. Yang
> bergerak adalah daftar pekerjaan Fase 2 dan status dua konflik.

---

> ## URUTAN EKSEKUSI DITUKAR 2026-08-26 — PLAN-03 duluan, PLAN-02 menunggu titik operasi
>
> **Alasannya proses, bukan hasil:** PLAN-02 terblokir kalibrasi `f_min`; PLAN-03 tidak
> bergantung padanya. Urutan Fase 1 → Fase 2 ditetapkan sebelum dua hal diketahui — bahwa
> `conv1` membuang 98,6% separasi (D6), dan bahwa kalibrasi `f_min` tidak menghasilkan
> kandidat.
>
> **Nomor fase tidak diganti.** Mengganti nomornya memutus rujukan silang di enam dokumen.
> Yang berubah cuma urutan eksekusinya:
>
> ```
> Fase 0 (selesai) → Fase 2a PLAN-03 → Fase 1 PLAN-02 → Fase 2b PLAN-04 → Fase 3 → Fase 4
>                       (sekarang)      (menunggu       (kondisional,
>                                        titik operasi)  §0c)
> ```
>
> **Kegagalan kalibrasi itu temuan, bukan hambatan.** Pada titik operasi v4 tidak ada `f_min`
> yang sekaligus feasible dan mengikat bagi baseline referensi non-GNN. Bandwidth tetap: dua
> constraint berebut PRB yang sama, dan Gate A ronde 6 lolos justru karena resilient
> constraint belum aktif. Bentuknya identik dengan temuan `delta` (rentang 1,6pp di bawah
> derau 2,0pp), cuma di sumbu berbeda. Detail dan angkanya di `PREREG-V5.md` §0 dan
> `results/CALIBRATE_FMIN.md`; PLAN-02 Q5 mengikat dokumen itu.
>
> **Opsi "keluarga referensi dideklarasikan ulang" ditolak** — memilih keluarga yang
> kebetulan lolos adalah memilih berdasarkan hasil, bentuk yang sama dengan godaan aturan
> peringkat `gnn-mappo` yang sudah ditolak (PLAN-02 §Larangan 1, aturan lintas-fase 3).
> Kalau titik operasi digeser, kriterianya **a priori** dan bebas nama algoritma; bentuknya
> dikunci di `PREREG-V5.md` §0. Tidak dikerjakan sekarang — menggesernya membuat seluruh v4
> tidak lagi sebanding, jadi itu keputusan manusia yang dicatat di ledger.
>
> **Satu alasan substantif untuk menunda, bukan sekadar menghindari kebuntuan.** Kalibrasi
> `f_min` mengukur distribusi `embb_p5` yang bisa dicapai, dan policy v4 lockstep —
> `prev_alloc` std antar-node 0,0000 di 49/50 checkpoint. Distribusi itu berasal dari policy
> yang tidak berdiferensiasi. Untuk GNN arsitekturnya memang rusak; untuk `ippo` argumen ini
> tidak berlaku langsung, tapi tension dua-constraint mungkin terlihat berbeda sesudah policy
> bisa berdiferensiasi.
>
> Kode PLAN-02 tidak sia-sia dan tidak dibuang: terimplementasi, teruji, `mode: none`
> bit-identik dengan v4. Ia menunggu titik operasi.

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

**Nomor fase di atas adalah struktur ketergantungan, bukan urutan eksekusi.** Sejak
2026-08-26 urutannya Fase 0 → **Fase 2a (PLAN-03)** → Fase 1 (PLAN-02, menunggu titik
operasi) → Fase 2b (PLAN-04) → Fase 3 → Fase 4. Alasannya di blok URUTAN EKSEKUSI DITUKAR di
atas.

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
| Sebaran node per tahap (D6, 2026-08-25) | Di layer mana representasi kolaps? |

**Gerbang keputusan:** hasil Fase 0 menentukan **urutan** Fase 2 — dan sejak D2c dan D6,
PLAN-04 tidak lagi digerbangi oleh D2 melainkan oleh hasil PLAN-03 §5 (PLAN-04 §0c).

### FASE 1 — Per-cell resilient constraint (wave v5) — kode selesai, wave menunggu

Unitnya per-**gNB**, bukan per-UE (PLAN-02 Q1). Kode terimplementasi dan teruji; wave menunggu
keputusan titik operasi (Q5, `PREREG-V5.md` §0). Dieksekusi **sesudah** Fase 2a.

Perubahan paling didukung: dua jalur analisis independen menunjuk solusi yang sama.
- Preseden empiris: resilient RRM (TSP 2023)
- Penjelasan mekanistik: constraint agregat secara struktural mengizinkan pengorbanan minoritas

Objective tidak diubah. Constraint kedua ditambahkan di atas infrastruktur primal-dual yang ada.

### FASE 2 — Arsitektur (wave v6) — dikerjakan sekarang

Empat arm, ditetapkan 2026-08-26 (`PREREG-V6.md`): `gat` (v4, pembanding), `gatres`
(residual menjangkau input), `gatedge` (`interference_coupling` sebagai kolom edge kedua),
`gatres-edge` (keduanya). Arm gabungan ikut **karena dua mekanismenya diperkirakan
berurutan, bukan paralel** — edge feature menambah informasi pembeda yang masuk ke `conv1`,
dan `conv1` yang membuang 98,6% separasi itu; tanpa residual, informasi tambahannya tidak
punya jalan keluar. Prediksi bahwa `gatedge` sendirian mungkin null dinyatakan di muka.

- **Selalu:** edge feature eksplisit (path loss + kopling interferensi; `distance_norm` dicoret, P3) + GATv2
- **Kondisional:** auxiliary loss, hanya kalau §5 sudah memperbaiki representasi **tapi KPI
  tetap datar** (PLAN-04 §0c — menggantikan syarat "Fase 0 mengonfirmasi collapse")
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
