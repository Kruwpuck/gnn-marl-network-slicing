# Struktur paper — metodologi sebagai hasil utama (2026-08-17)

Diputuskan manusia (`Habb`), dicatat di `runs/2026-08-05-run01/ledger.md`. Dokumen ini
menetapkan **posisi** temuan dalam paper, bukan angka baru: setiap angka disalin dari file
report yang di-generate, sumbernya disebut per baris (`docs/HANDOVER.md` §11).

> **Urutan berubah 2026-08-17.** Versi sebelumnya menaruh tesis performa di §1 dan
> metodologi di §2. Setelah instansi ketiga ditemukan, urutannya dibalik: **cacat pembacaan
> naik jadi hasil utama**. Alasannya bukan bahwa temuan performanya lemah, melainkan bahwa
> tiga instansi independen dalam satu wave adalah **pola**, dan pola itu berlaku untuk
> siapa pun yang menjalankan eksperimen RL — sementara angka retensi dan collapse rate di
> sini berlaku untuk task ini saja.

---

## 1. HASIL UTAMA — protokol pembacaan yang salah menghasilkan baris yang terlihat valid

**Temuan.** Dalam satu wave, tiga cacat protokol pembacaan yang **independen** masing-masing
menghasilkan keluaran yang lolos setiap uji kewajaran yang bisa diterapkan pembaca setelah
kejadian: jumlah baris benar, kolom benar, CI terbentuk, KPI agregat masuk akal. Tidak satu
pun mengumumkan dirinya, dan tidak satu pun tertangkap oleh diagnostik murah. Ketiganya
ketahuan hanya lewat pemeriksaan yang menargetkan protokolnya, bukan angkanya.

**Klaimnya:** protokol pembacaan adalah keputusan **pra-registrasi setara pemilihan
metrik**. Bukan "gate pada pembacaan stokastik" — itu terlalu sempit, dan instansi 2 dan 3
di bawah justru terjadi *di dalam* pembacaan yang sudah disebut stokastik.

### 1.1 Tiga instansi

| # | Cacat | Kenapa terlihat valid | Bagaimana akhirnya ketahuan |
|---|---|---|---|
| 1 | **argmax pada PPO** — aksi diambil dari modus, bukan dari campuran policy | IQM lengkap dengan CI bootstrap; angkanya cuma "rendah", bukan mustahil | Diantisipasi: `scripts/policy_confidence.py` mengukur `agree` = 0.198–0.215, argmax bukan perilaku policy. P3 dibekukan 2026-08-08, **sebelum** satu pun hasil v4 ada |
| 2 | **ε=1.0 pada keluarga DQN** — `epsilon` tidak ada di `state_dict` dan tidak disimpan `_save_model()`, jadi 20 dari 20 checkpoint dimuat pada ε=1.0, yaitu aksi acak seragam | KPI agregat nyaris tidak bergerak: `central-dqn` timely throughput 67.50 greedy / 64.66 ε=1.0 / 67.49 ε=0.05. Tabel utama tidak terlihat salah | Lewat instansi 3, bukan lewat tabelnya sendiri |
| 3 | **ε=1.0 membuat `central-dqn` "jalan" di topologi yang mustahil** | 150 baris data per checkpoint di n_gnb=10 dan 20, format sempurna | Ketahuan justru karena **mustahil**: `obs_dim = n_gnb * 8` terkunci di 40, jadi arsitektur itu tidak bisa dievaluasi di sana sama sekali. Jaringannya tidak pernah dipanggil, sehingga tidak ada yang melempar shape error |

Instansi 3 yang paling tajam, dan ia bukan sisipan dari instansi 2: cacat yang sama
menghasilkan **dua kegagalan berbeda jenis** — satu merusak angka, satu memalsukan
keberadaan hasil. Yang kedua lebih berbahaya karena tidak ada nilai yang bisa dicurigai.

Ukuran kegagalan diagnostik murah, terukur: entropi keempat algoritma PPO merentang hanya
**0.100 nat** (2.191–2.291 terhadap plafon `ln 11` = 2.398) sementara celah pembacaan
throughput-nya merentang **46.28 Mbps** (`results/READOUT_COMPARISON.md`). Degenerasi argmax
seragam; akibatnya pada KPI bergantung arsitektur. Entropi karena itu **tidak** memprediksi
model mana yang dirusak argmax — menyaring angka "yang kelihatan wajar" bukan pengaman.

### 1.2 Tiga kontrafaktual — apa yang akan ditulis paper ini

Bukan latihan berandai-andai. Ketiganya adalah kesimpulan yang **sudah tertulis** di draf
laporan sebelum cacatnya ditemukan.

**Kontrafaktual 1 — arsitektur yang salah dinyatakan kalah.**
Dibaca greedy (`results/RLIABLE_v4_greedy.md`): `gnn-mappo_gat` IQM =
**16.7700** [0.0002, 53.9184] lawan `ippo` **68.0649** [65.6590, 70.9259] →
**proposed WORSE, CI terpisah**. Dibaca dengan protokol yang sah atas bobot yang **sama
persis** (`results/RLIABLE_v4_primary.md`): **67.9627** [67.3482, 68.5825] lawan
**68.5563** [68.2290, 69.0997] → **COMPARABLE**. Lebar CI greedy sendiri sudah jadi tanda:
[0.0002, 53.9184] bukan pengukuran, itu dua rezim episode yang dirata-ratakan.

**Kontrafaktual 2 — temuan dinyatakan universal padahal spesifik-keluarga.**
Di bawah ε=1.0 seluruh keluarga DQN terbaca kolaps 5 dari 5 seed, sehingga trade-off
cell-edge akan ditulis sebagai **properti umum arsitektur**. Di bawah pembacaan yang sah
(`results/STABILITY_v4_primary.md`): `central-dqn` 0 dari 5, `gnn-madqn_gat` 2 dari 5,
`gnn-madqn_sage` **0 dari 5**, `idqn` 2 dari 5. Kontra-contoh yang membatalkan bentuk
universal itu — `gnn-madqn_sage` — **tidak akan pernah muncul**.

**Kontrafaktual 3 — temuan struktural terhapus dari paper.**
`central-dqn` terbaca OK di n_gnb=10 dan 20 dengan 150 baris per checkpoint. Kalau itu
dibiarkan, klaim tingkat 1 di §2 — bahwa baseline terpusat **kategoris** tidak dapat
dievaluasi di luar topologi latihnya — hilang, sebab datanya seolah membuktikan sebaliknya.
Yang hilang bukan sekadar angka, melainkan **satu-satunya klaim di paper ini yang tidak
butuh statistik**.

### 1.3 Pertahanan yang benar-benar bekerja

Satu-satunya yang menahan ketiganya bukan kehati-hatian membaca angka, melainkan aturan
struktural:

1. **Kunci protokol sebelum melihat hasil.** P3 dibekukan 2026-08-08; penetapan pembacaan
   keluarga DQN 2026-08-16 mendeklarasikan aturan keputusannya (rasio sd antar-episode > 2.0
   membatalkan argmax) **sebelum** angka DQN dihitung, justru karena konsekuensinya adalah
   memakai angka yang sudah terlihat.
2. **Setiap laporan menyebut protokolnya sendiri.** `scripts/readout_audit.py` memindai
   setiap laporan yang memuat angka collapse rate dan **exit non-zero** kalau ada yang tidak
   mendeklarasikan pembacaannya; hasilnya `results/READOUT_PROVENANCE.md`. Saat pertama
   dijalankan ia menemukan 2 dari 15 laporan tanpa label.
3. **Putuskan dari arsitektur, bukan dari isi disk.** `scripts/zeroshot_eval.py` sekarang
   memaksa `CANNOT_RUN` untuk `central-*` di topologi non-5 apa pun file yang ada, dengan
   alasan per baris (`obs_dim mismatch: trained 40, required 160`). Membaca filesystem untuk
   menjawab pertanyaan arsitektur adalah yang membuat instansi 3 lolos.
4. **Karantina, bukan hapus.** Data cacat disimpan dengan sebabnya
   (`results/quarantine_eps1.0/README.md`), dan 60 CSV yang memang dihapus punya manifest
   path (`results/eval_zeroshot_v4/DELETED_MANIFEST.md`).

### 1.4 Batas kejujuran temuan ini

Ketiga cacat ditemukan **oleh tim yang sama yang membuatnya**, dan dua di antaranya lolos
berbulan-bulan. Temuan ini karena itu bukan "kami punya proses yang baik" melainkan
sebaliknya: proses yang wajar pun meloloskannya, dan yang menangkapnya adalah pemeriksaan
yang menargetkan protokol alih-alih angka. Ditulis apa adanya sesuai D3.

**Penempatan:** section hasil utama pertama, sebelum hasil performa apa pun.

---

## 2. Tesis tiga tingkat (hasil performa)

Klaim disusun dari yang paling tidak terbantahkan ke yang paling bergantung asumsi. Bentuk
ini juga yang patuh C3: yang lintas-keluarga hanya tingkat 1 dan 2, dan keduanya bukan
perbandingan statistik antar-keluarga.

### Tingkat 1 — Struktural

**Baseline terpusat kategoris tidak dapat dievaluasi di luar topologi latihnya.**
`central-dqn`/`central-ppo` punya `obs_dim = n_gnb * 8` terkunci saat training (40 di
n_gnb=5; dibutuhkan 80 di n=10, 160 di n=20) — CANNOT_RUN di seluruh topologi non-5, di
kedua arm (`results/ZEROSHOT_v4_primary.md`, kolom `reason` mencetak alasannya per baris).
Kategoris, tanpa CI, tanpa n, tidak terbantahkan pada 5 seed.

### Tingkat 2 — Mekanistik

**Transferabilitas ditentukan parameter sharing per-agen dengan observasi lokal — bukan
message passing, dan bukan pula informasi tetangga.** Ablasi tiga tingkat, retensi
throughput per-gNB di n_gnb=20 relatif n=5, arm kepadatan konstan
(`results/ZEROSHOT_v4_primary.md`):

| tingkat informasi | algoritma | retensi |
|---|---|---|
| tanpa info tetangga | `ippo` | 0.954 |
| info tetangga, tanpa message passing | `mlp-knn-ppo` | 0.949 |
| message passing penuh | `gnn-mappo_gat` | 0.942 |
| | `gnn-mappo_sage` | 0.950 |
| | `gnn-madqn_gat` | 0.951 |
| | `gnn-madqn_sage` | 0.955 |
| | `idqn` | 0.952 |

Rentang ketujuh arsitektur yang bisa transfer: **1.3 pp**. **Temuan positif, bukan null:**
baseline k-NN dibangun untuk memisahkan dua faktor yang selama ini menempel — "punya
informasi tetangga" dan "melakukan message passing". Setelah dipisah keduanya jatuh di sisi
yang sama, dan yang tersisa sebagai penjelasan adalah faktor ketiga yang dimiliki
ketujuhnya. **Berdiri di kedua keluarga.**

Catatan integritas: `mlp-knn-ppo` ditambahkan setelah hasil v4 dilihat, jadi dilarang masuk
gate pra-registrasi mana pun; perannya membuat perbandingan zero-shot adil.

**Bukti pendukung — arm area tetap.** Bukan sekadar confound yang dilaporkan demi
integritas #3. Di n_gnb=20 dengan `area_size` ditahan 500 m, retensi 0.559–0.603 — rentang
**4.4 pp**. Penurunannya besar dan **seragam**: seluruh arsitektur terdegradasi setara, jadi
message passing tidak memberi ketahanan tambahan bahkan di rezim yang lebih menantang.

### Tingkat 3 — Trade-off keluarga PPO (turunan, spesifik-keluarga)

> Di keluarga PPO, kami mengamati trade-off antara proteksi cell-edge dan transferabilitas:
> `central-ppo` satu-satunya yang tidak pernah kolaps tetapi secara struktural tidak dapat
> dievaluasi di luar topologi latihnya, sementara seluruh varian per-agen transfer pada
> retensi ~0.95 dengan collapse rate 4–5 dari 5 seed. Trade-off ini **tidak berlaku di
> keluarga DQN**: `gnn-madqn_sage` mencapai 0 dari 5 seed kolaps dan retensi 0.955 secara
> bersamaan. Kami melaporkan trade-off sebagai temuan spesifik-keluarga, bukan properti umum
> arsitektur.

Collapse rate, pembacaan primer per keluarga, ambang 0.01 Mbps pada `embb_p5_mbps`, unit =
seed (`results/STABILITY_v4_primary.md`); retensi dari `results/ZEROSHOT_v4_primary.md`:

| keluarga | algoritma | kolaps | Wilson 95% | retensi n=20 |
|---|---|---|---|---|
| PPO | `central-ppo` | 0 dari 5 | [0.00, 0.43] | CANNOT_RUN |
| PPO | `gnn-mappo_gat` | 4 dari 5 | [0.38, 0.96] | 0.942 |
| PPO | `gnn-mappo_sage` | 5 dari 5 | [0.57, 1.00] | 0.950 |
| PPO | `ippo` | 5 dari 5 | [0.57, 1.00] | 0.954 |
| PPO | `mlp-knn-ppo` | 5 dari 5 | [0.57, 1.00] | 0.949 |
| DQN | `central-dqn` | 0 dari 5 | [0.00, 0.43] | CANNOT_RUN |
| DQN | `gnn-madqn_gat` | 2 dari 5 | [0.12, 0.77] | 0.951 |
| DQN | `gnn-madqn_sage` | 0 dari 5 | [0.00, 0.43] | 0.955 |
| DQN | `idqn` | 2 dari 5 | [0.12, 0.77] | 0.952 |

Bentuk penulisan wajib "kolaps k dari N seed", tidak pernah "kolaps X% waktu".

**Keluarga PPO: terpisah.** Wilson `central-ppo` [0.00, 0.43] lepas dari [0.38, 0.96] dan
[0.57, 1.00]. Agregatnya dibayar ke arah sebaliknya (`results/RLIABLE_v4_primary.md`):
`gnn-mappo_gat` 67.9627 [67.3482, 68.5825] dan `gnn-mappo_sage` 68.8968 [67.7281, 70.1377]
keduanya **proposed BETTER, CI terpisah** terhadap `central-ppo` 64.2827 [64.1337, 64.3955].
Jadi model per-agen membeli throughput dan SLA agregat dengan mengorbankan UE cell-edge.
**Bukan kemenangan message passing:** `ippo` dan `mlp-knn-ppo` kolaps 5 dari 5 seperti varian
GNN, jadi garis pemisahnya terpusat lawan per-agen.

**Keluarga DQN: patah.** `gnn-madqn_sage` 0 dari 5 dan retensi 0.955 sekaligus, dan tidak
kalah dari `central-dqn` (juga 0 dari 5) pada cell-edge sementara `central-dqn` tidak bisa
transfer. Ada arsitektur per-agen yang menang di kedua sumbu. Batas kejujuran: seluruh Wilson
keluarga DQN beririsan di n=5, jadi perbedaan di dalam keluarga itu **tidak** terpisah secara
statistik — `gnn-madqn_sage` membantah trade-off sebagai klaim universal, bukan memenangkan
klaim tandingan. Kontra-contoh itu berdiri di **kedua** pembacaan yang sah: 0 dari 5 pada
argmax (primer) dan 0 dari 5 pada ε=0.05 (`results/STABILITY_v4_dqn_eps005.md`).

**Batas klaim:** komparatif saja. Karakterisasi — berapa persis rate-nya, bagaimana bentuk
bimodalitas per-seed — belum diklaim; C4 (≥ 20 seed) GAGAL di 5 seed. **Perluasan seed
keluarga PPO ke 20 sedang berjalan** (dideklarasikan 2026-08-17 atas dasar biaya keluarga);
setelah selesai, klaim karakterisasi boleh dibuat **untuk keluarga PPO saja**, dan keluarga
DQN tetap komparatif-saja. Bagian ini wajib ditulis ulang dengan angka n=20 saat itu.

---

## 3. Future work

**`gnn-madqn_sage` — satu-satunya arsitektur yang tidak pernah kolaps sekaligus bisa
transfer** (0 dari 5 seed, retensi 0.955). Di n=5 ia tidak terpisah secara statistik dari apa
pun di keluarganya, jadi ia hipotesis, bukan klaim. Kalau C4 pernah dipenuhi untuk keluarga
DQN, itu hipotesis pertama yang layak diuji: apakah kombinasi backbone SAGE dengan anggaran
DQN benar-benar menghindari trade-off yang mengikat keluarga PPO, atau 0 dari 5 itu undian
beruntung pada 5 seed.

- **Atensi terstruktur tanpa pengaruh kausal.** Per node penerima, mean rho = −0.5308,
  median −0.8000, 78.4% node-step negatif (`results/ATTENTION_v4_stoch.md`) — arah yang
  diprediksi cerita mekanisme. Tetapi ablasi kausal nyaris nol, dan menaikkan derajat node
  dari 4 ke 19 di n_gnb=20 justru mengecilkannya. Kenapa struktur yang benar tidak berubah
  jadi pengaruh kausal adalah pertanyaan terbuka.
- **B3 gagal karena instrumen, bukan karena algoritma seragam.** `urllc_delay_p99` tersensor
  keras oleh deadline drop: dukungannya hanya 11 titik kisi dan ujung atasnya dinding keras,
  sementara paket yang akan membentuk ekor pembeda justru yang dibuang
  (`results/B3_DELAY_CENSORING.md`, pembacaan primer per keluarga). Bagian yang dibuang
  berbeda tajam antar-algoritma — `idqn` censored **17.65%** lawan `ippo` 6.05% — sehingga
  perbedaan perilaku antrean muncul di `sla_satisfaction_pct` (B2 lolos 11.98 pp) alih-alih
  di p99 (B3 gagal 1.01 ms). Titik operasi dengan deadline lebih longgar akan menguji ulang
  B3 sebagai gate diskriminasi yang sungguhan.
