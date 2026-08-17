# Struktur paper — tesis tiga tingkat (2026-08-17)

Diputuskan manusia (`Habb`), dicatat juga di `runs/2026-08-05-run01/ledger.md`. Dokumen ini
menetapkan **posisi** temuan dalam paper, bukan angka baru: setiap angka di bawah disalin
dari file report yang di-generate, dengan sumbernya disebut per baris
(`docs/HANDOVER.md` §11).

> **Menggantikan versi 2026-08-16.** Versi itu menaikkan collapse rate jadi hasil utama
> dengan angka pembacaan lama, dan kalimat intinya — "seluruh varian proposed kolaps" —
> ternyata benar untuk keluarga PPO dan salah untuk keluarga DQN. Koreksinya dicatat di
> ledger 2026-08-16T18:00. Keputusannya sendiri tidak dibatalkan; strukturnya diturunkan
> derajatnya (bukan dipersempit cakupannya) jadi tiga tingkat di bawah.

---

## 1. Tesis tiga tingkat

Klaim disusun dari yang paling tidak terbantahkan ke yang paling bergantung asumsi. Bentuk
ini juga yang patuh C3 (dua keluarga anggaran tidak boleh digabung dalam satu klaim
statistik): yang lintas-keluarga hanya tingkat 1 dan 2, dan keduanya bukan perbandingan
statistik antar-keluarga.

### Tingkat 1 — Struktural (hasil utama)

**Baseline terpusat kategoris tidak dapat dievaluasi di luar topologi latihnya.**

`central-dqn` dan `central-ppo` punya `obs_dim = n_gnb * 8` yang terkunci saat training
(40 di n_gnb=5). Di n_gnb=10 dibutuhkan 80, di n_gnb=20 dibutuhkan 160 — CANNOT_RUN di
seluruh topologi non-5, di kedua arm (`results/ZEROSHOT_v4_primary.md`, kolom `reason`
mencetak `obs_dim mismatch: trained 40, required 80/160` per baris).

Ini kategoris, bukan statistik: tidak ada CI, tidak ada n, tidak terbantahkan pada 5 seed.
Inilah klaim terkuat yang jujur, dan persis yang §Fallback `handoff/goal1.md` prediksi
sebelum grid dijalankan.

### Tingkat 2 — Mekanistik (hasil utama)

**Transferabilitas ditentukan parameter sharing per-agen dengan observasi lokal — bukan
message passing, dan bukan pula informasi tetangga.**

Ablasi tiga tingkat, retensi throughput per-gNB di n_gnb=20 relatif n_gnb=5, arm kepadatan
konstan (`results/ZEROSHOT_v4_primary.md`):

| tingkat informasi | algoritma | retensi |
|---|---|---|
| tanpa info tetangga | `ippo` | 0.954 |
| info tetangga, tanpa message passing | `mlp-knn-ppo` | 0.949 |
| message passing penuh | `gnn-mappo_gat` | 0.942 |
| | `gnn-mappo_sage` | 0.950 |
| | `gnn-madqn_gat` | 0.951 |
| | `gnn-madqn_sage` | 0.955 |
| | `idqn` | 0.952 |

Rentang seluruh tujuh arsitektur yang bisa transfer: **1.3 pp**. Menambahkan informasi
tetangga tidak menambah apa-apa, dan menambahkan message passing di atasnya juga tidak.

Ini **temuan positif, bukan null**: baseline k-NN dibangun justru untuk memisahkan dua
faktor yang selama ini menempel — "punya informasi tetangga" dan "melakukan message
passing". Setelah dipisahkan, keduanya jatuh di sisi yang sama, dan yang tersisa sebagai
penjelasan adalah faktor ketiga yang dimiliki ketujuhnya: kebijakan per-agen dengan
parameter berbagi dan observasi lokal. Klaimnya **berdiri di kedua keluarga**.

Catatan integritas yang wajib ikut tercetak: `mlp-knn-ppo` ditambahkan **setelah** hasil v4
dilihat, jadi ia dilarang masuk gate pra-registrasi mana pun. Perannya hanya membuat
perbandingan zero-shot adil, dan hasilnya dilaporkan ke arah mana pun ia jatuh.

#### Bukti pendukung: arm area tetap

Bukan sekadar confound yang dilaporkan demi integritas #3. Di arm area tetap (`area_size`
ditahan 500 m sehingga menaikkan `n_gnb` juga menguatkan kopling), n_gnb=20 memberi retensi
0.559–0.603 — rentang **4.4 pp** (`results/ZEROSHOT_v4_primary.md`). Penurunannya besar dan
**seragam**: seluruh arsitektur terdegradasi setara. Jadi message passing tidak memberi
ketahanan tambahan bahkan di rezim yang lebih menantang, yang memperkuat tingkat 2 alih-alih
sekadar mengulangnya.

### Tingkat 3 — Trade-off keluarga PPO (turunan, spesifik-keluarga)

Kalimat yang dipakai di paper:

> Di keluarga PPO, kami mengamati trade-off antara proteksi cell-edge dan transferabilitas:
> `central-ppo` satu-satunya yang tidak pernah kolaps (0 dari 5 seed) tetapi secara
> struktural tidak dapat dievaluasi di luar topologi latihnya, sementara seluruh varian
> per-agen transfer pada retensi ~0.95 dengan collapse rate 4–5 dari 5 seed. Trade-off ini
> **tidak berlaku di keluarga DQN**: `gnn-madqn_sage` mencapai 0 dari 5 seed kolaps dan
> retensi 0.955 secara bersamaan, meskipun perbedaan collapse rate dalam keluarga DQN tidak
> terpisah secara statistik pada n=5. Kami melaporkan trade-off sebagai temuan
> spesifik-keluarga, bukan properti umum arsitektur.

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

Bentuk penulisan wajib "kolaps k dari 5 seed", tidak pernah "kolaps X% waktu" (keputusan
scoping C4, `handoff/goal1.md`).

**Di keluarga PPO trade-off-nya nyata dan terpisah:** Wilson `central-ppo` [0.00, 0.43]
lepas dari [0.38, 0.96] dan [0.57, 1.00]. Dan agregatnya memang dibayar ke arah sebaliknya —
proposed unggul atas kontrol terpusat di KPI agregat sambil kolaps di cell-edge
(`results/RLIABLE_v4_primary.md`):

| perbandingan | `timely_throughput_mbps` IQM [95% CI] | verdict vs `central-ppo` | kolaps |
|---|---|---|---|
| `gnn-mappo_gat` | 67.9627 [67.3482, 68.5825] | proposed BETTER (CI terpisah) | 4 dari 5 |
| `gnn-mappo_sage` | 68.8968 [67.7281, 70.1377] | proposed BETTER (CI terpisah) | 5 dari 5 |
| `ippo` | 68.5563 [68.2290, 69.0997] | — | 5 dari 5 |
| `central-ppo` | 64.2827 [64.1337, 64.3955] | — | 0 dari 5 |

Jadi di keluarga PPO kalimatnya bukan "proposed kalah", melainkan: **model per-agen membeli
throughput dan SLA agregat dengan mengorbankan UE cell-edge**, dan satu-satunya arsitektur
yang tidak melakukannya adalah kontrol terpusat, yang justru membayar di agregat sekaligus
tidak bisa dipindah topologi. Perlu ditulis jelas bahwa ini **bukan** kemenangan message
passing: `ippo` dan `mlp-knn-ppo` kolaps 5 dari 5 sama seperti varian GNN, jadi garis
pemisahnya adalah terpusat lawan per-agen, bukan GNN lawan baseline.

**Di keluarga DQN trade-off-nya patah.** `gnn-madqn_sage` kolaps 0 dari 5 dan retensi 0.955
sekaligus; ia bahkan tidak kalah dari `central-dqn` (juga 0 dari 5) pada cell-edge sementara
`central-dqn` tidak bisa transfer sama sekali. Jadi ada arsitektur per-agen yang menang di
kedua sumbu, dan pernyataan trade-off apa pun yang berbentuk umum langsung terbantah
olehnya. Batas kejujuran yang wajib ikut: seluruh Wilson CI keluarga DQN beririsan pada n=5
([0.00, 0.43] lawan [0.12, 0.77]), jadi perbedaan di dalam keluarga itu **tidak** terpisah
secara statistik — `gnn-madqn_sage` membantah trade-off sebagai klaim universal, bukan
memenangkan klaim tandingan.

**Batas klaim (wajib ikut tercetak):** klaim komparatif saja. Berapa persis collapse
rate-nya dan bentuk bimodalitas di dalam satu algoritma tidak diklaim — C4 (seed ≥ 20)
GAGAL di 5 seed.

**Penempatan:** tingkat 1 dan 2 di section hasil utama, sebelum tabel KPI agregat, supaya
pembaca melihat batas struktural dan hasil ablasi lebih dulu. Tingkat 3 menyusul sebagai
turunan, dengan kontra-contoh DQN di badan teks — bukan di catatan kaki.

---

## 2. Subsection metodologi — protokol pembacaan sebagai pra-registrasi

**Temuan.** Protokol pembacaan yang salah menghasilkan baris yang **terlihat valid**, dan
tidak ada diagnostik murah yang menandainya setelah kejadian. Karena itu protokol pembacaan
wajib dikunci sebelum hasil dilihat, setara dengan pemilihan metrik.

Wave ini memberi **dua** kasus independen dengan bentuk kegagalan yang sama, dan tidak satu
pun mengumumkan dirinya (`results/READOUT_COMPARISON.md`).

### Kasus 1 — argmax pada PPO

- Entropi 2.191–2.291 nat terhadap plafon `ln 11` = 2.398 — rentang **0.100 nat**.
- `p_max` 0.200–0.217, `agree` 0.198–0.215 — argmax bukan perilaku policy di keempatnya,
  dengan derajat yang praktis sama.
- Celah pembacaan `timely_throughput_mbps` (non-greedy − greedy): `central-ppo` −1.01,
  `ippo` +0.46, `gnn-mappo_sage` +28.77, `gnn-mappo_gat` +45.27 — rentang **46.28 Mbps**.

Degenerasi argmax seragam; akibatnya pada KPI bergantung arsitektur. Jadi menyaring angka
greedy "yang kelihatan wajar" bukan pengaman, dan entropi/`p_max` tidak bisa dipakai sebagai
penyaring.

**Kontrafaktual.** P3 dibekukan 2026-08-08, sebelum satu pun hasil v4 ada. Andai laporan
dibaca greedy (`results/RLIABLE_v4_greedy.md`):

- `gnn-mappo_gat` IQM = **16.7700** [0.0002, 53.9184] lawan `ippo` **68.0649**
  [65.6590, 70.9259] → **proposed WORSE (CI terpisah)**

Pembacaan primer atas checkpoint yang **sama persis** (`results/RLIABLE_v4_primary.md`):

- `gnn-mappo_gat` IQM = **67.9627** [67.3482, 68.5825] lawan `ippo` **68.5563**
  [68.2290, 69.0997] → **COMPARABLE**

Kesimpulan "GNN kalah telak" itu sepenuhnya artefak pembacaan: bobotnya identik, yang berbeda
hanya apakah aksi diambil dari campurannya atau dari modusnya. Lebar CI greedy sendiri sudah
jadi tanda — [0.0002, 53.9184] bukan pengukuran, itu dua rezim episode yang dirata-ratakan.

### Kasus 2 — ε=1.0 pada DQN

`epsilon` bukan bagian dari `state_dict` dan `_save_model()` tidak menyimpannya, jadi 20 dari
20 checkpoint DQN dimuat pada `epsilon = 1.0`: setiap pembacaan non-greedy keluarga DQN
adalah **aksi acak seragam**, bukan policy (kesepakatan aksi dengan argmax 0.097–0.100 lawan
1/11 = 0.091 untuk acak murni). Rinciannya di `results/quarantine_eps1.0/README.md`.

- **Kenapa lolos:** KPI agregat nyaris tidak bergerak — `central-dqn` timely throughput 67.50
  greedy / 64.66 di ε=1.0 / 67.49 di ε=0.05 (diagnostik pembanding tiga pembacaan, 20
  episode, ledger 2026-08-16T16:00; angka 150-episode ada di `READOUT_COMPARISON.md`). Tidak
  ada di tabel utama yang terlihat salah.
- **Kerusakannya:** terkonsentrasi di metrik cell-edge, dan di situ ia **membalik verdict**.
  `gnn-madqn_gat` seed42 `embb_p5_mbps` = 1.778019 greedy dan 1.679319 di ε=0.05, lawan
  0.000006 yang dilaporkan.
- **Cara ia ketahuan:** cacat yang sama membuat `central-dqn` mengeluarkan 150 baris data per
  checkpoint di topologi tempat `obs_dim`-nya membuatnya mustahil jalan. Jaringannya tidak
  pernah dipanggil, jadi tidak ada yang melempar error.

**Kontrafaktual kedua.** Tanpa penemuan ini, laporan akan menyatakan "seluruh varian proposed
kolaps" — benar untuk keluarga PPO, salah untuk keluarga DQN, dengan `gnn-madqn_sage` 0 dari
5. Yaitu: seluruh tingkat 3 di atas akan ditulis sebagai properti umum arsitektur.

Perlu ditulis apa adanya bahwa temuan lama lebih dramatis dan lebih menguntungkan narasi
"kontrol terpusat menang telak"; versi yang benar lebih lemah dan lebih rumit (D3).

### Klaim metodologisnya

Bukan "gate pada pembacaan stokastik", melainkan yang lebih kuat: **protokol pembacaan adalah
keputusan pra-registrasi setara dengan pemilihan metrik**, karena protokol yang salah
menghasilkan keluaran yang lolos setiap uji kewajaran yang bisa diterapkan pembaca setelah
kejadian. Dua kasus di atas berbeda mekanismenya (satu mode distribusi, satu eksplorasi yang
tidak pernah mati) tetapi identik bentuknya.

**Yang tetap dilaporkan, bukan disembunyikan:** tabel greedy lengkap ikut dicetak (P3
mewajibkan dua-duanya), file ε=1.0 dikarantina dan bukan dihapus, dan 60 CSV header-saja yang
dihapus punya manifest path lengkap (`results/eval_zeroshot_v4/DELETED_MANIFEST.md`).
Penetapan protokol DQN sendiri dijalankan dengan aturan keputusan yang dideklarasikan sebelum
pengukuran, karena konsekuensinya adalah memakai angka yang sudah terlihat
(`handoff/goal1.md`, "Penetapan protokol pembacaan keluarga DQN 2026-08-16").

**Penempatan:** subsection di bagian metodologi (protokol evaluasi), dengan kedua
kontrafaktual sebagai tabel — bukan appendix.

---

## 3. Future work

**`gnn-madqn_sage` adalah satu-satunya arsitektur yang tidak pernah kolaps sekaligus bisa
transfer** (0 dari 5 seed, retensi 0.955). Pada n=5 ia tidak terpisah secara statistik dari
apa pun di keluarganya, jadi ia bukan klaim — ia hipotesis. Kalau C4 (20 seed keluarga DQN)
pernah dipenuhi, itu hipotesis pertama yang layak diuji: apakah kombinasi backbone SAGE
dengan anggaran DQN benar-benar menghindari trade-off yang mengikat keluarga PPO, atau 0 dari
5 itu hanya undian yang beruntung pada 5 seed.

Dua hal lain yang jatuh dari wave ini dan belum dikejar:

- Atensi GAT terstruktur secara fisik (mean rho per node −0.5308, 78.4% node-step negatif)
  tetapi kausalnya nyaris nol di bawah ablasi (`results/ATTENTION_v4_stoch.md`). Menaikkan
  derajat node dari 4 ke 19 tidak membuatnya berpengaruh. Kenapa struktur yang benar tidak
  berubah jadi pengaruh kausal adalah pertanyaan terbuka.
- `urllc_delay_p99` tersensor keras oleh deadline drop di titik operasi ini
  (`results/B3_DELAY_CENSORING.md`), sehingga B3 gagal karena instrumennya, bukan karena
  algoritmanya seragam. Titik operasi dengan deadline lebih longgar akan menguji ulang B3
  sebagai gate diskriminasi yang sungguhan.
