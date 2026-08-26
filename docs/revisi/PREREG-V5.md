# Pra-registrasi wave v5 — per-cell resilient constraint

**Status:** metrik primer **DIKUNCI**. `f_min` **BELUM BEKU**, dan §0 menjelaskan kenapa itu
temuan tentang titik operasi, bukan kalibrasi yang gagal dijalankan. Wave v5 menunggu
keputusan manusia atas titik operasi, dicatat di ledger lebih dulu. Estafet **tidak** menunggu
wave ini — PLAN-03 jalan duluan (PLAN-00, blok 2026-08-26).

**Sumber:** `docs/revisi/PLAN-02-RESILIENT-CELLEDGE.md` §9
**Ditulis:** 2026-08-25, sebelum wave apa pun dijalankan
**Kode:** `envs/network_slicing_env.py` (blok `resilient`), `configs/experiment_config.yaml`,
`tests/test_resilient.py`, `scripts/calibrate_fmin.py`

---

## 0. Temuan: tidak ada `f_min` yang sekaligus feasible dan mengikat pada titik operasi v4

Kalibrasi `f_min` (PLAN-02 §7) **tidak menghasilkan kandidat**, dan skripnya menolak
memilih satu. Dua populasi diukur, hasil di `results/CALIBRATE_FMIN.md`:

| Populasi | Hasil |
|---|---|
| Alokasi statis, 9 fraksi | Pelanggaran terendah **0,0964** di frac 0,9 — di atas `delta = 0,085`. Nol fraksi feasible |
| Checkpoint v4 terlatih | Referensi `ippo` melanggar (**0,1013**) **dan** kolaps di **5/5** seed (`embb_p5` 0,0000) |

Yang statis memang diperkirakan gagal: `delta` dikalibrasi terhadap policy **terlatih**
(lihat blok `cmdp` di config — lantai statis 12,22%, lantai terlatih ~5,5%), pelajaran yang
sama dengan ronde 4 kalibrasi `delta` sendiri.

### Ini hasil, bukan kegagalan implementasi

Baris kedua tabel bukan `f_min` yang salah pilih — **dua constraint berebut PRB yang sama.**
Bandwidth tetap: melindungi cell-edge eMBB berarti mengambil PRB dari URLLC, jadi menambah
constraint kedua membuat yang pertama lebih sulit dipenuhi. Gate A ronde 6 lolos justru
**karena** resilient constraint belum aktif — kelolosannya bukan bukti bahwa keduanya bisa
hidup bersama pada titik operasi yang sama.

Jadi temuannya, dan begitulah ia dilaporkan:

> Pada titik operasi v4, tidak ada `f_min` yang sekaligus **feasible** (shortfall konvergen)
> dan **mengikat** (μ steady-state > 0) bagi baseline referensi non-GNN.

Bentuknya identik dengan temuan `delta` dulu — seluruh rentang yang bisa dijangkau, 1,6pp,
duduk di bawah derau antar-seed 2,0pp, jadi tidak ada nilai yang memisahkan apa pun. Sama
persis, cuma di sumbu berbeda: di sana rentangnya terlalu sempit untuk memisahkan, di sini
ruang feasible-nya kosong. Itu yang membuatnya temuan tentang **task**, bukan kecelakaan
kalibrasi. `scripts/calibrate_fmin.py` **tidak ditambal**: tidak ada ambang yang dilonggarkan
dan tidak ada kandidat yang dikarang.

### Godaan pertama yang ditolak — aturan peringkat

Aturan peringkat "eMBB rata-rata tertinggi di antara yang feasible", dijalankan atas seluruh
checkpoint, memilih **`gnn-mappo`** — yaitu menetapkan `f_min` dari distribusi yang bisa
dicapai metode *proposed* sendiri, menanamkan keunggulannya ke dalam constraint. PLAN-02
§Larangan 1 melarang persis itu, dan §7 langkah 3 menyebut referensinya non-GNN (`ippo`,
konsisten dengan Gate A). Pembatasan itu sekarang **dipaksakan di kode**
(`scripts/calibrate_fmin.py`, `REFERENCE_ALGOS`), bukan diserahkan ke aturan peringkat.
Kandidat `gnn-mappo` juga bersandar pada **1 dari 10** run; batas minimum 3 run non-kolaps
kini dinyatakan di muka supaya persentil tidak pernah berdiri di atas satu seed.

### Godaan kedua yang ditolak — mengganti keluarga referensi

Jalan keluar yang paling mudah dari §0 adalah mendeklarasikan ulang keluarga referensi
kalibrasi ke keluarga yang kebetulan bisa lolos. **Ditolak, dan ditolak dengan alasan yang
sama persis dengan godaan pertama:** memilih referensi berdasarkan siapa yang lolos adalah
memilih berdasarkan **hasil**, bukan berdasarkan properti task. §Larangan 1 tidak
membedakan apakah yang dipilih-berdasar-hasil itu nilai `f_min` atau keluarga yang
mendefinisikannya. Keputusan 2026-08-26: **opsi ini tidak ada dalam daftar.**

### Kalau titik operasi digeser, kriterianya a priori

Satu-satunya jalan yang defensible, dan bentuknya wajib begini — properti task, nol nama
algoritma:

> Titik operasi dipilih sedemikian rupa sehingga baseline referensi non-GNN dapat memenuhi
> kedua constraint (URLLC SLA dan minimum capacity) secara simultan dengan margin terukur.

Bukan "supaya kalibrasi lolos". Jalur teknisnya: turunkan `lambda_arrival` atau naikkan
`delta` sampai ada ruang, lalu kalibrasi `f_min` dengan protokol §7 yang **sudah ada**,
tidak diubah.

**Tidak dikerjakan sekarang** (keputusan 2026-08-26). Menggeser titik operasi membuat
seluruh v4 tidak lagi sebanding, jadi ia keputusan manusia yang dicatat di ledger sebelum
apa pun dibekukan. Sementara itu estafet lanjut lewat PLAN-03, yang tidak bergantung pada
§0 ini sama sekali (PLAN-00, blok 2026-08-26).

Satu alasan substantif untuk menunda, bukan sekadar menghindari kebuntuan: kalibrasi `f_min`
mengukur **distribusi `embb_p5` yang bisa dicapai**, dan policy v4 lockstep — `prev_alloc`
punya std antar-node 0,0000 di 49/50 checkpoint (`results/DIAG_INPUT_SEPARABILITY.md`).
Distribusi itu berasal dari policy yang tidak berdiferensiasi. Untuk GNN arsitekturnya
memang rusak; untuk `ippo` argumen ini tidak berlaku langsung, tapi tension dua-constraint
mungkin terlihat berbeda sesudah policy bisa berdiferensiasi.

---

## 1. Hipotesis

> Per-cell resilient constraint dengan learnable slack menurunkan collapse rate cell-edge
> pada seluruh arsitektur. Karena constraint bersifat lokal per-gNB dan dapat dikomputasi
> lewat message passing, arsitektur GNN diperkirakan mencapai collapse rate lebih rendah
> pada throughput yang setara dibanding baseline MLP per-agen.

Ditulis ulang dari §9 dalam istilah **per-cell**. Perubahan unit dari UE ke gNB dijelaskan
di blok koreksi PLAN-02; hipotesisnya justru **makin kuat** dalam bentuk ini, karena gNB
**adalah** node graf — tidak perlu agregasi apa pun antara unit constraint dan unit
message passing.

## 2. Metrik primer — dikunci

1. `cell_edge_collapse_rate` — proporsi binomial, CI Wilson
2. `embb_p5_mbps` — level dan distribusi
3. `timely_throughput_mbps` — harga yang dibayar
4. `sla_satisfaction_pct` — memastikan constraint URLLC tidak rusak
5. Retensi zero-shot 10/20 gNB
6. Feasibility rate zero-shot — apakah pelanggaran di bawah `delta` bertahan di topologi baru

Definisi keenamnya **tidak diubah**. `embb_p5_mbps` khususnya tetap persentil-5 atas sampel
(gNB × slot) seperti di v4 — metrik gate, tidak disentuh.

**Metrik per-UE yang sungguhan, kalau nanti dibuat, adalah sekunder dan dipra-registrasi
terpisah.** `handoff/goal1.md` integritas #4 melarang promosi metrik post-hoc.

## 3. Arm

| Arm | Isi |
|---|---|
| `resilient=none` | baseline; observasi tetap 8 kolom sehingga arm ini **bit-identik dengan v4** (`tests/test_resilient.py::test_mode_none_is_identical_to_no_resilient_block`) |
| `resilient=fixed` | `f_min` tetap, `z ≡ 0` |
| `resilient=learned` | usulan penuh |

`fixed` adalah uji kejujuran §8: kalau hasilnya sama dengan `learned`, klaim "learnable
slack" tidak didukung dan mekanismenya cukup disebut minimum-rate constraint biasa.

μ **tidak** masuk observasi di v5 (ditunda ke Fase 2) — menambah kolom ke-9 mengubah
`in_channels` untuk 8 algoritma, bentrok dengan PLAN-03 §7, dan merusak sifat bit-identik
arm `none` yang jadi uji regresi paling berguna dari wave ini.

## 4. Seed

PPO ≥ 20 seed. DQN sesuai anggaran device-jam. Aturan biaya yang sudah dibekukan berlaku.

## 5. Yang dilaporkan apa pun hasilnya

- Collapse turun tapi throughput ikut turun → laporkan trade-off
- GNN tidak lebih baik dari MLP → hasil sah
- Zero-shot rusak → laporkan sebagai biaya mekanisme
- v5 tidak memperbaiki apa pun → hasil sah
- Seed yang kolaps **tidak dibuang**
- Hasil v4 tetap dilaporkan penuh sebagai pembanding

## 6. Yang sudah diverifikasi sebelum pra-registrasi ini ditulis

| Uji | Hasil |
|---|---|
| `pytest -q` | 106 lolos (96 lama tidak berubah status, 10 baru) |
| `resilient=none` identik dengan config tanpa blok `resilient` | 200 step; obs, reward, `embb_thr_bps`, dan kunci `info` identik |
| Gate C1, harness lama tidak diubah | 9/9 (3 seed × 3 floor mode) |
| μ dan z tidak masuk `state_dict` | hidup di `get_cmdp_state()`, dict biasa; round-trip diverifikasi |
| Checkpoint v4 pra-resilient bisa di-resume | kunci `mu`/`z` absen → nilai awal dipakai, bukan exception |
| Penalti tidak ditelan klip `[-10,10]` | fraksi step terklip di bawah 5% |
