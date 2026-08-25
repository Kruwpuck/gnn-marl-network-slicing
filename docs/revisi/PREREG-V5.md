# Pra-registrasi wave v5 — per-cell resilient constraint

**Status:** metrik primer **DIKUNCI**. `f_min` **BELUM BEKU** — lihat §0. Wave tidak boleh
dijalankan sebelum §0 diselesaikan manusia dan dicatat di ledger.

**Sumber:** `docs/revisi/PLAN-02-RESILIENT-CELLEDGE.md` §9
**Ditulis:** 2026-08-25, sebelum wave apa pun dijalankan
**Kode:** `envs/network_slicing_env.py` (blok `resilient`), `configs/experiment_config.yaml`,
`tests/test_resilient.py`, `scripts/calibrate_fmin.py`

---

## 0. Yang memblokir wave ini

Kalibrasi `f_min` (PLAN-02 §7) **tidak menghasilkan kandidat**, dan skripnya menolak
memilih satu. Dua populasi diukur, hasil di `results/CALIBRATE_FMIN.md`:

| Populasi | Hasil |
|---|---|
| Alokasi statis, 9 fraksi | Pelanggaran terendah **0,0964** di frac 0,9 — di atas `delta = 0,085`. Nol fraksi feasible |
| Checkpoint v4 terlatih | Referensi `ippo` melanggar (**0,1013**) **dan** kolaps di **5/5** seed (`embb_p5` 0,0000) |

Yang statis memang diperkirakan gagal: `delta` dikalibrasi terhadap policy **terlatih**
(lihat blok `cmdp` di config — lantai statis 12,22%, lantai terlatih ~5,5%), pelajaran yang
sama dengan ronde 4 kalibrasi `delta` sendiri.

Yang terlatih adalah **penghalang sungguhan**. Keluarga referensi tidak bisa memenuhi
constraint dan menghindari kolaps cell-edge sekaligus — kondisi yang sama yang membuat
proyek ini berstatus NOT DONE (Gate B3 gagal, C4 gagal untuk keluarga DQN). Mengkalibrasi
lantai laju terhadap referensi yang dirinya sendiri kolaps tidak mungkin.

**Satu godaan yang ditolak, dan alasannya perlu dibaca sebelum §0 diputuskan.** Aturan
peringkat "eMBB rata-rata tertinggi di antara yang feasible", dijalankan atas seluruh
checkpoint, memilih **`gnn-mappo`** — yaitu menetapkan `f_min` dari distribusi yang bisa
dicapai metode *proposed* sendiri, menanamkan keunggulannya ke dalam constraint. PLAN-02
§Larangan 1 melarang persis itu, dan §7 langkah 3 menyebut referensinya non-GNN (`ippo`,
konsisten dengan Gate A). Pembatasan itu sekarang **dipaksakan di kode**
(`scripts/calibrate_fmin.py`, `REFERENCE_ALGOS`), bukan diserahkan ke aturan peringkat.
Kandidat `gnn-mappo` juga bersandar pada **1 dari 10** run; batas minimum 3 run non-kolaps
kini dinyatakan di muka supaya persentil tidak pernah berdiri di atas satu seed.

**Keputusan yang dibutuhkan (manusia, dicatat di ledger sebelum apa pun dibekukan):**
titik operasi yang digeser, atau keluarga referensi kalibrasi yang dideklarasikan ulang.
Keduanya mengubah pembandingan dengan v4 dan tidak boleh diputuskan agent.

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
