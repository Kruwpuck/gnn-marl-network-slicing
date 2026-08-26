# Pra-registrasi wave v6 — residual ke input + kopling interferensi pada edge

**Status:** metrik primer **DIKUNCI**. Tidak ada penghalang: keempat arm bisa dilatih hari ini.
**Sumber:** `docs/revisi/PLAN-03-EDGE-FEATURES.md` §2 dan §5, temuan D6
(`results/DIAG_INPUT_SEPARABILITY.md`)
**Ditulis:** 2026-08-26, sebelum satu pun arm v6 dilatih
**Kode:** `gnn/gat_backbone.py`, `gnn/__init__.py`, `envs/network_slicing_env.py`
(kolom edge kedua), `tests/test_gnn_v6.py`, `scripts/run_wave.py`

---

## 0. Kenapa wave ini jalan sementara v5 menunggu

PLAN-02 terblokir kalibrasi `f_min` dan itu dicatat sebagai temuan tentang titik operasi
(`PREREG-V5.md` §0). PLAN-03 tidak bergantung pada keluaran PLAN-02, jadi urutan eksekusinya
ditukar (PLAN-00, blok 2026-08-26). Wave ini berjalan pada titik operasi v4 yang **tidak
digeser sedikit pun**: `delta`, `lambda_arrival`, `buffer.urllc_max_bits`,
`dual_update_every`, `floor.mode`, dan `resilient.mode: none` semuanya seperti v4.

## 1. Hipotesis, per arm

Ditulis sebelum satu pun arm dilatih, termasuk **prediksi null yang dinyatakan di muka**:

> Kami memprediksi `gatres` memperbaiki separasi embedding, `gatedge` sendirian menunjukkan
> efek terbatas karena informasi tambahan tetap dihomogenkan `conv1`, dan `gatres-edge`
> menunjukkan efek terbesar. Verdict dilaporkan per-arm apa pun hasilnya.

Dasar mekanistiknya, bukan sekadar kelengkapan faktorial: D6 mengukur `conv1` membuang
**98,6%** separasi node (rasio `rel_spread` input→`conv1` 0,0145 untuk `gat`, aktivasinya
0,9861 — jadi yang membuang adalah agregasinya, bukan ELU-nya). Fitur edge menambah informasi
pembeda yang **masuk ke `conv1`**; kalau `conv1` tetap menghomogenkan keluarannya, informasi
itu hilang di tempat yang sama. Dua mekanisme ini karena itu diperkirakan **berurutan**, bukan
paralel: residual menyelamatkan separasi, lalu fitur edge punya sesuatu untuk disumbangkan
pada separasi yang selamat. Kalau benar, efek gabungannya superaditif — dan itu tidak bisa
terbaca dari dua arm terisolasi saja.

Konsekuensinya untuk pembacaan: **`gatedge` yang COMPARABLE tidak boleh dibaca sebagai "fitur
edge tidak berguna".** Bacaan yang sah adalah "efeknya tidak punya jalan keluar", dan yang
memisahkan kedua bacaan itu adalah arm gabungan.

Prediksi tambahan yang bisa jatuh sendiri, dari D6 juga: `gatres` diperkirakan mendekati
`sage` pada `rel_spread` akhir (`sage` 0,0348 lawan `gat` 0,0000), karena `SAGEConv`
menyimpan bobot root terpisah (`gnn/sage_backbone.py:19`) dan residual pada `gat`
menambahkan kembali sifat itu. Kalau tidak mendekati, prediksinya salah dan dilaporkan salah.

### Kenapa kolom edge menambah sesuatu meski turunannya sudah ada di node

Keberatan yang paling wajar terhadap §2: SINR sudah ada di observasi node, dan kopling
interferensi adalah salah satu bahan penyusunnya — jadi apa yang baru? Jawabannya bukan
"besarannya bukan karangan", melainkan bentuk informasinya:

> SINR pada observasi node adalah **agregat seluruh interferer** — satu penjumlahan yang
> membuang identitas penyumbangnya. Kolom edge memberi **dekomposisi per-tetangga** dari
> agregat yang sama: informasi yang secara matematis hilang di dalam penjumlahan itu. Agen
> tahu "SINR saya jelek", tapi tanpa fitur edge tidak tahu "gara-gara tetangga mana".

Penjumlahan itu bukan kiasan — `envs/network_slicing_env.py` menghitung interferensi sebagai
`np.sum(interferer_mw[i] * embb_fracs)`, dan yang sampai ke observasi cuma hasil penjumlahannya
lewat SINR. Koordinasi yang mensyaratkan "kurangi alokasi **terhadap tetangga tertentu**" tidak
punya dasar informasi di observasi node, dan itulah yang message passing seharusnya sediakan.
Kalimat ini juga masuk paper (PLAN-06 §7, bab arsitektur).

### Prediksi lockstep — terukur, dan bebas KPI

> Pada arm `gatres`, std antar-node `prev_alloc` dan `prev_alloc_lag2` **> 0** (ambang relatif
> D6 `VARY_REL = 0.01`, **tidak diubah**) pada mayoritas checkpoint. Pembandingnya `gat` v4,
> yang 0,0000 di **49/50** checkpoint. Kalau terpenuhi, observasi efektif naik dari 6 ke 8
> dimensi.

Dua sifat yang membuatnya berharga. **Bebas KPI:** ia mengukur perilaku, bukan hasil, jadi ia
bisa terpenuhi meski throughput datar — dan kalau begitu, ia bukti mekanistik bahwa perbaikan
arsitektur benar-benar mengubah perilaku policy, bukan cuma menggeser angka. **Nol kode baru:**
`scripts/diag_input_separability.py` sudah melaporkan sebaran per kolom observasi pada tahap
`input`, jadi prediksi ini terbaca dari laporan D6 yang memang sudah wajib diulang di §5.

Bentuk gagalnya dicatat juga: lockstep pecah **tanpa** separasi embedding membaik berarti
rantai kausal yang diasumsikan (§5 → separasi → lockstep pecah) salah arah, dan itu dilaporkan
apa adanya.

## 2. Arm

| Arm | `edge_dim` | residual | Isi |
|---|---|---|---|
| `gat` | 1 | tidak | v4, **pembanding**. **Bukan run baru** — checkpoint `_v4` yang sudah ada, lihat blok di bawah |
| `gatres` | 1 | ya | residual menjangkau **input** pada kedua layer, `alpha = 0.1` |
| `gatedge` | 2 | tidak | kolom edge kedua = `interference_coupling` |
| `gatres-edge` | 2 | ya | keduanya |

**Isi `edge_dim`, dinyatakan eksplisit supaya arm terbaca sebagai apa adanya:**

| dim | Isi | Asal | Berubah dari v4? |
|---|---|---|---|
| 0 | path loss dB gNB→gNB | `envs/channel_model.py:124`, fungsi posisi gNB saja | **tidak** |
| 1 | `interference_coupling` dB = `PL(dst→UE dst) − PL(src→UE dst)` | rata-rata path loss per-UE, besaran yang sama yang masuk SINR di `envs/network_slicing_env.py` | baru |

Positif berarti interferer lebih kuat dari serving link di UE tujuan. Arah **penting**:
`coupling[i→j] ≠ coupling[j→i]`, diuji di `tests/test_gnn_v6.py`. `distance_norm` **tidak**
ditambahkan (PLAN-03 P3: reparametrisasi bijektif dari path loss, nol informasi baru), jadi
dua fitur edge, bukan tiga.

### `gat` bukan run baru — checkpoint v4 dipakai ulang

Keputusan 2026-08-26: arm `gat` **tidak dilatih ulang**. Yang dipakai `gnn-mappo_gat_v4`
(**20 seed**) dan `gnn-madqn_gat_v4` (**5 seed**), checkpoint wave v4 apa adanya. Menghemat
~166 job-jam PPO plus arm DQN-nya, dan melatih ulang pembanding yang dinamikanya terbukti
identik hanya menambah derau seed, bukan validitas.

Syaratnya satu: **dinamika environment harus identik antara saat v4 dilatih dan sekarang.**
Diperiksa, bukan diasumsikan:

| Yang diperiksa | Hasil |
|---|---|
| **Uji numerik langsung** | Pohon di `c73de09` (keadaan kode saat wave v4) dan HEAD, config yang sama, 200 langkah dengan barisan aksi yang sama: `obs` (202×5×8), `reward`, `embb_thr_bps`, kolom path loss `edge_attr`, `edge_index`, dan himpunan kunci `info` **identik bit-per-bit** |
| Kode env selama wave | Tersentuh terakhir `c73de09` (4 Agustus); wave v4 jalan 8–18 Agustus. Beku sepanjang wave |
| Perubahan env sesudahnya | Dua. `b48dd79` — tiap baris yang menyentuh dinamika ada **di dalam** `if resilient_mode != "none"`, termasuk `_rate_window`, blok dual, kunci `info`, pencacah klip. `1a127f7` — `edge_attr` tidak pernah masuk `step()` |
| Aliran RNG | Kolom kopling dihitung dari `_pl_matrix`, **nol undian acak**, jadi topologi per-episode tidak bergeser |
| Blok config `env`/`cmdp`/`floor`/`buffer` | Tidak berubah sejak `aad4198` (6 Agustus), **sebelum** wave v4 |
| Config sesudah v4 | `405d76f` menambah blok `agent:` (bit-identik dengan default yang digantikan, dijaga `tests/test_hparams_identity.py`); `de481aa` menambah `knn_k` (baseline `mlp-knn-ppo` saja) |
| Titik operasi | Wave v4 = `floor.mode=none`, `delta=0.085`, `lambda_arrival=60000`, `urllc_max_bits=307200`, `dual_update_every=12500` — **persis** titik operasi wave v6 |

Gerbangnya dinyatakan sebelum uji dijalankan: beda sekecil apa pun berarti `gat` dilatih ulang
sebagai arm keempat, dan selisihnya dilaporkan sebagai temuan. Uji lolos, jadi jalur itu tidak
terpakai.

**Konsekuensi yang harus dijaga:** checkpoint `_v4` kini **pembanding aktif**, bukan arsip.
Integritas artefaknya diperiksa sebelum dan sesudah tiap pekerjaan.

**Environment identik untuk 8 algoritma dan keempat arm.** Env selalu memancarkan kedua kolom;
yang membedakan arm adalah `edge_dim` backbone, yang meng-slice kolom pertama saja kalau
nilainya 1. Tidak ada config per-arm, jadi tidak ada cara memasangkan config yang salah dengan
checkpoint saat evaluasi, dan aturan lintas-fase 1 (perubahan environment diterapkan identik ke
8 algoritma) terpenuhi **secara struktural**. `mlp-knn-ppo` tetap memeringkat tetangga dengan
kolom 0, jadi ablasi tiga tingkat tidak bergeser.

`sage` **tidak** disentuh: ia kontrol alami untuk prediksi §1, dan `SAGEConv` memang tidak
menerima `edge_attr` sama sekali (`gnn/sage_backbone.py:19-20`).

## 3. Metrik primer — dikunci

Enam metrik v5 apa adanya, definisi **tidak diubah**:

1. `cell_edge_collapse_rate` — proporsi binomial, CI Wilson
2. `embb_p5_mbps` — level dan distribusi
3. `timely_throughput_mbps`
4. `sla_satisfaction_pct`
5. Retensi zero-shot 10/20 gNB
6. Feasibility rate zero-shot

Ditambah dua **metrik representasi**, didaftarkan di sini supaya bukan promosi post-hoc
(`handoff/goal1.md` integritas #4) — keduanya sudah terdefinisi dan terimplementasi di
`scripts/diag_input_separability.py`, tidak dibuat baru untuk wave ini:

7. `rel_spread` per tahap (`input`, `conv1_pre`, `conv1_act`, `conv2`)
8. `eff_rank` per tahap

Metrik 7–8 yang menggerbangi PLAN-04 §0c: representasi membaik + KPI datar → PLAN-04;
representasi membaik + KPI membaik → selesai; representasi tidak membaik → §5 belum benar.

## 4. Seed

PPO 20 seed (42–61), DQN 5 seed (42–46) — **sama dengan wave v4, dan sama untuk tiap arm**;
tidak ada pengecualian per-arm. Aturan biaya yang sudah dibekukan berlaku.

Yang dilatih cuma **tiga** arm: `gatres`, `gatedge`, `gatres-edge`. `gat` memakai checkpoint v4
(lihat §2), jadi wave ini 75 job, bukan 100.

## 5. Yang wajib dijalankan sesudah wave, sebelum klaim apa pun

- **D2b diulang pada checkpoint v6** (`shuffle_edge_attr`, `scripts/diag_gnn_reliance.py:113`).
  Uji `allclose` di `tests/test_gnn_v6.py` membuktikan `edge_attr` berpengaruh pada **keluaran
  backbone**; itu **bukan** bukti policy memakainya (PLAN-03 §0). D2b menemukan atribut edge v4
  tidak terpakai di 0/25 checkpoint, dan arm baru tidak boleh mengulang kondisi itu tanpa
  ketahuan.
- **D6 diulang pada checkpoint v6** — itu metrik 7–8, dan pembandingnya angka `gat` yang sudah
  ada.
- Analisis atensi (PLAN-06 §2) boleh diulang sesudah kolom kedua ada, dengan ablasi kausal.

## 6. Yang dilaporkan apa pun hasilnya

- `gatedge` null → dilaporkan, dengan bacaan §1 dinyatakan eksplisit, bukan disimpulkan jadi
  "fitur edge tidak berguna"
- Representasi membaik tapi KPI tidak → hasil sah, dan itu justru gerbang PLAN-04 §0c
- KPI membaik tapi zero-shot rusak → dilaporkan sebagai biaya mekanisme
- v6 tidak memperbaiki apa pun → hasil sah
- Seed yang kolaps **tidak dibuang**
- Hasil v4 tetap dilaporkan penuh sebagai pembanding; `gat` tidak ditimpa

## 7. Yang sudah diverifikasi sebelum pra-registrasi ini ditulis

| Uji | Hasil |
|---|---|
| `pytest -q` | **123 lolos** (106 lama tidak berubah status, 17 baru) |
| `gat` membaca kolom 0 saja | keluaran identik antara `edge_attr` (E,2) dan (E,1) |
| Kolom 0 masih path loss `channel_model` | `array_equal` terhadap `build_interference_graph` |
| `state_dict` `gat` tidak bertambah kunci | `proj*` hanya ada di arm residual |
| Peringkat `mlp-knn-ppo` tidak bergeser | `knn_features` identik dengan/tanpa kolom kedua |
| Uji wajib §2 | `edge_attr` ter-nol mengubah keluaran `gatedge` |
| Residual menjangkau input | bobot `conv1`/`conv2` dinolkan: `gat` `rel_spread` 0, `gatres` > 0 |
| Kopling berarah dan bukan salinan kolom 0 | korelasi < 0,999; `coupling[i→j] ≠ coupling[j→i]` |
| Checkpoint v4 masih bisa dimuat | `gnn-mappo_gat_floornone_seed42.pt` dimuat, embedding identik dengan jalur v4 |
| Gate C1, harness lama tidak diubah | 9/9 (3 seed × 3 floor mode) |
| `parse_run_name` memisahkan keempat arm | diuji; sebelum perbaikan, `gatres` **salah parse jadi tag** |
| **Dinamika env identik dengan saat v4 dilatih** | pohon `c73de09` lawan HEAD, config sama, 200 langkah aksi sama: `obs`, `reward`, `embb_thr_bps`, kolom path loss, `edge_index`, kunci `info` **identik bit-per-bit** — ini yang menggerbangi pemakaian ulang `gat` |
