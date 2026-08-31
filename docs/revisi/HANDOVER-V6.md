# HANDOVER — wave v6 (arm arsitektur), keadaan per 2026-08-31

Serah-terima untuk jalur **revisi v6**: tiga arm arsitektur baru (`gatres`, `gatedge`,
`gatres-edge`) melawan pembanding `gat`. Ini **bukan** pengganti `docs/HANDOVER.md`, yang
memegang jalur Rev 2 (Fase 0–2, kalibrasi) dan masih berlaku apa adanya.

Semua angka di dokumen ini disalin dari file yang di-generate, bukan diketik dari ingatan
(`docs/HANDOVER.md` §11).

---

## 1. Keadaan sekarang

| | |
|---|---|
| Wave PPO v6 | **SELESAI** — `Wave done in 71.27h. all jobs ok`, 60 job (3 arm × 20 seed 42–61), nol gagal |
| Jalan | 2026-08-27 12:03:04 sampai 2026-08-30 11:19:30 |
| Checkpoint `_v6` | **120** file (60 job × `_best`/`_last`) |
| CSV metrik `_v6` | **60** file |
| Proses python | **0**; `results/logs/.wave_v6.lock` sudah terlepas sendiri |
| Wave DQN v6 | **BELUM DIJALANKAN** — 15 job, perkiraan ~10 jam |
| Evaluasi & laporan | **BELUM SATU PUN** dijalankan atas checkpoint `_v6` |
| Commit belum di-push | **2** (`5a980fe`, `d5b8581`) |

Belum ada satu pun KPI v6 yang dibaca. Tidak ada klaim apa pun yang boleh dibuat dari keadaan
ini.

## 2. Perintah berikutnya

Wave DQN — boleh jalan sekarang, lock per-tag sudah lepas:

```
.venv/Scripts/python.exe scripts/run_wave.py --seeds 42,43,44,45,46 \
    --floor-mode none --tag _v6 --max-parallel 6 \
    --algos gnn-madqn_gatres,gnn-madqn_gatedge,gnn-madqn_gatres-edge
```

`--floor-mode none` **wajib dinyatakan eksplisit**: defaultnya `dynamic`, dan itu bukan titik
operasi beku v4/v6. Salah di sini membuat seluruh wave tidak sebanding dengan v4 tanpa
ketahuan sampai evaluasi.

Resep pelepasan yang terbukti bertahan (wave PPO 71 jam selesai dengan cara ini): jalankan
terlepas dari sesi mana pun, arahkan stdout ke `results/logs/stdout/`, dan matikan tidur mesin
lebih dulu.

```powershell
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
Start-Process -FilePath (Resolve-Path ".venv\Scripts\python.exe").Path `
    -ArgumentList "scripts/run_wave.py","--seeds","42,43,44,45,46","--floor-mode","none",
                  "--tag","_v6","--max-parallel","6",
                  "--algos","gnn-madqn_gatres,gnn-madqn_gatedge,gnn-madqn_gatres-edge" `
    -RedirectStandardOutput "results\logs\stdout\wave_v6_dqn.out" `
    -RedirectStandardError  "results\logs\stdout\wave_v6_dqn.err" `
    -WindowStyle Hidden -PassThru
```

Kegagalan paling umum di run multi-hari bukan crash, melainkan mesin tidur atau sesi induk
mati. Keduanya ditutup di atas.

## 3. Kewajiban pasca-wave, sebelum klaim apa pun

Dari PREREG-V6 §5, disalin bukan diringkas:

1. `scripts/evaluate_checkpoints.py` atas keenam arm.
2. `scripts/rliable_report.py --tag "_v6,_v4"` — **dua** tag, karena arm ada di `_v6` sementara
   pembanding `gat` dan seluruh baseline ada di `_v4`. Kalau satu algoritma punya data eval di
   lebih dari satu tag, `per_seed_means` menggagalkan run dengan `SystemExit`; itu disengaja,
   supaya dua wave tidak pernah dirata-ratakan diam-diam.
3. **D2b diulang pada checkpoint v6** — `shuffle_edge_attr`, `scripts/diag_gnn_reliance.py:113`.
   Uji `allclose` di `tests/test_gnn_v6.py` membuktikan `edge_attr` mengubah keluaran
   **backbone**; itu bukan bukti policy memakainya. Pada v4, atribut edge tidak terpakai di
   **0/25** checkpoint, dan arm baru tidak boleh mengulang kondisi itu tanpa ketahuan.
4. **D6 diulang pada checkpoint v6** — metrik 7–8 PREREG-V6, pembandingnya angka `gat` yang
   sudah ada.

## 4. Yang sudah dikunci sebelum wave — jangan dibuka lagi

- **Hipotesis per-arm dan prediksi lockstep** tertulis di PREREG-V6 §1 **sebelum** wave,
  termasuk prediksi bahwa `gatedge` sendirian mungkin null.
- **PREREG-V6 §6** mendaftar apa yang dilaporkan apa pun hasilnya: `gatedge` null, representasi
  membaik tapi KPI datar, KPI membaik tapi zero-shot rusak, dan "v6 tidak memperbaiki apa pun"
  — semuanya hasil sah. Seed yang kolaps tidak dibuang. Hasil v4 tetap dilaporkan penuh.
- **Pembanding `gat` adalah checkpoint v4 yang dipakai ulang**, bukan run baru. Keputusan itu
  digerbangi uji identitas numerik, bukan pembacaan kode: pohon `c73de09` lawan HEAD, config
  sama, 200 langkah dengan barisan aksi sama — `obs`, `reward`, `embb_thr_bps`, kolom path loss,
  `edge_index`, dan kunci `info` identik bit-per-bit (PREREG-V6 §7). Karena itu artefak `_v4`
  berstatus **pembanding aktif, bukan arsip**; jangan ditimpa, dipindah, atau dibersihkan.

## 5. Koreksi yang harus terbawa: insiden "jalan-ganda" tidak pernah terjadi

Pada 2026-08-26 wave v6 dihentikan karena hitungan proses dibaca sebagai dua wave berjalan
bersamaan. **Pembacaan itu salah.**

`.venv\Scripts\python.exe` adalah **stub peluncur**, bukan salinan interpreter: 274.712 byte
dengan hash berbeda dari `Python311\python.exe` yang 103.192 byte. Tiap pemanggilan venv karena
itu selalu **dua** proses, dan yang kedua adalah **anak** dari yang pertama (`ParentProcessId`
cocok), dengan si anak yang benar-benar menjalankan skripnya. Satu induk + 6 trainer = 7 pasang
= **14 proses** — persis angka yang dibaca sebagai dua wave. Selisih 9 ms antara dua
`run_wave.py` berargumen identik adalah jarak stub memanggil anaknya; "4 proses per backbone"
adalah 2 job × 2 proses, bukan 2 penulis per job.

Akibatnya: **wave sehat dimatikan dan 4,2 jam latihan hilang tanpa sebab.** Artefaknya utuh
bukan karena `--resume` beruntung menyelamatkan dua penulis, melainkan karena penulisnya memang
selalu satu.

**Pelajaran yang dibawa: hitungan proses bukan instrumen integritas data.** Yang mengikat ada
tiga, dan ketiganya bersih sejak awal: nilai langkah duplikat, inversi langkah di luar batas
resume, dan checkpoint yang gagal dimuat. Tercatat sebagai baris 7 di PLAN-06 §5 — satu-satunya
dari tujuh instansi yang arahnya terbalik (melaporkan **sakit padahal sehat**), dan
satu-satunya yang alarm palsunya langsung memicu tindakan destruktif.

## 6. Lockfile `run_wave.py` — cara pakai dan dua batasnya

`acquire_wave_lock` (`scripts/run_wave.py:92`) membuat `results/logs/.wave<tag>.lock` dengan
`os.O_EXCL`. Ia menjaga mode gagal yang **belum pernah teramati**: dua peluncuran sungguhan atas
satu tag, misalnya dari dua terminal. `O_EXCL` dipilih karena atomik — pemeriksaan cek-lalu-buat
akan meloloskan dua proses yang berangkat berdekatan.

Dua batas, harus diketahui sebelum orang menyangka kuncinya rusak:

1. **Kunci mati harus dihapus manual.** Proses yang dibunuh paksa atau mesin yang mati
   meninggalkan file lock. Pesan penolakannya menyebut pid dan argv pemegangnya, supaya
   menghapusnya jadi keputusan, bukan tebakan. Tidak ada deteksi pid-hidup.
2. **Lingkupnya per-tag.** Wave PPO dan DQN bertag sama tidak bisa jalan serentak meski nama
   filenya berbeda. Sejalan dengan urutan PPO-dulu-baru-DQN yang dipakai, tapi bukan
   kebetulan yang boleh diandalkan diam-diam.

Diuji di `tests/test_wave_lock.py`: peluncuran kedua mati dengan pesan yang menyebut lock dan
pid-nya, dan kunci benar-benar dilepas sesudah `unlink` sehingga penjaganya tidak berubah jadi
cacat berikutnya.

## 7. Artefak yang sengaja tidak di-commit

60 CSV metrik `_v6`, `results/eval_v6smoke/`, dan seluruh checkpoint dibiarkan untracked. Artefak
latihan tidak menyangga klaim, dan repo bukan tempat penyimpanannya; yang di-commit hanya
laporan yang di-generate. Delapan belas file sempat ter-staged dari sesi sebelumnya dan sudah
di-unstage (`git restore --staged`, nol perubahan working tree).

## 8. Utang terbuka

- **2 commit belum di-push** (`5a980fe`, `d5b8581`). User yang push; agent tidak pernah push.
- **`resilient.f_min_mbps` dan titik operasi** masih menunggu keputusan manusia (PREREG-V5 §0).
  Kalibrasi gagal menemukan `f_min` yang sekaligus feasible dan mengikat pada titik operasi v4 —
  itu **temuan**, bukan hambatan implementasi, dan tidak boleh ditambal. Kriteria a priori untuk
  menggeser titik operasi sudah dikunci di PREREG-V5 §0 tapi belum dieksekusi.
- **PAT GitHub yang pernah ditempel di chat belum dicabut** — cabut di
  `github.com/settings/tokens`.
- **Pertanyaan yang belum dijawab:** menambah test ke file yang dibuat di unit kerja yang sama
  itu boleh, atau harus selalu file terpisah? `handoff/goal1.md` melarang **mengedit** file
  test; sejauh ini penambahan selalu dibuat aditif dan file baru dipisah untuk aman.

## 9. Aturan kerja yang berlaku di jalur ini

- Jangan `git add -A`; stage file spesifik. Commit hanya kalau diminta. Agent tidak push.
- Identitas commit: `-c user.name=Habb -c user.email=ihabhasanainakmal0409@gmail.com`.
- Angka hasil selalu disalin dari file yang di-generate.
- `citation_audit.py --update` **tidak dipercaya untuk memperbaiki** anchor, hanya untuk
  melaporkan bahwa anchor bergeser — itu cacat #4 di PLAN-06 §5. Anchor yang melenceng dibaca
  ulang terhadap baris aslinya dan diperbaiki dengan tangan lebih dulu.
- Varian baru dapat nama baru; nol artefak lama ditimpa (PLAN-03 §Larangan 5).
