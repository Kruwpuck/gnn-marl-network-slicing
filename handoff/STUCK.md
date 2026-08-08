# STUCK
run: 2026-08-05-run01   iterasi: 6   level: 5 (pertanyaan metode)   waktu: 2026-08-08T13:05:00

Versi sebelumnya: iterasi 4 di commit `7193247`, iterasi 5 di commit `aad4198`.

## Ringkasan

Tiga keputusan manusia iterasi 5 sudah dibekukan di `handoff/goal1.md` dan
diselaraskan di `scripts/calibrate_trained.py`. Dua penguatan yang diminta sudah
diukur. Empat dari lima gate lolos.

A1 gagal — **dan bentuk kegagalannya baru**. Bukan lagi "constraint tidak
mengikat", melainkan gate yang jadi lebih presisi daripada pengendali yang
diujinya.

```
A1   GAGAL  |9.12 - 8.50| = 0.62pp  vs  2*SE 0.39pp / t*SE 0.54pp
A2a  LOLOS  lantai 5.49%, margin 3.01pp vs std 1.62pp
A2b  LOLOS  1.86pp vs std 1.62pp   (kontrol lam=0 10.98% -> lam-aktif 9.12%)
A3   LOLOS  inf:1   (late 8.78%, overflow 0.00%)
A4   LOLOS  1.62pp atas 80 window
```

## Bukti

### Constraint mengikat, berulang, di 5 dari 5 seed

```
seed   viol 20% akhir   viol 5% akhir   lam setimbang   held-out stokastik
42          8.32            8.51            1.87              8.85
43          8.29            8.44            3.72              8.77
44          9.28            9.53            0.80              9.86
45          8.81            8.99            3.66              9.02
46          8.34            8.65            0.39              9.10
rerata      8.61            8.82                              9.12     (delta 8.50)
sd_seed     0.43                                              0.48
SE_seed     0.19                                              0.19
```

Tiga pembacaan, tiga-tiganya di atas δ dengan selisih kecil dan **konsisten
arah**: +0.11, +0.32, +0.62 pp. Dual mendarat dekat sasaran dengan offset tunak.

λ setimbang berpencar hampir 10× (0.39–3.72). Itu bukan kegagalan: λ* adalah
harga yang dibutuhkan policy seed itu untuk duduk di δ, dan policy berbeda butuh
harga berbeda. Yang harus konsisten adalah variabel terkendali, dan ia konsisten
(sd 0.43 pp). Ini sekaligus menguatkan penurunan A2 lama dari "λ ≥ 5.0": di sini
λ=0.39 dan λ=3.72 sama-sama mengikat sempurna.

### Kenapa A1 gagal

`SE_seed` runtuh dari 1.1 pp (satu seed, antar-episode) ke **0.19 pp** (lima
seed). Pengukurannya jadi cukup presisi untuk mendeteksi bias kecil yang nyata.
Gate menuntut kesesuaian dalam 0.54 pp; pengendalinya punya offset 0.6 pp.

Arah cacatnya berbahaya: **tambah seed → pita menyempit → controller yang sama
gagal lebih parah.** Pada 20 seed wave, `t·SE` turun ke ~0.27 pp dan A1 mustahil
dilewati pengendali integral mana pun. Pola klasik uji hipotesis titik dengan
daya yang naik: cukup presisi, selalu menolak.

Pemilihan `t` lawan 1.96 tidak menyelamatkan apa pun — gagal di kedua konvensi.

### Perbaikan reproduktibilitas (memengaruhi semua angka di atas)

Sampling aksi menarik dari RNG global torch yang tidak pernah di-seed, jadi
pembacaan stokastik — angka primer yang baru — tidak reproducible: checkpoint
sama membaca 8.71 % lalu 8.65 % pada dua invokasi berturut. Diperbaiki dengan
`torch.manual_seed(EVAL_SEED_BASE + ep)` per episode, berlaku untuk kedua
protokol. Seluruh angka di dokumen ini sudah memakai versi yang diperbaiki.

### Entropi: hipotesis tidak terkonfirmasi, mekanismenya lain

```
λ      p_max  margin  entropi  agree | greedy  stoch    gap
0      0.194   0.042    2.281  0.199 | 12.57   10.57   2.00
1      0.167   0.049    2.341  0.171 | 12.55   10.30   2.25
5      0.238   0.068    2.204  0.237 | 10.83    8.87   1.96
12     0.265   0.088    2.161  0.259 |  9.65    6.69   2.96
30     0.331   0.209    2.096  0.327 | 94.06    5.49  88.56
seragam 0.091          maks 2.398
```

Tidak ada lonjakan entropi — entropi justru turun monoton sementara gap meledak,
dan di λ=30 policy paling percaya diri se-sweep (sd greedy antar-episode 0.11 pp:
mantap, konsisten, salah). Mekanisme sebenarnya: **argmax membawa ≤ 0.33 massa
aksi dan cocok dengan perilaku policy hanya 17–33 % langkah.** Kompetensi policy
ada di campurannya, bukan modusnya — konsisten dengan violation turun 5 pp tanpa
membayar eMBB, yang berarti perbaikannya penempatan PRB terhadap *waktu*, dan
satu aksi tetap tidak bisa bervariasi terhadap waktu.

Ambang bernomor untuk kegagalan-pembacaan ditolak: gap 2–3 pp itu offset
sistematis, bukan derau, jadi tidak ada SE yang sah jadi penyebutnya. 3×SE
berpasangan (0.8–2.4 pp) menandai seluruh regime sehat; 3×SE_episode (2.73 pp)
menandai λ=12 dan calib5 yang dua-duanya sehat.

## Keputusan yang dibutuhkan

Satu fork. Agent menolak mengambilnya sendiri karena usulan utamanya muncul
**setelah** melihat FAIL — persis situasi yang dijaga larangan integritas #1.

**E. Toleransi kesetaraan, bukan pita SE** *(rekomendasi)*. Nyatakan Δ yang
berarti secara praktis, lalu minta seluruh CI termuat di dalamnya:

```
A1 LOLOS jika  [mean - t*SE, mean + t*SE]  termuat dalam  [delta - D, delta + D]
```

dengan `D = 1 std window = 1.62 pp` — satuan yang sudah dideklarasikan di A2a dan
A2b, bukan angka baru. Artinya "mengikat = violation berada dalam resolusi yang
dilihat dual itu sendiri". Pita tidak menyusut ke nol seiring bertambahnya seed,
sehingga cacat struktural di atas hilang. Hasil: `[8.58, 9.66] ⊂ [6.88, 10.12]`
→ LOLOS.

**A. Ukur A1 pada violation training** — kuantitas yang benar-benar dikendalikan
dual. Miss 0.11 pp vs t·SE 0.54 pp → lolos. Celah ke held-out dilaporkan sebagai
artefak kalibrasi, yang memang sudah diwajibkan `goal1.md` §Mode pengukuran.

**F. Terima GAGAL.** Titik operasi tidak memenuhi A1 sebagaimana didefinisikan;
wave tidak dijalankan.

## Sudah dicoba dan gagal

| percobaan | hasil |
|---|---|
| `lambda_lr` 0.01 → 2.0 → 14.0 (ronde 1–3) | λ 1.03 → 18.59, perilaku policy tidak bergerak |
| δ=0.05 di titik operasi lama | δ di bawah lantai, λ → tak hingga, A2 lolos hampa |
| δ=0.15 dengan jangkar lantai statis (ronde 4) | λ → 0 dalam 100K step, A2b gagal 0.02 pp |
| kontrol A2b dipatok λ=1 (ronde 5) | kontras nol *by construction*; diperbaiki, kini λ=0 dan lolos |
| pita A1 satu sisi `[0.7δ, 1.0δ]` | menghukum dual yang berhasil; diganti dua sisi atas keputusan manusia |
| pita A1 dua sisi berbasis SE | menyempit seiring seed bertambah; gagal 0.62 vs 0.54 pp |
| ambang kegagalan-pembacaan 3×SE | menandai seluruh regime sehat — gap sistematis, bukan derau |
| opsi C, clip berprinsip | **tidak diperlukan** — kurva training di λ=30 sehat; kolapsnya degenerasi argmax |

## File tersentuh

- `handoff/goal1.md` — amandemen 2026-08-08 (A1 dua sisi, kontrol A2b λ=0, mode pengukuran P3)
- `scripts/calibrate_trained.py` — `--a1-seeds`, `--control-lambda` default 0.0, kedua eval tiap invokasi
- `scripts/evaluate_checkpoints.py` — `torch.manual_seed` per episode
- `scripts/policy_confidence.py` — baru; `results/policy_confidence.csv`
- `runs/2026-08-05-run01/ledger.md` — seluruh angka

## Commit terakhir hijau

`b929648` (pytest 80 passed)
