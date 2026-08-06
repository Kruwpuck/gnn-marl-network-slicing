# STUCK
run: 2026-08-05-run01   iterasi: 5   level: 5 (pertanyaan metode)   waktu: 2026-08-06T13:45:00

Versi sebelumnya: iterasi 3 di commit `50664fb`, iterasi 4 di commit `7193247`.

## Ringkasan

**Kalibrasinya berhasil.** Ronde 5 (`delta=0.085`, titik operasi O1) adalah pertama
kalinya sejak v3 dual benar-benar setimbang di sasaran:

```
training 20% terakhir : viol = 8.32 %   lam = 1.87   (delta = 8.50 %)
```

λ tidak meluncur ke 0 (v3, ronde 4) dan tidak lari ke tak hingga (ronde 3). Ia
mendarat dan berosilasi 1.27–1.64. A2a, A3, A4 lolos.

A1 dan A2b tercetak FAIL, tapi **keduanya berasal dari pilihan pengukuranku sendiri,
bukan dari properti task.** Itulah yang perlu diputuskan manusia sebelum wave.

## Bukti

### Uraian celah ronde 5

```
training 20% terakhir    8.32 %      <- yang dikendalikan dual, vs delta 8.50 %
held-out stokastik       8.71 %      celah dari training  +0.39 pp
held-out greedy         12.03 %      celah dari stokastik +3.32 pp
```

Seluruh selisih A1 (12.03 % vs pita [5.95 %, 8.50 %]) adalah artefak argmax. Dibaca
stokastik, A1 = 8.71 % — meleset 0.21 pp, jauh di dalam SE 1.1 pp.

### Kurva respons λ (ippo, 1M step per titik, 150 episode held-out)

| λ | greedy | stokastik | selisih | eMBB |
|---|---|---|---|---|
| 0 | 12.57 % | 10.57 % | −2.00 pp | 8.15 |
| 1 | 12.55 % | 10.30 % | −2.25 pp | 8.12 |
| 5 | 10.83 % | 8.87 % | −1.96 pp | 8.02 |
| 12 | 9.65 % | 6.69 % | −2.96 pp | 8.19 |
| 30 | 94.06 % | 5.49 % | −88.56 pp | 8.14 |

Tiga hal terbaca. Dual punya wewenang kendali nyata (5.1 pp, vs SE 1.1 pp). Tebing
degenerasi argmax ada antara λ=12 dan λ=30; di bawah itu greedy melacak stokastik
dengan selisih stabil 2–3 pp. Dan violation turun 5 pp **tanpa** membayar eMBB, jadi
perbaikannya penempatan waktu PRB — keterampilan adaptif yang seharusnya membedakan
arsitektur, dan ia tersedia di titik operasi ini.

### A2b dihitung ulang dengan kontrol yang benar

Kontrol dipatok di λ=1.0 sementara run setimbang di λ=1.87 — dua λ hampir sama,
kontrasnya nol *by construction*. Terhadap kontrol λ=0 yang benar-benar tak
terbatasi (`ippo_calib4`, stokastik 10.57 %):

```
|10.57 - 8.71| = 1.86 pp  >=  std 1.62 pp   -> LOLOS
```

Cacatnya di spesifikasi gate buatanku: A2b ditulis "run λ-beku" tanpa menetapkan
nilai patokan.

### Ringkasan ronde 5 dengan pembacaan stokastik + kontrol λ=0

| gate | hasil |
|---|---|
| A1 | 8.71 % vs [5.95 %, 8.50 %] — meleset 0.21 pp, di dalam derau |
| A2a | lantai 5.49 %, margin 3.01 pp vs std 1.62 pp — lolos |
| A2b | 1.86 pp vs std 1.62 pp — lolos |
| A3 | inf:1 (late 8.78 %, overflow 0.00 %) — lolos |
| A4 | 1.62 pp atas 80 window — lolos |

## Keputusan yang dibutuhkan

Tidak satu pun menyangkut algoritma mana yang unggul.

**1. Protokol pelaporan.** `evaluate_checkpoints.py` menjalankan argmax. Argmax dan
sampel berbeda 2–3 pp sistematis lalu putus total di antara λ=12 dan λ=30. Di wave,
8 algoritma × 20 seed akan mendarat di λ berbeda-beda; yang melewati tebing akan
melaporkan angka sampah padahal policy-nya sehat.

- **P1 greedy tetap primer** — standar RL, realistis untuk deployment. Tapi A1 diukur
  pada pembacaan yang meleset 2–3 pp, dan δ harus digeser ke ~0.11 supaya greedy masuk
  pita, yang mendekatkan titik kerja ke tebing.
- **P2 stokastik primer** — cocok dengan yang dioptimalkan training dan dikendalikan
  dual. δ=0.085 langsung konsisten. Kurang lazim sebagai angka utama di paper RL.
- **P3 laporkan dua-duanya, stokastik primer** *(rekomendasi)* — biaya satu pass eval
  tambahan. Greedy jadi uji realisme deployment, tebingnya jadi temuan tersendiri.
  Aturan yang bisa dipra-registrasi: selisih greedy−stokastik di atas ambang (misal
  10 pp) ditandai **kegagalan pembacaan**, bukan kegagalan policy, dan dilaporkan
  sebagai keduanya.

**2. Nilai patokan kontrol A2b.** Tetapkan λ=0 (benar-benar tak terbatasi), bukan
`lambda_init`. Tanpa ini A2b bisa gagal semata karena kontrol dan run kebetulan punya
λ yang mirip.

**3. Penempatan A1 terhadap kesetimbangan dual.** Dual menggiring violation *training*
ke δ, sehingga held-out mendarat di δ + ~0.4 pp, sementara pita A1 `[0.7δ, 1.0δ]`
menuntut held-out ≤ δ. Selalu meleset tipis, dan dengan SE 1.1 pp hasilnya jadi lemparan
koin tiap run — bukan gate yang kokoh. Pilihan:
- ukur A1 pada violation training (kuantitas yang benar-benar dikendalikan dual);
- lebarkan batas atas pita sebesar celah terukur;
- naikkan δ ~0.5 pp supaya held-out mendarat di dalam pita.

## Sudah dicoba dan gagal

| percobaan | hasil |
|---|---|
| `lambda_lr` 0.01 → 2.0 → 14.0 (ronde 1–3) | λ 1.03 → 18.59, perilaku policy tidak bergerak |
| δ=0.05 di titik operasi lama | δ di bawah lantai, λ → tak hingga, A2 lolos hampa |
| δ=0.15 dengan jangkar lantai statis (ronde 4) | λ → 0 dalam 100K step, A2b gagal 0.02 pp |
| kontrol A2b dipatok λ=1 (ronde 5) | kontras nol *by construction* |
| opsi C, clip berprinsip | **tidak diperlukan** — pada λ=30 kurva training sehat (viol 10.65 → 5.03, reward −881 → −72); kolapsnya degenerasi argmax, bukan gradien mati |

## File tersentuh

- `configs/experiment_config.yaml` — `delta` 0.15 → 0.085 (titik operasi kedua; keduanya dilaporkan)
- `scripts/evaluate_checkpoints.py` — `--stochastic` (diagnostik, default tetap greedy)
- `scripts/calibrate_trained.py` — `--control-lambda`
- `runs/2026-08-05-run01/ledger.md` — seluruh angka

## Commit terakhir hijau

`6ba2aab` (pytest 80 passed)
