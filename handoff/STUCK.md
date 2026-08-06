# STUCK
run: 2026-08-05-run01   iterasi: 4   level: 5 (pertanyaan metode)   waktu: 2026-08-06T07:10:00

Versi sebelumnya dokumen ini (diagnosis "A2 lolos hampa") ada di commit `50664fb`.
Diagnosis itu sudah digantikan oleh pengukuran di bawah.

## Ringkasan

Jalur B dijalankan sesuai keputusan: baseline referensi dipindah ke `ippo`, Gate A2
dipecah jadi A2a/A2b, mode pengukuran dinyatakan eksplisit. Semuanya sudah masuk
`goal1.md` + ledger.

Tapi pengukuran lantai ruang aksi menemukan sesuatu yang lebih dalam dari soal
feasibility, dan itu membatalkan asumsi jalur B **maupun** jalur A dalam bentuk
sederhananya:

**Di titik operasi sekarang, seluruh rentang violation yang bisa dicapai policy
hanya 1.6 pp (3.66%–5.26%), dan δ=0.05 duduk di tengahnya.** Tidak ada policy —
arsitektur apa pun — yang bisa menyetir constraint. Itu penjelasan 35/40
`COMPARABLE` di v3 yang lebih baik daripada dinamika dual.

## Perintah yang dijalankan

```
python scripts/probe_action_floor.py --episodes 150 --final-episodes 150
python scripts/probe_action_floor.py --episodes 150 --final-episodes 150 --floor-mode none
python scripts/probe_action_floor.py --episodes 60 --tiers 1,3,5,7,9 --lambda-arrival L
python scripts/probe_action_floor.py --episodes 60 --tiers 1,3,5,7,9 --lambda-arrival L --dbp-buffer 2
python scripts/diag_breakdown.py
```
Log: `runs/2026-08-05-run01/transcripts/{4,5,6,7,8,9}-*.log`

## Bukti

### 1. Lantai dinamis menelan seluruh daya setir aksi

150 episode, seed 0+, SE 0.63–0.96 pp, std antar-episode 7.7–11.7 pp:

| frac | viol% `floor=none` | viol% `floor=dynamic` | eMBB Mbps |
|---|---|---|---|
| 0.00 | 94.50 | 4.41 | 8.62 |
| 0.10 | 13.71 | 4.41 | 8.62 |
| 0.20 | 6.90 | 5.05 | 8.60 |
| 0.30 | 5.26 | 5.26 | 8.53 |
| 0.50 | 4.25 | 4.25 | 8.21 |
| 0.70 | 3.89 | 3.89 | 7.39 |
| 1.00 | 3.66 | 3.66 | 0.00 |

Lantai menghapus seluruh wilayah kelaparan. Lebih dari itu, pada frac 0.0–0.2 ia
**mengalahkan** alokasi tetap yang lebih besar (4.41% < 5.26% di frac 0.30) karena
`f_min` responsif terhadap backlog. Adaptivitas terbukti berharga di task ini —
tapi lantai sudah memberikannya gratis ke semua algoritma, jadi tidak ada yang
tersisa untuk dibedakan.

### 2. Drop terikat SINR/deadline, bukan PRB

`scripts/diag_breakdown.py`, λ=25000, per-gNB pooled%:

```
frac=0.5: [10.3  5.2  5.5  2.2  5.8]   jaringan 5.81%
frac=0.8: [10.3  5.2  5.1  2.0  5.0]   jaringan 5.49%
```

Menaikkan alokasi 60% (0.5 -> 0.8) menggerakkan gNB-0 sebesar **0.0 pp**. Sisa
violation bukan antrean yang kelaparan PRB, melainkan paket yang tidak sempat
terkirim pada laju yang tersedia. PRB tambahan tidak membelinya.

### 3. Ruang aksi per-gNB hampir tidak menurunkan lantai

Seed held-out, coordinate descent 2 ronde: lantai uniform 5.34%, lantai per-gNB
5.33%. Selisih 0.01 pp. Keunggulan per-gNB ada di **eMBB pada violation yang sama**
(2.77 vs 0.00 Mbps), bukan di constraint. Konsekuensi untuk jalur B: mengganti
baseline referensi ke `ippo` **tetap benar** (titik operasi tidak lagi ditentukan
ruang aksi tersempit), tapi tidak memulihkan feasibility — tidak ada yang perlu
dipulihkan, δ=0.05 memang sudah di atas lantai 3.66%.

### 4. Derau sampel setara dengan seluruh rentang yang bisa disetir

Vektor identik tier=10, dua populasi seed, 150 episode masing-masing:
3.66% (seed 0–149) vs 5.70% (seed 10000–10149). Selisih 2.0 pp, sementara seluruh
rentang yang bisa disetir 1.6 pp. `evaluate_checkpoints.py` default 30 episode
memberi ±2.8 pp. Gate B tidak akan bisa memisahkan algoritma di titik operasi ini,
berapa pun seed-nya.

> **Koreksi 2026-08-06T11:05.** Selisih 2.0 pp itu **derau sampel, bukan properti
> populasi seed**. Uji tuntas 400 episode per populasi pada alokasi identik memberi
> 14.18 % ±0.68 (seed 0+) vs 14.87 % ±0.70 (seed 10000+) — selisih 0.69 pp pada SE
> gabungan 0.98 pp, yaitu 0.7 σ, tidak signifikan. Yang tetap berlaku dari butir ini
> hanyalah kesimpulan praktisnya: dengan std antar-episode ~14 pp, eval 30 episode
> memberi ±2.6 pp dan tidak layak untuk gate berpresisi pp. Seluruh eval kalibrasi
> sejak ronde 4 memakai ≥ 150 episode.

### 5. Rezim yang mengembalikan daya setir — terukur

`floor=dynamic`, buffer diskalakan ke 2×DBP (aturan a priori yang sudah dipakai
config), 60 episode seed 0+:

| λ | urllc_max_bits | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 | rentang | overflow | lantai held-out |
|---|---|---|---|---|---|---|---|---|---|
| 25000 | 131072 | 4.41 | 5.26 | 4.25 | 3.89 | 3.70 | 1.6 pp | 0.00% | 5.70 ±0.71 |
| 40000 | 204800 | 10.47 | 10.61 | 7.95 | — | — | — | 0.00% | 8.36 ±1.26 |
| 60000 | 307200 | 19.43 | 17.83 | 13.17 | 11.33 | 10.33 | 9.1 pp | 0.00% | 12.10 ±1.56 |
| 90000 | 460800 | 32.94 | 29.29 | 20.43 | 17.15 | 15.64 | 17.3 pp | 0.00% | 17.77 ±1.80 |

Menaikkan λ **tanpa** menaikkan buffer gagal A3: λ=60000 dengan buffer lama memberi
late 8.21% vs overflow 6.19% (1.3:1, ambang 3:1). Menaikkan keduanya bersama menjaga
A3 (`overflow 0.00%` di seluruh tabel) sekaligus melemahkan lantai dinamis —
`q_ref = urllc_max_bits`, jadi buffer lebih besar berarti `f_min` lebih kecil pada
backlog yang sama. Pelemahan lantai itulah yang mengembalikan daya setir.

## Diagnosis

Tiga hal menumpuk, dan hanya yang ketiga bisa diperbaiki tanpa mengubah fisika:

1. Drop terikat deadline/SINR di sebagian gNB — tidak bisa dibeli dengan PRB.
2. Lantai dinamis memberi adaptivitas gratis, menghapus keterampilan yang seharusnya
   membedakan arsitektur.
3. Beban terlalu ringan relatif terhadap buffer, sehingga wilayah tempat alokasi
   berpengaruh seluruhnya berada di bawah `f_min`.

## Sudah dicoba dan gagal

| percobaan | hasil |
|---|---|
| `lambda_lr` 0.01 -> 2.0 -> 14.0 | λ 1.03 -> 18.59, perilaku policy tidak bergerak |
| `dual_update_every` 2000 -> 5000 | A4 lolos; tidak berhubungan dengan daya setir |
| baseline referensi -> `ippo` (jalur B) | benar dan tetap dipakai, tapi lantai per-gNB = lantai uniform (5.33 vs 5.34) |
| naikkan λ saja (40k/60k/90k) | A3 gagal: overflow mendominasi (sampai 18.95% vs late 5.96%) |

## Keputusan yang dibutuhkan

Semuanya menyentuh `C6` (titik operasi dibekukan) atau desain eksperimen, jadi
tidak boleh diputuskan agent.

- **O1 — λ=60000 + `urllc_max_bits`=307200 (rekomendasi).** Rentang 9.1 pp, ~5.6×
  std window (1.62 pp) dan ~6× SE. A3 aman. eMBB tetap responsif (9.01 -> 4.57).
  δ diusulkan ≈ lantai held-out 12.10 + margin 3 pp = **0.15**, A1 jadi [10.5%, 15%].
- **O2 — λ=90000 + `urllc_max_bits`=460800.** Rentang 17.3 pp, diskriminasi paling
  kuat, tapi δ harus ≈ 0.21.
- **O3 — `floor.mode` jadi faktor utama `none`/`static`, λ tetap.** Tidak menyentuh
  C6 selain δ, dan v3 sudah punya ablasi `_floornone`. Tapi ablasi itu **juga**
  menghasilkan COMPARABLE dengan rentang sempit (timely 30.72 vs 30.75), jadi
  bukti empirisnya lemah.
- **O4 — terima dan beralih ke `goal1.md` §Fallback.** Rentang 1.6 pp adalah hasil
  sah dan bisa dilaporkan sebagai batas fase: pada 5 gNB dengan lantai dinamis,
  koordinasi tidak punya ruang. Butir 5 fallback persis tentang ini.

Tegangan yang harus dinyatakan di paper apa pun pilihannya: agar constraint benar-benar
mengikat, δ harus dipasang di 12–21% violation, sementara `slices.urllc.reliability`
di config menyatakan 0.999. Environment ini tidak bisa memberi 99.9% pada alokasi mana
pun — drop-nya terikat deadline. Itu properti task yang harus dilaporkan, bukan angka
yang boleh dipilih diam-diam.

## File tersentuh

- `handoff/goal1.md` — amandemen A2a/A2b, mode pengukuran, baseline referensi `ippo`
- `scripts/calibrate_trained.py` — `REFERENCE_BASELINE`, `--floor-pct`, `--control-tag`, `--as-control`
- `scripts/probe_action_floor.py` — baru

Config **tidak** disentuh pada iterasi ini: `lambda_arrival`, `delta`, `urllc_max_bits`
masih persis seperti di commit `50664fb`.

## Commit terakhir hijau

`5049198` (pytest 80 passed)
