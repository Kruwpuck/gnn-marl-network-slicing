# STUCK
run: 2026-08-05-run01   iterasi: 3   level: 5 (pertanyaan metode)   waktu: 2026-08-06T01:05:00

## Ringkasan

Gate A mencetak 4/4 PASS di ronde 3, tapi **A2 lolos secara hampa**. λ naik 18× tanpa
mengubah perilaku policy sama sekali. Melaporkan ini sebagai "Gate A lolos" akan
melanggar goal1.md §Integritas (verdict tidak boleh dinaikkan derajatnya, angka tidak
boleh dipaksa cocok dengan target).

## Perintah yang dijalankan

```
python scripts/calibrate_trained.py --seed 42 --tag _calib   # lambda_lr=0.01, dual_update_every=2000
python scripts/calibrate_trained.py --seed 42 --tag _calib2  # lambda_lr=2.0,  dual_update_every=5000
python scripts/calibrate_trained.py --seed 42 --tag _calib3  # lambda_lr=14.0, dual_update_every=5000
```

## Bukti

Rata-rata 20% step terakhir, `results/logs/central-ppo_calib{,2,3}_seed42.csv`:

| ronde | λ | violation % | eMBB Mbps | timely Mbps | delay p99 ms | ep_reward |
|---|---|---|---|---|---|---|
| 1 | 1.03 | 6.15 | 7.82 | 29.82 | 8.81 | 565.4 |
| 2 | 3.19 | 5.98 | 7.82 | 29.87 | 8.82 | 502.4 |
| 3 | 18.59 | 5.91 | 7.85 | 29.87 | 8.98 | 57.5 |

λ naik 18×. Violation turun 0.24pp — di bawah std window (1.61pp), jadi tidak bisa
dibedakan dari derau. eMBB, timely throughput, dan p99 tidak bergerak sama sekali.
Anjloknya `ep_reward` murni suku penalti `λ·violation`, bukan perubahan kebijakan.

Hasil gate ronde 3 (disalin dari `transcripts/3-gateA-calib3-central-ppo.log`):

```
GATE A -- central-ppo_calib3_seed42  (delta=0.0500, lambda_arrival=25000, dual_update_every=5000)
  [PASS] A1 violation in [0.7d, 1.0d]     4.14% vs [3.50%, 5.00%]
  [PASS] A2 lam_ss >= 5.0                 18.594
  [PASS] A3 deadline:overflow >= 3:1      inf:1  (late 5.89%, overflow 0.00%)
  [PASS] A4 window std < 2.0pp            1.62pp over 200 windows
```

## Diagnosis

Dual mengejar violation dari policy **stokastik saat training** (5.9%), sementara A1
diukur pada policy **deterministik saat evaluasi held-out** (4.14%). Selisihnya ~1.8pp
dan tidak pernah menutup, jadi gap `violation − delta` tetap positif selamanya dan λ
mengintegrasi tanpa henti. λ besar bukan tanda constraint mengikat; itu tanda
integrator yang tidak pernah mencapai titik tetap.

Dua penguat, keduanya terukur:

1. **Batas ruang aksi baseline referensi.** `central-ppo` memancarkan satu tier PRB
   untuk 5 gNB (`training/train_baselines.py`: `actions = np.full(n_gnb, central_action)`).
   Sapuan statis (`scripts/calibrate_load.py`) menunjukkan alokasi seragam mentok di
   6.04% pada frac=0.80 dan 6.26% pada frac=0.50 — di atas `delta=0.05`. Policy terlatih
   sudah di 5.91%, praktis di lantai kemampuannya. Tidak ada λ yang bisa membeli
   perbaikan yang tidak tersedia di ruang aksi.

2. **Clipping reward.** `envs/network_slicing_env.py:_compute_reward` menjepit
   `r = r_obj − λ·violation` ke `[-10, 10]`. Dengan `r_obj ≈ 2.75/step` dan λ=18.6,
   setiap step dengan violation tinggi tembus batas bawah dan terjepit. Setelah
   terjepit, turunan reward terhadap violation jadi nol — persis sinyal yang seharusnya
   dipakai dual untuk menyetir. Jepitan ini dirancang saat λ≈1.

## Sudah dicoba dan gagal

| percobaan | hasil |
|---|---|
| `lambda_lr` 0.01 → 2.0 | λ 1.03 → 3.19, perilaku tidak berubah |
| `lambda_lr` 2.0 → 14.0 | λ 3.19 → 18.59, perilaku tetap tidak berubah |
| `dual_update_every` 2000 → 5000 | A4 lolos (2.74 → 1.61pp). Tidak berhubungan dengan A2 |

## Yang belum dicoba dan kenapa

- **Menaikkan `lambda_arrival`** — melebarkan gap, tapi memindahkan titik operasi yang
  sudah membuat A1 lolos dua ronde berturut-turut. Butuh keputusan manusia (C6).
- **Mengganti baseline referensi** ke algoritma per-gNB — mengubah dokumen goal yang
  sudah dibekukan. Dilarang dilakukan agent (§Larangan operasional).
- **Menaikkan batas clip reward** — mengubah definisi reward, dengan demikian mengubah
  semua hasil v3 yang bisa dibandingkan. Butuh keputusan manusia.
- **Mengukur violation dual pada rollout deterministik** — mengubah semantik env dan
  menambah biaya per step. Perubahan metode, bukan perbaikan bug.

## File tersentuh

- `configs/experiment_config.yaml` — `cmdp.lambda_lr`, `cmdp.dual_update_every`
- `scripts/calibrate_trained.py` — baseline referensi, ambang A1, pengukuran A2/A3/A4, `--tag`

## Commit terakhir hijau

`93d1dde`
