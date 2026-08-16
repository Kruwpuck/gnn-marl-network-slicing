# Struktur paper — dua keputusan (2026-08-16)

Diputuskan manusia (`Habb`), dicatat juga di `runs/2026-08-05-run01/ledger.md`. Dokumen ini
menetapkan **posisi** dua temuan dalam paper, bukan angka baru: setiap angka di bawah
disalin dari file report yang di-generate, dengan sumbernya disebut per baris.

---

## 1. Collapse rate cell-edge naik jadi HASIL UTAMA

Bukan lagi metrik robustness pendukung. Ini jawaban atas pertanyaan riset — meski bukan
jawaban yang diharapkan. Aturan pelaporan D3 berlaku penuh: kekalahan proposed ditulis
dengan aturan yang sama dengan kemenangan.

**Temuan.** Kontrol terpusat melindungi cell-edge; koordinasi lewat message-passing tidak.

Collapse rate, readout stokastik, ambang 0.01 Mbps pada `embb_p5_mbps`, unit = seed
(`results/STABILITY_v4_stoch.md`):

| algoritma | kolaps | Wilson 95% |
|---|---|---|
| `central-ppo` | 0 dari 5 seed | [0.00, 0.43] |
| `central-dqn` | 1 dari 5 seed | [0.04, 0.62] |
| `gnn-mappo_gat` | 4 dari 5 seed | [0.38, 0.96] |
| `gnn-madqn_gat`, `gnn-madqn_sage`, `gnn-mappo_sage`, `idqn`, `ippo` | 5 dari 5 seed | [0.57, 1.00] |

Penulisan wajib bentuk "k dari 5 seed" (keputusan scoping C4, `handoff/goal1.md`).

**Yang membuatnya temuan, bukan sekadar tabel:** agregatnya tidak dibayar. Pada readout
stokastik (`results/RLIABLE_v4_stoch.md`) proposed justru **unggul** atas baseline terpusat
di KPI agregat sambil kolaps di cell-edge:

| perbandingan | `timely_throughput_mbps` IQM [95% CI] | verdict | collapse |
|---|---|---|---|
| `gnn-mappo_gat` | 67.9627 [67.3482, 68.5825] | — | 4 dari 5 |
| `ippo` | 68.5563 [68.2290, 69.0997] | COMPARABLE | 5 dari 5 |
| `central-ppo` | 64.2827 [64.1337, 64.3955] | proposed BETTER | 0 dari 5 |
| `gnn-madqn_gat` | 67.0816 [67.0724, 67.1221] | — | 5 dari 5 |
| `idqn` | 67.0774 [67.0028, 67.1032] | COMPARABLE | 5 dari 5 |
| `central-dqn` | 61.8649 [61.8159, 61.9354] | proposed BETTER | 1 dari 5 |

Jadi kalimatnya bukan "proposed kalah", melainkan: **proposed membeli throughput dan SLA
agregat dengan mengorbankan UE cell-edge**, dan satu-satunya arsitektur yang tidak
melakukannya adalah kontrol terpusat, yang justru membayar di agregat. Trade-off itu yang
jadi hasil, bukan salah satu sisinya saja.

**Pola yang sama sudah muncul di v3, pada algoritma berbeda.** `results/RLIABLE.md` (wave
v3): `ippo` `embb_p5_mbps` IQM = 0.0000 [0.0000, 0.0000] sementara `timely_throughput_mbps`
30.5941 COMPARABLE terhadap `gnn-mappo_gat` 30.4868. Di v3 hanya baseline independen yang
kolaps; di v4, pada task yang sudah punya daya diskriminasi, seluruh varian proposed ikut.
Ini menaikkan temuan dari anekdot satu wave jadi pola lintas-wave.

**Batas klaim (wajib ikut tercetak):** klaim komparatif saja. Berapa persis rate-nya dan
bentuk bimodalitas di dalam satu algoritma tidak diklaim — C4 (seed ≥ 20) GAGAL di 5 seed.

**Penempatan:** section hasil utama, sebelum tabel KPI agregat, supaya pembaca melihat
trade-off-nya dulu dan bukan menemukannya sebagai catatan kaki.

---

## 2. Temuan readout naik jadi SUBSECTION METODOLOGI

**Temuan.** Tidak ada diagnostik murah yang memprediksi model mana yang dirusak argmax,
sehingga protokol pembacaan wajib dikunci **sebelum** hasil dilihat.

Bukti (`results/READOUT_COMPARISON.md`, empat algoritma PPO, satu-satunya yang punya
distribusi aksi):

- Entropi 2.191–2.291 nat terhadap plafon `ln 11` = 2.398 — rentang **0.100 nat**.
- `p_max` 0.200–0.217, `agree` 0.198–0.215 — argmax bukan perilaku policy di keempatnya, dengan derajat yang praktis sama.
- Celah pembacaan `timely_throughput_mbps` (stokastik − greedy): `central-ppo` −1.01, `ippo` +0.46, `gnn-mappo_sage` +28.77, `gnn-mappo_gat` +45.27 — rentang **46.28 Mbps**.

Degenerasi argmax seragam; akibatnya pada KPI bergantung arsitektur. Karena itu menyaring
angka greedy "yang kelihatan wajar" bukan pengaman, dan entropi/`p_max` tidak bisa dipakai
sebagai penyaring.

**Kontrafaktual — inti argumennya.** Protokol P3 dibekukan 2026-08-08, sebelum satu pun
hasil v4 ada. Andai laporan tetap dibaca greedy (`results/RLIABLE_v4_greedy.md`), keluarga
PPO terbaca:

- `gnn-mappo_gat` IQM = **16.7700** [0.0002, 53.9184] lawan `ippo` **68.0649** [65.6590, 70.9259] → **proposed WORSE (CI terpisah)**

Pembacaan stokastik atas checkpoint yang **sama persis** (`results/RLIABLE_v4_stoch.md`):

- `gnn-mappo_gat` IQM = **67.9627** [67.3482, 68.5825] lawan `ippo` **68.5563** [68.2290, 69.0997] → **COMPARABLE**

Kesimpulan "GNN kalah telak" itu sepenuhnya artefak pembacaan: bobotnya identik, yang
berbeda hanya apakah aksi diambil dari campurannya atau dari modusnya. Lebar CI greedy
sendiri sudah jadi tanda — [0.0002, 53.9184] bukan pengukuran, itu dua rezim episode yang
dirata-ratakan.

**Yang tetap dilaporkan, bukan disembunyikan:** tabel greedy lengkap ikut dicetak (P3
mewajibkan dua-duanya). Kolaps argmax adalah temuan tentang degenerasi policy — sebagian
arsitektur kolaps, sebagian tidak, dan itu informasi nyata. Tanda pendukung: sd antar-episode
`sla_violation_pct` pada readout greedy justru **membengkak** untuk `gnn-mappo_gat`/`_sage`
(34.95 / 37.82 pp lawan 11.04 / 10.45 stokastik), berbeda dari kasus λ = 30 di kalibrasi di
mana sd greedy malah runtuh ke 0.11 pp. Dua bentuk kegagalan pembacaan yang berlawanan
tandanya, dua-duanya tidak terprediksi oleh entropi.

**Penempatan:** subsection di bagian metodologi (protokol evaluasi), dengan kontrafaktual
di atas sebagai tabel — bukan appendix. Klaim metodologisnya: *protokol pembacaan adalah
keputusan pra-registrasi setara dengan pemilihan metrik.*
