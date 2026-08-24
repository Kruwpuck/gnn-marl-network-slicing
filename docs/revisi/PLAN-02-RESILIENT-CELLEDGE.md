# Fase 1 — Per-UE Resilient Constraint (wave v5)

**Fase:** 1
**Prasyarat:** PLAN-01 selesai (tidak ada gerbang yang memblokir — fase ini jalan apa pun hasil diagnostik)
**Keluaran:** collapse rate baru, feasibility per-UE, checkpoint v5
**Estafet:** PLAN-03 (Fase 2)
**Referensi:** PLAN-07-CMDP-NOTES.md
**Master:** PLAN-00-MASTER.md

---

## Kenapa ini perubahan yang paling didukung

Dua jalur analisis independen menunjuk solusi yang sama:

1. **Preseden empiris** — resilient RRM (TSP 2023) menunjukkan per-user minimum-capacity constraint dengan learnable slack melindungi cell-edge **tanpa** merusak transferabilitas
2. **Penjelasan mekanistik** — constraint agregat secara struktural mengizinkan pengorbanan minoritas (lihat PLAN-07 §2)

Konvergensi dua jalur ini layak ditulis di paper.

---

## 1. Sumber

> N. NaderiAlizadeh, M. Eisen, A. Ribeiro, "Learning Resilient Radio Resource Management Policies With Graph Neural Networks," **IEEE Transactions on Signal Processing, vol. 71, pp. 995–1009, 2023**. DOI: 10.1109/TSP.2023.3255547. arXiv:2203.11012. Kode: https://github.com/navid-naderi/Resilient_RRM_GNN

Isi relevan: per-user minimum-capacity constraints adaptif lewat learnable slack variables, direformulasi di domain Lagrangian dual, dilatih primal-dual bersama parameter policy.

**Koreksi sitasi:** sumber NotebookLM menyebut "IEEE TWC 2025" — salah. Venue dan tahun yang benar seperti di atas.

---

## 2. Kenapa bukan α-fair

| | α-fair utility | Resilient constraint |
|---|---|---|
| Sifat perubahan | mengganti objective | menambah constraint |
| Skala reward | berubah drastis (`log(x)` vs `min(rate/t_ref, 5)`) | tidak berubah |
| Klip `[-10,10]` | pecah, kalibrasi ulang | aman |
| Kalibrasi λ | pecah, Gate A diulang penuh | λ existing tetap valid |
| Infrastruktur | perlu baru | primal-dual sudah ada |
| Transferabilitas | tidak dibahas | klaim eksplisit paper |

Risiko konkret: ronde 3 kalibrasi pernah gagal karena klip mematikan gradien saat skala reward dan λ tidak cocok. Mengganti objective mengulang risiko itu.

α-fair disimpan sebagai kandidat ablation opsional (§8).

---

## 3. Formulasi

Objective tetap throughput. Constraint URLLC existing (δ, λ) **tetap berjalan tanpa perubahan**. Ini constraint kedua yang berdiri sendiri.

```
max_θ   E[ Σ_t r_throughput(s_t, a_t) ]

s.t.    E[r_u] ≥ f_min − z_u     ∀u ∈ UE
        z_u ≥ 0
```

Lagrangian:
```
L(θ, μ, z) = E[Σ_t r_throughput]
             − Σ_u μ_u · (f_min − z_u − r_u)
             − c · Σ_u z_u
```

---

## 4. Update rule

```python
# 1. Primal — policy (pola sama, reward diganti r_penalized)
θ ← ascent(r_penalized)

# 2. Dual per-UE (pola identik dengan update λ existing)
μ_u ← max(0, μ_u + α_μ · (f_min − z_u − r_u))

# 3. Slack per-UE (BARU)
z_u ← max(0, z_u + α_z · (μ_u − c))
```

Reward ter-penalti:
```python
shortfall_u = max(0.0, f_min - z_u - rate_u)
r_penalized = r_throughput - sum(mu_u * shortfall_u for u in ues)
```

**Timescale:** `α_z < α_μ < α_θ`. Slack bergerak paling lambat — kalau terlalu cepat, ia melonggar mengejar policy dan constraint jadi hampa (pola kegagalan yang sama dengan λ vacuous di v3).

**Frekuensi:** setiap `dual_update_every` step, samakan dengan λ existing supaya window-variance yang sudah divalidasi tetap berlaku.

**Inisialisasi:** μ dan z mulai dari 0 atau nilai kecil (0.1) — agen fokus throughput di iterasi awal sebelum constraint mengikat.

---

## 5. μ dan z terhadap transferabilitas (KOREKSI — konflik K1)

Versi awal dokumen ini menyatakan "μ dan z tidak boleh ikut transfer". Itu kurang presisi. Yang benar:

| Peran μ | Status |
|---|---|
| **Input** — μ_u dipetakan sebagai node/edge attribute pada graf | **AMAN dan disarankan.** Dimensi node feature tetap; jumlah node yang berubah. Policy jadi bisa "melihat" tekanan constraint per-UE dan meresponsnya |
| **Parameter** — μ_u tersimpan di `state_dict` checkpoint | **MERUSAK.** Dimensi model jadi bergantung jumlah UE, klaim transferabilitas runtuh |

### Aturan implementasi
- μ dan z **dihitung dan disimpan di training loop**, bukan sebagai `nn.Parameter` di model
- Boleh diumpankan ke graf sebagai fitur input
- **Jangan** masuk `state_dict` policy
- Saat evaluasi zero-shot: μ dan z diinisialisasi ulang dari nol, bukan di-load

### Uji wajib
Verifikasi `state_dict` policy identik dimensinya sebelum dan sesudah perubahan ini.

---

## 6. Simetri lintas algoritma

Constraint diimplementasikan di **environment / training loop bersama**, bukan di kelas agen.

- Seluruh 8 algoritma menerima `r_penalized` yang dihitung kode yang sama
- `f_min`, `c`, `α_μ`, `α_z`, nilai awal μ dan z: satu sumber di `configs/experiment_config.yaml`
- Tidak ada nilai per-algoritma

**Gate C1 dijalankan ulang** setelah perubahan: aksi identik → `(floor_applied, lam, mu, z, delta, violation_rate, reward)` identik bit-per-bit, minimal 3 seed × seluruh floor-mode.

---

## 7. Kalibrasi f_min (sebelum wave)

Terlalu rendah → tidak mengikat (kegagalan v3 terulang). Terlalu tinggi → infeasible (kegagalan ronde 3 terulang).

**Protokol** (mengikuti pola Gate A yang sudah terbukti):

1. Sapuan statis: ukur distribusi `embb_p5_mbps` yang bisa dicapai di titik operasi sekarang
2. Ambil persentil-25 dari distribusi run non-kolaps sebagai kandidat `f_min`
3. Latih 1 baseline referensi **non-GNN** (`ippo`, konsisten dengan Gate A) di kandidat itu
4. Cek dua gerbang:
   - **Feasible:** shortfall rata-rata konvergen ke sekitar nol, bukan tumbuh monoton
   - **Mengikat:** rata-rata μ steady-state > 0 terukur, DAN `embb_p5` bergerak ≥ 1 std window antara μ=0 dan μ aktif
5. Kalau tidak lolos, sesuaikan `f_min`, ulangi

Beku sebelum wave penuh. Catat di ledger. **Jangan** pilih `f_min` berdasarkan algoritma mana yang diuntungkan.

---

## 8. Ablation

| Arm | Tujuan |
|---|---|
| `resilient=none` | baseline, sama dengan v4 |
| `resilient=fixed` | `f_min` tetap tanpa slack (`z_u ≡ 0`) |
| `resilient=learned` | usulan penuh |

Arm `fixed` adalah uji kejujuran: kalau hasilnya sama dengan `learned`, klaim "learnable slack" tidak didukung dan mekanismenya cukup disebut minimum-capacity constraint biasa.

α-fair (α=1) bisa jadi arm keempat, tapi **hanya kalau** kalibrasi klip dan λ dijalankan penuh terpisah untuk arm itu.

---

## 9. Pra-registrasi (tulis SEBELUM wave)

**Hipotesis:**
> Per-UE resilient constraint dengan learnable slack menurunkan collapse rate cell-edge pada seluruh arsitektur. Karena constraint bersifat lokal per-UE dan dapat dikomputasi lewat message passing, arsitektur GNN diperkirakan mencapai collapse rate lebih rendah pada throughput yang setara dibanding baseline MLP per-agen.

**Metrik primer** (kunci sebelum wave):
1. `cell_edge_collapse_rate` — proporsi binomial, CI Wilson
2. `embb_p5_mbps` — level dan distribusi
3. `timely_throughput_mbps` — harga yang dibayar
4. `sla_satisfaction_pct` — memastikan constraint URLLC tidak rusak
5. Retensi zero-shot 10/20 gNB
6. **Feasibility rate zero-shot** — apakah violation di bawah δ bertahan di topologi baru (uji eksplisit klaim literatur yang belum terbukti di datamu)

**Yang dilaporkan apa pun hasilnya:**
- Collapse turun tapi throughput ikut turun → laporkan trade-off
- GNN tidak lebih baik dari MLP → hasil sah
- Zero-shot rusak → laporkan sebagai biaya mekanisme

**Seed:** ikuti aturan biaya yang sudah dibekukan. PPO ≥20 seed. DQN sesuai anggaran device-jam.

---

## 10. Urutan eksekusi

1. Implementasi μ, z, `r_penalized` di environment/training loop bersama
2. Uji unit: `r_penalized` identik lintas 8 algoritma untuk aksi sama
3. Uji dimensi: `state_dict` policy tidak berubah (§5)
4. Gate C1 ulang, 9/9
5. Kalibrasi `f_min` (§7), bekukan, catat di ledger
6. Tulis pra-registrasi (§9), commit **sebelum** wave
7. Wave v5 penuh
8. Evaluasi: rliable IQM + Wilson CI, per keluarga
9. Zero-shot 10/20 gNB pada checkpoint v5
10. Bandingkan v4 vs v5 eksplisit

---

## 11. Risiko

| Risiko | Mitigasi |
|---|---|
| Slack melonggar sampai constraint hampa | `α_z << α_μ`; gerbang "mengikat" §7 |
| μ divergen (infeasible) | gerbang "feasible" §7; turunkan `f_min` |
| Throughput anjlok dalam | laporkan apa adanya; jangan tune `f_min` ke hasil |
| Constraint URLLC existing terganggu | monitor λ dan `sla_satisfaction_pct`; laporkan sebagai interaksi antar-constraint |
| Reward ter-penalti kena klip | monitor fraksi step terklip; kalau >5%, pisahkan klip objective dari klip penalti |
| Zero-shot rusak | verifikasi §5 sebelum wave; kalau tetap rusak, itu temuan |

---

## 12. Larangan

1. `f_min`, `c`, `α_μ`, `α_z` dipilih dengan justifikasi properti task, tidak pernah properti hasil
2. Constraint identik ke 8 algoritma — tidak ada nilai per-algoritma
3. μ dan z **tidak boleh** jadi `nn.Parameter` atau masuk `state_dict`
4. Metrik primer dikunci §9 sebelum wave
5. Jangan ubah `α_λ`, `λ_init`, `λ_max`, `dual_update_every` (konflik K5)
6. Hasil v4 tetap dilaporkan penuh
7. Kalau v5 tidak memperbaiki apa pun, itu hasil sah
8. Seed yang kolaps tidak dibuang
