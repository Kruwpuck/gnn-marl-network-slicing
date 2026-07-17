[← Revisi 1 (v2 coupled)](03_revisi1-v2-coupled.md) | [Index](00_INDEX.md) | [Bug-fix & kalibrasi →](05_bugfix-topologi-kalibrasi.md)

# 04 — Revisi 2: Env v3 (CMDP + Packet-Level URLLC + PRB Floor)

**Commit:** `87374bc` (2026-07-17)
**Sumber justifikasi:** deep-research report (`compass_artifact_wf-b0f2cba3-f928-5ed1-9aea-fa533e8cac21_text_markdown.md`) yang mereview literatur 2021-2025 tentang constrained RL untuk network slicing.

## Recap 4 masalah dari v2

1. URLLC starvation masih terjadi di semua run meski reward sudah log-scale.
2. SLA proposed kalah tipis dari central baseline — dicurigai artefak scalarization (bobot bisa saling ditukar).
3. Jain's fairness rendah, sebagian struktural, butuh framing + KPI cell-edge.
4. Single seed, belum publishable.

## Investigasi tambahan sebelum redesain: 3 cacat model yang belum pernah terdiagnosis

Sebelum mengubah reward, dilakukan audit ulang kode environment dan ditemukan tiga cacat struktural yang menjelaskan *kenapa* starvation bertahan meski reward sudah diperbaiki di v2:

1. **Tidak ada durasi slot eksplisit.** `traffic.step(dt=1.0)` dipanggil tanpa `dt` eksplisit — slot implisit = 1 detik. Tapi SLA bound = 10 **milidetik**. Delay dihitung dari formula drain-time fluida (`queue/rate*1000`), bukan latensi paket per-slot, dan tidak terikat ke resolusi waktu manapun. Model tidak punya mekanisme untuk membuktikan klaim SLA 10 ms — delay yang dilaporkan bukan delay paket sungguhan.
2. **Reward memakai kapasitas Shannon, bukan bit yang benar-benar terkirim.** `w1 * embb_rates.mean()`, dan `embb_rates` adalah kapasitas dari alokasi bandwidth, bukan `min(queue, rate)`. Agent dibayar karena *mengalokasikan* bandwidth ke eMBB walau antrean eMBB kosong — insentif langsung untuk melapar-kan URLLC terlepas dari demand aktual. Ini kandidat kuat akar starvation yang bertahan lintas v1 dan v2.
3. **`slices.urllc.reliability: 0.999` di config tidak pernah dipakai di kode manapun** — itu spesifikasi keandalan per-paket (99.9% paket harus sampai tepat waktu), mustahil dievaluasi tanpa model paket individual (antrean v1/v2 keduanya fluida, bukan per-paket).

## Arah desain v3

Ubah masalah dari *reward shaping berbobot tetap* (rapuh, bisa saling ditukar antar term) menjadi **Constrained MDP (CMDP)**: throughput sebagai objective yang dioptimasi, SLA URLLC sebagai **hard constraint** lewat Lagrangian dual yang dipelajari (bukan bobot tetap). Ditambah model antrean URLLC per-paket dengan deadline drop, dan PRB floor via action projection.

**Prinsip kunci: semua perubahan diterapkan identik ke 8 algoritma** — CMDP, floor, dan queue model masuk seluruhnya ke `envs/network_slicing_env.py`, bukan ke policy/agent manapun. Kedua training loop (`train_proposed.py`, `train_baselines.py`) hanya memakai skalar `reward` dari `env.step()`, jadi tidak ada algoritma yang melihat antarmuka berbeda atau diuntungkan secara tidak adil.

## Perubahan teknis

### 1. Slot eksplisit

`slot_duration_ms: 1.0` — semua traffic generator memanggil `.step(dt=slot_s)` secara eksplisit. Episode tetap 200 slot = 200 ms.

### 2. URLLC FIFO per-paket dengan deadline drop

Antrean URLLC diganti dari skalar bit menjadi `deque` per-paket berisi `(arrival_slot, bits_remaining)`. Tiap step:
- Paket yang menunggu lebih lama dari `urllc_max_delay_ms` (10 ms) di-drop (deadline drop, head-of-line).
- Paket yang melebihi kapasitas buffer di-drop (tail drop / overflow).
- Sisa paket dilayani FIFO dengan bit-budget slot ini.

Konsekuensi penting: delay paket yang **berhasil terkirim** otomatis terbatas ≤ 10 ms (dijamin oleh deadline drop), sehingga metrik P99 delay jadi bermakna. Pelanggaran SLA kini muncul sebagai **drop rate**, persis gaya evaluasi 3GPP:

$$\text{violation\_rate} = \frac{\text{dropped\_late} + \text{dropped\_overflow}}{\text{arrived}}, \quad \text{reliability} = 1 - \text{violation\_rate}$$

### 3. eMBB reward pakai bit terkirim, bukan kapasitas

`embb_served = min(queue_embb, embb_rate * slot_s)` — memperbaiki cacat #2 di atas. Agent kini hanya dibayar untuk throughput yang benar-benar dikirim ke antrean yang benar-benar berisi data.

### 4. Reward CMDP (Lagrangian, RCPO two-timescale)

$$r = \underbrace{w_1 \cdot \frac{\bar{T}^{eMBB}_{served}}{T_{ref}} + w_4 \cdot SE}_{\text{objective (throughput saja)}} - \lambda \cdot \text{violation\_rate}$$

$\lambda$ **bukan bobot tetap** — dipelajari sebagai dual variable, di-update tiap `dual_update_every` step ke arah target constraint $\delta$:

$$\lambda \leftarrow \text{clip}(\lambda + \alpha_\lambda(\text{mean\_violation} - \delta),\; 0,\; \lambda_{max})$$

$\lambda$ **persisten lintas episode reset** (bukan direset tiap episode) — dua timescale: policy belajar cepat (tiap step), $\lambda$ bergerak lambat (tiap 2000 step, ~10 episode). Ini pendekatan standar RCPO (Reward Constrained Policy Optimization) untuk constrained RL, dipilih daripada bobot tetap karena constraint keras (SLA) secara struktural tidak bisa "ditukar" dengan throughput — kalau melanggar delta, $\lambda$ naik sampai constraint dominan, apapun nilai throughput-nya.

### 5. PRB floor via action projection

Sebelum bandwidth displit, alokasi minimum URLLC dipaksakan:

- `floor.mode: none` — tidak ada floor (ablation)
- `floor.mode: static` — floor tetap (`floor_base`)
- `floor.mode: dynamic` — floor naik seiring backlog: $f_{min} = \text{clip}(base + k \cdot \frac{backlog}{Q_{ref}},\, lo,\, hi)$

Floor diterapkan **identik di dalam `env.step()`**, sebelum agen manapun melihat hasilnya — bukan sebuah "bantuan" khusus untuk proposed.

### 6. Observasi diperbarui

Kolom yang usang (delay linear, kini selalu ≤10ms karena deadline drop sehingga tak informatif) diganti `urllc_violation_rate` EWMA — sinyal starvation yang lebih relevan pasca packet-level model. $\lambda$ **tidak** dimasukkan ke observasi (sama untuk semua agent tiap step, hanya menambah noise).

### 7. Protokol evaluasi yang lebih ketat

- **5 seed** (42, 43, 44, 45, 46) — bukan 1.
- **Held-out evaluation**: model final dievaluasi greedy (bukan sampling) pada seed ≥10,000, terpisah dari seed training.
- **rliable**: IQM (Interquartile Mean) + stratified bootstrap 95% CI, dibandingkan **dalam keluarga budget masing-masing** (DQN vs DQN, PPO vs PPO — tidak pernah dipool lintas keluarga). Kriteria jujur: CI overlap → klaim "comparable", bukan "superior".
- **Ablation floor** (`none`/`static`/`dynamic`) pada subset 4 algoritma (`gnn-madqn_sage`, `gnn-mappo_gat`, `idqn`, `central-ppo`) — untuk memastikan keunggulan GNN (jika ada) bertahan pada floor yang sama, bukan cuma muncul karena floor menutupi kelemahan.

## Config v3 (kunci)

```yaml
env:
  ue_radius_m: 100          # lihat file 05 — parameter dari bug-fix topologi
  slot_duration_ms: 1.0
  slice_coupled_interference: true

reward:
  w1: 0.4   # eMBB throughput terkirim
  w4: 0.1   # spectral efficiency
  # w2, w3 dihapus — delay/violation kini constraint, bukan reward term

traffic:
  urllc:
    lambda_arrival: 25000.0   # pkt/detik, dikalibrasi — lihat file 05
    packet_size_bits: 256
  embb:
    lambda_burst: 30.0
    lambda_idle: 2.0
    r_burst_to_idle: 20.0
    r_idle_to_burst: 10.0

buffer:
  urllc_max_bits: 40960
  embb_max_bits: 50_000_000

cmdp:
  enabled: true
  delta: 0.12                # dikalibrasi — lihat file 05
  lambda_init: 1.0
  lambda_lr: 0.01
  lambda_max: 100.0
  dual_update_every: 2000

floor:
  mode: dynamic
  base: 0.10
  k: 0.30
  lo: 0.10
  hi: 0.40
```

Nilai `traffic.urllc.lambda_arrival`, `cmdp.delta`, dan `env.ue_radius_m` bukan tebakan — semuanya hasil proses kalibrasi empiris yang diceritakan lengkap di file 05, termasuk bug topologi yang ditemukan di tengah proses.

## Verifikasi kode (sebelum training wave)

- 80/80 test pass (`tests/test_env.py` + suite lain) — termasuk 12 test baru khusus v3: konservasi paket (arrived = delivered + dropped_late + dropped_overflow + in-flight), delay terkirim terbatas ≤ deadline, $\lambda$ naik saat violation > $\delta$, floor projection benar, eMBB reward pakai bit terkirim bukan kapasitas.
- Smoke test 4 kombinasi algoritma (DQN proposed/baseline, PPO proposed/baseline) 3,000 step — CSV terisi, reward finite, $\lambda$ bergerak, violation rate turun dari nilai awal.

## Data & artefak

- `envs/network_slicing_env.py` — docstring lengkap mendokumentasikan desain v3 dan bug-fix (lihat file 05)
- `configs/experiment_config.yaml` — config aktif
- `tests/test_env.py` — 18 test env (12 baru untuk v3)
- Hasil training wave: **belum tersedia saat dokumen ini ditulis** — lihat file 06 untuk status.

---

[← Revisi 1 (v2 coupled)](03_revisi1-v2-coupled.md) | [Index](00_INDEX.md) | [Bug-fix & kalibrasi →](05_bugfix-topologi-kalibrasi.md)
