# GOAL

Rekalibrasi environment v3 (`gnn-marl-network-slicing`) sehingga constraint CMDP benar-benar mengikat pada policy **terlatih**, lalu jalankan wave v4 (8 algoritma × N seed, perlakuan identik) supaya eksperimen punya daya diskriminasi untuk menguji apakah arsitektur GNN berpengaruh — **bukan** untuk membuat GNN menang.

Wave v3 menghasilkan 35/40 `COMPARABLE` karena saturasi KPI: kalibrasi beban dilakukan terhadap policy statis, sementara policy terlatih menemukan kebijakan jauh di dalam batas `delta=0.12` (violation 3.5–4.5%, λ konvergen 0.58–1.02 dari plafon 100). Task terlalu longgar → arsitektur tidak punya ruang berbeda. Itu cacat desain eksperimen, dan itu yang diperbaiki di sini.

---

## Kriteria selesai

Wave v4 dinyatakan **selesai** hanya jika **semua** gate di bawah lolos. Kriteria ini dikunci **sebelum** wave dijalankan (pre-registrasi) dan **tidak boleh diubah setelah melihat hasil**.

### A. Gate kalibrasi (harus lolos SEBELUM wave penuh)

| # | Kriteria | Ambang | Cara ukur |
|---|---|---|---|
| A1 | Constraint mengikat pada policy terlatih | `[mean ± t₍0.975,n−1₎ · SE_seed] ⊂ [δ − Δ, δ + Δ]` dengan `Δ = 1 std window` | latih baseline referensi (`ippo`) pada n ≥ 5 seed sampai konvergen; `SE_seed` = sd antar-seed / √n |
| A2a | Constraint feasible | lantai violation ruang aksi baseline referensi ≤ `δ − 1 std window` | sapuan statis per-gNB (`scripts/probe_action_floor.py`), tanpa training |
| A2b | Dual punya efek pada perilaku | violation held-out bergeser ≥ 1 std window antara kontrol **λ dipatok 0** dan run λ steady-state | dua run identik, kontrol dengan `lambda_lr = 0` **dan** `lambda_init = 0` |
| A3 | Mekanisme drop benar | `deadline_drop : overflow_drop` ≥ 3 : 1 | agregat seluruh run kalibrasi |
| A4 | Variance window stabil | std violation level-window < 2.0 pp | window = `dual_update_every` step |

> Baseline referensi dipilih **sebelum** kalibrasi dan wajib baseline (non-GNN), supaya titik operasi tidak pernah ditentukan oleh perilaku model proposed.

> **Mode pengukuran (eksplisit, direvisi 2026-08-08).** A1, A2a, A2b diukur pada policy **stokastik held-out**; A3 dan A4 pada log **training (stokastik)**, karena keduanya properti dinamika training. Pembacaan **greedy/argmax dilaporkan berdampingan di setiap tabel hasil**, tidak pernah dihilangkan, tetapi bukan angka gate. Selisih training-vs-held-out wajib dilaporkan sebagai artefak kalibrasi, bukan dibiarkan implisit.

> **Amandemen 2026-08-06** (dicatat di `runs/2026-08-05-run01/ledger.md`, sebelum wave dijalankan). A2 lama — "`λ` steady-state ≥ 5.0" — salah spesifikasi: ia mengukur **besaran** dual, bukan apakah constraint mengikat. Ronde kalibrasi 3 lolos A2 secara hampa (λ 1.03 → 18.59, perilaku policy tidak bergerak) justru karena `δ` berada **di bawah** lantai yang bisa dicapai ruang aksi `central-ppo`; pada constraint infeasible, λ → ∞ adalah perilaku Lagrangian yang benar, bukan dual yang malas. A2 dipecah jadi A2a (feasibility, syarat perlu) + A2b (sensitivitas, yang sebenarnya dimaksud). Baseline referensi A1 dipindah `central-ppo` → `ippo`: `central-ppo` menyiarkan satu tier PRB ke 5 gNB, jadi titik operasi ikut ditentukan ruang aksi tersempit di wave. `ippo` tetap non-GNN sehingga prinsip A1 utuh. Kedua amandemen dipilih atas dasar properti *task* (feasibility, daya sensitivitas), bukan properti *hasil* — tidak ada hasil per-algoritma v4 yang dilihat saat amandemen ini dibuat.

> **Amandemen 2026-08-08** (dicatat di `runs/2026-08-05-run01/ledger.md`, sebelum wave dijalankan). Tiga perbaikan spesifikasi, semuanya properti pengukuran, tidak satu pun dipilih dengan melihat algoritma mana yang unggul — hasil per-algoritma v4 belum ada.
>
> **(1) A1 jadi uji dua sisi.** A1 menguji apakah constraint mengikat. Constraint yang mengikat menghasilkan violation **di sekitar** δ, bukan di bawahnya: dual adalah pengendali integral yang menggiring violation **ke** sasaran. Pita satu sisi `[0.7δ, 1.0δ]` dirancang saat masalahnya violation jauh di bawah δ (v3, `λ → 0`); masalah itu sudah selesai, dan pita lama justru menghukum keberhasilan mekanismenya — pada ronde 5 lima seed mendarat di 8.61 % training terhadap δ = 8.50 %, yaitu sasaran terkena, tetapi pita menuntut ≤ δ sehingga sekitar separuh seed gagal karena definisi. Diganti pita dua sisi berbasis SE terukur.
>
> Dua rincian statistik menyertainya, keduanya ditetapkan atas dasar prinsip sebelum hasilnya dilihat. **Penyebutnya SE antar-seed, bukan antar-episode:** A1 adalah pernyataan tentang policy terlatih di titik operasi ini, dan SE 150-episode dari satu seed punya nol derajat kebebasan antar-seed. Terukur: sd antar-seed 0.48 pp pada held-out stokastik lawan sd antar-episode 11–14 pp — dua kuantitas yang berbeda ordo, dan yang relevan adalah yang pertama. **Nilai kritisnya `t`, bukan 1.96:** `SE_seed` diestimasi dari n−1 derajat kebebasan, bukan diketahui; dengan n = 5, `t₍0.975,4₎ = 2.776`. Memakai 1.96 berarti berpura-pura SE-nya pasti padahal ia sendiri berderau.
>
> **(1b) Bentuk akhir A1: uji kesetaraan, bukan uji hipotesis titik.** Toleransi kesetaraan diskalakan pada **derau proses** (std window), bukan presisi estimasi mean (SE), karena offset tunak pengendali tidak menyusut dengan jumlah seed; kriteria berbasis SE mustahil dilewati secara asimtotik terlepas dari performa sistem. Pita `|x − δ| ≤ k·SE` juga memberi hadiah pada instrumen yang lebih berderau — makin kacau pengukuran, makin lebar pitanya, makin gampang lolos. Bentuk pemuatan (`CI ⊂ pita`) menghapus kedua cacat sekaligus: interval yang lebih lebar dari pita tidak bisa termuat di dalamnya, jadi pembacaan yang tidak presisi ditolak oleh ketidakpresisiannya sendiri. `Δ` memakai satuan yang sudah dideklarasikan di A2a dan A2b, bukan angka baru.
>
> **Uji falsifikasi, dijalankan sebelum bentuk ini dibekukan.** Sebuah gate yang meloloskan semua rezim bukan gate. Pita `δ ± 1 std window` diuji pada tiga titik operasi cacat yang sudah diketahui, semuanya dengan protokol pengukuran yang sama (`ippo`, stokastik held-out, 150 episode): **v3** (δ=0.12, longgar, λ→0) — mean 5.19 %, meleset 6.81 pp lawan Δ 2.70 pp, **DITOLAK**; **ronde 4** (δ=0.15, longgar, λ→0) — mean 10.57 %, meleset 4.43 pp lawan Δ 1.40 pp, **DITOLAK**; **ronde 3** (δ=0.05, infeasible, λ→∞) — CI `[1.54, 6.75]` lebih lebar dari pita `[3.38, 6.62]`, **DITOLAK** lewat klausa presisi. Titik operasi ronde 6 lolos (CI `[8.58, 9.66]` ⊂ `[6.95, 10.05]`). Catatan jujur: ronde 3 ditolak karena presisi, bukan lokasi — selisih mean-nya 0.86 pp sebenarnya di dalam Δ, sehingga dengan 5 seed ia mungkin lolos A1. Itu bukan lubang: rezim infeasible adalah wilayah kerja **A2a** (lantai ruang aksi ≤ δ − 1 std), yang memang menolak ronde 3, dan A1 tidak pernah diklaim sebagai pendeteksi infeasibility.
>
> **Untuk metodologi paper.** Offset tunak sekitar 0.6 pp antara violation held-out dan `δ` adalah perilaku baku pengendali integral pada plant berderau, bukan tanda constraint gagal mengikat.
>
> **(2) Kontrol A2b dipatok λ = 0.** Spesifikasi lama menulis "run λ-beku" tanpa menetapkan nilainya, jadi ia jatuh ke `lambda_init`. Ronde 5 mematoknya di 1.0 melawan run yang setimbang di 1.87 — dua harga yang hampir sama, sehingga kontrasnya nol *by construction* dan A2b gagal karena cacat spesifikasi, bukan karena dual tidak bekerja. Kontrol yang benar adalah run yang benar-benar tak terbatasi.
>
> **(3) Protokol pelaporan: stokastik primer, greedy tetap dilaporkan.** Bukan preferensi gaya, melainkan konsekuensi hasil ukur (`scripts/policy_confidence.py`, 2026-08-08): sepanjang λ = 0…30, aksi argmax hanya membawa 0.17–0.33 massa probabilitas dari 11 aksi, dan **cocok dengan aksi yang benar-benar diambil policy hanya pada 17–33 % langkah**. Kompetensi policy ada di campurannya, bukan di modusnya — konsisten dengan temuan bahwa violation turun 5 pp tanpa membayar eMBB, yang berarti perbaikannya penempatan PRB terhadap **waktu**, dan satu aksi tetap tidak bisa bervariasi terhadap waktu. Argmax menghapus mekanisme yang mengerjakan tugasnya. Hipotesis awal bahwa kolaps greedy datang dari lonjakan entropi **tidak terkonfirmasi**: entropi justru turun landai (2.34 → 2.10 nat terhadap plafon `ln 11` = 2.398) sementara selisih greedy−stokastik meledak 2.0 → 88.6 pp, dan di λ = 30 policy paling percaya diri se-sweep (`p_max` 0.331, margin 0.209) — argmax di sana bukan memilih di atas derau, melainkan memilih dengan mantap dan salah. Tidak ada ambang bernomor untuk "kegagalan pembacaan": selisih 2–3 pp itu offset sistematis yang selalu hadir, jadi tidak ada SE yang sah jadi penyebutnya. Dipakai dua tanda kualitatif: `agree < 0.5` (argmax bukan perilaku policy) dan sd greedy antar-episode runtuh (λ = 30: 0.11 pp lawan 11–14 pp normal, yakni pembacaan berhenti merespons state).

### B. Gate diskriminasi (dievaluasi SETELAH wave)

| # | Kriteria | Ambang v4 | Nilai v3 |
|---|---|---|---|
| B1 | `timely_throughput_mbps` rentang relatif antar-algoritma | ≥ 5% | 1.0% |
| B2 | `sla_satisfaction_pct` rentang antar-algoritma | ≥ 5 pp | 1.1 pp |
| B3 | `urllc_delay_p99` rentang antar-algoritma | ≥ 2 ms | 0.4 ms |
| B4 | Jumlah KPI tersaturasi | ≤ 1 dari 5 | 4 dari 5 |

### C. Gate validitas (wajib, tidak bisa ditawar)

- [ ] **C1** — Uji identitas perlakuan PASS: aksi identik → `(floor_applied, lam, delta, violation_rate, reward)` identik bit-per-bit, diuji di ≥ 3 seed × seluruh floor-mode
- [ ] **C2** — Nol hyperparameter di-set per-algoritma; seluruhnya dari satu `configs/experiment_config.yaml`
- [ ] **C3** — Keluarga DQN (200K) dan PPO (1M) tidak pernah digabung dalam klaim statistik apa pun
- [ ] **C4** — Seed ≥ 20 untuk KPI cell-edge bimodal; collapse rate dilaporkan sebagai proporsi binomial + CI Wilson/Clopper–Pearson
- [ ] **C5** — Evaluasi held-out (seed ≥ 10000) terpisah penuh dari seed training
- [ ] **C6** — Titik operasi (`lambda_arrival`, `delta`, `buffer.urllc_max_bits`) dibekukan dan di-commit **sebelum** wave penuh dijalankan

### D. Kriteria pelaporan

- [ ] **D1** — Wave v3 dan v4 **dua-duanya** dilaporkan. v3 dibingkai sebagai studi kalibrasi/pilot, bukan dihapus
- [ ] **D2** — Verdict CI dipatuhi apa adanya: CI tumpang tindih → `COMPARABLE`, bukan "superior"
- [ ] **D3** — Kekalahan proposed dilaporkan dengan aturan yang sama dengan kemenangan
- [ ] **D4** — Kalau setelah v4 masih `COMPARABLE`, itu **hasil sah**. Fallback: paper metodologi + zero-shot + stabilitas (lihat §Fallback)

> **BUKAN kriteria selesai:** "GNN-MARL mengungguli baseline." Klaim itu adalah *hipotesis yang diuji*, bukan target yang dikejar. Loop tidak boleh melanjutkan iterasi hanya karena hasilnya belum menguntungkan proposed.

### Keputusan scoping 2026-08-15 (bukan amandemen ambang)

Gate B v4 lolos B1 (11.43 % ≥ 5 %), B2 (8.70 pp ≥ 5 pp), B4 (1 dari 5 ≤ 1), dan **gagal B3** (rentang `urllc_delay_p99` 1.01 ms < 2 ms). Ambang B3 **tetap 2 ms** dan statusnya **tetap GAGAL** di setiap laporan. Ambang itu tidak diamandemen: mengubahnya sekarang berarti memindahkan tiang gawang setelah melihat hasil, yaitu larangan integritas #4. Ini berbeda dari amandemen A1 2026-08-08, yang dipilih karena kriteria lamanya mustahil dilewati secara asimtotik (SE menyusut √n, offset tunak pengendali tidak) dan ditetapkan sebelum ada hasil per-algoritma — B3 tidak mustahil, ia hanya gagal terukur.

Yang berubah hanya **cakupan klaim**, karena Gate B adalah diagnostik daya diskriminasi *task*, bukan uji hipotesis tentang arsitektur. Tiga dari empat kriteria lolos, dan B4 turun dari 4/5 KPI tersaturasi (v3) ke 1/5 (v4), sehingga task v4 memang punya daya diskriminasi. Yang dikatakan B3 adalah: **`urllc_delay_p99` bukan KPI pembeda di rezim beban ini.**

- Klaim dibangun di atas KPI yang lolos: `timely_throughput_mbps`, `sla_satisfaction_pct`, `embb_p5_mbps`, `cell_edge_collapse_rate`.
- `urllc_delay_p99` **dilaporkan penuh** (tabel, CI, semua algoritma) tetapi **tidak dipakai menopang klaim apa pun**, dengan catatan eksplisit bahwa B3 gagal pra-registrasi.
- Catatan jujur atas daftar di atas: `embb_p5_mbps` justru KPI yang B4 hitung sebagai satu-satunya yang tersaturasi (0.0000 identik pada 5 dari 8 algoritma). Klaim di atasnya karena itu hanya sah dalam bentuk diskretnya, `cell_edge_collapse_rate`; level `embb_p5_mbps` dilaporkan apa adanya beserta saturasinya, tidak diklaim.
- Mekanisme kegagalan B3 didiagnosis terpisah di `results/B3_DELAY_CENSORING.md` (sensor deadline pada 10 ms + kuantisasi slot 1 ms). Diagnosis itu **tidak mengubah** verdict B3; ia bahan metodologi paper.

Tidak ada angka, definisi metrik, atau ambang di dokumen ini yang diubah oleh keputusan ini. Diputuskan manusia (`Habb`), dicatat di `runs/2026-08-05-run01/ledger.md`.

### Keputusan scoping C4 2026-08-16 (bukan amandemen ambang)

Wave v4 menjalankan **5 seed** per algoritma; C4 menuntut ≥ 20 untuk KPI cell-edge bimodal. Ambang C4 **tetap ≥ 20** dan statusnya **tetap GAGAL** di setiap laporan. Klausa keduanya (proporsi binomial + CI Wilson) terpenuhi. Seperti pada B3, yang dipecah adalah **cakupan klaim**, bukan ambangnya:

- **Klaim komparatif — sah, tetap dilaporkan, tetapi hanya di keluarga PPO.** "Model proposed kolaps lebih sering daripada baseline terpusat." Klaim ini hanya butuh pemisahan, bukan estimasi rate yang presisi.
  > **Dikoreksi 2026-08-17.** Bukti yang dikutip di sini semula `results/STABILITY_v4_stoch.md` — pembacaan yang untuk keluarga DQN ternyata aksi acak ε = 1.0 (lihat §Penetapan protokol pembacaan keluarga DQN 2026-08-16). Di bawah pembacaan primer per keluarga (`results/STABILITY_v4_primary.md`): **keluarga PPO** `gnn-mappo_sage`, `ippo`, `mlp-knn-ppo` kolaps 5 dari 5 (Wilson `[0.57, 1.00]`) dan `gnn-mappo_gat` 4 dari 5 (`[0.38, 0.96]`) lawan `central-ppo` 0 dari 5 (`[0.00, 0.43]`) — interval tidak beririsan, klaim bertahan. **Keluarga DQN** seluruh intervalnya beririsan (`[0.00, 0.43]` lawan `[0.12, 0.77]`) dan `gnn-madqn_sage` justru 0 dari 5, jadi klaim komparatif **tidak dibuat** di keluarga itu. Ambang C4 tidak disentuh dan statusnya tetap GAGAL. Perlu ikut tercetak bahwa di keluarga PPO garis pemisahnya terpusat lawan per-agen, bukan GNN lawan baseline: `ippo` dan `mlp-knn-ppo` kolaps 5 dari 5 sama seperti varian GNN.
- **Klaim karakterisasi — tidak dibuat.** Berapa persisnya collapse rate sebuah algoritma, dan bagaimana bentuk distribusi bimodal per-seed *di dalam* satu algoritma, tidak diklaim di mana pun. Itulah yang syarat ≥ 20 ada untuk menjawab, dan 5 seed tidak bisa menjawabnya seberapa pun lebar jarak antar-algoritma.
- **Aturan penulisan, mengikat seluruh laporan dan paper:** tulis **"kolaps di k dari 5 seed"**, jangan "kolaps X % waktu". Dengan n = 5 hanya ada 6 nilai rate yang mungkin (0, 0.2, 0.4, 0.6, 0.8, 1.0); bentuk persen menyiratkan presisi kontinu yang datanya tidak punya.

**Aturan biaya perluasan seed — dideklarasikan sebelum dijalankan, atas dasar biaya keluarga, bukan hasil.** Kalau anggaran mengizinkan, keluarga budget yang **lebih murah per seed** dinaikkan ke 20 seed, keempat algoritma di dalamnya sekaligus (2 proposed + 2 baseline, tanpa kecuali), dan keluarga satunya tetap di 5 seed dengan C4-nya tetap GAGAL. Keluarga mana yang lebih murah ditentukan dari `elapsed_sec` terakhir 40 CSV training wave v4, bukan dari perkiraan:

| keluarga | budget | total job-jam | rata-rata per job | biaya 1 seed (4 algoritma) | 15 seed tambahan |
|---|---|---|---|---|---|
| DQN | 200K langkah | 387.84 | 19.39 | 77.57 | 1163.6 |
| PPO | 1M langkah | 41.63 | 2.08 | 8.32 | 124.8 |

Yang lebih murah adalah **PPO**, 9.3× lebih murah per seed — kebalikan dari dugaan bahwa 200K langkah pasti lebih murah dari 1M. Sebabnya bukan jumlah langkah melainkan frekuensi update: DQN belajar di **setiap** langkah atas batch 64 transisi lewat loop Python per-sampel, PPO satu kali per 512 langkah. Angka ini fakta biaya implementasi, tidak menyentuh hasil algoritma mana pun.

**Keputusan: perluasan dilewati.** Wave v4 sudah memakai 429.47 job-jam terhadap anggaran ±450, jadi sisa 20.5 job-jam sementara opsi termurah butuh 124.8. Dicatat jujur: dengan pembacaan anggaran sebagai *device*-jam (satu GPU, 40 job selesai 147.22 jam wall-clock pada paralelisme 6) perluasan PPO justru muat (≈43 jam wall-clock). Kedua pembacaan dilaporkan; memilih pembacaan yang kebetulan meloloskan rencana adalah bentuk lain dari memindahkan tiang gawang, jadi eksekusinya milik keputusan manusia, bukan default agent. Klaim komparatif tidak bergantung padanya. Catatan integritas: memperluas keluarga PPO **memperkuat temuan yang merugikan proposed** (`central-ppo` 0 dari 5 lawan proposed 4–5 dari 5), sehingga aturan biaya ini tidak bisa dibaca sebagai memilih keluarga yang menguntungkan proposed.

> **Diperbarui 2026-08-16 oleh §Kebijakan akuntansi anggaran.** Pembacaan anggaran sudah ditetapkan tetap sebagai device-jam, jadi ambiguitas dua pembacaan di atas tidak lagi terbuka: perluasan PPO (≈ 43 device-jam dari sisa ≈ 303) **muat**, dan statusnya **ditunda**, bukan dilewati — manusia menetapkan zero-shot dikerjakan lebih dulu atas dasar nilai per GPU-jam. Kebijakan itu dipilih sebelum konsekuensinya dipakai dan berlaku dua arah.

Tidak ada angka, definisi metrik, atau ambang di dokumen ini yang diubah oleh keputusan ini. Diputuskan manusia (`Habb`), dicatat di `runs/2026-08-05-run01/ledger.md`.

### Penetapan protokol pembacaan keluarga DQN 2026-08-16 (dideklarasikan SEBELUM diukur)

P3 (beku 2026-08-08) menetapkan pembacaan stokastik sebagai primer, tetapi seluruh buktinya PPO: `p_max`, entropi, dan `agree` hanya terdefinisi bila ada distribusi aksi, dan `results/READOUT_COMPARISON.md` sendiri mencetak "(DQN: no action distribution)". Arti "pembacaan stokastik" untuk DQN tidak pernah ditetapkan manusia; kode diam-diam menghasilkan `epsilon = 1.0`, yaitu aksi acak seragam (kesepakatan dengan argmax 0.097–0.100 lawan 1/11 = 0.091 untuk acak murni). Angka DQN di kolom stokastik karena itu bukan pembacaan policy sama sekali.

**Konvensi (ditetapkan sebelum hasil diagnostik dilihat).** Pembacaan stokastik keluarga DQN adalah **ε = 0.05**, yaitu `epsilon_min`, lantai eksplorasi yang benar-benar dipakai policy di akhir training. Alasannya konsistensi behaviour policy, bukan biaya: itu satu-satunya nilai ε yang benar-benar dialami agen saat perilakunya terbentuk. Nilainya dibaca dari `agent.dqn.epsilon_min` di `configs/experiment_config.yaml`, bukan konstanta baru.

**Yang TIDAK boleh diasumsikan.** "Argmax adalah policy DQN, ε cuma eksplorasi" ditolak sebagai premis. Kolaps yang terukur pada PPO bukan soal stokastisitas policy melainkan lintasan deterministik yang terkunci di rezim degenerat tanpa derau untuk keluar — CI greedy `gnn-mappo_gat` `[0.0002, 53.9184]` adalah dua rezim episode yang dirata-ratakan, dengan sd antar-episode 34.95 pp lawan 11.04 pp stokastik. Mekanisme itu berlaku sama untuk kebijakan greedy DQN di lingkungan multi-agen berkopling. Bedanya hanya PPO sudah diukur dan DQN belum, dan **belum diukur bukan berarti aman**. Ketiadaan diagnostik murah (entropi/`p_max`) justru memperkuat kehati-hatian, karena entropi sudah terbukti tidak memprediksi model mana yang dirusak argmax (rentang 0.100 nat lawan celah pembacaan 46.28 Mbps).

**Aturan keputusan, dikunci sebelum angka DQN yang baru dihitung.** Tanda degenerasi yang dipakai adalah sd antar-episode `sla_violation_pct`, karena ia tidak butuh distribusi aksi — hanya varians hasil — dan sudah terbukti memisahkan kasus PPO. Ambangnya diturunkan dari data PPO yang sudah ada, bukan dari DQN:

| pembacaan PPO | sd greedy | sd stokastik | rasio |
|---|---|---|---|
| `central-ppo` | 13.80 | 14.25 | 0.97 |
| `ippo` | 13.42 | 11.37 | 1.18 |
| `gnn-mappo_gat` | 34.95 | 11.04 | 3.17 |
| `gnn-mappo_sage` | 37.82 | 10.45 | 3.62 |

- **Rasio `sd_greedy / sd_(ε=0.05)` > 2.0** → pembacaan greedy dinyatakan degenerat. (PPO memisah bersih di 1.18 lawan 3.17.)
- Uji absolut penyerta: **sd greedy > 25 pp** (titik tengah 13.80 dan 34.95). Dicatat jujur bahwa sd greedy DQN sudah terlihat di `results/READOUT_COMPARISON.md` (14.16–19.66 pp) dan semuanya di bawah 25 pp, jadi uji yang benar-benar mengikat adalah **rasio**, yang penyebutnya belum ada saat aturan ini ditulis.

Konsekuensi yang mengikat, dua-duanya diterima di muka:

- **Rasio ≤ 2.0** → tidak ada kolaps greedy pada DQN; argmax ditetapkan sebagai pembacaan primer keluarga DQN, sekarang dengan bukti dan bukan asumsi. Kolom stokastik DQN yang lama (ε = 1.0) dibuang sebagai cacat instrumen.
- **Rasio > 2.0** → argmax tidak sah sebagai pembacaan primer DQN; kedua pembacaan (argmax dan ε = 0.05) dilaporkan berdampingan seperti PPO, dan seluruh eval stokastik DQN dijalankan ulang di ε = 0.05.

**Hasil diagnostik (dijalankan setelah aturan di atas dikunci), 20 checkpoint × 150 episode:**

| algo | sd greedy | sd (ε = 0.05) | rasio | verdict |
|---|---|---|---|---|
| `central-dqn` | 15.51 | 15.39 | 1.01 | ok |
| `gnn-madqn_gat` | 15.41 | 15.22 | 1.01 | ok |
| `gnn-madqn_sage` | 14.16 | 14.04 | 1.01 | ok |
| `idqn` | 19.66 | 17.85 | 1.10 | ok |

0 dari 4 melewati ambang, dan jaraknya tidak marginal: 1.01–1.10 lawan ambang 2.0, sementara kasus PPO yang degenerat duduk di 3.17 dan 3.62. **Argmax ditetapkan sebagai pembacaan primer keluarga DQN**, dengan bukti. Kolom stokastik DQN yang lama (ε = 1.0) dibuang sebagai cacat instrumen, bukan sebagai hasil.

**Konsekuensi yang jauh lebih besar dari soal protokol.** Collapse rate `embb_p5_mbps` keluarga DQN di bawah pembacaan yang sah berbeda tajam dari yang pernah dilaporkan:

| algo | ε = 1.0 (cacat, pernah dilaporkan) | greedy (primer) | ε = 0.05 |
|---|---|---|---|
| `central-dqn` | 1 dari 5 | 0 dari 5 | 0 dari 5 |
| `gnn-madqn_gat` | 5 dari 5 | 2 dari 5 | 3 dari 5 |
| `gnn-madqn_sage` | 5 dari 5 | 0 dari 5 | 0 dari 5 |
| `idqn` | 5 dari 5 | 2 dari 5 | 2 dari 5 |

> **Sumber, ditambahkan 2026-08-17.** Kolom **greedy (primer)** dari `results/STABILITY_v4_primary.md`; kolom **ε = 0.05** dari `results/STABILITY_v4_dqn_eps005.md`, yang sebelumnya tidak punya file di belakangnya — angkanya lahir dari diagnostik ad hoc dan baru sekarang di-generate. Keduanya cocok persis dengan tabel ini. Kolom ε = 1.0 tidak punya file yang sah karena datanya dikarantina; ia dicantumkan hanya sebagai catatan atas apa yang pernah dilaporkan. Perlu ditegaskan supaya tidak ada ambiguitas: **angka collapse rate DQN yang dipakai di seluruh klaim adalah kolom greedy**, bukan ε = 1.0 dan bukan ε = 0.05. Kontra-contoh `gnn-madqn_sage` 0 dari 5 berdiri di kedua pembacaan yang sah. Provenance seluruh laporan yang memuat angka collapse ada di `results/READOUT_PROVENANCE.md`, di-generate `scripts/readout_audit.py` yang gagal (exit non-zero) kalau ada laporan tanpa label protokol.

Angka keluarga PPO tidak tersentuh cacat ini dan tetap berlaku. Yang bertahan dari temuan lama: baseline terpusat tidak pernah kolaps (0 dari 5 di kedua keluarga). Yang **tidak** bertahan: pernyataan bahwa seluruh varian proposed kolaps — itu benar untuk keluarga PPO dan salah untuk keluarga DQN, dengan `gnn-madqn_sage` 0 dari 5. Seluruh laporan yang memuat angka lama wajib dihitung ulang, dan bagian §Keputusan scoping C4 2026-08-15 di dokumen ini dikoreksi di tempat oleh catatan ini, bukan dihapus.

**Gate B dihitung ulang di bawah pembacaan yang sah** (`scripts/gate_b_report.py`, baru — sebelumnya Gate B dihitung manual dan satu-satunya jejaknya baris ledger, sehingga tidak bisa diturunkan ulang ketika pembacaannya berubah). Skrip itu lebih dulu diverifikasi mereproduksi angka yang tercatat di bawah pembacaan lama, persis sampai dua desimal:

| pembacaan | B1 (≥ 5%) | B2 (≥ 5 pp) | B3 (≥ 2 ms) | B4 (≤ 1 dari 5) |
|---|---|---|---|---|
| stokastik lama (DQN cacat, angka yang tercatat) | 11.43% | 8.70 pp | 1.01 ms | 1 dari 5 |
| **primer per keluarga (sah)** | **16.33%** | **11.98 pp** | **1.01 ms** | **0 dari 5** |
| greedy semua algoritma (pembanding) | 200.30% | 56.05 pp | 3.59 ms | 0 dari 5 |

**Tidak ada verdict Gate B yang berbalik**: B1, B2, B4 tetap LOLOS dan B3 tetap GAGAL, jadi §Keputusan scoping 2026-08-15 berdiri utuh. Dua hal berubah dan dua-duanya wajib ikut tercetak. Pertama, B1 dan B2 justru **menguat** — daya diskriminasi task lebih besar dari yang dilaporkan, bukan lebih kecil. Kedua, **B4 turun ke 0 dari 5**: `embb_p5_mbps` tidak lagi tersaturasi, karena nilai 0.0000 identik pada 5 algoritma itu sebagian artefak aksi acak. Karena itu catatan jujur di §Keputusan scoping 2026-08-15 — bahwa klaim atas `embb_p5_mbps` hanya sah dalam bentuk diskretnya — **tidak lagi berlaku**; level `embb_p5_mbps` sekarang KPI yang tidak tersaturasi dan boleh diklaim seperti KPI lolos lainnya. Baris greedy dicantumkan hanya sebagai pembanding dan tidak dipakai apa pun: angkanya meledak justru karena kolaps argmax pada keluarga PPO, yang persis alasan P3 ada.

Tidak ada ambang gate atau definisi metrik yang diubah oleh penetapan ini. Diputuskan manusia (`Habb`), dicatat di `runs/2026-08-05-run01/ledger.md`.

### Kebijakan akuntansi anggaran 2026-08-16 (tetap, berlaku dua arah)

Anggaran `± 450 GPU-jam` di §Batasan dibaca sebagai **device-jam**: jam okupansi GPU, yaitu wall-clock mesin ini yang punya satu GPU. Bukan penjumlahan jam tiap job.

Alasannya properti pengukuran, bukan hasil: menjumlahkan job-jam menghitung ganda satu perangkat yang dipakai 6 job paralel. Wave v4 yang sama terbaca 429.47 job-jam tetapi hanya **147.22 device-jam** — selisihnya bukan pekerjaan tambahan, hanya cara menghitung. Terpakai sejauh ini **147.22 dari ± 450**, sisa ≈ 303.

Kebijakan ini **tidak boleh ditinjau ulang per-keputusan**. Ia berlaku sama ketika ia mengizinkan sesuatu yang kita inginkan dan ketika ia melarangnya; memilih pembacaan per kasus adalah bentuk lain dari memindahkan tiang gawang. Konsekuensi yang sudah diketahui saat kebijakan ini ditetapkan, dan tetap ditetapkan:

- Perluasan seed keluarga PPO (aturan biaya 2026-08-16, ≈ 43 device-jam) **muat**. Dikerjakan setelah zero-shot, karena nilai per GPU-jam zero-shot lebih tinggi.
- Baseline adaptasi k-NN keluarga PPO (§Fallback poin 2, ≈ 4.5 job-jam) **muat**.
- Kalau kelak sebuah eksperimen yang kita inginkan melewati sisa 303 device-jam, ia ditolak dengan kebijakan yang sama ini, bukan dihitung ulang dengan pembacaan lain.

Diputuskan manusia (`Habb`), dicatat di `runs/2026-08-05-run01/ledger.md`.

### Eksekusi perluasan seed keluarga PPO 2026-08-17 (dideklarasikan SEBELUM training jalan)

Aturan biayanya sudah ditetapkan 2026-08-16 atas dasar biaya keluarga, bukan hasil; section ini menetapkan **eksekusinya**, dan ditulis sebelum satu job pun dijalankan supaya tidak ada keputusan cakupan yang lahir setelah melihat angka.

**Dasar keputusan.** Keluarga yang lebih murah per seed dinaikkan. Terukur dari `elapsed_sec` 40 CSV training wave v4, itu **PPO**: 8.32 job-jam per seed untuk empat algoritmanya lawan 77.57 untuk DQN, yaitu 9.3× lebih murah. Ini fakta biaya implementasi (DQN belajar tiap langkah, PPO sekali per 512), tidak menyentuh hasil algoritma mana pun.

**Cakupan — empat algoritma pra-registrasi, tanpa kecuali:** `gnn-mappo_gat`, `gnn-mappo_sage`, `ippo`, `central-ppo`. Seed **47–61** (15 tambahan, total 20). `mlp-knn-ppo` **tidak** ikut: ia ditambahkan setelah hasil v4 dilihat dan bukan bagian himpunan yang aturan biaya 2026-08-16 sebutkan, jadi memasukkannya berarti memperluas himpunan yang dideklarasikan setelah melihat hasil. Ia tetap 5 seed dan tetap dilaporkan di tingkat 2 struktur paper, yang klaimnya soal retensi bukan collapse rate.

**Anggaran.** ≈ 124.8 job-jam ≈ **43 device-jam** dari sisa ≈ 303 (kebijakan device-jam 2026-08-16). Muat, dan dihitung dengan kebijakan yang sudah ditetapkan sebelum konsekuensinya dipakai.

**Perlakuan identik (integritas #2).** `--floor-mode none` dan `--tag _v4` diberikan eksplisit, bukan dibiarkan jatuh ke default, supaya 15 seed baru menerima perlakuan yang persis sama dengan 5 seed pertama. Titik operasi tidak disentuh dan kuncinya tidak berubah sejak `aad4198` (C6).

**Gate B dibekukan di 5 seed untuk kedelapan algoritma.** B1–B4 didefinisikan sebagai rentang mean lintas 8 algoritma; menghitungnya dengan n = 20 untuk PPO dan n = 5 untuk DQN mencampur presisi estimasi yang berbeda, sehingga rentangnya bisa bergeser karena alasan statistik alih-alih karena daya diskriminasi task berubah. Angka Gate B tetap seperti tercatat (B1 16.33%, B2 11.98 pp, B3 1.01 ms, B4 0 dari 5). Seed tambahan dipakai **hanya** untuk karakterisasi C4 keluarga PPO.

**Catatan integritas.** Memperluas keluarga PPO hanya bisa membuat kesimpulan **lebih merugikan proposed**: `central-ppo` kolaps 0 dari 5 lawan proposed 4–5 dari 5, dan n yang lebih besar mempersempit Wilson di kedua sisi. Aturan biaya ini karena itu tidak bisa dibaca sebagai memilih keluarga yang menguntungkan proposed.

**Konsekuensi yang diterima di muka, dua arah:**

- Kalau di n = 20 pemisahan bertahan → klaim **karakterisasi** keluarga PPO boleh dibuat, dalam bentuk "kolaps k dari 20 seed". Itu yang C4 minta, dan C4 jadi LOLOS untuk keluarga PPO saja.
- Kalau di n = 20 Wilson `central-ppo` mulai beririsan dengan proposed → **klaim komparatif keluarga PPO dicabut**, bukan dipertahankan dengan angka n = 5. Tidak ada opsi memakai n yang lebih menguntungkan.
- Keluarga DQN tetap 5 seed dan C4-nya tetap GAGAL. Tidak ada klaim karakterisasi di sana.

Diputuskan manusia (`Habb`), dicatat di `runs/2026-08-05-run01/ledger.md`.

### Status Gate G4 2026-08-17 — diturunkan ulang dari file, bukan dari ingatan

Setiap kriteria di §Kriteria selesai diturunkan ulang terhadap file yang di-generate hari ini, di bawah pembacaan primer per keluarga. Kolom "diverifikasi" menyebut apa yang benar-benar dijalankan ulang, bukan apa yang dulu dilaporkan.

| # | Verdict | Bukti (file yang di-generate) | Diverifikasi 2026-08-17 |
|---|---|---|---|
| A1–A4 | **LOLOS** | ledger 2026-08-08T15:00 di titik operasi beku; A1 CI `[8.58, 9.66]` ⊂ pita `[6.88, 10.12]`, A2a margin 3.01 pp, A2b 2.13 pp, A3 `inf:1`, A4 1.62 pp | tidak dijalankan ulang — Gate A properti kalibrasi pra-wave dan tidak tersentuh cacat pembacaan DQN (diukur pada `ippo`) |
| B1 | **LOLOS** 16.33% (≥ 5%) | `results/GATE_B_v4_primary.md` | dihitung ulang |
| B2 | **LOLOS** 11.98 pp (≥ 5 pp) | `results/GATE_B_v4_primary.md` | dihitung ulang |
| B3 | **GAGAL** 1.01 ms (< 2 ms) | `results/GATE_B_v4_primary.md`, mekanisme di `results/B3_DELAY_CENSORING.md` | dihitung ulang; B3 dijalankan ulang di bawah pembacaan primer |
| B4 | **LOLOS** 0 dari 5 (≤ 1) | `results/GATE_B_v4_primary.md` | dihitung ulang |
| C1 | **LOLOS** | `scripts/test_treatment_identity.py` | dijalankan ulang hari ini, **9/9** (seed 42/43/44 × floor `none`/`static`/`dynamic`, 200 langkah) |
| C2 | **PARSIAL** | `results/GATE_C.md`; remediasi di `configs/experiment_config.yaml` §`agent` + `agents/hparams.py` | `pytest tests/ -q` hijau 92/92, termasuk `tests/test_hparams_identity.py`. Verdict untuk wave sebagaimana dijalankan tetap PARSIAL — biner yang menghasilkan checkpoint ini tidak membaca YAML. Menaikkannya keputusan manusia |
| C3 | **LOLOS**, satu kasus ditandai | `results/GATE_B_v4_primary.md` §"C3 supplement" | split per keluarga sekarang **di-generate**, bukan disalin tangan; tidak ada verdict B yang berbalik di set mana pun |
| C4 | **GAGAL** | `results/STABILITY_v4_primary.md` (5 seed, bukan ≥ 20) | klaim di-scoping ulang: komparatif hanya di keluarga PPO, lihat koreksi 2026-08-17 di atas |
| C5 | **LOLOS** | `EVAL_SEED_BASE = 10_000` (`scripts/evaluate_checkpoints.py`), seed training 42–46 | dibaca ulang dari kode |
| C6 | **LOLOS** | `git diff aad4198 HEAD -- configs/experiment_config.yaml` hanya menyentuh blok `agent:` yang ditambahkan setelah wave | pemeriksaan per-kunci, bukan per-file |
| D1 | **LOLOS** | `results/RLIABLE.md` (v3) tetap ada berdampingan dengan `results/RLIABLE_v4_primary.md` | v3 tidak ditimpa |
| D2 | **LOLOS** | `results/RLIABLE_v4_primary.md` — verdict tegas turun 23 → 14 setelah koreksi pembacaan, seluruhnya mengikuti CI | — |
| D3 | **LOLOS** | `handoff/paper_structure.md` menulis kekalahan proposed dengan aturan yang sama, termasuk koreksi diri yang melemahkan narasi | — |
| D4 | **berlaku** | §Fallback poin 2, 3, 4 dikerjakan: `results/ZEROSHOT_v4_primary.md`, `results/STABILITY_v4_primary.md` (collapse + CVaR), `results/ATTENTION_v4_stoch.md` | — |

**Status: `wave tuntas dengan dua gate gagal tercatat dan klaim ter-scoping` — bukan `done`.** §Kriteria selesai menyatakan selesai hanya kalau **semua** gate lolos; B3 dan C4 GAGAL dan C2 PARSIAL. Ketiganya tercatat apa adanya dengan klaim yang sudah dipersempit, dan tidak satu pun ambang disentuh. Keputusan akhir "selesai atau tidak" milik manusia, bukan agent (`Plan_escalation_loop.md`).

Temuan G4 yang ikut diperbaiki hari ini, karena verifikasi ini menemukannya alih-alih mengasumsikannya: `results/GATE_C.md` masih mengutip angka Gate B pembacaan lama (B1 11.43%, B2 8.70 pp, B4 1 dari 5) dan collapse rate ε = 1.0, dua-duanya sudah dikoreksi di tempat dengan sumbernya ditunjuk.

---

## Batasan

- Dilarang menyentuh: `.venv\`, dan **file yang sudah ada** di `results\checkpoints\`, `results\logs\`, `results\eval\` (12 hari wall-clock GPU, tidak bisa dibuat ulang). Boleh dibaca. Dilarang menimpa/menghapus. Run baru wajib pakai tag berbeda sehingga tidak bertabrakan nama
- Interpreter: `.venv\Scripts\python.exe`. Jangan pakai `python` sistem
- Training berat: jalankan di `localhost` (mesin ini, RTX 3060 12 GB). Tidak ada VPS di jalur kerja ini — cek `nvidia-smi` dulu, jangan tabrakan dengan run yang sedang jalan
- Maks iterasi: `15`
- Setiap perubahan environment/objective **wajib** diterapkan identik ke 8 algoritma (`gnn-madqn_gat`, `gnn-madqn_sage`, `gnn-mappo_gat`, `gnn-mappo_sage`, `idqn`, `central-dqn`, `ippo`, `central-ppo`)
- Budget compute: wave penuh ± 450 GPU-jam. Jangan mulai wave sebelum Gate A lolos
- Kalibrasi wajib pakai policy **terlatih**, bukan policy statis (ini akar kegagalan v3)

---

## Larangan keras

### Operasional
- Dilarang mengedit/menghapus file test
- Dilarang mengubah definisi metrik atau ambang kriteria selesai di dokumen ini
- Dilarang `--force`, `reset --hard`, `rm -rf`, `git push`

### Integritas riset — sama mengikatnya dengan larangan operasional

1. **Dilarang menyetel parameter ke hasil.** `lambda_arrival`, `delta`, `buffer.urllc_max_bits`, `n_gnb`, `ue_radius_m` hanya boleh dipilih dengan justifikasi properti *task* (constraint mengikat, kontensi nyata, deadline jadi mekanisme dominan). **Tidak pernah** dengan justifikasi properti *hasil* (algoritma mana yang unggul).
2. **Dilarang perlakuan asimetris.** Tidak ada tambahan kapasitas, step training, kanal observasi, atau floor-mode berbeda untuk model proposed.
3. **Dilarang regime-shopping.** Kalau beberapa titik operasi diuji, laporkan seluruh grid — bukan hanya yang menguntungkan.
4. **Dilarang promosi metrik post-hoc.** Metrik primer dikunci di dokumen ini sebelum wave. `embb_p5_mbps` sudah dipromosikan jadi kelas satu **berdasarkan alasan a priori** (proksi cell-edge yang justru dilindungi mekanisme koordinasi), bukan karena hasil v3.
5. **Dilarang membuang seed yang kolaps** atau mengganti seed sampai stabil. Collapse adalah data.
6. **Dilarang mengklaim gain latensi URLLC** selama drop masih didominasi overflow (lihat Gate A3).
7. **Dilarang menaikkan derajat verdict.** `COMPARABLE` tetap `COMPARABLE`.

---

## Metrik primer (dikunci sebelum wave)

1. `timely_throughput_mbps`
2. `sla_satisfaction_pct`
3. `urllc_delay_p99`
4. `embb_p5_mbps` — cell-edge, kelas satu
5. `jains_fairness`
6. `cell_edge_collapse_rate` — **baru**, outcome diskret (Bernoulli per seed), CI Wilson

Statistik: rliable IQM + stratified bootstrap 95% CI + probability of improvement, **per keluarga budget**. Collapse rate pakai CI binomial, bukan mean kontinu.

---

## Fallback (kalau v4 tetap COMPARABLE)

Bukan kegagalan. Paper bergeser ke kontribusi metodologi + temuan negatif:

1. Benchmark slicing Gymnasium/PettingZoo terkalibrasi + **protokol kalibrasi terhadap policy terlatih** (rilis publik)
2. **Zero-shot topology generalization** (latih 5 gNB → uji 20/50) — baseline MLP/central secara struktural tidak bisa ikut; ini klaim terkuat yang jujur
3. **Collapse rate / CVaR** sebagai metrik robustness kelas satu
4. **Bukti mekanisme**: korelasi bobot atensi GAT vs matriks interferensi sebenarnya, divalidasi lewat ablasi kausal (bukan korelasi saja)
5. Batas fase beban di mana koordinasi mulai terpisah — hasil ilmiah positif meski 5 gNB ada di bawah batas itu

Target venue fallback: ICBINB, reproducibility/benchmark track, atau TNSM sebagai empirical study.

---

## Referensi internal

| Dokumen | Isi |
|---|---|
| `docs/journey/07_hasil-evaluasi-v3.md` | Hasil v3 lengkap + akar masalah saturasi |
| `research/2026-08-05-run01-gnn-marl-fair-chance.md` | Riset redesign: rezim beban, zero-shot, stabilitas, batas etika |
| `configs/experiment_config.yaml` | Sumber tunggal seluruh hyperparameter bersama |
| `scripts/test_treatment_identity.py` | Uji Gate C1 |

---

**Dibekukan pada:** `2026-08-05T22:26:37`
**Run:** `2026-08-05-run01`
**Ditandatangani manusia:** `Habb` — perubahan pada dokumen ini setelah wave dimulai wajib dicatat di ledger dengan alasan eksplisit.