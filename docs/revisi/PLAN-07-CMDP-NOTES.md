# Catatan Referensi — Formulasi CMDP & Credit Assignment

**Fase:** bukan fase — dokumen referensi
**Dipakai oleh:** PLAN-02 (Fase 1) dan PLAN-06 (Fase 4)
**Master:** PLAN-00-MASTER.md

---

## 1. Peringatan sitasi

**Circular citation:** rujukan `[cite: 37]` untuk kalimat "Ini menjelaskan mengapa setup terdistribusi Anda mengalami kolaps pada persentil-5" merujuk dokumen internal yang di-upload ke NotebookLM. Jangan kutip. Mekanismenya masuk akal, tapi sumber eksternalnya harus dicari sendiri.

**Angka dual learning rate tidak konsisten:** NotebookLM menyebut `α_λ` = 2.0 (APS-GNN) dan 10⁻³–10⁻⁴ (JCPGNN-M) — selisih 3–4 orde magnitudo tanpa penjelasan. Kemungkinan salah baca atau mencampur skema update berbeda. **Jangan pakai mentah.**

**Belum terverifikasi:** JCPGNN-M, APS-GNN, EExApp, IC-GMRO, TELGEN, P-DGN.

---

## 2. TEMUAN UTAMA — penjelasan mekanistik cell-edge collapse

Ini yang dipakai PLAN-02 sebagai justifikasi kedua.

### Mekanisme

Constraint saat ini bersifat **network-wide**: δ = 8.5% violation rate agregat, satu λ global.

Konsekuensi struktural:

> Mematikan throughput satu UE lemah di cell-edge demi memberikan seluruh spektrum ke UE dekat gNB hanya memberi **penalti kecil pada rata-rata global**, tapi menaikkan throughput agregat secara drastis.

Dengan 5 gNB: kalau 1 gNB dikorbankan penuh, kontribusinya ke violation agregat cuma 20% dari total — bisa diserap di bawah δ selama 4 gNB lain sehat.

### Implikasi

**Ini bukan kegagalan optimisasi.** Policy menemukan solusi yang benar untuk objective yang ditulis. Objective-nya yang mengizinkan collapse.

Kamu punya penjelasan mekanistik untuk collapse rate — bukan sekadar observasi empiris. Dan penjelasan itu memprediksi arah perbaikannya: pindah dari constraint agregat ke constraint per-entitas.

Itu persis PLAN-02. **Dua jalur analisis independen menunjuk solusi yang sama** — layak ditulis di paper.

---

## 3. Yang sudah benar di implementasi — JANGAN DIUBAH (konflik K5)

| Praktik | Status |
|---|---|
| Projected SGA, `λ ← max(0, λ + α·g)` | Sudah |
| Dual learning rate << primal | Sudah (`α_λ << α_θ`) |
| Dual clipping / bound | Sudah (`λ_max = 100`) |
| Update λ per-batch, bukan per-step | Sudah (`dual_update_every`) |
| Violation dilaporkan pada held-out | Sudah (seed ≥ 10000) |
| Dua protokol pembacaan dilaporkan | Sudah (P3) |

**Jangan mengubah `α_λ`, `λ_init`, `λ_max`, `dual_update_every`.** Sudah lolos Gate A, dan angka literatur pembandingnya tidak konsisten.

### Saran yang ditolak

| Saran | Alasan penolakan |
|---|---|
| λ diinisialisasi 0.0 (bukan 1.0) | Kalibrasi sudah dilakukan dengan 1.0 dan Gate A lolos. Mengubahnya berarti mengulang seluruh kalibrasi tanpa alasan kuat |
| Conservative violation margin (δ_target 8.0% vs δ 8.5%) | Mengubah titik operasi → Gate A harus diulang. Sistem sudah lolos dan constraint sudah mengikat. Jangan utak-atik yang sudah bekerja |
| Action feasibility masking | Sudah punya versinya — floor dinamis lewat action projection |

---

## 4. Framing readout — dua lapis

Dipakai di PLAN-06 §4.

Literatur menyatakan:
> Deterministic evaluation (argmax) **wajib dilaporkan** untuk mensimulasikan uji operasional nyata — operator seluler tidak menginginkan keputusan alokasi bandwidth yang acak. Jika deterministik memicu greedy collapse, itu **harus dilaporkan transparan sebagai batas validitas model**.

Ini mendukung keputusan mempertahankan laporan greedy (P3), **dan memberi framing yang lebih kuat**: bukan artefak pembacaan, melainkan batas validitas operasional.

Detail dan tabel data ada di PLAN-06 §4.

---

## 5. Detail teknis per-UE constraint (melengkapi PLAN-02)

### Ledakan variabel dual
Dengan N UE, sistem memelihara N Lagrange multiplier paralel. Literatur menyebut GNN-MARL mengatasinya dengan memetakan variabel dual sebagai **atribut node pada graf**.

### Klarifikasi konflik K1

Ini yang mengoreksi PLAN-02 §5:

| Peran μ | Status |
|---|---|
| **Input** (node/edge attribute) | Aman dan disarankan — dimensi node feature tetap, jumlah node yang berubah |
| **Parameter** (tersimpan di `state_dict`) | Merusak transferabilitas |

Yang dilarang hanya yang kedua. Yang pertama justru desain bagus — policy bisa "melihat" tekanan constraint per-UE dan meresponsnya.

### Nilai awal
Multiplier per-UE diinisialisasi 0 atau kecil (0.1), agar agen fokus throughput di iterasi awal. Konsisten dengan PLAN-02 §4.

---

## 6. Klaim yang harus ditolak

### "GNN-CMDP mempertahankan feasibility zero-shot pada topologi 5×–20× lebih besar"

Sudah diuji di v4. Hasilnya null: retensi GNN 0.942–0.955, MLP per-agen 0.949–0.954, rentang total 1.3pp.

**Tindakan:** di wave v5 (PLAN-02), uji ulang klaim ini secara eksplisit sebagai hipotesis yang dipra-registrasi — apakah feasibility rate bertahan zero-shot, dan apakah GNN bertahan lebih baik dari MLP per-agen.

Tapi **jangan kutip klaimnya sebagai temuan literatur yang mendukung** — datamu sendiri belum menunjukkannya.

### "APS-GNN mengaktifkan 50–70% AP lebih sedikit"

Belum terverifikasi, domain berbeda (AP selection di cell-free massive MIMO, bukan slicing PRB). Jangan kutip.

---

## 7. Larangan

1. Jangan kutip `[cite: 37]` atau dokumen internal apa pun sebagai sumber literatur
2. Jangan pakai angka `α_λ` dari blok ini
3. Jangan ubah parameter dual yang sudah lolos Gate A (konflik K5)
4. Jangan kutip JCPGNN-M, APS-GNN, EExApp, IC-GMRO, TELGEN sebelum diverifikasi
5. Jangan kutip klaim "GNN-CMDP zero-shot 5×–20×" sebagai pendukung
6. Framing "batas validitas operasional" tidak boleh menyembunyikan bahwa proposed kalah di dimensi itu (D3)
