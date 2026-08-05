Skill: caveman (ultra), ponytail (full), safety-guard, terminal-ops.

Baca handoff/goal1.md dan STATE.json.
Kerjakan iterasi berikutnya menuju kriteria selesai.

Interpreter WAJIB: .venv\Scripts\python.exe — `python` sistem tidak punya torch/gymnasium/PyG.
Sebelum menjalankan training berat: cek `nvidia-smi`. Ada run lain jalan = jangan mulai.

ATURAN:
- Solusi paling malas yang jalan. YAGNI. Jangan tambah dependency tanpa alasan tertulis.
- Jangan analisis panjang. Eksekusi, ukur, laporkan.
- Setiap perubahan berhasil: git commit dengan `-c user.name=Habb -c user.email=ihabhasanainakmal0409@gmail.com`.
  Stage file spesifik. Dilarang `git add -A`.
- Gagal 1-2x: perbaiki sendiri.
- Gagal 3x: baca docs lokal, `--help`, `pip show`.
- Gagal 4x: WebSearch maks 3 query.
- Gagal 5x: BERHENTI. Tulis handoff/STUCK.md format lengkap (lihat Plan_escalation_loop.md FASE 4). Jangan lanjut.
- Error menyebut API/signature/versi berubah: langsung tulis STUCK.md, tandai level 4.
- OOM / kuota / hardware: langsung tulis handoff/BLOCKED.md, berhenti.

INTEGRITAS RISET (dari handoff/goal1.md — sama mengikatnya dengan larangan operasional):
- Dilarang menyetel parameter ke hasil. Justifikasi hanya boleh properti task, tidak pernah properti hasil.
- Dilarang perlakuan asimetris antar 8 algoritma.
- Angka hasil selalu disalin dari file report yang di-generate, tidak pernah diketik dari ingatan.
- Dilarang membuang seed yang kolaps. Collapse adalah data.
- COMPARABLE tetap COMPARABLE.

DILARANG: edit/hapus file test, ubah definisi metrik atau ambang di handoff/goal1.md,
menulis/menghapus di .venv\, results\checkpoints\, results\logs\, results\eval\,
`git push`, `git reset --hard`, `rm -rf`, `--force`.

Catat tiap aksi ke runs/2026-08-05-run01/ledger.md dengan format:
## <ISO8601> | lvl <n> | <AKSI> | <HASIL> | <sha atau ->
Perbarui STATE.json setiap akhir iterasi.
