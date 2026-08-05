Skill: research-ops, literature-review, parallel-execution-optimizer.

Baca handoff/STUCK.md, handoff/BRIEF.md, handoff/goal1.md.

Pecah jadi 3-5 sub-pertanyaan. Spawn subagent paralel via Task.
BATAS KERAS: maksimal 5 subagent. Maksimal 8 web_fetch per subagent.
Lewat batas = berhenti dan laporkan.

Sumber: WebSearch, WebFetch, arXiv API, Semantic Scholar API (on-demand saja).
Baca sumber PENUH. Konflik antar sumber ditulis eksplisit, jangan dirata-rata.
Bagian "Yang TIDAK ketemu" wajib diisi jujur.

Tulis handoff/RESEARCH.md format wajib (Plan_escalation_loop.md FASE 4).
Arsipkan salinan ke research/2026-08-05-run01-<slug>.md.

Kalau hasil riset TIDAK memberi jalan yang jelas:
tulis handoff/BLOCKED.md lengkap dengan PROMPT SIAP TEMPEL, lalu BERHENTI.

DILARANG KERAS: menyentuh file kode apa pun (envs/, training/, gnn/, scripts/, configs/, tests/).
