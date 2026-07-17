"""Analyze training CSV logs and generate results/RESULTS.md + results/results_summary.csv.

Unlike make_paper_figures.py (which only plots), this script *normalizes* reward
across algorithm families (DQN logs per-episode, PPO logs per-rollout -> different
step counts per row) before computing any comparison, and writes a narrative report
with diagnosis of anomalies found in the data.

  python scripts/analyze_results.py [--log-dir results/logs] [--out-dir results]
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from training.metrics_logger import CSV_COLUMNS  # reuse canonical schema, don't redefine

FINAL_WINDOW_FRAC = 0.10
MILESTONES = (0.10, 0.25, 0.50, 0.75, 1.00)
BUDGET_BY_FAMILY = {"dqn": 200_000, "ppo": 1_000_000}


def load_config(root: Path) -> dict:
    with open(root / "configs" / "experiment_config.yaml") as f:
        return yaml.safe_load(f)


def classify(stem: str):
    """(algo, proposed/baseline, backbone, dqn/ppo) - dqn/ppo decided from data, not name."""
    proposed = stem.startswith("gnn-")
    backbone = ""
    for b in ("gat", "sage", "gcn"):
        if f"_{b}_" in stem or stem.endswith(f"_{b}"):
            backbone = b
    algo = stem.split("_seed")[0]
    return algo, ("proposed" if proposed else "baseline"), backbone


def load_run(path: Path) -> dict | None:
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames or []
        if header != CSV_COLUMNS:
            missing = set(CSV_COLUMNS) - set(header)
            print(f"  [warn] {path.name}: header does not match training/metrics_logger.CSV_COLUMNS "
                  f"(missing={missing or 'none'}, order may differ)")
        rows = []
        for r in reader:
            try:
                r["step"] = int(float(r["step"]))
                r["ep_reward"] = float(r["ep_reward"])
            except (ValueError, KeyError, TypeError):
                continue
            rows.append(r)
    if len(rows) < 2:
        return None
    rows.sort(key=lambda r: r["step"])
    return {"rows": rows}


def _f(row: dict, key: str) -> float | None:
    v = row.get(key, "")
    if v in ("", None):
        return None
    try:
        return float(v)
    except ValueError:
        return None


def compute_metrics(stem: str, run: dict, cfg: dict) -> dict:
    rows = run["rows"]
    n = len(rows)
    steps = np.array([r["step"] for r in rows], dtype=np.float64)
    diffs = np.diff(steps)
    steps_per_row = float(np.median(diffs)) if len(diffs) else 1.0
    steps_per_row = max(steps_per_row, 1.0)

    ep_reward = np.array([r["ep_reward"] for r in rows], dtype=np.float64)
    reward_per_step = ep_reward / steps_per_row

    ma100_raw = np.array([_f(r, "ep_reward_ma100") for r in rows], dtype=np.float64)
    normalized_ma = ma100_raw / steps_per_row  # linear op: MA(x)/c == MA(x/c) for constant c

    algo, family, backbone = classify(stem)
    dqn_family = any(_f(r, "epsilon") is not None for r in rows)
    algo_family = "dqn" if dqn_family else "ppo"

    fw_n = max(1, int(round(n * FINAL_WINDOW_FRAC)))
    final_rows = rows[-fw_n:]
    final_reward_per_step = reward_per_step[-fw_n:]

    peak_idx = int(np.nanargmax(normalized_ma))
    peak_ma = float(normalized_ma[peak_idx])
    peak_step = int(steps[peak_idx])
    final_ma = float(np.nanmean(normalized_ma[-fw_n:]))
    degradation = peak_ma - final_ma

    # trend over the last 25% of rows
    tail_n = max(2, int(round(n * 0.25)))
    tail_steps = steps[-tail_n:]
    tail_ma = normalized_ma[-tail_n:]
    slope = float(np.polyfit(tail_steps, tail_ma, 1)[0]) if len(set(tail_steps)) > 1 else 0.0
    total_change = slope * (tail_steps[-1] - tail_steps[0])
    scale = max(abs(np.nanmean(tail_ma)), 1e-6)
    if total_change > 0.05 * scale:
        trend = "improving"
    elif total_change < -0.05 * scale:
        trend = "degrading"
    else:
        trend = "plateau"

    # convergence: first step where normalized_ma enters a 5% band around the final-window mean
    target = final_ma
    band = 0.05 * abs(target) if abs(target) > 1e-6 else 0.05
    conv_step = None
    for s, v in zip(steps, normalized_ma):
        if not np.isnan(v) and abs(v - target) <= band:
            conv_step = int(s)
            break

    def _mean_final(key):
        vals = [_f(r, key) for r in final_rows]
        vals = [v for v in vals if v is not None]
        return float(np.mean(vals)) if vals else None

    embb = _mean_final("embb_mbps_mean")
    spec = _mean_final("spectral_eff_bps_hz")
    embb_p5 = _mean_final("embb_p5_mbps")
    delay_mean = _mean_final("urllc_delay_ms_mean")   # mean delay of DELIVERED packets only (<= deadline)
    delay_p95 = _mean_final("urllc_delay_p95")
    delay_p99 = _mean_final("urllc_delay_p99")
    drop_late = _mean_final("urllc_drop_late_pct")
    drop_overflow = _mean_final("urllc_drop_overflow_pct")
    sla = _mean_final("sla_satisfaction_pct")
    timely_thr = _mean_final("timely_throughput_mbps")
    fairness = _mean_final("jains_fairness")
    lam_final = _mean_final("lam")

    urllc_delta = float(cfg.get("cmdp", {}).get("delta", 0.02))
    flag_urllc_starving = sla is not None and sla < 100.0 * (1.0 - urllc_delta) - 5.0  # >5pp below CMDP target
    flag_low_sla = sla is not None and sla < 90
    flag_degraded = degradation > max(0.05 * abs(peak_ma), 0.01) and peak_ma > final_ma

    return {
        "algo": algo, "family": family, "backbone": backbone, "algo_family": algo_family,
        "stem": stem, "n_rows": n, "final_step": int(steps[-1]),
        "steps_per_row": steps_per_row, "budget_target": BUDGET_BY_FAMILY[algo_family],
        "reward_per_step_final_mean": float(np.mean(final_reward_per_step)),
        "reward_per_step_final_std": float(np.std(final_reward_per_step)),
        "reward_per_episode_equiv": float(np.mean(final_reward_per_step)) * 200,
        "peak_ma_reward_per_step": peak_ma, "peak_step": peak_step,
        "final_ma_reward_per_step": final_ma, "degradation": degradation,
        "trend_label": trend, "slope_last25pct": slope,
        "convergence_step": conv_step,
        "embb_mbps_mean": embb, "spectral_eff_bps_hz": spec, "embb_p5_mbps": embb_p5,
        "urllc_delay_ms_mean": delay_mean, "urllc_delay_p95": delay_p95, "urllc_delay_p99": delay_p99,
        "urllc_drop_late_pct": drop_late, "urllc_drop_overflow_pct": drop_overflow,
        "sla_satisfaction_pct": sla, "timely_throughput_mbps": timely_thr,
        "jains_fairness": fairness, "lam_final": lam_final,
        "flag_urllc_starving": flag_urllc_starving, "flag_low_sla": flag_low_sla,
        "flag_degraded": flag_degraded,
        "milestones": _milestones(rows, steps, normalized_ma),
    }


def _milestones(rows, steps, normalized_ma):
    n = len(rows)
    out = []
    for f in MILESTONES:
        i = min(int(f * n), n - 1)
        out.append({"frac": f, "step": int(steps[i]), "reward_per_step_ma": float(normalized_ma[i])})
    return out


def write_summary_csv(metrics: list[dict], path: Path):
    fields = ["algo", "family", "backbone", "algo_family", "n_rows", "final_step",
              "steps_per_row", "budget_target",
              "reward_per_step_final_mean", "reward_per_step_final_std", "reward_per_episode_equiv",
              "peak_ma_reward_per_step", "peak_step", "final_ma_reward_per_step", "degradation",
              "trend_label", "slope_last25pct", "convergence_step",
              "embb_mbps_mean", "spectral_eff_bps_hz", "embb_p5_mbps",
              "urllc_delay_ms_mean", "urllc_delay_p95", "urllc_delay_p99",
              "urllc_drop_late_pct", "urllc_drop_overflow_pct",
              "sla_satisfaction_pct", "timely_throughput_mbps", "jains_fairness", "lam_final",
              "flag_urllc_starving", "flag_low_sla", "flag_degraded"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for m in sorted(metrics, key=lambda x: x["stem"]):
            w.writerow(m)
    print("summary csv ->", path.name)


def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    if isinstance(v, bool):
        return "YA" if v else "tidak"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(metrics: list[dict], cfg: dict, root: Path) -> str:
    by_stem = {m["stem"]: m for m in metrics}
    dqn_group = [m for m in metrics if m["algo_family"] == "dqn"]
    ppo_group = [m for m in metrics if m["algo_family"] == "ppo"]

    # Primary ranking is on operator KPIs, not reward/step: under the CMDP formulation
    # the reward's scale drifts with the learned Lagrangian lambda (env._lam), so
    # reward/step is NOT comparable across algorithms/runs whose lambda trajectories
    # differ — it's kept only as a per-run convergence diagnostic (§5).
    def _thr(m):
        return m["timely_throughput_mbps"] if m["timely_throughput_mbps"] is not None else -1.0

    best_throughput_at_sla = max(metrics, key=_thr)
    best_sla = max((m for m in metrics if m["sla_satisfaction_pct"] is not None),
                   key=lambda m: m["sla_satisfaction_pct"])
    best_fairness = max((m for m in metrics if m["jains_fairness"] is not None),
                        key=lambda m: m["jains_fairness"])
    best_p5 = max((m for m in metrics if m["embb_p5_mbps"] is not None),
                  key=lambda m: m["embb_p5_mbps"])
    best_dqn = max(dqn_group, key=_thr) if dqn_group else None
    best_ppo = max(ppo_group, key=_thr) if ppo_group else None

    proposed = [m for m in metrics if m["family"] == "proposed"]
    baseline = [m for m in metrics if m["family"] == "baseline"]
    proposed_worse = all(
        _thr(m) < max(_thr(b) for b in baseline if b["algo_family"] == m["algo_family"])
        for m in proposed
    ) if proposed and baseline else False

    lines = []
    A = lines.append

    A("# Hasil Training GNN-MARL Network Slicing (env v3: CMDP + packet-level URLLC)\n")
    A(f"*Generated otomatis oleh `scripts/analyze_results.py` dari {len(metrics)} run di `results/logs/`. "
      f"Untuk perbandingan lintas-seed yang benar (IQM + bootstrap CI, held-out eval episodes) lihat "
      f"`results/RLIABLE.md`, dihasilkan dari `scripts/evaluate_checkpoints.py` + `scripts/rliable_report.py` "
      f"— dokumen ini memakai training-time CSV logs (mixed exploration/lambda-drift), cocok untuk "
      f"diagnosis konvergensi tapi bukan untuk klaim signifikansi.*\n")

    # 1. Ringkasan Eksekutif
    A("## 1. Ringkasan Eksekutif\n")
    A(f"- **Timely throughput tertinggi keseluruhan:** `{best_throughput_at_sla['stem']}` "
      f"({_fmt(best_throughput_at_sla['timely_throughput_mbps'],3)} Mbps, SLA "
      f"{_fmt(best_throughput_at_sla['sla_satisfaction_pct'],1)}%)")
    A(f"- **SLA satisfaction terbaik:** `{best_sla['stem']}` ({_fmt(best_sla['sla_satisfaction_pct'],1)}%)")
    A(f"- **Cell-edge (P5) eMBB throughput terbaik:** `{best_p5['stem']}` ({_fmt(best_p5['embb_p5_mbps'],3)} Mbps)")
    if best_dqn:
        A(f"- **Terbaik di keluarga DQN (budget setara, {BUDGET_BY_FAMILY['dqn']:,} step):** `{best_dqn['stem']}`")
    if best_ppo:
        A(f"- **Terbaik di keluarga PPO (budget setara, {BUDGET_BY_FAMILY['ppo']:,} step):** `{best_ppo['stem']}`")
    A("")
    if proposed_worse:
        A("**Temuan utama:** pada timely throughput (KPI operator utama di bawah CMDP), "
          "**algoritma proposed belum mengungguli baseline sepadan** di eksperimen ini. Lihat §8 "
          "untuk diagnosis dan §10 untuk rekomendasi sebelum data ini dipakai di paper.\n")

    # 2. Setup
    A("## 2. Setup Eksperimen\n")
    e = cfg["env"]
    A(f"- Jumlah gNB: **{e['n_gnb']}**, bandwidth: **{e['bandwidth_hz']/1e6:.0f} MHz**, "
      f"skenario: {e['scenario']}, slot: **{e.get('slot_duration_ms', 1.0)} ms**")
    A(f"- Episode length: **{cfg['training']['episode_length']} slot** "
      f"({cfg['training']['episode_length'] * e.get('slot_duration_ms', 1.0):.0f} ms per episode)")
    A(f"- URLLC max delay (SLA deadline): **{cfg['slices']['urllc']['max_delay_ms']} ms** — "
      f"paket per-gNB di FIFO, di-drop kalau nunggu > deadline (deadline-drop) atau backlog > "
      f"`buffer.urllc_max_bits` (overflow-drop). Delay yang dilaporkan (`urllc_delay_ms_mean/p95/p99`) "
      f"HANYA dari paket yang **terkirim** — jadi selalu <= deadline by construction; pelanggaran SLA "
      f"muncul sebagai `sla_violation_pct` (drop), bukan delay yang meledak.")
    A(f"- eMBB min throughput target (t_ref reward): **{cfg['slices']['embb']['min_throughput_mbps']} Mbps**")
    r = cfg["reward"]
    cmdp = cfg.get("cmdp", {})
    floor = cfg.get("floor", {})
    if cmdp.get("enabled", True):
        A(f"- **Objective/constraint (CMDP):** reward = w1={r['w1']} (eMBB throughput terkirim) + "
          f"w4={r['w4']} (spectral efficiency) `- lambda * violation_rate`. `lambda` adalah dual variable "
          f"yang **dipelajari** (bukan bobot tetap), diupdate tiap {cmdp.get('dual_update_every')} step "
          f"menuju target `delta={cmdp.get('delta')}` (lr={cmdp.get('lambda_lr')}), **persist lintas "
          f"episode** dalam satu run. Reward per-step di-clip ke `[-10, +10]`. **PENTING:** karena skala "
          f"reward bergeser seiring lambda, `reward/step` TIDAK sebanding antar run dengan trajektori "
          f"lambda berbeda — lihat §5 (dipakai sebagai diagnostik konvergensi saja, bukan §1/§4/§6).")
    else:
        A(f"- **Reward (scalarized, cmdp.enabled=false):** w1={r['w1']} (eMBB) - w2={r.get('w2')} "
          f"(violation) + w4={r['w4']} (spectral efficiency), clip `[-10,+10]`.")
    A(f"- **PRB floor URLLC:** mode=`{floor.get('mode')}` "
      f"(base={floor.get('base')}, k={floor.get('k')}, range=[{floor.get('lo')},{floor.get('hi')}]) — "
      f"diterapkan via action projection di `env.step()`, identik untuk 8 algoritma.")
    coupled = bool(e.get("slice_coupled_interference", False))
    A(f"- **Coupled interference (reuse-1):** {'AKTIF' if coupled else 'nonaktif'} — interferensi "
      f"antar-gNB terikat pada fraksi alokasi slice tetangga.")
    A(f"- **UE-association bug-fix (v2->v3):** `ue_radius_m={e.get('ue_radius_m')}` — UE sekarang "
      f"di-drop dalam radius ini dari gNB pemiliknya sendiri. Sebelumnya (v1/v2) posisi UE dan gNB "
      f"ditarik independen dari `uniform(0, area_size)` yang sama, jadi UE index milik gNB i tidak "
      f"berkorelasi jarak dengan gNB i sama sekali — bukan skenario 3GPP UMa yang valid (UE seharusnya "
      f"di-drop di sel yang melayaninya). Diagnostik (`scripts/diag_topology_sweep.py`) menunjukkan "
      f"~72% gNB dulunya interference-limited (SIR negatif) akibat bug ini, tidak tergantung `area_size` "
      f"(scaling area seragam menggeser path-loss own-link dan interference-link dengan konstanta yang "
      f"sama, saling meniadakan di rasio SIR) dan cuma lemah tergantung `n_gnb`. Setelah fix: ~92-98% "
      f"gNB punya SIR positif. **Konsekuensi metodologis: seluruh hasil v1 (`results/v1_uncoupled/`) "
      f"dan v2 (`results/v2_scalarized/`) TIDAK representatif dan tidak lagi jadi baseline yang valid** "
      f"— diarsip untuk referensi historis/ablasi saja. Hanya hasil v3 (dokumen ini) yang sah dilaporkan "
      f"di paper.")
    A(f"- Budget training: DQN family = {BUDGET_BY_FAMILY['dqn']:,} step, "
      f"PPO family = {BUDGET_BY_FAMILY['ppo']:,} step")
    A(f"- Run dianalisis: {', '.join(sorted(by_stem))}\n")

    # 3. Catatan metodologis
    A("## 3. ⚠️ Catatan Metodologis (baca sebelum menafsirkan angka)\n")
    A("1. **`reward/step` bukan KPI perbandingan yang sah di bawah CMDP** (§2) — dipertahankan di "
      "§5 hanya untuk melihat apakah training konvergen, bukan untuk ranking algoritma. Ranking "
      "utama (§1, §4, §6) memakai KPI operator: `timely_throughput_mbps`, `sla_satisfaction_pct`, "
      "`embb_p5_mbps` (cell-edge), `jains_fairness`.")
    A("2. **Budget training timpang** (DQN 200K vs PPO 1M step). Perbandingan yang metodologis "
      "sah adalah **di dalam keluarga yang sama** (§6), bukan lintas keluarga.")
    A("3. **Angka di dokumen ini berasal dari training-time CSV** (rata-rata jendela 10% baris "
      "terakhir dari `MetricsLogger`), bercampur eksplorasi (epsilon/policy stokastik) dan drift "
      "lambda. Untuk klaim di paper, pakai `results/RLIABLE.md` (held-out greedy eval, multi-seed, "
      "bootstrap CI).")
    A("4. **Kalau hanya 1 seed dianalisis**, selisih kecil antar algoritma **belum bisa diklaim "
      "signifikan secara statistik** — lihat §10 dan `results/RLIABLE.md` untuk hasil multi-seed.\n")

    # 4. Tabel hasil utama
    A("## 4. Tabel Hasil Utama (KPI Operator)\n")
    A("| algo | family | final_step | timely thr. Mbps | SLA sat. % | eMBB Mbps (P5) | "
      "delay ms (mean/p95/p99, delivered-only) | Jain's fairness | lambda | anomali |")
    A("|---|---|---|---|---|---|---|---|---|---|")
    for m in sorted(metrics, key=lambda x: -_thr(x)):
        anomalies = []
        if m["flag_urllc_starving"]:
            anomalies.append("SLA di bawah target CMDP")
        if m["flag_low_sla"]:
            anomalies.append("SLA rendah")
        if m["flag_degraded"]:
            anomalies.append("degradasi dari peak (reward)")
        anom_str = ", ".join(anomalies) if anomalies else "-"
        A(f"| `{m['stem']}` | {m['family']} | {m['final_step']:,} | "
          f"{_fmt(m['timely_throughput_mbps'],3)} | {_fmt(m['sla_satisfaction_pct'],1)} | "
          f"{_fmt(m['embb_mbps_mean'],2)} ({_fmt(m['embb_p5_mbps'],2)}) | "
          f"{_fmt(m['urllc_delay_ms_mean'],2)} / {_fmt(m['urllc_delay_p95'],2)} / "
          f"{_fmt(m['urllc_delay_p99'],2)} | {_fmt(m['jains_fairness'],3)} | "
          f"{_fmt(m['lam_final'],2)} | {anom_str} |")
    A("")

    # 5. Tren per algoritma (reward/step, diagnostik konvergensi saja)
    A("## 5. Analisis Tren per Algoritma (diagnostik konvergensi — reward/step, bukan KPI perbandingan)\n")
    A("Tren dihitung dari kemiringan (slope) `reward_per_step` (MA-100 ternormalisasi) pada 25% "
      "baris terakhir training. **Reward di sini tidak sebanding antar algoritma** (§2, §3) — pakai "
      "kolom ini hanya untuk cek training sudah konvergen/stabil sebelum ambil metrik final-window.\n")
    A("| algo | trend | peak reward/step (step) | final reward/step | convergence step |")
    A("|---|---|---|---|---|")
    for m in sorted(metrics, key=lambda x: x["stem"]):
        conv = f"{m['convergence_step']:,}" if m["convergence_step"] is not None else "belum konvergen"
        A(f"| `{m['stem']}` | {m['trend_label']} | "
          f"{_fmt(m['peak_ma_reward_per_step'])} (step {m['peak_step']:,}) | "
          f"{_fmt(m['final_ma_reward_per_step'])} | {conv} |")
    A("")

    # 6. Proposed vs Baseline per keluarga
    A("## 6. Proposed vs Baseline per Keluarga (budget setara, KPI operator)\n")
    for fam_name, group, budget in (("DQN", dqn_group, BUDGET_BY_FAMILY["dqn"]),
                                     ("PPO", ppo_group, BUDGET_BY_FAMILY["ppo"])):
        A(f"### Keluarga {fam_name} (@ {budget:,} step)\n")
        A("| algo | family | timely thr. Mbps | SLA % | eMBB P5 Mbps |")
        A("|---|---|---|---|---|")
        for m in sorted(group, key=lambda x: -_thr(x)):
            A(f"| `{m['stem']}` | {m['family']} | {_fmt(m['timely_throughput_mbps'],3)} | "
              f"{_fmt(m['sla_satisfaction_pct'],1)} | {_fmt(m['embb_p5_mbps'],3)} |")
        winner = max(group, key=_thr) if group else None
        if winner:
            verdict = "PROPOSED MENANG" if winner["family"] == "proposed" else "BASELINE MENANG"
            A(f"\n**Pemenang keluarga {fam_name} (timely throughput): `{winner['stem']}` ({verdict})** "
              f"— verifikasi klaim ini di `results/RLIABLE.md` (CI overlap = 'comparable', bukan 'menang').\n")

    # 7. KPI Jaringan
    A("## 7. Analisis KPI Jaringan\n")
    A(f"- Timely throughput tertinggi: `{best_throughput_at_sla['stem']}` "
      f"({_fmt(best_throughput_at_sla['timely_throughput_mbps'],3)} Mbps)")
    A(f"- Fairness (Jain's index) tertinggi: `{best_fairness['stem']}` "
      f"({_fmt(best_fairness['jains_fairness'],3)})")
    A(f"- Cell-edge (P5) eMBB tertinggi: `{best_p5['stem']}` ({_fmt(best_p5['embb_p5_mbps'],3)} Mbps)")
    A("- **Delay yang dilaporkan (§4) hanya untuk paket yang terkirim** (deadline-drop menjamin "
      "<= deadline). Kualitas URLLC nyata dibaca dari `sla_satisfaction_pct` (drop rate), bukan delay.\n")

    # 8. Diagnosis
    A("## 8. Temuan & Anomali (Diagnosis)\n")
    starving = [m for m in metrics if m["flag_urllc_starving"]]
    if starving:
        bits = []
        for m in sorted(starving, key=lambda x: (x["sla_satisfaction_pct"] or 100)):
            bits.append("`{}` ({}) SLA {}%, drop late {}% / overflow {}%".format(
                m["stem"], m["family"], _fmt(m["sla_satisfaction_pct"], 1),
                _fmt(m["urllc_drop_late_pct"], 2), _fmt(m["urllc_drop_overflow_pct"], 2)))
        A(f"**SLA di bawah target CMDP** ({', '.join(bits)}) — di bawah packet-level + CMDP + floor "
          f"(v3), pelanggaran SLA muncul sebagai drop rate terukur langsung, bukan delay yang meledak "
          f"tak terbatas (masalah v1/v2). Kalau lambda (§4 kolom terakhir) belum naik cukup tinggi "
          f"untuk menekan violation ke `delta`, itu tanda `lambda_lr`/`dual_update_every` perlu "
          f"ditinjau, atau budget training terlalu pendek untuk dual variable konvergen.\n")
    drop_dominant_overflow = [m["stem"] for m in metrics
                              if m["urllc_drop_overflow_pct"] and m["urllc_drop_late_pct"]
                              and m["urllc_drop_overflow_pct"] > m["urllc_drop_late_pct"]]
    if drop_dominant_overflow:
        A(f"**Drop didominasi overflow (buffer penuh), bukan deadline** pada "
          f"{', '.join('`{}`'.format(s) for s in drop_dominant_overflow)} — pertimbangkan menaikkan "
          f"`buffer.urllc_max_bits` kalau overflow tidak diinginkan sebagai mekanisme utama "
          f"(deadline-drop lebih representatif untuk SLA URLLC).\n")

    # 9. Kesimpulan
    A("## 9. Kesimpulan: Hasil Terbaik\n")
    A(f"- **Timely throughput tertinggi lintas semua run:** `{best_throughput_at_sla['stem']}` "
      f"({_fmt(best_throughput_at_sla['timely_throughput_mbps'],3)} Mbps)")
    if best_dqn:
        A(f"- **Terbaik keluarga DQN (adil, budget sama):** `{best_dqn['stem']}`")
    if best_ppo:
        A(f"- **Terbaik keluarga PPO (adil, budget sama):** `{best_ppo['stem']}`")
    A(f"- **SLA satisfaction terbaik:** `{best_sla['stem']}` ({_fmt(best_sla['sla_satisfaction_pct'],1)}%)")
    A("")
    A("**Catatan kehati-hatian:** angka di atas dari training-time CSV (1 seed per run kalau "
      "cuma 1 log per algo ada di `results/logs/`). Klaim \"algoritma X signifikan lebih baik dari "
      "Y\" harus dirujuk ke `results/RLIABLE.md` (multi-seed, held-out eval, bootstrap CI) — lihat §10.\n")

    # 10. Rekomendasi
    A("## 10. Rekomendasi Sebelum Paper\n")
    A("1. ~~Ganti reward linear-clip dengan log-scale~~ — **superseded**: v3 memindahkan delay/violation "
      "dari reward term ke CMDP constraint (§2), jadi isu saturasi tidak relevan lagi.")
    A("2. ~~Tambah PRB floor URLLC~~ — **sudah diterapkan** (§2, action projection di `env.step()`).")
    A("3. **Verifikasi hasil di `results/RLIABLE.md`** — IQM + bootstrap CI per keluarga budget, "
      "bandingkan tiap varian GNN vs baseline sepadan sebelum klaim menang di paper.")
    A("4. **Kalau CI proposed vs baseline overlap**, laporkan sebagai \"comparable\", bukan \"lebih baik\" "
      "— jangan cherry-pick titik tunggal dari tabel §4/§6.")
    A("5. **Kalau semua run masih menunjukkan SLA di bawah target CMDP** (§8), pertimbangkan menaikkan "
      "`cmdp.lambda_lr` atau menurunkan `cmdp.dual_update_every` supaya lambda konvergen lebih cepat "
      "relatif ke budget training yang tersedia.\n")

    # 11. Indeks file
    A("## 11. Indeks File\n")
    A("- Figures: `results/figures/*.png` (generated oleh `scripts/make_paper_figures.py`)")
    A("- `results/figures/paper_metrics_long.csv` — data tidy per algo/metric/step")
    A("- `results/results_summary.csv` — tabel training-time dari dokumen ini (machine-readable)")
    A("- `results/eval/*_eval.csv` — held-out greedy evaluation per episode (`evaluate_checkpoints.py`)")
    A("- `results/RLIABLE.md` — IQM + bootstrap CI, klaim statistik yang sah untuk paper")
    A("- `results/v2_scalarized/` — arsip hasil v2 (scalarized reward, fluid queue, delay-clip tanpa "
      "deadline). **INVALIDATED**: dijalankan sebelum UE-association bug-fix (§2) — bukan baseline yang sah.")
    A("- `results/v1_uncoupled/` — arsip hasil v1 (uncoupled interference, reward clip `[-2,+2]`). "
      "**INVALIDATED**: sama, dijalankan sebelum bug-fix (§2).")
    A("- `results/checkpoints/*_last.pt`, `*_best.pt` — checkpoint tiap run (`_best.pt` TIDAK dipakai "
      "untuk eval di bawah CMDP — reward turun seiring lambda naik, jadi 'best' menyesatkan; eval "
      "pakai `_last.pt`/`.pt` final)\n")

    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", default="results/logs")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    log_dir = (ROOT / args.log_dir) if not Path(args.log_dir).is_absolute() else Path(args.log_dir)
    out_dir = (ROOT / args.out_dir) if not Path(args.out_dir).is_absolute() else Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(ROOT)

    csvs = sorted(p for p in log_dir.glob("*.csv") if "seed999" not in p.stem)
    if not csvs:
        print(f"No CSV logs in {log_dir}. Run training first.")
        sys.exit(0)

    metrics = []
    for p in csvs:
        run = load_run(p)
        if run is None:
            print(f"  [skip] {p.name}: too few rows")
            continue
        metrics.append(compute_metrics(p.stem, run, cfg))

    print(f"Loaded {len(metrics)} runs: {', '.join(sorted(m['stem'] for m in metrics))}\n")

    write_summary_csv(metrics, out_dir / "results_summary.csv")

    md = render_markdown(metrics, cfg, ROOT)
    md_path = out_dir / "RESULTS.md"
    md_path.write_text(md, encoding="utf-8")
    print("report ->", md_path)
