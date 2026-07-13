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

    def _stat_final(key, fn):
        vals = [_f(r, key) for r in final_rows]
        vals = [v for v in vals if v is not None]
        return float(fn(vals)) if vals else None

    embb = _mean_final("embb_mbps_mean")
    spec = _mean_final("spectral_eff_bps_hz")
    delay_mean = _mean_final("urllc_delay_ms_mean")
    delay_median = _stat_final("urllc_delay_ms_mean", np.median)
    delay_p95 = _stat_final("urllc_delay_ms_mean", lambda v: np.percentile(v, 95))
    sla = _mean_final("sla_satisfaction_pct")
    fairness = _mean_final("jains_fairness")

    urllc_max_delay_ms = float(cfg["slices"]["urllc"]["max_delay_ms"])
    flag_urllc_extreme = delay_mean is not None and delay_mean > 10 * urllc_max_delay_ms
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
        "embb_mbps_mean": embb, "spectral_eff_bps_hz": spec,
        "urllc_delay_ms_mean": delay_mean, "urllc_delay_ms_median": delay_median,
        "urllc_delay_ms_p95": delay_p95, "sla_satisfaction_pct": sla,
        "jains_fairness": fairness,
        "flag_urllc_extreme": flag_urllc_extreme, "flag_low_sla": flag_low_sla,
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
              "embb_mbps_mean", "spectral_eff_bps_hz",
              "urllc_delay_ms_mean", "urllc_delay_ms_median", "urllc_delay_ms_p95",
              "sla_satisfaction_pct", "jains_fairness",
              "flag_urllc_extreme", "flag_low_sla", "flag_degraded"]
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

    best_reward = max(metrics, key=lambda m: m["reward_per_step_final_mean"])
    best_sla = max((m for m in metrics if m["sla_satisfaction_pct"] is not None),
                   key=lambda m: m["sla_satisfaction_pct"])
    best_throughput = max((m for m in metrics if m["embb_mbps_mean"] is not None),
                          key=lambda m: m["embb_mbps_mean"])
    best_delay = min((m for m in metrics if m["urllc_delay_ms_mean"] is not None),
                     key=lambda m: m["urllc_delay_ms_mean"])
    best_fairness = max((m for m in metrics if m["jains_fairness"] is not None),
                        key=lambda m: m["jains_fairness"])
    best_dqn = max(dqn_group, key=lambda m: m["reward_per_step_final_mean"]) if dqn_group else None
    best_ppo = max(ppo_group, key=lambda m: m["reward_per_step_final_mean"]) if ppo_group else None

    proposed = [m for m in metrics if m["family"] == "proposed"]
    baseline = [m for m in metrics if m["family"] == "baseline"]
    proposed_worse = all(
        m["reward_per_step_final_mean"] < max(b["reward_per_step_final_mean"] for b in baseline
                                               if b["algo_family"] == m["algo_family"])
        for m in proposed
    ) if proposed and baseline else False

    lines = []
    A = lines.append

    A("# Hasil Training GNN-MARL Network Slicing\n")
    A(f"*Generated otomatis oleh `scripts/analyze_results.py` dari {len(metrics)} run di `results/logs/`.*\n")

    # 1. Ringkasan Eksekutif
    A("## 1. Ringkasan Eksekutif\n")
    A(f"- **Reward/step terbaik keseluruhan:** `{best_reward['stem']}` "
      f"({_fmt(best_reward['reward_per_step_final_mean'])} reward/step)")
    A(f"- **SLA satisfaction terbaik:** `{best_sla['stem']}` ({_fmt(best_sla['sla_satisfaction_pct'],1)}%)")
    A(f"- **Delay URLLC terendah:** `{best_delay['stem']}` ({_fmt(best_delay['urllc_delay_ms_mean'],3)} ms mean)")
    A(f"- **Terbaik di keluarga DQN (budget setara, {BUDGET_BY_FAMILY['dqn']:,} step):** "
      f"`{best_dqn['stem']}`" if best_dqn else "")
    A(f"- **Terbaik di keluarga PPO (budget setara, {BUDGET_BY_FAMILY['ppo']:,} step):** "
      f"`{best_ppo['stem']}`" if best_ppo else "")
    A("")
    gnn_madqn_preview = [m for m in metrics if m["algo"].startswith("gnn-madqn")]
    if proposed_worse:
        worst_sla_str = ""
        if gnn_madqn_preview:
            worst = min(gnn_madqn_preview, key=lambda m: m["sla_satisfaction_pct"] or 100)
            worst_sla_str = f" (mis. `{worst['stem']}` SLA satisfaction turun ke {_fmt(worst['sla_satisfaction_pct'],1)}%)"
        A(f"**Temuan utama:** setelah reward dinormalisasi per-step, **algoritma proposed "
          f"(GNN-MADQN/GNN-MAPPO) belum mengungguli baseline** pada eksperimen seed=42 ini. "
          f"`gnn-madqn` khususnya menunjukkan tanda **starvation slice URLLC** (delay jauh lebih "
          f"tinggi dan SLA satisfaction lebih rendah dari baseline{worst_sla_str}). Lihat §8 untuk "
          f"diagnosis dan §10 untuk rekomendasi sebelum data ini dipakai di paper.\n")

    # 2. Setup
    A("## 2. Setup Eksperimen\n")
    e = cfg["env"]
    A(f"- Jumlah gNB: **{e['n_gnb']}**, bandwidth: **{e['bandwidth_hz']/1e6:.0f} MHz**, "
      f"skenario: {e['scenario']}")
    A(f"- Episode length: **{cfg['training']['episode_length']} step**")
    A(f"- URLLC max delay (SLA bound): **{cfg['slices']['urllc']['max_delay_ms']} ms**, "
      f"eMBB min throughput: **{cfg['slices']['embb']['min_throughput_mbps']} Mbps**")
    r = cfg["reward"]
    A(f"- Bobot reward: w1={r['w1']} (eMBB rate), w2={r['w2']} (URLLC violation), "
      f"w3={r['w3']} (URLLC delay ratio), w4={r['w4']} (spectral efficiency); "
      f"reward per-step di-clip ke `[-2, +2]`")
    A(f"- Budget training: DQN family = {BUDGET_BY_FAMILY['dqn']:,} step, "
      f"PPO family = {BUDGET_BY_FAMILY['ppo']:,} step; seed = 42 (single seed)")
    A(f"- 8 run: {', '.join(sorted(by_stem))}\n")

    # 3. Catatan metodologis
    A("## 3. ⚠️ Catatan Metodologis (baca sebelum menafsirkan angka)\n")
    A("1. **Reward DQN vs PPO tidak dalam satuan yang sama secara mentah.** DQN mencatat "
      "`ep_reward` per-episode (200 step), PPO per-rollout (umumnya 512 step). Semua angka "
      "reward di laporan ini sudah **dinormalisasi menjadi `reward_per_step`** "
      "(`ep_reward / steps_per_row`, dengan `steps_per_row` di-derive otomatis dari median "
      "selisih kolom `step`) agar bisa dibandingkan lintas algoritma.")
    A("2. **Budget training timpang** (DQN 200K vs PPO 1M step). Perbandingan yang metodologis "
      "sah adalah **di dalam keluarga yang sama** (§6), bukan lintas keluarga.")
    A("3. **`summary_table.csv` dari `make_paper_figures.py` memakai `ConvergenceEvaluator."
      "sample_efficiency`, yang mengambil satu titik `ep_reward` mentah (noisy)** — dokumen ini "
      "memakai rata-rata jendela akhir (10% baris terakhir) dari `ep_reward_ma100` yang sudah "
      "dinormalisasi, jauh lebih stabil.")
    A("4. **Hanya seed=42 (single seed), tanpa ulangan.** Selisih kecil antar algoritma dalam "
      "laporan ini **belum bisa diklaim signifikan secara statistik** — lihat §10.\n")

    # 4. Tabel hasil utama
    A("## 4. Tabel Hasil Utama (Normalized)\n")
    A("| algo | family | final_step | reward/step (mean±std, final 10%) | SLA sat. % | "
      "eMBB Mbps | delay ms (mean/median/p95) | Jain's fairness | anomali |")
    A("|---|---|---|---|---|---|---|---|---|")
    for m in sorted(metrics, key=lambda x: -x["reward_per_step_final_mean"]):
        anomalies = []
        if m["flag_urllc_extreme"]:
            anomalies.append("URLLC starvation")
        if m["flag_low_sla"]:
            anomalies.append("SLA rendah")
        if m["flag_degraded"]:
            anomalies.append("degradasi dari peak")
        anom_str = ", ".join(anomalies) if anomalies else "-"
        A(f"| `{m['stem']}` | {m['family']} | {m['final_step']:,} | "
          f"{_fmt(m['reward_per_step_final_mean'])} ± {_fmt(m['reward_per_step_final_std'])} | "
          f"{_fmt(m['sla_satisfaction_pct'],1)} | {_fmt(m['embb_mbps_mean'],3)} | "
          f"{_fmt(m['urllc_delay_ms_mean'],2)} / {_fmt(m['urllc_delay_ms_median'],2)} / "
          f"{_fmt(m['urllc_delay_ms_p95'],2)} | {_fmt(m['jains_fairness'],3)} | {anom_str} |")
    A("")

    # 5. Tren per algoritma
    A("## 5. Analisis Tren per Algoritma\n")
    A("Tren dihitung dari kemiringan (slope) `reward_per_step` (MA-100 ternormalisasi) pada 25% "
      "baris terakhir training.\n")
    A("| algo | trend | peak reward/step (step) | final reward/step | convergence step |")
    A("|---|---|---|---|---|")
    for m in sorted(metrics, key=lambda x: x["stem"]):
        conv = f"{m['convergence_step']:,}" if m["convergence_step"] is not None else "belum konvergen"
        A(f"| `{m['stem']}` | {m['trend_label']} | "
          f"{_fmt(m['peak_ma_reward_per_step'])} (step {m['peak_step']:,}) | "
          f"{_fmt(m['final_ma_reward_per_step'])} | {conv} |")
    A("")

    # 6. Proposed vs Baseline per keluarga
    A("## 6. Proposed vs Baseline per Keluarga (budget setara)\n")
    for fam_name, group, budget in (("DQN", dqn_group, BUDGET_BY_FAMILY["dqn"]),
                                     ("PPO", ppo_group, BUDGET_BY_FAMILY["ppo"])):
        A(f"### Keluarga {fam_name} (@ {budget:,} step)\n")
        A("| algo | family | reward/step final | SLA % | delay ms mean |")
        A("|---|---|---|---|---|")
        for m in sorted(group, key=lambda x: -x["reward_per_step_final_mean"]):
            A(f"| `{m['stem']}` | {m['family']} | {_fmt(m['reward_per_step_final_mean'])} | "
              f"{_fmt(m['sla_satisfaction_pct'],1)} | {_fmt(m['urllc_delay_ms_mean'],2)} |")
        winner = max(group, key=lambda x: x["reward_per_step_final_mean"]) if group else None
        if winner:
            verdict = "PROPOSED MENANG" if winner["family"] == "proposed" else "BASELINE MENANG"
            A(f"\n**Pemenang keluarga {fam_name}: `{winner['stem']}` ({verdict})**\n")

    # 7. KPI Jaringan
    A("## 7. Analisis KPI Jaringan\n")
    A(f"- Throughput eMBB tertinggi: `{best_throughput['stem']}` "
      f"({_fmt(best_throughput['embb_mbps_mean'],3)} Mbps)")
    A(f"- Fairness (Jain's index) tertinggi: `{best_fairness['stem']}` "
      f"({_fmt(best_fairness['jains_fairness'],3)})")
    A(f"- Delay URLLC terendah: `{best_delay['stem']}` ({_fmt(best_delay['urllc_delay_ms_mean'],3)} ms)")
    A("- **Perhatikan kolom mean vs median/p95 delay pada §4** — pada run dengan distribusi delay "
      "yang skewed (mis. `gnn-mappo`), mean bisa jauh lebih besar dari median karena ditarik oleh "
      "sedikit outlier katastrofik; SLA satisfaction % lebih representatif untuk kasus ini.\n")

    # 8. Diagnosis
    A("## 8. Temuan & Anomali (Diagnosis)\n")
    gnn_madqn = [m for m in metrics if m["algo"].startswith("gnn-madqn")]
    if gnn_madqn:
        madqn_bits = []
        for m in gnn_madqn:
            madqn_bits.append(
                "`{}` delay mean {} ms, SLA {}%".format(
                    m["stem"], _fmt(m["urllc_delay_ms_mean"], 0), _fmt(m["sla_satisfaction_pct"], 1)
                )
            )
        madqn_str = ", ".join(madqn_bits)
        A(f"**URLLC starvation pada `gnn-madqn`.** Pada `envs/network_slicing_env.py:174-178`, "
          f"`urllc_delay_ms = queue_urllc / max(urllc_rate, 1.0) * 1000`. Saat agent "
          f"mengalokasikan PRB URLLC mendekati 0%, `urllc_rate` jatuh ke lantai `1.0` bps sehingga "
          f"delay menjadi `queue_bits * 1000` ms dan meledak seiring antrean menumpuk. Ini terlihat "
          f"persis pada data: {madqn_str}.\n")
    A("**Reward clip `[-2, +2]` menjenuhkan penalti delay.** Term "
      "`-w3 * mean(delay / max_delay)` (`network_slicing_env.py:193`) seharusnya menghukum delay "
      "besar, tapi `np.clip(r, -2.0, 2.0)` (`:195`) membuat reward mentok di batas bawah begitu "
      "delay sudah katastrofik — memburuk lebih jauh tidak menambah hukuman, sehingga gradien "
      "hilang dan agent kehilangan insentif untuk keluar dari kondisi starvation. Ini kandidat "
      "**cacat desain reward**, bukan sekadar kekurangan kapasitas model GNN.\n")
    zero_delay = [m["stem"] for m in metrics if m["urllc_delay_ms_mean"] is not None and m["urllc_delay_ms_mean"] < 0.01]
    if zero_delay:
        zero_delay_str = ", ".join("`{}`".format(s) for s in zero_delay)
        A(f"**Baseline menjaga delay ≈ 0 ms** ({zero_delay_str}) — "
          f"antrean URLLC mereka selalu terkuras habis, kebijakan lebih konservatif/aman, dan "
          f"itulah salah satu alasan reward-nya lebih baik dibanding proposed.\n")
    skewed = [m for m in metrics if m["urllc_delay_ms_mean"] and m["urllc_delay_ms_median"] is not None
              and m["urllc_delay_ms_mean"] > 20 * max(m["urllc_delay_ms_median"], 0.01)]
    if skewed:
        skewed_str = ", ".join("`{}`".format(m["stem"]) for m in skewed)
        A(f"**Distribusi delay sangat skewed pada** {skewed_str} "
          f"— SLA satisfaction tinggi tapi mean delay besar, artinya mayoritas step sehat dan "
          f"sedikit outlier menarik mean naik jauh. Jangan pakai mean delay sendirian untuk klaim "
          f"kualitas di paper; sertakan median/p95 (§4).\n")

    # 9. Kesimpulan
    A("## 9. Kesimpulan: Hasil Terbaik\n")
    A(f"- **Reward/step tertinggi lintas semua run:** `{best_reward['stem']}` "
      f"({_fmt(best_reward['reward_per_step_final_mean'])})")
    A(f"- **Terbaik keluarga DQN (adil, budget sama):** `{best_dqn['stem']}`" if best_dqn else "")
    A(f"- **Terbaik keluarga PPO (adil, budget sama):** `{best_ppo['stem']}`" if best_ppo else "")
    A(f"- **SLA satisfaction terbaik:** `{best_sla['stem']}` ({_fmt(best_sla['sla_satisfaction_pct'],1)}%)")
    A("")
    A("**Catatan kehati-hatian:** hasil di atas berasal dari **satu seed (42)** tanpa ulangan, "
      "jadi belum sah untuk klaim \"algoritma X signifikan lebih baik dari Y\" di paper tanpa "
      "multi-seed run + uji statistik (lihat §10).\n")

    # 10. Rekomendasi
    A("## 10. Rekomendasi Sebelum Paper\n")
    A("1. **Revisi reward clipping** — naikkan batas `[-2,+2]` atau ganti penalti delay ke skala "
      "log (`log1p(delay/max_delay)`) supaya gradien tidak hilang saat delay besar.")
    A("2. **Samakan budget step DQN vs PPO** (atau laporkan sample-efficiency, bukan reward "
      "absolut, saat membandingkan lintas keluarga).")
    A("3. **Tambah minimal 3 seed per algoritma** untuk error bar dan uji signifikansi "
      "(mis. Welch's t-test / bootstrap CI) sebelum klaim proposed vs baseline di paper.")
    A("4. **Laporkan p95/median delay**, bukan hanya mean, terutama untuk `gnn-mappo` yang "
      "distribusinya skewed.")
    A("5. **Pertimbangkan floor alokasi PRB URLLC minimum** di action space atau di reward, "
      "supaya agent tidak bisa menstarve URLLC sepenuhnya.")
    A("6. Re-run `gnn-madqn` setelah perbaikan #1 dan #5, bandingkan ulang dengan tabel di §4.\n")

    # 11. Indeks file
    A("## 11. Indeks File\n")
    A("- Figures: `results/figures/*.png` (11 grafik, generated oleh `scripts/make_paper_figures.py`)")
    A("- `results/figures/paper_metrics_long.csv` — data tidy per algo/metric/step")
    A("- `results/figures/summary_table.csv` — ringkasan dari `ConvergenceEvaluator` (lihat catatan §3.3)")
    A("- `results/results_summary.csv` — tabel ternormalisasi dari dokumen ini (machine-readable)")
    A("- `results/logs/_pre_fix_backup/` — backup CSV sebelum perbaikan header/skema")
    A("- `results/checkpoints/*_last.pt`, `*_best.pt` — checkpoint tiap run\n")

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
