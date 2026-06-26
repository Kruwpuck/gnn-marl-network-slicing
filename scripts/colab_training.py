"""
GNN-MARL Training Script for Google Colab
==========================================
Upload file ini ke Colab, atau copy-paste blok per blok ke Colab cell.

TIPS:
- Runtime > Change runtime type > GPU  (T4 gratis, A100 butuh Colab Pro)
- Colab free timeout ~90 menit — results disave ke Drive per algo selesai
- Kalau disconnect, tinggal rerun blok yang belum selesai (skip yang sudah ada CSV-nya)
- Estimasi T4: PPO ~45 menit/algo, DQN ~2 jam/algo (CPU bottleneck)
- Estimasi A100: PPO ~15 menit/algo, DQN ~45 menit/algo
"""

import os, sys, subprocess, time, csv, glob

# ============================================================
# BLOK 1 — Setup (jalankan sekali di awal)
# ============================================================

REPO_URL  = "https://github.com/YOUR_USERNAME/gnn-marl-network-slicing.git"  # TODO ganti
DRIVE_DIR = "/content/drive/MyDrive/gnn-marl-thesis"

def setup_colab():
    # Mount Google Drive supaya hasil tidak hilang saat disconnect
    from google.colab import drive
    drive.mount('/content/drive')

    # Clone repo ke Drive (persist antar session)
    if not os.path.exists(f"{DRIVE_DIR}/.git"):
        subprocess.run(["git", "clone", REPO_URL, DRIVE_DIR], check=True)
    else:
        subprocess.run(["git", "-C", DRIVE_DIR, "pull"], check=True)
    os.chdir(DRIVE_DIR)
    print("CWD:", os.getcwd())

    # Install PyTorch CUDA
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch==2.4.0", "--index-url",
                    "https://download.pytorch.org/whl/cu124"], check=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                    "torch-geometric==2.8.0", "numpy==1.26.4", "scipy==1.13.0",
                    "gymnasium==1.1.0", "pettingzoo==1.25.0", "pyyaml==6.0.2"],
                   check=True)

    import torch
    print(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    else:
        print("WARNING: GPU not detected. Change runtime type!")

    # Smoke test
    subprocess.run([sys.executable, "-m", "pytest", "tests/", "-q", "--tb=short"], check=True)
    print("Setup OK!")

# setup_colab()   # ← uncomment dan run


# ============================================================
# BLOK 2 — Helper
# ============================================================

def _csv_exists(algo, backbone=None):
    """Cek apakah CSV sudah ada (skip kalau sudah selesai)."""
    seed = 42
    if backbone:
        name = f"{algo}_{backbone}_seed{seed}.csv"
    else:
        name = f"{algo}_seed{seed}.csv"
    path = f"results/logs/{name}"
    if not os.path.exists(path):
        return False
    with open(path) as f:
        n = sum(1 for _ in f) - 1
    return n > 10  # anggap selesai kalau ada >10 episode


def train(algo, backbone=None, steps=1_000_000, seed=42, skip_if_done=True):
    """Run satu training job. Skip otomatis kalau CSV sudah ada."""
    if skip_if_done and _csv_exists(algo, backbone):
        label = f"{algo}/{backbone}" if backbone else algo
        print(f"[SKIP] {label} — CSV already exists")
        return

    os.makedirs("results/logs", exist_ok=True)
    if backbone:
        cmd = [sys.executable, "training/train_proposed.py",
               "--algo", algo, "--backbone", backbone,
               "--steps", str(steps), "--seed", str(seed)]
        label = f"{algo}/{backbone}"
    else:
        cmd = [sys.executable, "training/train_baselines.py",
               "--algo", algo, "--steps", str(steps), "--seed", str(seed)]
        label = algo

    print(f"\n{'='*55}")
    print(f"  {label}  ({steps:,} steps)")
    print(f"{'='*55}")
    t0 = time.time()
    subprocess.run(cmd, check=True)
    print(f"  Selesai dalam {(time.time()-t0)/60:.1f} menit")


# ============================================================
# BLOK 3A — PPO (jalankan ini dulu, paling cepat)
# ============================================================
# T4 estimasi: ~45 menit per algo = ~3 jam total
# A100 estimasi: ~15 menit per algo = ~1 jam total

def run_ppo(steps=1_000_000):
    train("gnn-mappo", backbone="gat",  steps=steps)
    train("gnn-mappo", backbone="sage", steps=steps)
    train("ippo",                       steps=steps)
    train("central-ppo",                steps=steps)

# run_ppo()   # ← uncomment dan run


# ============================================================
# BLOK 3B — DQN (lebih lambat, 200K steps cukup)
# ============================================================
# T4 estimasi: ~2 jam per algo = ~8 jam total
# A100 estimasi: ~45 menit per algo = ~3 jam total
# TIPS: Jalankan satu-satu kalau khawatir timeout

def run_dqn(steps=200_000):
    train("gnn-madqn", backbone="gat",  steps=steps)
    train("gnn-madqn", backbone="sage", steps=steps)
    train("idqn",                       steps=steps)
    train("central-dqn",                steps=steps)

# run_dqn()   # ← uncomment dan run


# Atau satu per satu:
# train("gnn-madqn", backbone="gat",  steps=200_000)
# train("gnn-madqn", backbone="sage", steps=200_000)
# train("idqn",                       steps=200_000)
# train("central-dqn",                steps=200_000)


# ============================================================
# BLOK 4 — Evaluasi (jalankan setelah semua training selesai)
# ============================================================

def run_evaluation():
    print("\n=== Phase 2a: Convergence Curves ===")
    subprocess.run([sys.executable, "evaluation/convergence_eval.py",
                    "--log-dir", "results/logs"], check=True)

    print("\n=== Phase 2b: Network Performance ===")
    subprocess.run([sys.executable, "evaluation/network_perf_eval.py"], check=True)

    print("\n=== Phase 2c: Zero-Shot Generalization ===")
    subprocess.run([sys.executable, "evaluation/zero_shot_eval.py"], check=True)

    print("\n=== Phase 3: Ablation Studies ===")
    for abl in ["backbone", "reward", "depth", "burstiness"]:
        subprocess.run([sys.executable, "ablation/backbone_ablation.py",
                        "--ablation", abl, "--steps", "200000"], check=True)

    print("\nSemua evaluasi selesai! Cek folder results/")

# run_evaluation()   # ← uncomment dan run


# ============================================================
# BLOK 5 — Cek progress kapan saja
# ============================================================

def check_progress():
    log_dir = "results/logs"
    if not os.path.exists(log_dir):
        print("Belum ada logs.")
        return
    print(f"{'File':45s} {'Episodes':>10} {'Last Step':>12} {'Last Reward':>12}")
    print("-" * 85)
    for f in sorted(glob.glob(f"{log_dir}/*.csv")):
        with open(f) as fp:
            rows = list(csv.reader(fp))
        n = len(rows) - 1
        if n > 0:
            last = rows[-1]
            step = last[0]
            rew  = last[2] if len(last) > 2 else "-"
            print(f"{os.path.basename(f):45s} {n:>10} {step:>12} {rew:>12}")
        else:
            print(f"{os.path.basename(f):45s} {'(empty)':>10}")

# check_progress()   # ← uncomment dan run kapan saja
