# Smoke check for a fresh env: imports + CUDA, pytest, Gate C1 (9 cells), citation audit,
# 1100-step training of three algorithms on GPU (past DQN replay_start and two PPO
# rollouts), and held-out eval of one real v6 checkpoint. Stops at the first failure.
# Training artifacts use tag _smoke and seed 999 (the repo's smoke seed,
# scripts/evaluate_checkpoints.py) and are deleted at the end; no other file under
# results/ is touched.
#
#   powershell -ExecutionPolicy Bypass -File scripts\smoke.ps1 [-VenvDir <path>]
param([string]$VenvDir = "$env:USERPROFILE\.venvs\gnn-marl")
$ErrorActionPreference = "Stop"
$py = Join-Path $VenvDir "Scripts\python.exe"
Set-Location (Split-Path $PSScriptRoot -Parent)

function Step($name) { Write-Host "`n=== $name" -ForegroundColor Cyan }
function Run { & $py @args; if ($LASTEXITCODE) { throw "FAILED (exit $LASTEXITCODE): python $args" } }

$evalDir = Join-Path $env:TEMP "gnn-marl-smoke-eval"
try {
    Step "imports + CUDA"
    Run -c "import torch, torch_geometric, gymnasium, rliable; assert torch.cuda.is_available(), 'CUDA not available'; print(torch.__version__, torch.cuda.get_device_name(0))"

    Step "pytest"
    Run -m pytest -q

    Step "Gate C1: treatment identity, 3 seeds x 3 floor modes"
    foreach ($seed in 42, 43, 44) {
        foreach ($floor in "none", "static", "dynamic") {
            Run scripts/test_treatment_identity.py --seed $seed --floor-mode $floor --steps 200
        }
    }

    Step "citation audit"
    Run scripts/citation_audit.py

    Step "training, 1100 steps each"
    $common = "--steps", "1100", "--seed", "999", "--tag", "_smoke", "--ckpt-interval", "1000"
    Run training/train_proposed.py --algo gnn-mappo --backbone gatres-edge @common
    Run training/train_proposed.py --algo gnn-madqn --backbone gat @common
    Run training/train_baselines.py --algo ippo @common

    Step "eval of a real v6 checkpoint, 2 episodes"
    Run scripts/evaluate_checkpoints.py --run results/logs/gnn-mappo_gatres-edge_v6_seed42.pt --episodes 2 --out-dir $evalDir

    Write-Host "`nSMOKE PASS" -ForegroundColor Green
}
finally {
    Remove-Item results/logs/*_smoke_seed999*, results/checkpoints/*_smoke_seed999* -ErrorAction SilentlyContinue
    Remove-Item $evalDir -Recurse -ErrorAction SilentlyContinue
}
