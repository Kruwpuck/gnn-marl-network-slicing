# Build the Python env this repo was validated on (requirements-lock.txt, Python 3.11.9,
# torch 2.4.1+cu124). Needs uv. The venv lives outside the repo by default because the
# laptop checkout sits in a Google Drive folder: 5 GB of site-packages would sync, and
# Drive drops desktop.ini into every directory it touches.
#
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1 [-VenvDir <path>]
param([string]$VenvDir = "$env:USERPROFILE\.venvs\gnn-marl")
$ErrorActionPreference = "Stop"
$repo = Split-Path $PSScriptRoot -Parent

function Run { & $args[0] $args[1..($args.Count - 1)]; if ($LASTEXITCODE) { throw "failed: $args" } }

Run uv python install 3.11.9
Run uv venv $VenvDir --python 3.11.9 --allow-existing
$py = Join-Path $VenvDir "Scripts\python.exe"
Run uv pip install --python $py -r (Join-Path $repo "requirements-lock.txt") `
    --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
Run uv pip install --python $py -e $repo --no-deps
Run $py -c "import torch, torch_geometric as g; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'pyg', g.__version__)"
Write-Host "env ready: $py"
