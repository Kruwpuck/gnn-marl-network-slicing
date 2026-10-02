# T3 (docs/revisi/PREREG-T3.md): PPO stopped at 200K under the exact v6 1M config, one run at
# a time, seeds 42-46 x 3 arms. Safe to re-launch: --resume continues an interrupted run, and
# a finished run resumes at its final step and exits. Stops at the first failure. Keeps the
# machine awake while running (no permanent power setting is changed).
#
#   powershell -ExecutionPolicy Bypass -File scripts\run_t3.ps1 [-VenvDir <path>]
param([string]$VenvDir = "$env:USERPROFILE\.venvs\gnn-marl")
$ErrorActionPreference = "Stop"
$py = Join-Path $VenvDir "Scripts\python.exe"
Set-Location (Split-Path $PSScriptRoot -Parent)

Add-Type -Namespace Win32 -Name Power -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint f);'
[void][Win32.Power]::SetThreadExecutionState([uint32]"0x80000001")  # ES_CONTINUOUS | ES_SYSTEM_REQUIRED

$env:PYTHONUNBUFFERED = "1"  # progress lines reach the log as they happen
New-Item -ItemType Directory -Force results/logs/stdout | Out-Null
foreach ($seed in 42..46) {
    foreach ($arm in "gatres", "gatedge", "gatres-edge") {
        $run = "gnn-mappo_${arm}_v6at200k_seed$seed"
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] start $run"
        # Start-Process, not `& $py ... *> log`: under PowerShell 5.1 with Stop, a redirected
        # native stderr line (any Python warning) becomes a terminating error.
        $p = Start-Process $py -NoNewWindow -Wait -PassThru `
            -RedirectStandardOutput "results/logs/stdout/$run.log" `
            -RedirectStandardError "results/logs/stdout/$run.err" -ArgumentList (
                "training/train_proposed.py --algo gnn-mappo --backbone $arm " +
                "--steps 1000000 --stop-at 200192 --seed $seed --resume --ckpt-interval 25000 " +
                "--config configs/generated/floor_none.yaml --tag _v6at200k")
        if ($p.ExitCode) { throw "$run failed (exit $($p.ExitCode)), see results/logs/stdout/$run.err" }
        Write-Host "[$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')] done  $run"
    }
}
Write-Host "T3 ALL DONE"
