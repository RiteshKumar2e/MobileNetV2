# Final runs for the paper (options 1-4). Run from this folder:
#   powershell -ExecutionPolicy Bypass -File run_final.ps1            # all steps, in order
#   powershell -ExecutionPolicy Bypass -File run_final.ps1 -Step 2    # just one step
# Every step applies the same setting to SDF and to the baselines (fair comparison).
# Finished runs are skipped, so after a crash or Ctrl+C start the same command again.
# Baselines per step = the strongest ones by VAL mAP50 in the 640 / seed-42 runs.
param([ValidateSet("all", "1", "2", "3", "4")][string]$Step = "all")

Set-Location $PSScriptRoot
Start-Transcript -Append -Path runs\run_final_log.txt | Out-Null

function Run([string[]]$flags) {
    Write-Host "`n=== python train.py $($flags -join ' ') ===" -ForegroundColor Cyan
    & python -u train.py @flags
    if ($LASTEXITCODE -ne 0) { Write-Host "!!! failed (exit $LASTEXITCODE), continuing" -ForegroundColor Red }
}

# 1) TTA numbers for the 16 runs that already exist (~30 min, no training)
if ($Step -in "all", "1") {
    Run @("--data", "neu", "--model", "all", "--eval-only")
    Run @("--data", "gc10", "--model", "all", "--eval-only", "--imgsz", "640")
}

# 2) NEU, 3 seeds (42 exists; adds 1 and 2): SDF + top-4 baselines (~16 h)
if ($Step -in "all", "2") {
    foreach ($m in "sdf", "yolov9s.pt", "yolo11n.pt", "yolo11s.pt", "yolo12s.pt") {
        Run @("--data", "neu", "--model", $m, "--seeds", "1,2", "--skip-done")
    }
}

# 3) bigger models, NEU 640: SDF-m vs YOLO11m / YOLOv9m / YOLO12m (~12 h)
if ($Step -in "all", "3") {
    foreach ($m in "sdf", "yolo11s.pt", "yolov9s.pt", "yolo12s.pt") {
        Run @("--data", "neu", "--model", $m, "--scale", "m", "--skip-done")
    }
}

# 4) GC10 at 1024 (batch 8): SDF + top-3 baselines (~15-19 h EACH, ~65 h total)
if ($Step -in "all", "4") {
    foreach ($m in "sdf", "yolo11s.pt", "yolo11n.pt", "yolov9s.pt") {
        Run @("--data", "gc10", "--model", $m, "--imgsz", "1024", "--skip-done")
    }
}

Stop-Transcript | Out-Null
