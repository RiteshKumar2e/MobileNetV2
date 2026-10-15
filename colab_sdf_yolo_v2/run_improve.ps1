# SDF-YOLO v2 improvement runs (see IMPROVE_PLAN.md). Run from colab_sdf_yolo_v2/:
#   powershell -ExecutionPolicy Bypass -File run_improve.ps1 -Phase 1
#   powershell -ExecutionPolicy Bypass -File run_improve.ps1 -Phase 2 -Data gc10 -Winner "--coco-head --rfs 0.3"
# Runs go one after another. A run that already has runs/<name>/summary.json is
# skipped (--skip-done), so after a crash just start the same command again.
param([ValidateSet("1", "2")][string]$Phase = "1",
      [ValidateSet("neu", "gc10")][string]$Data = "gc10",
      # phase 2: flags of that dataset's phase-1 winner (baselines ignore --coco-head)
      [string]$Winner = "")

Set-Location $PSScriptRoot

function Run([string[]]$flags) {
    Write-Host "`n=== python train.py $($flags -join ' ') ===" -ForegroundColor Cyan
    & python train.py @flags --skip-done
    if ($LASTEXITCODE -ne 0) { Write-Host "!!! run failed (exit $LASTEXITCODE), continuing" -ForegroundColor Red }
}

if ($Phase -eq "1") {
    # SDF-YOLO-s screening on both datasets, seed 42. Every run starts the Detect
    # head from COCO (--coco-head), like the baselines. Pick winners by VAL mAP50.
    Run @("--data", "neu", "--model", "sdf", "--coco-head")
    Run @("--data", "gc10", "--model", "sdf", "--coco-head")
    Run @("--data", "gc10", "--model", "sdf", "--coco-head", "--rfs", "0.3")
    Run @("--data", "neu", "--model", "sdf", "--coco-head", "--epochs", "150")
    Run @("--data", "gc10", "--model", "sdf", "--coco-head", "--epochs", "120")
    Run @("--data", "gc10", "--model", "sdf", "--coco-head", "--imgsz", "1024", "--batch", "8")
    Run @("--data", "gc10", "--model", "sdf", "--coco-head", "--imgsz", "1024", "--batch", "8", "--rfs", "0.3")
}
else {
    # the winning setting for SDF + the 3 strongest baselines of that dataset, seeds 42/1/2
    $w = @($Winner -split '\s+' | Where-Object { $_ })
    $base = if ($Data -eq "gc10") { "yolo12s.pt", "yolov10s.pt", "yolo11s.pt" } else { "yolov8s.pt", "yolo11n.pt", "yolov10s.pt" }
    foreach ($seed in 42, 1, 2) {
        foreach ($m in @("sdf") + $base) {
            Run (@("--data", $Data, "--model", $m, "--seed", "$seed") + $w)
        }
    }
}

& python make_comparison.py
