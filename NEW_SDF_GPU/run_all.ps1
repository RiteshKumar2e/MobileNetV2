# Windows version of run_all.sh. Run from NEW_SDF_GPU\:
#   powershell -ExecutionPolicy Bypass -File run_all.ps1                 # everything
#   powershell -ExecutionPolicy Bypass -File run_all.ps1 -Step 3 -Gc10Flags "--p2-head"
param([ValidateSet("all", "0", "1", "2", "3", "4")][string]$Step = "all",
      [string]$NeuVoc = "..\NEU-DET", [string]$Gc10Voc = "..\GC10-DET", [string]$Seeds = "42,1,2",
      [string]$NeuFlags = "", [string]$Gc10Flags = "")
Set-Location $PSScriptRoot
$Aug = "mixup=0.1 degrees=5 flipud=0.5 scale=0.7"
function Run([string[]]$flags) {
    Write-Host "`n=== python train.py $($flags -join ' ') ===" -ForegroundColor Cyan
    & python -u train.py @flags --skip-done
    if ($LASTEXITCODE -ne 0) { Write-Host "!!! failed (exit $LASTEXITCODE), continuing" -ForegroundColor Red }
}
function Words([string]$s) { if ($s) { return $s.Split(" ") } else { return @() } }

if ($Step -in "all", "0") {
    python prepare_data.py --data neu --voc $NeuVoc
    python prepare_data.py --data gc10 --voc $Gc10Voc
}
if ($Step -in "all", "1") {
    foreach ($d in "neu", "gc10") {
        Run @("--data", $d, "--model", "sdf")
        Run @("--data", $d, "--model", "sdf", "--pki-kernels", "3,3,3,5,5")
        Run @("--data", $d, "--model", "sdf", "--p2-head")
        Run @("--data", $d, "--model", "sdf", "--no-aifi")
        Run @("--data", $d, "--model", "sdf", "--arch", "v1", "--caa-init", "identity")
        Run @("--data", $d, "--model", "yolo11s.pt")
    }
}
if ($Step -in "all", "2") {
    foreach ($d in "neu", "gc10") {
        Run @("--data", $d, "--model", "sdf", "--seeds", $Seeds)
        Run @("--data", $d, "--model", "sdf", "--seeds", $Seeds, "--pki-levels", "none")
        Run @("--data", $d, "--model", "sdf", "--seeds", $Seeds, "--no-aifi")
        Run @("--data", $d, "--model", "sdf", "--seeds", $Seeds, "--msdw-kernels", "none")
        Run @("--data", $d, "--model", "yolo11s.pt", "--seeds", $Seeds)
    }
}
if ($Step -in "all", "3") {
    Run (@("--data", "neu", "--model", "sdf", "--scale", "l", "--epochs", "300", "--seeds", $Seeds,
           "--extra", $Aug) + (Words $NeuFlags))
    Run (@("--data", "gc10", "--model", "sdf", "--scale", "l", "--epochs", "300", "--imgsz", "1024",
           "--rfs", "0.3", "--seeds", $Seeds, "--extra", $Aug) + (Words $Gc10Flags))
    foreach ($m in "yolo11s.pt", "yolo12s.pt", "yolov8s.pt") {
        Run @("--data", "neu", "--model", $m, "--scale", "l", "--epochs", "300", "--seeds", $Seeds, "--extra", $Aug)
        Run @("--data", "gc10", "--model", $m, "--scale", "l", "--epochs", "300", "--imgsz", "1024",
              "--rfs", "0.3", "--seeds", $Seeds, "--extra", $Aug)
    }
}
if ($Step -in "all", "4") {
    python make_comparison.py --out ..\paper_results_v3
    python make_comparison.py --out ..\paper_results_v3_large --neu "epochs=300" --gc10 "imgsz=1024 epochs=300 rfs=0.3"
}
