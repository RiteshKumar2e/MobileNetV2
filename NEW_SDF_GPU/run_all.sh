#!/usr/bin/env bash
# Full pipeline on a GPU machine (Linux/Colab). Run from NEW_SDF_GPU/:
#   bash run_all.sh            # everything
#   STEP=2 bash run_all.sh     # one step
# Finished runs are skipped (--skip-done), so after a crash just rerun.
# Every training setting goes to SDF-YOLO AND to the baselines (fair comparison).
set -u
STEP=${STEP:-all}
NEU_VOC=${NEU_VOC:-../NEU-DET}
GC10_VOC=${GC10_VOC:-../GC10-DET}
SEEDS=${SEEDS:-42,1,2}
AUG="mixup=0.1 degrees=5 flipud=0.5 scale=0.7"     # same for every model
run() { echo "=== python train.py $*"; python -u train.py "$@" --skip-done || echo "!!! failed: $*"; }

# 0) data from the frozen splits
if [[ $STEP == all || $STEP == 0 ]]; then
  python prepare_data.py --data neu --voc "$NEU_VOC"
  python prepare_data.py --data gc10 --voc "$GC10_VOC"
fi

# 1) SDF-YOLO v2 screening, s scale, 640, seed 42 -> pick per dataset by VAL mAP50
if [[ $STEP == all || $STEP == 1 ]]; then
  for d in neu gc10; do
    run --data $d --model sdf                                  # v2 default (PKI 3,5,7,9,11)
    run --data $d --model sdf --pki-kernels 3,3,3,5,5
    run --data $d --model sdf --p2-head
    run --data $d --model sdf --no-aifi
    run --data $d --model sdf --arch v1 --caa-init identity    # published SDF neck
    run --data $d --model yolo11s.pt
  done
fi

# 2) ablation of the v2 components, 3 seeds (s scale, 640)
if [[ $STEP == all || $STEP == 2 ]]; then
  for d in neu gc10; do
    run --data $d --model sdf --seeds $SEEDS
    run --data $d --model sdf --seeds $SEEDS --pki-levels none
    run --data $d --model sdf --seeds $SEEDS --no-aifi
    run --data $d --model sdf --seeds $SEEDS --msdw-kernels none
    run --data $d --model yolo11s.pt --seeds $SEEDS
  done
fi

# 3) high-accuracy setting: L scale, 300 epochs, strong aug; GC10 at 1024 + rfs 0.3.
#    Put the step-1 winner flags in NEU_FLAGS / GC10_FLAGS, e.g. GC10_FLAGS="--p2-head"
if [[ $STEP == all || $STEP == 3 ]]; then
  NEU_FLAGS=${NEU_FLAGS:-}
  GC10_FLAGS=${GC10_FLAGS:-}
  run --data neu  --model sdf --scale l --epochs 300 --seeds $SEEDS --extra "$AUG" $NEU_FLAGS
  run --data gc10 --model sdf --scale l --epochs 300 --imgsz 1024 --rfs 0.3 --seeds $SEEDS --extra "$AUG" $GC10_FLAGS
  for m in yolo11s.pt yolo12s.pt yolov8s.pt; do               # -> yolo11l / yolo12l / yolov8l
    run --data neu  --model $m --scale l --epochs 300 --seeds $SEEDS --extra "$AUG"
    run --data gc10 --model $m --scale l --epochs 300 --imgsz 1024 --rfs 0.3 --seeds $SEEDS --extra "$AUG"
  done
fi

# 4) tables + figures
if [[ $STEP == all || $STEP == 4 ]]; then
  python make_comparison.py --out ../paper_results_v3
  python make_comparison.py --out ../paper_results_v3_large --neu "epochs=300" --gc10 "imgsz=1024 epochs=300 rfs=0.3"
fi
