#!/bin/bash
# usage: train_mw.sh <gpu> <seed>...
cd /mnt/hdd4/TrustedInjection/experiments/tagged_propagation_20261031
gpu=$1; shift
for s in "$@"; do
  TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu /mnt/hdd4/venvs/hipporag/bin/python train_pairce.py --rel --w-anti-b 0.3 --data mw_train/pairs --seed $s --out /mnt/hdd4/dujddx/tagged_pairce_mw_antib0.3_s${s}_20261104 > /mnt/hdd4/dujddx/mw/train_s$s.log 2>&1
  echo "SEEDDONE $s rc=$? $(date +%H:%M)" >> /mnt/hdd4/dujddx/mw/train.log
done
