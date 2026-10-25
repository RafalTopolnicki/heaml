#!/bin/bash
# Extension of the TiScNbZrHfTaMoWYReRuLa random dataset (999 full-simplex entries, seeds 100..1099, already
# converted to random.ratios/sra.TiScNbZrHfTaMoWYReRuLa.kp10.ew0.6.slim.gl): +1 full-simplex sample (-> 1000) and
# 100 face compositions for every k = 11 (all but one element) ... 2 (binaries), i.e. 1001 new compositions.
#   full simplex (k=12)  : 1 sample, seed 3000
#   k-faces, k = 2..11   : 100 per k, seed 50000+1000k (no faces existed for this element set)
# Results go to a separate directory and are merged by scripts/make_slim_dataset.py <WORKDIR> <...>.slim.gl.
# Usage (tcsh: env VAR=x cmd):  slurm/submit_random_gl_TiScNbZrHfTaMoWYReRuLa.sh
#   env CALIB=1 ...   calibration (task 1 of k11/k5/k2, one round)       env DRYRUN=1 ...   plan only
#   env ROUNDS=8 ...  fewer, longer tasks                                see slurm/submit_random_families.sh
WORKDIR=${WORKDIR:-random.ratios/sra.TiScNbZrHfTaMoWYReRuLa.kp10.ew0.6.gl}
ELEMENTS="Ti,Sc,Nb,Zr,Hf,Ta,Mo,W,Y,Re,Ru,La"
CALIB_FAMILIES=${CALIB_FAMILIES:-k11 k5 k2}
# name : N_NONZERO (empty = full simplex) : NSAMPLES : SEED : CONCURRENT per task (4 for k>=7, 5 for k<=6)
FAMILIES=(
  "full::1:3000:1"
  "k11:11:100:61000:4"
  "k10:10:100:60000:4"
  "k9:9:100:59000:4"
  "k8:8:100:58000:4"
  "k7:7:100:57000:4"
  "k6:6:100:56000:5"
  "k5:5:100:55000:5"
  "k4:4:100:54000:5"
  "k3:3:100:53000:5"
  "k2:2:100:52000:5"
)
source "$(dirname "$0")/submit_random_families.sh"
