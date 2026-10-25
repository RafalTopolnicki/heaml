#!/bin/bash
# New TiNbZrHfTaMoWYReRu random dataset for the GL re-optimisation (full pipeline; eta is recomputed in GL by
# scripts/make_slim_dataset.py afterwards and merged with the 492 old entries of the .slim.gl set).
#   full simplex (k=10)  : 800 samples, seed 2000      (old set used seeds 100..1099 -> disjoint)
#   k-faces, k = 2..9    : top-up to 100 per k         (old faces used 20000+1000k+i -> new base 40000+1000k)
# Usage (tcsh: env VAR=x cmd):  slurm/submit_random_gl_TiNbZrHfTaMoWYReRu.sh
#   env CALIB=1 ...   calibration (task 1 of full/k5/k2, one round)      env DRYRUN=1 ...   plan only
#   env ROUNDS=8 ...  fewer, longer tasks                                see slurm/submit_random_families.sh
WORKDIR=${WORKDIR:-random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl}
ELEMENTS="Ti,Nb,Zr,Hf,Ta,Mo,W,Y,Re,Ru"
# name : N_NONZERO (empty = full simplex) : NSAMPLES : SEED : CONCURRENT per task (4 for k>=7, 5 for k<=6)
FAMILIES=(
  "full::800:2000:4"
  "k9:9:100:49000:4"
  "k8:8:100:48000:4"
  "k7:7:100:47000:4"
  "k6:6:68:46000:5"
  "k5:5:45:45000:5"
  "k4:4:30:44000:5"
  "k3:3:25:43000:5"
  "k2:2:40:42000:5"
)
source "$(dirname "$0")/submit_random_families.sh"
