#!/bin/bash
# Extension of the TiScNbZrHfTaMoWYReRuLa random dataset (999 full-simplex entries already converted to
# random.ratios/sra.TiScNbZrHfTaMoWYReRuLa.kp10.ew0.6.slim.gl): 1001 compositions listed in
# slurm/compositions/TiScNbZrHfTaMoWYReRuLa_gl.csv (+1 full simplex, 100 faces for every k = 11 ... 2).
# Results go to a separate directory, merged later by scripts/make_slim_dataset.py <WORKDIR> <...>.slim.gl.
#   slurm/submit_random_gl_TiScNbZrHfTaMoWYReRuLa.sh                 36 tasks x <=28 interleaved rows, 6 concurrent (5 rounds)
#   env NTASKS=25 slurm/submit_random_gl_TiScNbZrHfTaMoWYReRuLa.sh   (tcsh: env VAR=x cmd)
FILE=${FILE:-slurm/compositions/TiScNbZrHfTaMoWYReRuLa_gl.csv}
WORKDIR=${WORKDIR:-random.ratios/sra.TiScNbZrHfTaMoWYReRuLa.kp10.ew0.6.gl}
NTASKS=${NTASKS:-36}; CONCURRENT=${CONCURRENT:-6}; JOBNAME=rnd12
# run directly on the login node (it calls sbatch itself); under sbatch $0 is a spool copy without submit_complist.sh
[ -n "${SLURM_JOB_ID:-}" ] && { echo "this submit script calls sbatch itself: run it on the login node WITHOUT sbatch, e.g.  slurm/submit_random_gl_TiScNbZrHfTaMoWYReRuLa.sh"; exit 2; }
source "$(dirname "$0")/submit_complist.sh"
