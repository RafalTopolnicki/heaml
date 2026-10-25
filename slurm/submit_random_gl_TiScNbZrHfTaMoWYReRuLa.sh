#!/bin/bash
# Extension of the TiScNbZrHfTaMoWYReRuLa random dataset (999 full-simplex entries already converted to
# random.ratios/sra.TiScNbZrHfTaMoWYReRuLa.kp10.ew0.6.slim.gl): 1001 compositions listed in
# slurm/compositions/TiScNbZrHfTaMoWYReRuLa_gl.csv (+1 full simplex, 100 faces for every k = 11 ... 2).
# Results go to a separate directory, merged later by scripts/make_slim_dataset.py <WORKDIR> <...>.slim.gl.
#   slurm/submit_random_gl_TiScNbZrHfTaMoWYReRuLa.sh                 37 tasks x 4 workers claiming rows dynamically (all run at once under the cpu cap)
#   env NTASKS=20 slurm/submit_random_gl_TiScNbZrHfTaMoWYReRuLa.sh   (tcsh: env VAR=x cmd)
FILE=${FILE:-slurm/compositions/TiScNbZrHfTaMoWYReRuLa_gl.csv}
WORKDIR=${WORKDIR:-random.ratios/sra.TiScNbZrHfTaMoWYReRuLa.kp10.ew0.6.gl}
NTASKS=${NTASKS:-37}; CONCURRENT=${CONCURRENT:-4}; JOBNAME=rnd12
source "$(dirname "$0")/submit_complist.sh"
