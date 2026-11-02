#!/bin/bash
# Binary lines in GL (gg + lloyd, consts.py defaults; same settings as the set-B database) for the recalibration of
# lambda_es (Tc_corr) and the check of the binary ranking Nb-Zr > Nb-Ti > Ti-V (DOCS/ROZMOWA_2026-11-02_optB_VTi.md).
# 51 compositions in slurm/compositions/binaries_TiNbZrHfTaMoWYVSc_gl.csv, columns as in the set-B list (10 elements,
# zeros elsewhere), so the results can also be merged into databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv:
#   TiV  Ti10..90 step 10 + Ti15, Ti25 (exp: Cheng 1962, Ti15/25/50/70)       11
#   VNb  V85 70 50 31 20 10 (exp: Corsan 1970)                                   6
#   VTa  V90 76 50 30 10 (exp: Corsan 1970)                                      5
#   VMo  V95 90 (exp: Kuentzler 1986)                                            2
#   NbZr, NbTi, NbHf  Nb90..10 step 10 (exp: Hulm & Blaugher 1961)           3 x 9
# Experimental values: results/Tc_literature_V_additions.csv and DOCS/LITERATURE.md (sections 3.2, 3.5).
# Default 17 tasks x 3 interleaved rows, all 3 concurrent (one round; binaries take ~1-3 h).
#   slurm/submit_binaries_gl_TiNbZrHfTaMoWYVSc.sh
#   env DRYRUN=1 slurm/submit_binaries_gl_TiNbZrHfTaMoWYVSc.sh          plan only (tcsh: env VAR=x cmd)
FILE=${FILE:-slurm/compositions/binaries_TiNbZrHfTaMoWYVSc_gl.csv}
WORKDIR=${WORKDIR:-validation.ratios/binaries.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl}
NTASKS=${NTASKS:-17}; CONCURRENT=${CONCURRENT:-3}; JOBNAME=bin_gl
# run directly on the login node (it calls sbatch itself); under sbatch $0 is a spool copy without submit_complist.sh
[ -n "${SLURM_JOB_ID:-}" ] && { echo "this submit script calls sbatch itself: run it on the login node WITHOUT sbatch, e.g.  slurm/submit_binaries_gl_TiNbZrHfTaMoWYVSc.sh"; exit 2; }
source "$(dirname "$0")/submit_complist.sh"
