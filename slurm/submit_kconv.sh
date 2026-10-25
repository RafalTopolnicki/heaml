#!/bin/bash
# Submit a k-point convergence array (slurm/topola_kconv.sbatch): one task per bzqlty in BZLIST.
#   slurm/submit_kconv.sh                                  # Ta34Nb33Hf8Zr14Ti11 (Jaskiewicz alloy, a=3.36 A), bz 20 30 40 50 60
#   env BZLIST="15 25 35 45 55" slurm/submit_kconv.sh      # odd values (tcsh: use env)
#   env NAME=Nb_a6.20 ELEMENTS=41 CONC=1.0 A0=6.20 REL=nrl XC=vbh BZLIST="20 30 40 60" slurm/submit_kconv.sh
#   env NAME=V ELEMENTS=23 CONC=1.0 A0=5.662839 slurm/submit_kconv.sh
# Topola only computes: afterwards rsync results/test.kconv/<NAME>_* to the workstation and run
# scripts/eta_rmta_tests/17_kconv_table.py there (see slurm/README.md).
set -euo pipefail
cd "$(dirname "$0")/.." || exit 1
mkdir -p slurm/logs
BZLIST=${BZLIST:-"20 30 40 50 60"}
for bz in ${BZLIST}; do [ "${bz}" -le 60 ] || { echo "bzqlty ${bz} > 60 refused"; exit 1; }; done
n=$(wc -w <<< "${BZLIST}")
export BZLIST                       # values with spaces go through the environment (--export=ALL), not --export=K=V
for v in NAME ELEMENTS CONC A0 REL XC EW MXL OUT AKAIMODBIN; do [ -n "${!v:-}" ] && export "$v"; done
ACCOUNT=${ACCOUNT:-}
echo "array 1-${n}: ${NAME:-TNHZT_a3.36} bzqlty ${BZLIST}"
sbatch ${ACCOUNT:+--account="${ACCOUNT}"} --array="1-${n}" --export=ALL slurm/topola_kconv.sbatch
