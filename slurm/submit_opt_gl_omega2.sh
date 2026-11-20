#!/bin/bash
# Sensitivity of the optimisation to the phonon-moment convention and mu* (2026-11-20).
# Production target Tc_mu0.1 uses <omega^2> = (3/5) omega_D^2 and mu* 0.1; the alternative Tc_f0.5_mu0.16 uses
# <omega^2> = omega_D^2 / 2 (lambda x 1.2) with mu* recalibrated on the literature baseline (median calc/exp = 1 at
# mu* 0.16 for sets A, B, C).  Key 'Tc_f<f>_mu<mu>' is computed by src.process_kkr.add_tc_variants (consts.TC_OMEGA2_VARIANTS)
# and accepted by run_opt_compositions.py --target.  Everything else as in the production runs: GL eta, theta_D mix,
# Stoner penalty off, max 70 %, required components >= 5 %, beta 2, early stop 15, seed 100, CHAIN 3.
#   SET=C : set C (8 el.), Tc_f0.5_mu0.16 at mincomp 2 4 6 8; the Tc_mu0.1 reference runs exist (submit_opt_gl_TiNbZrHfTaMoWY.sh).
#   SET=D : set D (12 el.), both targets (Tc_mu0.1 reference and Tc_f0.5_mu0.16) at mincomp 2 4 6 8 10 12.
# Output opt.ratios.gl/<db>.f0.5.mu0.16.acqbeta2.stonerbeta0.mixdebye.mincomp<m>.r0.05.max0.7.face0.1 (and .mu0.1. for
# the D reference).  Run on the login node with bash, NOT with sbatch:
#   env SET=C bash slurm/submit_opt_gl_omega2.sh
#   env SET=D ONLY="D_f05_mc2 D_mu01_mc2" bash slurm/submit_opt_gl_omega2.sh
#   env SET=C DRY=1 bash slurm/submit_opt_gl_omega2.sh
[ -n "${SLURM_JOB_ID:-}" ] && { echo "run on the login node WITHOUT sbatch: env SET=C bash slurm/submit_opt_gl_omega2.sh"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
SET=${SET:?SET=C or SET=D}; CHAIN=${CHAIN:-3}; ONLY=${ONLY:-}; DRY=${DRY:-0}
case "${SET}" in
C)  DB=databases/sra.TiNbZrHfTaMoWY.kp10.ew0.6.gl.csv
    export ELEMENTS=Ti,Nb,Zr,Hf,Ta,Mo,W,Y
    # tag        target           tag-in-dirname  mincomp
    RUNS="
C_f05_mc2    Tc_f0.5_mu0.16   f0.5.mu0.16     2
C_f05_mc4    Tc_f0.5_mu0.16   f0.5.mu0.16     4
C_f05_mc6    Tc_f0.5_mu0.16   f0.5.mu0.16     6
C_f05_mc8    Tc_f0.5_mu0.16   f0.5.mu0.16     8
" ;;
D)  DB=databases/sra.TiNbZrHfTaMoWYReRuVSc.kp10.ew0.6.gl.csv
    export ELEMENTS=Ti,Nb,Zr,Hf,Ta,Mo,W,Y,Re,Ru,V,Sc
    RUNS="
D_mu01_mc2   Tc_mu0.1         mu0.1           2
D_mu01_mc4   Tc_mu0.1         mu0.1           4
D_mu01_mc6   Tc_mu0.1         mu0.1           6
D_mu01_mc8   Tc_mu0.1         mu0.1           8
D_mu01_mc10  Tc_mu0.1         mu0.1           10
D_mu01_mc12  Tc_mu0.1         mu0.1           12
D_f05_mc2    Tc_f0.5_mu0.16   f0.5.mu0.16     2
D_f05_mc4    Tc_f0.5_mu0.16   f0.5.mu0.16     4
D_f05_mc6    Tc_f0.5_mu0.16   f0.5.mu0.16     6
D_f05_mc8    Tc_f0.5_mu0.16   f0.5.mu0.16     8
D_f05_mc10   Tc_f0.5_mu0.16   f0.5.mu0.16     10
D_f05_mc12   Tc_f0.5_mu0.16   f0.5.mu0.16     12
" ;;
*)  echo "SET must be C or D"; exit 2 ;;
esac
[ -s "${DB}" ] || { echo "missing ${DB} (git pull?)"; exit 2; }
mkdir -p slurm/logs
BETA=2
n=0
while read -r tag target dtag mc; do
    [ -z "${tag}" ] && continue
    if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${tag} "* ]]; then continue; fi
    NAME=$(basename "${DB}" .csv).${dtag}.acqbeta${BETA}.stonerbeta0.mixdebye.mincomp${mc}.r0.05.max0.7.face0.1
    export NAME INIT_DB=${DB} MINCOMP=${mc} BETA TARGET=${target}
    prev=""
    for ((k = 1; k <= CHAIN; k++)); do
        dep=(); [ -n "${prev}" ] && dep=(--dependency=afterany:${prev})
        if [ "${DRY}" = 1 ]; then
            echo "sbatch --job-name=opt_${tag} ${dep[*]} slurm/topola_opt.sbatch   NAME=${NAME} TARGET=${target} MINCOMP=${mc} ELEMENTS=${ELEMENTS}"
            prev="<${tag}_${k}>"
        else
            prev=$(sbatch --parsable --export=ALL --job-name="opt_${tag}" "${dep[@]}" slurm/topola_opt.sbatch) || exit 3
            echo "${tag}: job ${prev} (${k}/${CHAIN})  -> opt.ratios.gl/${NAME}"
        fi
    done
    n=$((n + 1))
done <<< "${RUNS}"
echo "${n} runs x ${CHAIN} chained jobs"
