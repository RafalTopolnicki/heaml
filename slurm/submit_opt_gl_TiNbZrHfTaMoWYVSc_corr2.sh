#!/bin/bash
# GL optimisation runs on set B = Ti,Nb,Zr,Hf,Ta,Mo,W,Y,V,Sc with the REVISED Tc_corr (2026-11-02), mincomp 2, 3 and 7.
# Same settings as slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mc23.sh / _mc7.sh (GL eta, theta_D mix, mu* 0.1, Stoner penalty
# off, max 70 % per element, required components >= 5 %, beta 2, 10 champions per iteration, early stop after 15
# iterations, seed 100, init databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv), only the target differs:
#   Tc_corr (2026-11-02)  bcc theta_D of Sc only (215 K; Ti keeps hcp 420 K as in set A) + electron-paramagnon coupling
#                         on the V sites only, alpha_V = 0.4 (consts TC_CORR_THETA_BCC_K, TC_CORR_SF_ALPHA;
#                         fit scripts/refit_lambda_es.py --theta_bcc Sc=215).  Without V and Sc: Tc_corr == Tc_mu0.1.
# Three variants to compare: Tc_mu0.1 (tag mu0.1), Tc_corr of 2026-11-01 (tag corr.mu0.1, Ti 300 K / Sc 215 K,
# alpha 0.3 on all elements) and this one (tag corr2.mu0.1).  The new tag keeps the old corr runs from being resumed.
# Output opt.ratios.gl/<init>.corr2.mu0.1.acqbeta2...mincomp<m>..., logs slurm/logs/opt_<tag>_<jobid>.out.
# Run on the login node with bash, NOT with sbatch:
#   bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_corr2.sh                       all 3 runs, CHAIN=3 (3 x 2 days)
#   env ONLY="B_mc7_corr2" bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_corr2.sh     selected tags
#   env DRY=1 bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_corr2.sh             print what would be submitted
[ -n "${SLURM_JOB_ID:-}" ] && { echo "this submit script calls sbatch itself: run it on the login node WITHOUT sbatch, e.g.  bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_corr2.sh"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
CHAIN=${CHAIN:-3}; ONLY=${ONLY:-}; DRY=${DRY:-0}
DB=databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv
export ELEMENTS=Ti,Nb,Zr,Hf,Ta,Mo,W,Y,V,Sc        # via the environment (--export=ALL), never inside --export=ALL,VAR=a,b
# tag  target  mincomp  beta
RUNS="
B_mc2_corr2  Tc_corr  2  2
B_mc3_corr2  Tc_corr  3  2
B_mc7_corr2  Tc_corr  7  2
"
[ -s "${DB}" ] || { echo "missing ${DB} (git pull?)"; exit 2; }
# the checkout must have the revised constants, otherwise the runs would optimise the old Tc_corr under the new name
grep -q '^TC_CORR_THETA_BCC_K = {"Sc": 215.0}' scripts/src/consts.py && grep -q '^TC_CORR_SF_ALPHA = {"V": 0.4}' scripts/src/consts.py \
    || { echo "scripts/src/consts.py does not have the 2026-11-02 Tc_corr constants (git pull?)"; exit 2; }
mkdir -p slurm/logs
n=0
while read -r tag target mc beta; do
    [ -z "${tag}" ] && continue
    if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${tag} "* ]]; then continue; fi
    NAME=$(basename "${DB}" .csv).corr2.mu0.1.acqbeta${beta}.stonerbeta0.mixdebye.mincomp${mc}.r0.05.max0.7.face0.1
    export NAME INIT_DB=${DB} MINCOMP=${mc} BETA=${beta} TARGET=${target}
    prev=""
    for ((k = 1; k <= CHAIN; k++)); do
        dep=(); [ -n "${prev}" ] && dep=(--dependency=afterany:${prev})
        if [ "${DRY}" = 1 ]; then
            echo "sbatch --job-name=opt_${tag} ${dep[*]} slurm/topola_opt.sbatch   NAME=${NAME} TARGET=${target} MINCOMP=${mc} BETA=${beta} ELEMENTS=${ELEMENTS}"
            prev="<${tag}_${k}>"
        else
            prev=$(sbatch --parsable --export=ALL --job-name="opt_${tag}" "${dep[@]}" slurm/topola_opt.sbatch) || exit 3
            echo "${tag}: job ${prev} (${k}/${CHAIN})  -> opt.ratios.gl/${NAME}"
        fi
    done
    n=$((n + 1))
done <<< "${RUNS}"
echo "${n} runs x ${CHAIN} chained jobs"
