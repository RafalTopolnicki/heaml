#!/bin/bash
# GL optimisation runs on set B = Ti,Nb,Zr,Hf,Ta,Mo,W,Y,V,Sc (2026-11-01), same settings as the set-A runs
# (slurm/submit_opt_gl_TiNbZrHfTaMoWYReRu.sh): GL eta (G + Lloyd), theta_D mix, mu* 0.1, Stoner penalty off, max 70 % per
# element, required components >= 5 %, beta 2, 10 champions per iteration, early stop after 15 iterations, seed 100.
# mincomp 7 (do many required elements make the two targets converge to similar compositions?), each with two targets:
#   Tc_mu0.1  McMillan without corrections
#   Tc_corr   bcc theta_D for Ti/Sc + electron-paramagnon coupling (process_kkr.compute_tc_corrected, consts TC_CORR_*)
# Init database databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv for both (Tc_* are recomputed on reading).
# Output opt.ratios.gl/<init>.<mu0.1|corr.mu0.1>.acqbeta2...mincomp<m>..., logs slurm/logs/opt_<tag>_<jobid>.out.
# Run on the login node with bash, NOT with sbatch:
#   bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mc7.sh                  all runs below, CHAIN=3 (3 x 2 days)
#   env ONLY="B_mc7" bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mc7.sh     without corrections only
#   env ONLY="B_mc7_corr" bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mc7.sh     with corrections only
#   env DRY=1 bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mc7.sh        print what would be submitted
[ -n "${SLURM_JOB_ID:-}" ] && { echo "this submit script calls sbatch itself: run it on the login node WITHOUT sbatch, e.g.  bash slurm/submit_opt_gl_TiNbZrHfTaMoWYVSc_mc7.sh"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
CHAIN=${CHAIN:-3}; ONLY=${ONLY:-}; DRY=${DRY:-0}
DB=databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv
export ELEMENTS=Ti,Nb,Zr,Hf,Ta,Mo,W,Y,V,Sc        # via the environment (--export=ALL), never inside --export=ALL,VAR=a,b
# tag  target  mincomp  beta
RUNS="
B_mc7       Tc_mu0.1 7  2
B_mc7_corr  Tc_corr  7  2
"
[ -s "${DB}" ] || { echo "missing ${DB} (git pull?)"; exit 2; }
mkdir -p slurm/logs
n=0
while read -r tag target mc beta; do
    [ -z "${tag}" ] && continue
    if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${tag} "* ]]; then continue; fi
    case "${target}" in Tc_mu0.1) ttag=mu0.1 ;; Tc_corr) ttag=corr.mu0.1 ;; *) echo "unknown target ${target}"; exit 2 ;; esac
    NAME=$(basename "${DB}" .csv).${ttag}.acqbeta${beta}.stonerbeta0.mixdebye.mincomp${mc}.r0.05.max0.7.face0.1
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
