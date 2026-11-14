#!/bin/bash
# Submits slurm/topola_lda_champions.sbatch: the 27 optimisation champions (set A Tc_mu0.1, set B corr2, set B Tc_mu0.1,
# mincomp 2..10; rows of slurm/compositions/td_champions_gl.csv without the el_* rows) recomputed with LDA from scratch
# (lattice scan + final SCF + eta + Tc; only xc differs from production).  Rows whose results.json exists are skipped.
# Run on the login node with bash, NOT with sbatch:
#   bash slurm/submit_lda_champions.sh                          all pending champions, xc vwn (closest to the Perdew-Wang LDA of Jaskiewicz 2019)
#   env DRY=1 bash slurm/submit_lda_champions.sh                print the pending rows only
#   env ONLY="A_mu0.1_mc2 B_corr2_mc2" bash slurm/submit_lda_champions.sh     selected tags
#   env XC=vbh bash slurm/submit_lda_champions.sh               von Barth-Hedin as Jaskiewicz 2016 (own OUTROOT)
# Results: lda.champions.<xc>/<run>/<composition>/results.json; logs slurm/logs/lda_cmp_<jobid>_<task>.out.
[ -n "${SLURM_JOB_ID:-}" ] && { echo "run this on the login node with bash, not with sbatch"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
LIST=${LIST:-slurm/compositions/td_champions_gl.csv}; XC=${XC:-vwn}; OUTROOT=${OUTROOT:-lda.champions.${XC}}
MAXPAR=${MAXPAR:-27}; DRY=${DRY:-0}; ONLY=${ONLY:-}
case "${XC}" in vwn|mjw|vbh) ;; *) echo "XC=${XC} is not an LDA keyword of AkaiKKR (vwn, mjw, vbh)"; exit 2 ;; esac
[ -s "${LIST}" ] || { echo "missing ${LIST} (git pull?)"; exit 2; }
mkdir -p slurm/logs

rows=(); n=0
while IFS=, read -r tag run target tc nel elements conc outdir dirname; do
    n=$((n + 1))
    [ -z "${run}" ] && continue                                   # el_* reference rows
    if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${tag} "* ]]; then continue; fi
    if [ -f "${OUTROOT}/${run}/${dirname}/results.json" ]; then
        echo "[done]    ${tag}"
    else
        echo "[pending] row ${n}  ${tag}  ${elements} / ${conc}"
        rows+=("${n}")
    fi
done < <(tail -n +2 "${LIST}")

[ "${#rows[@]}" -gt 0 ] || { echo "nothing to submit"; exit 0; }
export LIST XC OUTROOT ROWS="${rows[*]}"
if [ "${DRY}" = 1 ]; then
    echo "sbatch --array=1-${#rows[@]}%${MAXPAR} slurm/topola_lda_champions.sbatch   LIST=${LIST} XC=${XC} OUTROOT=${OUTROOT} ROWS=\"${ROWS}\""
    exit 0
fi
job=$(sbatch --parsable --export=ALL --array=1-${#rows[@]}%${MAXPAR} slurm/topola_lda_champions.sbatch) || exit 3
echo "array ${job}: ${#rows[@]} tasks (rows ${ROWS}), xc ${XC} -> ${OUTROOT}"
