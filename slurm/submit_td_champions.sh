#!/bin/bash
# Submits slurm/topola_td_array.sbatch for the rows of LIST that have no structure_summary.json yet
# (bcc/fcc/hcp comparison of the optimisation champions + the 12 pure-element references).
# The list is generated on the workstation and committed:
#   python scripts/td_champion_list.py          -> slurm/compositions/td_champions_gl.csv
# (set A GL Tc_mu0.1 mc2–10, set B GL corr2 mc2–10, set B GL Tc_mu0.1 runs that have early-stopped; rerun it and
#  resubmit when more set-B Tc_mu0.1 runs finish — rows already done are skipped).
# Run on the login node with bash, NOT with sbatch:
#   bash slurm/submit_td_champions.sh                      all pending rows, at most MAXPAR tasks at a time
#   env DRY=1 bash slurm/submit_td_champions.sh            print the pending rows only
#   env ONLY="A_mu0.1_mc2 el_V" bash slurm/submit_td_champions.sh     selected tags
# Results: thermodynamic/<run>/<composition>/{bcc,fcc,hcp}/ + structure_summary.json,
#          thermodynamic/elements.sra.ew0.6.topola/<El>1.0000/; logs slurm/logs/td_cmp_<jobid>_<task>.out.
[ -n "${SLURM_JOB_ID:-}" ] && { echo "run this on the login node with bash, not with sbatch"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
LIST=${LIST:-slurm/compositions/td_champions_gl.csv}; MAXPAR=${MAXPAR:-20}; DRY=${DRY:-0}; ONLY=${ONLY:-}
[ -s "${LIST}" ] || { echo "missing ${LIST} (git pull?)"; exit 2; }
grep -q "^KKR_PARAMS_LATTICE = {" scripts/src/consts.py || { echo "KKR_PARAMS_LATTICE not found in consts.py"; exit 2; }
mkdir -p slurm/logs

rows=(); n=0
while IFS=, read -r tag run target tc nel elements conc outdir dirname; do
    n=$((n + 1))
    if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${tag} "* ]]; then continue; fi
    if [ -f "${outdir}/${dirname}/structure_summary.json" ]; then
        echo "[done]    ${tag}"
    else
        echo "[pending] row ${n}  ${tag}  ${elements} / ${conc}"
        rows+=("${n}")
    fi
done < <(tail -n +2 "${LIST}")

[ "${#rows[@]}" -gt 0 ] || { echo "nothing to submit"; exit 0; }
export LIST ROWS="${rows[*]}"
if [ "${DRY}" = 1 ]; then
    echo "sbatch --array=1-${#rows[@]}%${MAXPAR} slurm/topola_td_array.sbatch   LIST=${LIST} ROWS=\"${ROWS}\""
    exit 0
fi
job=$(sbatch --parsable --export=ALL --array=1-${#rows[@]}%${MAXPAR} slurm/topola_td_array.sbatch) || exit 3
echo "array ${job}: ${#rows[@]} tasks (rows ${ROWS})"
