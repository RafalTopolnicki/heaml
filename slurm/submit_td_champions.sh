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
#   env LIST=slurm/compositions/td_re_ew1.0.csv EW=1.0 bash slurm/submit_td_champions.sh
#        Re-containing champions (A mc9, mc10), control A mc2 and their element references with ew 1.0 Ry
#        (ew 0.6 breaks pure Re); with EW set every outdir must start with thermodynamic.ew<EW>/, without EW
#        none may start with thermodynamic.ew (checked below, so that energy windows are never mixed in one directory).
# Results: thermodynamic/<run>/<composition>/{bcc,fcc,hcp}/ + structure_summary.json,
#          thermodynamic/elements.sra.ew0.6.topola/<El>1.0000/; logs slurm/logs/td_cmp_<jobid>_<task>.out.
[ -n "${SLURM_JOB_ID:-}" ] && { echo "run this on the login node with bash, not with sbatch"; exit 2; }
cd "$(dirname "$0")/.." || exit 1
LIST=${LIST:-slurm/compositions/td_champions_gl.csv}; MAXPAR=${MAXPAR:-20}; DRY=${DRY:-0}; ONLY=${ONLY:-}; EW=${EW:-}
[ -s "${LIST}" ] || { echo "missing ${LIST} (git pull?)"; exit 2; }
grep -q "^KKR_PARAMS_LATTICE = {" scripts/src/consts.py || { echo "KKR_PARAMS_LATTICE not found in consts.py"; exit 2; }
mkdir -p slurm/logs

rows=(); n=0
while IFS=, read -r tag run target tc nel elements conc outdir dirname; do
    n=$((n + 1))
    if [ -n "${EW}" ] && [[ "${outdir}" != "thermodynamic.ew${EW}/"* ]]; then
        echo "EW=${EW} but row ${n} (${tag}) writes to ${outdir} (not under thermodynamic.ew${EW}/)"; exit 2
    fi
    if [ -z "${EW}" ] && [[ "${outdir}" == thermodynamic.ew* ]]; then
        echo "row ${n} (${tag}) writes to ${outdir}: set EW to the energy window of that directory"; exit 2
    fi
    if [ -n "${ONLY}" ] && [[ " ${ONLY} " != *" ${tag} "* ]]; then continue; fi
    if [ -f "${outdir}/${dirname}/structure_summary.json" ]; then
        echo "[done]    ${tag}"
    else
        echo "[pending] row ${n}  ${tag}  ${elements} / ${conc}"
        rows+=("${n}")
    fi
done < <(tail -n +2 "${LIST}")

[ "${#rows[@]}" -gt 0 ] || { echo "nothing to submit"; exit 0; }
export LIST EW ROWS="${rows[*]}"
if [ "${DRY}" = 1 ]; then
    echo "sbatch --array=1-${#rows[@]}%${MAXPAR} slurm/topola_td_array.sbatch   LIST=${LIST} EW=${EW:-default} ROWS=\"${ROWS}\""
    exit 0
fi
job=$(sbatch --parsable --export=ALL --array=1-${#rows[@]}%${MAXPAR} slurm/topola_td_array.sbatch) || exit 3
echo "array ${job}: ${#rows[@]} tasks (rows ${ROWS})"
