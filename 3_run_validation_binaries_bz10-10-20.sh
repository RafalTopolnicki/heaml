#!/bin/bash
# Binary validation lines (see 3_run_validation_binaries.sh) at the production k-point setting:
# bzqlty lattice 10 / Debye 10 / final SCF 20 (consts.py defaults, no --bzqlty flag).
# validation.ratios/binaries.sra.ew0.6 was computed with --bzqlty 10 for all three steps; the final SCF
# (eta, lambda) is not converged there (Nb60Ta40: Tc 16.9 K @bz10 -> 11.9 K @bz20).
# Lattice and Debye are identical (bzqlty 10) -> NOT recomputed: --reuse_from copies lattice/ and debye/
# from ${olddir}/<line>/<name> (params are checked) and runs only the final SCF (bzqlty 20) + McMillan.
# Entries with an existing results.json in ${outputdir} are skipped (18 were already done by the k-point
# test: NbZr 60/80, NbTa, NbMo, MoRe 60/80, pure Nb/Ta/Mo), so the script can be re-run safely.
# Optional: ONLY="NbTi/Nb70Ti30" ./3_run_validation_binaries_bz10-10-20.sh  runs a single entry (test).

NJOBS=${NJOBS:-16}
export OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-12}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ew=0.6
params="--overwrite_params --ew ${ew} --rel sra --debye_mode kkr"
outputdir="validation.ratios/binaries.sra.ew${ew}.bz10-10-20"
olddir="validation.ratios/binaries.sra.ew${ew}"   # bz10 run: source of lattice/ and debye/

run() {  # $1 = line, $2 = name, rest = process_hea.py arguments
    local line=$1 name=$2; shift 2
    local wd="${outputdir}/${line}/${name}"
    if [ -n "${ONLY}" ] && [ "${ONLY}" != "${line}/${name}" ]; then return; fi
    if [ -f "${wd}/results.json" ]; then echo "[skip] ${line}/${name}"; return; fi
    mkdir -p "${wd}"
    echo "[$(date +%F_%T)] start ${line}/${name}"
    local reuse=()
    if [ -f "${olddir}/${line}/${name}/run_params.json" ] && [ -f "${olddir}/${line}/${name}/lattice/lattice_summary.txt" ] \
       && [ -f "${olddir}/${line}/${name}/debye/debye_summary.json" ]; then
        reuse=(--reuse_from "${olddir}/${line}/${name}")
    else
        echo "[full] ${line}/${name}: no lattice/Debye in ${olddir}, computing everything"
    fi
    python scripts/process_hea.py "$@" --workdir "${wd}" ${params} "${reuse[@]}" > "${wd}/process_hea.log" 2>&1 \
        || echo "[FAILED] ${line}/${name} (see ${wd}/process_hea.log)"
    echo "[$(date +%F_%T)] done  ${line}/${name}"
}

throttle() {  # keep at most NJOBS background jobs
    while [ "$(jobs -rp | wc -l)" -ge "${NJOBS}" ]; do wait -n; done
}

binary() {  # $1 = line name, $2 = el A, $3 = el B, rest = at.% of A
    local line=$1 A=$2 B=$3; shift 3
    for a in "$@"; do
        local b=$((100 - a))
        throttle
        run "${line}" "${A}${a}${B}${b}" --element_labels ${A} ${B} --concentrations ${a} ${b} &
    done
}

# ── pure-element end points (bcc) ────────────────────────────────────────────
for el in Nb Zr Ti Mo Ta Hf; do
    throttle
    run pure "${el}" --element_labels ${el} --concentrations 100 &
done

# ── Nb–Zr: line of all opt champions (exp. max ~11 K at Nb75Zr25) ────────────
binary NbZr Nb Zr 90 80 70 60 50 40 30 20 10

# ── Nb–Ti: exp. max ~10 K at Nb65–75 ─────────────────────────────────────────
binary NbTi Nb Ti 90 80 70 60 50 40 30 20 10

# ── Mo–Re: second Matthias peak (exp. ~12 K at ~Mo60Re40); KKR gives ~0 so far ─
binary MoRe Mo Re 95 90 85 80 75 70 65 60 55

# ── Nb–Mo: Tc drops with Mo ──────────────────────────────────────────────────
binary NbMo Nb Mo 90 80 70 60 50 40 30 20 10

# ── Nb–Ta: isoelectronic, monotonic in exp.; Nb60Ta40 not k-converged at bz20 ─
binary NbTa Nb Ta 80 60 40 20

# ── Nb–Hf: bcc side only; exp. max ~9.5–10 K at Hf 20–25 ────────────────────
binary NbHf Nb Hf 90 80 70 60 50

wait
echo "=== All done. Results in: ${outputdir} ==="
