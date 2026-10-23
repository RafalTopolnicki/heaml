#!/bin/bash
# BCC / FCC / HCP total-energy comparison (scripts/structure_compare.py) for the pure elements that are
# missing in thermodynamic/elements.sra.ew0.6 (reference energies for dE / dG of V- and Sc-containing alloys):
#   V  (bcc ground state; new element, added to elements.py 2026-10-23)
#   Sc (hcp ground state; was in the 12-element set but never run through structure_compare)
# Optional: ELEMENTS="V Sc La Re" ./3_structure_compare_elements_VSc.sh   (Re: the existing run is broken —
# bcc scan not bracketing the minimum, hcp unconverged — and should be redone; La is missing as well).
# KKR settings = KKR_PARAMS_LATTICE defaults (sra, pbe, ew 0.6, bzqlty 10), identical to
# 3_structure_compare_mu0.2_champions.sh, so the numbers are directly comparable with the other elements
# (output: thermodynamic/elements.sra.ew0.6/<El>1.0000/{bcc,fcc,hcp}/ + structure_summary.json).
# Grid: 7 lattice points per structure, hcp c/a in {1.50 1.57 1.633 1.70 1.77} (defaults).
# The bcc E0 per atom of V is also the ebulk reference used in elements.py (−1898.622 Ry, from the
# 0_compare_with_known_V.sh pure-V run); compare the two as a consistency check.

set -uo pipefail
export OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-12}
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

outdir="thermodynamic"
ELEMENTS=${ELEMENTS:-"V Sc"}
NJOBS=${NJOBS:-2}

throttle() { while [ "$(jobs -rp | wc -l)" -ge "${NJOBS}" ]; do wait -n; done; }

for el in ${ELEMENTS}; do
    if [ -f "${outdir}/elements.sra.ew0.6/${el}1.0000/structure_summary.json" ] && [ "${FORCE:-0}" != "1" ]; then
        echo "[skip] ${el} (structure_summary.json exists; FORCE=1 to redo)"; continue
    fi
    throttle
    echo "[$(date +%F_%T)] start ${el}"
    ( python scripts/structure_compare.py \
          --element_labels ${el} \
          --concentrations 1.0 \
          --outdir "${outdir}/elements.sra.ew0.6" > "${outdir}/elements.sra.ew0.6/structure_compare_${el}.log" 2>&1 \
      || echo "[FAILED] ${el} (see ${outdir}/elements.sra.ew0.6/structure_compare_${el}.log)"
      echo "[$(date +%F_%T)] done  ${el}" ) &
done
wait
echo "=== done: ${ELEMENTS} -> ${outdir}/elements.sra.ew0.6 ==="
