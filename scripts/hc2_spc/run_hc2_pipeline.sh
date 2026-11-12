#!/bin/bash
# Hc2 (WHH/Maki, dirty limit) for ONE system dir made by make_spc_inputs.py — full AkaiKKR A(k,E) chain:
#   scf -> fs rays (47 radial rays) -> fs_average (crossings) -> star x,y,z lines -> gradient line -> cone LSQ
#   -> fs_average --vstar (final: fs/fs_summary.json) -> pack/clean the big spc files.
# Same parameters as the pilot (DOCS/ROZMOWA_2026-11-03_Hc2.md §5–7): alloys, sub=fs (edelt 0.001), --n 0,
# --delta 0.1 for star/grad/cone.  Resumable: each finished step leaves <sysdir>/.hc2_steps/<step>.done.
# Final state in <sysdir>/hc2_status.txt: DONE | FAILED <step>.
#
#   scripts/hc2_spc/run_hc2_pipeline.sh hc2_runs/baseline.sra.ew0.6.bz10-10-20.gl/HfNbTa
# env: AKAIMODBIN (specx), PYTHON (default python3), DELTA (0.1), KEEP_SPC=1 (keep raw star/cone spc in the tar)
set -uo pipefail
SYS=${1:?usage: run_hc2_pipeline.sh <sysdir>}
HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PYTHON:-python3}
DELTA=${DELTA:-0.1}
export AKAIMODBIN=${AKAIMODBIN:-/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx}
[ -x "${AKAIMODBIN}" ] || { echo "no specx at ${AKAIMODBIN}"; exit 1; }
[ -f "${SYS}/scf.inp" ] || { echo "no ${SYS}/scf.inp (run make_spc_inputs.py first)"; exit 1; }
SYS=$(cd "${SYS}" && pwd)
ST="${SYS}/.hc2_steps"
mkdir -p "${ST}"
LOG="${SYS}/hc2_pipeline.log"
POT=$(awk '/^go /{print $2; exit}' "${SYS}/scf.inp")

log() { echo "[$(date +%F_%T)] $*" | tee -a "${LOG}"; }
fail() { log "FAILED $1"; echo "FAILED $1" > "${SYS}/hc2_status.txt"; exit 1; }
step() {   # step <name> <command...>   (skipped when .done exists)
    local name=$1; shift
    if [ -f "${ST}/${name}.done" ]; then return 0; fi
    log "start ${name}"
    local t0=${SECONDS}
    ( "$@" ) >> "${LOG}" 2>&1 || fail "${name}"
    touch "${ST}/${name}.done"
    log "done  ${name} ($((SECONDS - t0)) s)"
}

scf() {
    cd "${SYS}"
    "${AKAIMODBIN}" < scf.inp > scf.out
    [ -f "${POT}" ] || { echo "no potential ${POT}"; return 1; }
    if grep -q "itr=499" scf.out || grep -qi "no convergence" scf.out; then echo "SCF not converged"; return 1; fi
    grep -q "def=" scf.out || { echo "no Lloyd DOS (def=) in scf.out"; return 1; }
}

echo "RUNNING" > "${SYS}/hc2_status.txt"
log "system ${SYS}  specx ${AKAIMODBIN}  python $(command -v "${PY}")  delta ${DELTA}"
step scf          scf
step fs_make      "${PY}" "${HERE}/make_fs_rays.py" "${SYS}"
step fs_run       bash "${SYS}/fs/run_fs.sh"
step fs_avg       "${PY}" "${HERE}/fs_average.py" "${SYS}" --sub fs
[ -f "${SYS}/fs/fs_summary_normals.json" ] || cp "${SYS}/fs/fs_summary.json" "${SYS}/fs/fs_summary_normals.json"
step star_make    "${PY}" "${HERE}/diag_star.py" make "${SYS}" --sub fs --n 0 --delta "${DELTA}"
step star_run     bash "${SYS}/star/run_star.sh"
step star_an      "${PY}" "${HERE}/diag_star.py" analyse "${SYS}" --sub fs
step grad_make    "${PY}" "${HERE}/diag_star.py" make_grad "${SYS}" --sub fs --delta "${DELTA}"
step grad_run     bash "${SYS}/star/run_grad.sh"
step grad_an      "${PY}" "${HERE}/diag_star.py" analyse_grad "${SYS}" --sub fs
step cone_make    "${PY}" "${HERE}/diag_star.py" make_cone "${SYS}" --sub fs --delta "${DELTA}"
step cone_run     bash "${SYS}/star/run_cone.sh"
step cone_an      "${PY}" "${HERE}/diag_star.py" analyse_cone "${SYS}" --sub fs
step final        "${PY}" "${HERE}/fs_average.py" "${SYS}" --sub fs --vstar

pack() {   # few inodes / little disk on Topola: big spc files -> tar.gz (fs) or deleted (star/cone, unless KEEP_SPC=1)
    cd "${SYS}"
    tar -czf fs/fs_runs.tar.gz fs/fs_[0-9]*.inp fs/fs_[0-9]*.out fs/fs_[0-9]*_up.spc fs/run_fs.sh
    rm -f fs/fs_[0-9]*.inp fs/fs_[0-9]*.out fs/fs_[0-9]*_up.spc
    if [ "${KEEP_SPC:-0}" = 1 ]; then
        tar -czf star/star_runs.tar.gz star/*_[0-9]*.inp star/*_[0-9]*.out star/*_up.spc star/run_*.sh
    else
        tar -czf star/star_runs.tar.gz star/*_[0-9]*.inp star/*_[0-9]*.out star/run_*.sh
    fi
    rm -f star/*_[0-9]*.inp star/*_[0-9]*.out star/*_up.spc
    tar -czf scf_aux.tar.gz fort.* "${POT}.info" 2>/dev/null && rm -f fort.* "${POT}.info"
    rm -f ./*_dn.spc fs/*_dn.spc
}
step pack pack
echo "DONE" > "${SYS}/hc2_status.txt"
log "DONE ${SYS}"
