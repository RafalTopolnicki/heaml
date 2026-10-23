#!/bin/bash
# OpenMP scaling test of specx on ONE exclusive Topola node: the same lattice-stage input is run with
# 28, 14, 7, 4, 2 and 1 threads (OMP_THREAD_LIMIT) and the wall time / CPU time are tabulated.
# Decides how many alloys to pack per core set (PER_TASK in submit_compare.sh, NJOBS in the opt runs).
#   sbatch slurm/topola_specx_scaling.sbatch
#   sbatch --export=ALL,INPUT=results/.../lattice/lattice_6.3.inp,THREADS="28 14 7" slurm/topola_specx_scaling.sbatch
# The input is copied to slurm/logs/specx_scaling_<jobid>/ and its `go` line is rewritten to a local
# potential file, so nothing in the running result directories is touched.  Output: a table at the end
# of slurm/logs/specx_scaling_<jobid>.out.
#SBATCH --job-name=specx_scaling
#SBATCH --account=g107-2962
#SBATCH --partition=topola
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclusive
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=slurm/logs/%x_%j.out
set -uo pipefail

ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}" || exit 1
SPECX="${ROOT}/cpa2002v010.potential2026/specx"
INPUT=${INPUT:-$(ls results/comparison.sra.ew0.6.bz10-10-20/V9.5Nb29.8Ta60.7/lattice/lattice_*.inp 2>/dev/null | head -1)}
THREADS=${THREADS:-"28 14 7 4 2 1"}
[ -x "${SPECX}" ] || { echo "specx not found: ${SPECX}"; exit 1; }
[ -n "${INPUT}" ] && [ -f "${INPUT}" ] || { echo "no input file (INPUT=${INPUT:-unset})"; exit 1; }

T="slurm/logs/specx_scaling_${SLURM_JOB_ID}"
mkdir -p "${T}"
sed -E 's#^go .*#go scaling.pot#' "${INPUT}" > "${T}/scaling.inp"   # fresh potential inside the test dir
echo "host $(hostname)  cpus ${SLURM_CPUS_PER_TASK}  input ${INPUT}"
lscpu | grep -E "Model name|^CPU\(s\)|Thread|Core|Socket" || true
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "${T}" || exit 1

printf "\n%-8s %-10s %-10s %-10s %s\n" threads wall_s cpu_s eff_cores specx_reports
for n in ${THREADS}; do
    rm -f scaling.pot fort.*
    export OMP_THREAD_LIMIT=${n}
    TIMEFORMAT='%R %U %S'
    { time "${SPECX}" < scaling.inp > "out_${n}.txt" 2>&1 ; } 2> "time_${n}.txt"
    read -r wall usr sys < "time_${n}.txt"
    rep=$(grep "elapsed time" "out_${n}.txt" | tail -1 | sed -E 's/.*\(\s*([0-9]+) threads\).*/\1 threads/')
    awk -v n="${n}" -v w="${wall}" -v u="${usr}" -v s="${sys}" -v r="${rep:-?}" \
        'BEGIN{printf "%-8s %-10.0f %-10.0f %-10.1f %s\n", n, w, u+s, (u+s)/w, r}'
done
echo "done; outputs in ${ROOT}/${T}"
