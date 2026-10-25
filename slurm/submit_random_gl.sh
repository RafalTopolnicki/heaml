#!/bin/bash
# Submit the new TiNbZrHfTaMoWYReRu random dataset (full pipeline, variant-independent: eta is recomputed in GL by
# scripts/make_slim_dataset.py afterwards) as job arrays of slurm/topola_random_array.sbatch, one array per family:
#   full simplex (k=10)            : 800 samples, seed 2000   (old set used seeds 100..1099 -> disjoint)
#   k-faces, k = 2..9              : top-up to 100 per k      (old faces used 20000+1000k+i -> new base 40000+1000k)
# Edit FAMILIES below to change counts.  Every array task computes PER_TASK compositions concurrently on 14 cpus
# (10-component alloys are ~2x a 5-component one; PER_TASK=4 for k>=7, 5 otherwise), 12 h limit per task.
#   slurm/submit_random_gl.sh                 # submit everything (MAXPAR tasks of each array at once, default 15)
#   env CALIB=1 slurm/submit_random_gl.sh     # calibration: only task 1 of the k=10, k=5 and k=2 arrays
#   env MAXPAR=30 slurm/submit_random_gl.sh   # tcsh: env VAR=x cmd
set -euo pipefail
ROOT=/lu/topola/home/rtop/heaml
cd "${ROOT}"
mkdir -p slurm/logs
WORKDIR=${WORKDIR:-random.ratios/sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl}
ELEMENTS="Ti,Nb,Zr,Hf,Ta,Mo,W,Y,Re,Ru"
MAXPAR=${MAXPAR:-15}
# family : N_NONZERO (empty = full simplex) : NSAMPLES : SEED : PER_TASK
FAMILIES=(
  "full::800:2000:4"
  "k9:9:100:49000:4"
  "k8:8:100:48000:4"
  "k7:7:100:47000:4"
  "k6:6:68:46000:5"
  "k5:5:45:45000:5"
  "k4:4:30:44000:5"
  "k3:3:25:43000:5"
  "k2:2:40:42000:5"
)
total=0
for fam in "${FAMILIES[@]}"; do
    IFS=: read -r name k n seed per <<< "${fam}"
    ntask=$(( (n + per - 1) / per )); total=$(( total + n ))
    arr="1-${ntask}%${MAXPAR}"
    if [ -n "${CALIB:-}" ]; then
        case "${name}" in full|k5|k2) arr="1" ;; *) continue ;; esac
    fi
    printf "%-5s k=%-4s %4d samples  seed %-6s %d/task -> %3d tasks  array %s\n" "${name}" "${k:-10}" "${n}" "${seed}" "${per}" "${ntask}" "${arr}"
    sbatch --job-name="rnd_${name}" --array="${arr}" ${ACCOUNT:+--account="${ACCOUNT}"} \
        --export=ALL,WORKDIR="${WORKDIR}",ELEMENTS="${ELEMENTS}",SEED="${seed}",NSAMPLES="${n}",PER_TASK="${per}",N_NONZERO="${k}",MIN_FRAC=0.05 \
        slurm/topola_random_array.sbatch
done
echo "total ${total} compositions -> ${WORKDIR}"
