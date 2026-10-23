#!/bin/bash
# Literature benchmark rerun at the production k-point setting of the ML data:
# bzqlty lattice 10 / Debye 10 / final SCF 20 (consts.py defaults, no --bzqlty flag).
# The old results/comparison.sra.ew0.6 ran with --bzqlty 10, which (old process_hea default) forced the
# final SCF to 10 as well; eta/lambda from bz10 are not converged (see validation.ratios/*.bz10-10-*).
# Generated 2026-10-22 from 0_compare_with_known.sh: ALL entries (also the ones commented out there,
# which were computed earlier at bz10). Entries with an existing results.json are skipped -> safe to re-run.
# Lattice and Debye are identical to the old run (bzqlty 10 both), so they are NOT recomputed:
# --reuse_from copies lattice/ and debye/ from ${olddir}/<name> (params are checked) and only the final SCF
# (bzqlty 20) + McMillan run (~10% of the full cost). The old directory is left untouched.

NJOBS=${NJOBS:-16}                              # parallel alloys; on SLURM set from cpus (see slurm/)
export OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-12}  # OpenMP threads per specx run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ew=0.6
params="--overwrite_params --ew ${ew} --rel sra --debye_mode kkr"
outputdir="results/comparison.sra.ew${ew}.bz10-10-20"
olddir="results/comparison.sra.ew${ew}"   # bz10 run: source of lattice/ and debye/

run() {  # $1 = workdir name, rest = process_hea.py arguments
    local name=$1; shift
    local wd="${outputdir}/${name}"
    if [ -n "${ONLY:-}" ]; then case " ${ONLY} " in *" ${name} "*) ;; *) return ;; esac; fi   # ONLY="a b c": only these entries (slurm array)
    if [ -f "${wd}/results.json" ]; then [ "${LIST:-0}" = "1" ] || echo "[skip] ${name}"; return; fi
    if [ "${LIST:-0}" = "1" ]; then echo "${name}"; return; fi   # LIST=1: print pending entries, run nothing (slurm/)
    mkdir -p "${wd}"
    echo "[$(date +%F_%T)] start ${name}"
    local reuse=()
    if [ -f "${olddir}/${name}/run_params.json" ] && [ -f "${olddir}/${name}/lattice/lattice_summary.txt" ] \
       && [ -f "${olddir}/${name}/debye/debye_summary.json" ]; then
        reuse=(--reuse_from "${olddir}/${name}")
    else
        echo "[full] ${name}: no lattice/Debye in ${olddir}, computing everything"
    fi
    python scripts/process_hea.py "$@" --workdir "${wd}" ${params} "${reuse[@]}" > "${wd}/process_hea.log" 2>&1 \
        || echo "[FAILED] ${name} (see ${wd}/process_hea.log)"
    echo "[$(date +%F_%T)] done  ${name}"
}

throttle() {  # keep at most NJOBS background jobs
    while [ "$(jobs -rp | wc -l)" -ge "${NJOBS}" ]; do wait -n; done
}


# Jaskiewicz 2019
throttle; run TaNb67HfZrTi33 --element_labels Ta Nb Hf Zr Ti --concentrations 0.335 0.335 0.11 0.11 0.11 &

# Pure elements
throttle; run Ta --element_labels Ta --concentrations 100 &
throttle; run Nb --element_labels Nb --concentrations 100 &
throttle; run Hf --element_labels Hf --concentrations 100 &
throttle; run Zr --element_labels Zr --concentrations 100 &
throttle; run Ti --element_labels Ti --concentrations 100 &
throttle; run Sc --element_labels Sc --concentrations 100 &
throttle; run W --element_labels W --concentrations 100 &
throttle; run Y --element_labels Y --concentrations 100 &
throttle; run La --element_labels La --concentrations 100 &
throttle; run Mo --element_labels Mo --concentrations 100 &
throttle; run Re --element_labels Re --concentrations 100 &
throttle; run Ru --element_labels Ru --concentrations 100 &





## Jaskiewicz 2016
throttle; run Ta34Nb33Hf8Zr14Ti11 --element_labels Ta Nb Hf Zr Ti --concentrations 34 33 8 14 11 &


## https://journals.aps.org/prb/pdf/10.1103/fnz7-bzmw
throttle; run NbHf75TiZr25 --element_labels Nb Hf Ti Zr --concentrations 37.5 37.5 12.5 12.5 &
throttle; run NbZr75TiHf25 --element_labels Nb Zr Ti Hf --concentrations 37.5 37.5 12.5 12.5 &
throttle; run NbTi75ZrHf25 --element_labels Nb Ti Zr Hf --concentrations 37.5 37.5 12.5 12.5 &

## https://www.mdpi.com/1996-1944/16/17/5814
throttle; run Nb34Ti33Zr14Ta11Hf8 --element_labels Nb Ti Zr Ta Hf --concentrations 34 33 14 11 8 &

##https://journals.aps.org/prb/abstract/10.1103/PhysRevB.106.184512
throttle; run NbTa67MoHfW33 --element_labels Nb Ta Mo Hf W --concentrations 33.5 33.5 11 11 11 &


## https://doi.org/10.3390/ma15031122 — Sc-containing quaternary/quinary/senary BCC HEAs
throttle; run ScHfNbTi --element_labels Sc Hf Nb Ti --concentrations 20 25 30 25 &
throttle; run ScHfNbZr --element_labels Sc Hf Nb Zr --concentrations 21 26 28 25 &
throttle; run ScHfTaTi --element_labels Sc Hf Ta Ti --concentrations 12 24 38 26 &
throttle; run ScNbTiZr --element_labels Sc Nb Ti Zr --concentrations 18 32 24 26 &
throttle; run ScTaTiZr --element_labels Sc Ta Ti Zr --concentrations 4 64 22 10 &
throttle; run ScHfNbTaTi --element_labels Sc Hf Nb Ta Ti --concentrations 8 22 25 25 20 &
throttle; run ScHfNbTaZr_bccL --element_labels Sc Hf Nb Ta Zr --concentrations 9 21 26 26 18 &
throttle; run ScHfNbTaZr_bccS --element_labels Sc Hf Nb Ta Zr --concentrations 15 22 22 20 21 &
throttle; run ScNbTaTiZr --element_labels Sc Nb Ta Ti Zr --concentrations 4 29 41 15 11 &
throttle; run ScHfNbTaTiZr --element_labels Sc Hf Nb Ta Ti Zr --concentrations 12 17 19 19 17 16 &


## https://doi.org/10.1103/PhysRevMaterials.2.034801 — Y/Mo substitution in TaNbHfZrTi
throttle; run NbY0.33Mo0.67HfZrTi --element_labels Ti Nb Zr Hf Mo Y --concentrations 11 33.5 11 11 22.445 11.055 &
throttle; run TaY0.33Mo0.67HfZrTi --element_labels Ti Zr Hf Ta Mo Y --concentrations 11 11 11 33.5 22.445 11.055 &
throttle; run TaNbY0.67Mo0.33ZrTi --element_labels Ti Nb Zr Ta Mo Y --concentrations 11 33.5 11 33.5 3.63 7.37 &
throttle; run TaNbHfY0.67Mo0.33Ti --element_labels Ti Nb Hf Ta Mo Y --concentrations 11 33.5 11 33.5 3.63 7.37 &
throttle; run TaNbHfZrY0.67Mo0.33 --element_labels Nb Zr Hf Ta Mo Y --concentrations 33.5 11 11 33.5 3.63 7.37 &
throttle; run TaNbHfZrTi_PRMat2018 --element_labels Ti Nb Zr Hf Ta --concentrations 11 33.5 11 11 33.5 &


## ── NEW ENTRIES FROM bcc_hea_superconducting_tc_summary.xls ─────────────────


## Marik 2018, J. Alloys Compd. 769, 1059 | doi:10.1016/j.jallcom.2018.08.039
## Nb20Re20Zr20Hf20Ti20 | Tc=5.3 K (bulk, transport+magnetization+heat capacity) | equiatomic 5-element BCC
throttle; run NbReZrHfTi --element_labels Nb Re Zr Hf Ti --concentrations 20 20 20 20 20 &
#
#
### Kim 2020, Acta Mater. 186, 250 | doi:10.1016/j.actamat.2020.01.007
## Ta(1/6)Nb(2/6)Hf(1/6)Zr(1/6)Ti(1/6) | Tc=7.9 K (resistive onset), 7.8 K (zero resistance) | single BCC phase
throttle; run TaNb2HfZrTi --element_labels Ta Nb Hf Zr Ti --concentrations 16.6667 33.3333 16.6667 16.6667 16.6667 &
#
#
### Kitagawa 2022, J. Alloys Compd. 924, 166473 | doi:10.1016/j.jallcom.2022.166473
## HfMoNbTiZr equiatomic | Tc=4.1 K (bulk) | single BCC phase
throttle; run HfMoNbTiZr --element_labels Hf Mo Nb Ti Zr --concentrations 20 20 20 20 20 &
#
#
### Zeng 2023, Adv. Quantum Technol. 6, 2300213 | doi:10.1002/qute.202300213
## TiHfNbTaMo equiatomic | Tc=3.42 K (experimental) | single homogeneous BCC phase (Im-3m), a=3.445 A
throttle; run TiHfNbTaMo --element_labels Ti Hf Nb Ta Mo --concentrations 20 20 20 20 20 &
#
#
## ── ACTIVATED 2026-10-19: entries below (Motla … von Rohr 2018 Sc/Mo) were never computed ──
### Motla 2022, Phys. Rev. B 105, 144501 | doi:10.1103/PhysRevB.105.144501
## Nb60Re10Zr10Hf10Ti10 | Tc=5.7 K (bulk, muSR+resistivity+magnetization) | single centrosymmetric BCC (Im-3m)
## CAVEAT: Nb=60 at.% is outside strict HEA definition (5-35 at.% per element); included for methodology testing
throttle; run Nb60Re10Zr10Hf10Ti10 --element_labels Nb Re Zr Hf Ti --concentrations 60 10 10 10 10 &
#
#
### Chakrabarty 2025, J. Appl. Phys. 137, 215901 | doi:10.1063/5.0265943
## (NbTa)0.55(HfTiZr)0.45 | Tc=7.2 K ambient pressure (max 10.1 K at 23.6 GPa) | single-phase BCC
throttle; run NbTa55HfTiZr45 --element_labels Nb Ta Hf Ti Zr --concentrations 27.5 27.5 15 15 15 &


## Hattori 2023, J. Alloys Metall. Syst. 3, 100020 | arXiv:2307.01958
## Ti-Hf-Nb-Ta-Re composition series | Tc from specific heat
## RE-001: Ti35Hf25Nb25Ta5Re10 | Tc=3.95 K | single BCC phase
throttle; run Ti35Hf25Nb25Ta5Re10 --element_labels Ti Hf Nb Ta Re --concentrations 35 25 25 5 10 &
## RE-002: Ti30Hf20Nb35Ta5Re10 | Tc=4.38 K | CAVEAT: weak segregation into two BCC phases
throttle; run Ti30Hf20Nb35Ta5Re10 --element_labels Ti Hf Nb Ta Re --concentrations 30 20 35 5 10 &
## RE-003: Ti25Hf15Nb35Ta15Re10 | Tc=4.1 K | CAVEAT: weak segregation into two BCC phases
throttle; run Ti25Hf15Nb35Ta15Re10 --element_labels Ti Hf Nb Ta Re --concentrations 25 15 35 15 10 &
## RE-004: Ti20Hf10Nb35Ta25Re10 | Tc=3.62 K | CAVEAT: two compositionally distinct BCC phases
throttle; run Ti20Hf10Nb35Ta25Re10 --element_labels Ti Hf Nb Ta Re --concentrations 20 10 35 25 10 &
## RE-005: Ti15Hf5Nb35Ta35Re10 | Tc=3.25 K | CAVEAT: two compositionally distinct BCC phases
throttle; run Ti15Hf5Nb35Ta35Re10 --element_labels Ti Hf Nb Ta Re --concentrations 15 5 35 35 10 &


## von Rohr 2018, Phys. Rev. Mater. 2, 034801 | doi:10.1103/PhysRevMaterials.2.034801
## Sc/Mo isoelectronic substitutions in TaNbHfZrTi (same paper as Y/Mo above)
# SUB-006: [{Sc0.33Mo0.67}Nb]0.67(HfZrTi)0.33 — Sc0.33Mo0.67 replaces Ta | Tc=4.4 K
throttle; run NbSc0.33Mo0.67HfZrTi --element_labels Sc Mo Nb Hf Zr Ti --concentrations 11.055 22.445 33.5 11 11 11 &
## SUB-007: [Ta{Sc0.33Mo0.67}]0.67(HfZrTi)0.33 — Sc0.33Mo0.67 replaces Nb | Tc=2.9 K
throttle; run TaSc0.33Mo0.67HfZrTi --element_labels Ta Sc Mo Hf Zr Ti --concentrations 33.5 11.055 22.445 11 11 11 &
## SUB-008: [TaNb]0.67({Sc0.67Mo0.33}ZrTi)0.33 — Sc0.67Mo0.33 replaces Hf | Tc=7.5 K
throttle; run TaNbSc0.67Mo0.33ZrTi --element_labels Ta Nb Sc Mo Zr Ti --concentrations 33.5 33.5 7.37 3.63 11 11 &
## SUB-009: [TaNb]0.67(Hf{Sc0.67Mo0.33}Ti)0.33 — Sc0.67Mo0.33 replaces Zr | Tc=6.6 K
throttle; run TaNbHfSc0.67Mo0.33Ti --element_labels Ta Nb Hf Sc Mo Ti --concentrations 33.5 33.5 11 7.37 3.63 11 &
## SUB-010: [TaNb]0.67(HfZr{Sc0.67Mo0.33})0.33 — Sc0.67Mo0.33 replaces Ti | Tc=7.5 K
throttle; run TaNbHfZrSc0.67Mo0.33 --element_labels Ta Nb Hf Zr Sc Mo --concentrations 33.5 33.5 11 11 7.37 3.63 &


## ── FROM: "High-Entropy Superconducting Materials" (IntechOpen review) ────────


## von Rohr et al. 2016, PNAS 113, E7144 — [TaNb]x[ZrHfTi]1-x composition series
## x=0.67 equiatomic already in script above (Jaskiewicz / TaNbHfZrTi_PRMat2018)
## x=0.70 | Tc=8.03 K | single BCC phase
throttle; run TaNb70ZrHfTi30 --element_labels Ta Nb Zr Hf Ti --concentrations 35 35 10 10 10 &
## x=0.60 | Tc=7.56 K | single BCC phase
throttle; run TaNb60ZrHfTi40 --element_labels Ta Nb Zr Hf Ti --concentrations 30 30 13.3333 13.3333 13.3333 &
## x=0.50 | Tc=6.46 K | single BCC phase
throttle; run TaNb50ZrHfTi50 --element_labels Ta Nb Zr Hf Ti --concentrations 25 25 16.6667 16.6667 16.6667 &
## x=0.16 | Tc=4.52 K | single BCC phase (Ti/Zr/Hf-rich extreme)
throttle; run TaNb16ZrHfTi84 --element_labels Ta Nb Zr Hf Ti --concentrations 8 8 28 28 28 &
#
#
### Wu et al. 2018, Natural Science 10, 110
## Ta20Nb20Hf20Zr20Ti20 equimolar | Tc=7.12 K | single BCC phase
throttle; run TaNbHfZrTi_equimolar --element_labels Ta Nb Hf Zr Ti --concentrations 20 20 20 20 20 &


### Sobota et al. 2022, Phys. Rev. B 106, 184512 | doi:10.1103/PhysRevB.106.184512
## (NbTa)0.67(MoHfW)0.33 | Tc=4.3 K | single BCC phase
# (duplicate of NbTa67MoHfW33, already queued)


## Krnel et al. 2022, Materials 15, 1122 | doi:10.3390/ma15031122
## Equimolar Sc-substituted pentenary series (Sc replaces one element of TaNbHfZrTi)
## Ta20Nb20Hf20Sc20Ti20 (Zr→Sc) | Tc=6.60 K | Bc2(0)=13.1 T | BCC
throttle; run TaNbHfScTi --element_labels Ta Nb Hf Sc Ti --concentrations 20 20 20 20 20 &
## Ta20Nb20Hf20Zr20Sc20 (Ti→Sc) | Tc=7.40 K (ρ, Krnel Table 4; 7.70 was wrong) | Bc2(0)=12.4 T | BCC
throttle; run TaNbHfZrSc --element_labels Ta Nb Hf Zr Sc --concentrations 20 20 20 20 20 &
## Ta20Nb20Sc20Zr20Ti20 (Hf→Sc) | Tc=7.90 K | Bc2(0)=19.3 T | BCC
throttle; run TaNbScZrTi --element_labels Ta Nb Sc Zr Ti --concentrations 20 20 20 20 20 &
## Ta16.67Nb16.67Hf16.67Zr16.67Ti16.67Sc16.67 equimolar senary | Tc=7.20 K | Bc2(0)=14.1 T | BCC
throttle; run TaNbHfZrTiSc --element_labels Ta Nb Hf Zr Ti Sc --concentrations 16.6667 16.6667 16.6667 16.6667 16.6667 16.6667 &

## ════════════════════════════════════════════════════════════════════════════
## ADDED 2026-10-19 — HEAs from results/Tc_literature.xlsx without KKR results
## Matched by composition against results/comparison.sra.ew0.6/*/results.json.
## Identical compositions reported by several papers are computed once (all rows listed).
## ════════════════════════════════════════════════════════════════════════════

## ── single-phase bcc candidates ──
## xlsx row 56: TaNb80ZrHfTi20 (von Rohr, 10.1073/pnas.1615926113, Tc=7.55 K, FIG-VEC (Fig. 4 marker coordinates; ±0.02 K), candidate)
throttle; run TaNb80ZrHfTi20 --element_labels Ti Zr Nb Hf Ta --concentrations 6.6667 6.6667 40 6.6667 40 &
## xlsx row 57: TaNb75ZrHfTi25 (von Rohr, 10.1073/pnas.1615926113, Tc=7.79 K, FIG-VEC (Fig. 4 marker coordinates; ±0.02 K), candidate)
throttle; run TaNb75ZrHfTi25 --element_labels Ti Zr Nb Hf Ta --concentrations 8.3333 8.3333 37.5 8.3333 37.5 &
## xlsx row 58: TaNb65ZrHfTi35 (von Rohr, 10.1073/pnas.1615926113, Tc=7.48 K, FIG-VEC (Fig. 4 marker coordinates; ±0.02 K), candidate)
throttle; run TaNb65ZrHfTi35 --element_labels Ti Zr Nb Hf Ta --concentrations 11.6667 11.6667 32.5 11.6667 32.5 &
## xlsx row 61: TaNb30ZrHfTi70 (von Rohr, 10.1073/pnas.1615926113, Tc=5.2 K, FIG-VEC (Fig. 4 marker coordinates; ±0.02 K), candidate)
throttle; run TaNb30ZrHfTi70 --element_labels Ti Zr Nb Hf Ta --concentrations 23.3333 23.3333 15 23.3333 15 &
## xlsx row 62: TaNb20ZrHfTi80 (von Rohr, 10.1073/pnas.1615926113, Tc=4.64 K, FIG-VEC (Fig. 4 marker coordinates; ±0.02 K), candidate)
throttle; run TaNb20ZrHfTi80 --element_labels Ti Zr Nb Hf Ta --concentrations 26.6667 26.6667 10 26.6667 10 &
## xlsx row 63, 101: Nb67HfZrTi33 (von Rohr, 10.1073/pnas.1615926113, Tc=9.36 K, FIG-VEC (Fig. 6 M(T) curve points; method check on x=0.33 gives +0.06 K vs Fig. 4), candidate); Nb67TiZrHf33_MA (Sobota, 10.3390/ma19112321, Tc=7 K, TAB (verified, PMC), caution – mechanically alloyed)
throttle; run Nb67HfZrTi33 --element_labels Ti Zr Nb Hf --concentrations 11 11 67 11 &
## xlsx row 64: TaNb67Hf33 (von Rohr, 10.1073/pnas.1615926113, Tc=7.42 K, FIG-VEC (Fig. 6 M(T) curve points; method check on x=0.33 gives +0.06 K vs Fig. 4), candidate)
throttle; run TaNb67Hf33 --element_labels Nb Hf Ta --concentrations 33.5 33 33.5 &
## xlsx row 65: TaNb67HfZr33 (von Rohr, 10.1073/pnas.1615926113, Tc=7.07 K, FIG-VEC (Fig. 6 M(T) curve points; method check on x=0.33 gives +0.06 K vs Fig. 4), candidate)
throttle; run TaNb67HfZr33 --element_labels Zr Nb Hf Ta --concentrations 16.5 33.5 16.5 33.5 &
## xlsx row 66, 92: NbTiZr (Senga, 10.1088/1361-6668/ae37e4, Tc=7.98 K, TAB (verified, arXiv 2602.19376), candidate); TiZrNb_Kosuth (Košuth, 10.1063/10.0027922, Tc=8.2 K, ABS, candidate)
throttle; run NbTiZr --element_labels Ti Zr Nb --concentrations 33.333 33.333 33.333 &
## xlsx row 67: HfNbTi (Senga, 10.1088/1361-6668/ae37e4, Tc=6.47 K, TAB (verified, arXiv 2602.19376), candidate)
throttle; run HfNbTi --element_labels Ti Nb Hf --concentrations 33.333 33.333 33.333 &
## xlsx row 68: HfNbZr (Senga, 10.1088/1361-6668/ae37e4, Tc=7.07 K, TAB (verified, arXiv 2602.19376), candidate)
throttle; run HfNbZr --element_labels Zr Nb Hf --concentrations 33.333 33.333 33.333 &
## xlsx row 69: HfNbTa (Senga, 10.1088/1361-6668/ae37e4, Tc=7.11 K, TAB (verified, arXiv 2602.19376), candidate)
throttle; run HfNbTa --element_labels Nb Hf Ta --concentrations 33.333 33.333 33.333 &
## xlsx row 70: (TiHfTa)1-xNbx_x0 (Li, 10.1088/1361-6668/ad9fa8, Tc=5.31 K, TAB (verified, arXiv 2501.02455), candidate)
throttle; run TiHfTa1-xNbx_x0 --element_labels Ti Hf Ta --concentrations 33.3333 33.3333 33.3333 &
## xlsx row 71: (TiHfTa)1-xNbx_x0.035 (Li, 10.1088/1361-6668/ad9fa8, Tc=5.61 K, TAB (verified, arXiv 2501.02455), candidate)
throttle; run TiHfTa1-xNbx_x0.035 --element_labels Ti Nb Hf Ta --concentrations 32.1667 3.5 32.1667 32.1667 &
## xlsx row 72: (TiHfTa)1-xNbx_x0.125 (Li, 10.1088/1361-6668/ad9fa8, Tc=5.82 K, TAB (verified, arXiv 2501.02455), candidate)
throttle; run TiHfTa1-xNbx_x0.125 --element_labels Ti Nb Hf Ta --concentrations 29.1667 12.5 29.1667 29.1667 &
## xlsx row 73: (TiHfTa)1-xNbx_x0.33 (Li, 10.1088/1361-6668/ad9fa8, Tc=7.06 K, TAB (verified, arXiv 2501.02455), candidate)
throttle; run TiHfTa1-xNbx_x0.33 --element_labels Ti Nb Hf Ta --concentrations 22.3333 33 22.3333 22.3333 &
## xlsx row 74: (TiHfTa)1-xNbx_x0.44 (Li, 10.1088/1361-6668/ad9fa8, Tc=7.41 K, TAB (verified, arXiv 2501.02455), candidate)
throttle; run TiHfTa1-xNbx_x0.44 --element_labels Ti Nb Hf Ta --concentrations 18.6667 44 18.6667 18.6667 &
## xlsx row 75: (TiHfTa)1-xNbx_x0.57 (Li, 10.1088/1361-6668/ad9fa8, Tc=8.19 K, TAB (verified, arXiv 2501.02455), candidate)
throttle; run TiHfTa1-xNbx_x0.57 --element_labels Ti Nb Hf Ta --concentrations 14.3333 57 14.3333 14.3333 &
## xlsx row 76: (TiHfTa)1-xNbx_x0.75 (Li, 10.1088/1361-6668/ad9fa8, Tc=8.82 K, TAB (verified, arXiv 2501.02455), candidate)
throttle; run TiHfTa1-xNbx_x0.75 --element_labels Ti Nb Hf Ta --concentrations 8.3333 75 8.3333 8.3333 &
## xlsx row 77: (TiHfTa)1-xNbx_x0.9 (Li, 10.1088/1361-6668/ad9fa8, Tc=9.11 K, TAB (verified, arXiv 2501.02455), candidate)
throttle; run TiHfTa1-xNbx_x0.9 --element_labels Ti Nb Hf Ta --concentrations 3.3333 90 3.3333 3.3333 &
## xlsx row 78: Ti25Hf25Nb25Ta25 (Zeng, 10.1007/s11433-023-2113-6, Tc=6.3 K, TAB (verified, arXiv 2308.14612), candidate)
throttle; run Ti25Hf25Nb25Ta25 --element_labels Ti Nb Hf Ta --concentrations 25 25 25 25 &
## xlsx row 79: Ti15Zr15Nb35Ta35 (Yuan, 10.3389/fmats.2018.00072, Tc=8 K, TAB (verified, full text), candidate)
throttle; run Ti15Zr15Nb35Ta35 --element_labels Ti Zr Nb Ta --concentrations 15 15 35 35 &
## xlsx row 80, 82: Nb25Ta25Ti25Zr25 (Sharma, 10.1103/8m3w-xc5x, Tc=7.75 K, TAB (verified, arXiv 2508.19584), candidate); TaNbZrTi_Krishnan (Krishnan, 10.1016/j.jallcom.2025.181813, Tc=8.12 K, ABS, candidate)
throttle; run Nb25Ta25Ti25Zr25 --element_labels Ti Zr Nb Ta --concentrations 25 25 25 25 &
## xlsx row 83, 84: TaNbHfZr_Krishnan (Krishnan, 10.1016/j.jallcom.2025.181813, Tc=7.64 K, ABS (Tc); SEC (Cp data), caution – θD suspicious); TaNbHfZr_Wu2024 (Wu, 10.1103/PhysRevLett.132.166002, Tc=8.1 K, SEC, candidate)
throttle; run TaNbHfZr --element_labels Zr Nb Hf Ta --concentrations 25 25 25 25 &
## xlsx row 85: Nb40Hf20Zr20Ti20 (Hidayati, 10.1016/j.actamat.2023.119420, Tc=7.99 K, SEC, candidate)
throttle; run Nb40Hf20Zr20Ti20 --element_labels Ti Zr Nb Hf --concentrations 20 20 40 20 &
## xlsx row 86: Ti50ZrNbHfTa50 (Sobota, 10.1016/j.actamat.2024.120666, Tc=6 K, SEC (PDF in papers_data/, not re-read), candidate)
throttle; run Ti50ZrNbHfTa50 --element_labels Ti Zr Nb Hf Ta --concentrations 50 12.5 12.5 12.5 12.5 &
## xlsx row 87: Ta20Nb21Hf20Zr20Ti19 (Vrtnik, 10.1016/j.jallcom.2016.11.417, Tc=6 K, SEC (two compilations agree), candidate)
throttle; run Ta20Nb21Hf20Zr20Ti19 --element_labels Ti Zr Nb Hf Ta --concentrations 19 20 21 20 20 &
## xlsx row 88: TaNb70HfZrTi50_Gao (Gao, 10.1063/5.0083365, Tc=7.13 K, SEC, candidate)
throttle; run TaNb70HfZrTi50_Gao --element_labels Ti Zr Nb Hf Ta --concentrations 13.8889 13.8889 29.1667 13.8889 29.1667 &
## xlsx row 89, 90, 91: HfNbZrTi_crystal (Sang, 10.1088/1361-6668/add097, Tc=6 K, SEC / ABS, caution – θD/γ suspicious); TiZrHfNb_Uporov (Uporov, 10.1016/j.intermet.2021.107394, Tc=6.3 K, SEC, candidate); TiZrNbHf_Kosuth (Košuth, 10.1063/10.0027922, Tc=6.3 K, ABS, candidate)
throttle; run TiZrHfNb --element_labels Ti Zr Nb Hf --concentrations 25 25 25 25 &
## xlsx row 93: Hf35Nb10Ta35Mo10W10 (Krishnan, 10.1088/1361-6668/ae01e7, Tc=4.06 K, SEC, candidate)
throttle; run Hf35Nb10Ta35Mo10W10 --element_labels Nb Mo Hf Ta W --concentrations 10 10 35 35 10 &
## xlsx row 94: Hf35Nb22Ta23Mo10W10 (Krishnan, 10.1088/1361-6668/ae01e7, Tc=4.82 K, SEC, candidate)
throttle; run Hf35Nb22Ta23Mo10W10 --element_labels Nb Mo Hf Ta W --concentrations 22 10 35 23 10 &
## xlsx row 95: Hf35Nb35Ta10Mo10W10 (Krishnan, 10.1088/1361-6668/ae01e7, Tc=5.91 K, SEC, candidate)
throttle; run Hf35Nb35Ta10Mo10W10 --element_labels Nb Mo Hf Ta W --concentrations 35 10 35 10 10 &
## xlsx row 96: Nb50Ti25W25 (Li, 10.1088/0256-307X/42/1/017401, Tc=4.2 K, TAB (verified, arXiv 2412.08361), candidate)
throttle; run Nb50Ti25W25 --element_labels Ti Nb W --concentrations 25 50 25 &
## xlsx row 97: Nb50Ti25Mo25 (Li, 10.1088/0256-307X/42/1/017401, Tc=2.8 K, TAB (verified, arXiv 2412.08361), candidate)
throttle; run Nb50Ti25Mo25 --element_labels Ti Nb Mo --concentrations 25 50 25 &
## xlsx row 98: Nb57Re10HfZrTi33 (Maran, arXiv:2607.23056, Tc=5.42 K, TAB (preprint), candidate – preprint)
throttle; run Nb57Re10HfZrTi33 --element_labels Ti Zr Nb Hf Re --concentrations 11 11 57 11 10 &
## xlsx row 99: Nb47Re20HfZrTi33 (Maran, arXiv:2607.23056, Tc=5.42 K, TAB (preprint), candidate – preprint)
throttle; run Nb47Re20HfZrTi33 --element_labels Ti Zr Nb Hf Re --concentrations 11 11 47 11 20 &

## ── caution: multiphase / inhomogeneous (see Status column in xlsx) ──
## xlsx row 81: NbTaTiZr_Wu2018 (Wu, 10.4236/ns.2018.103012, Tc=8.29 K, TAB (verified), caution – inhomogeneous)
throttle; run NbTaTiZr_Wu2018 --element_labels Ti Zr Nb Ta --concentrations 21.3 22.8 25.2 30.7 &
## xlsx row 100: YNbTiZrHf (Idczak, 10.1103/k1yr-t59z, Tc=6.1 K, ABS, caution – minor second phase)
throttle; run YNbTiZrHf --element_labels Ti Y Zr Nb Hf --concentrations 20 20 20 20 20 &
## xlsx row 102: ScHfNbTi_nominal (Krnel, 10.3390/ma15031122, Tc=6.6 K, TAB (Table 4), caution – multiphase)
throttle; run ScHfNbTi_nominal --element_labels Sc Ti Nb Hf --concentrations 25 25 25 25 &
## xlsx row 103: ScHfNbZr_nominal (Krnel, 10.3390/ma15031122, Tc=6.8 K, TAB (Table 4), caution – multiphase)
throttle; run ScHfNbZr_nominal --element_labels Sc Zr Nb Hf --concentrations 25 25 25 25 &
## xlsx row 104: ScHfTaTi_nominal (Krnel, 10.3390/ma15031122, Tc=6.4 K, TAB (Table 4), caution – multiphase)
throttle; run ScHfTaTi_nominal --element_labels Sc Ti Hf Ta --concentrations 25 25 25 25 &
## xlsx row 105, 107, 108: ScNbTiZr_nominal (Krnel, 10.3390/ma15031122, Tc=8.4 K, TAB (Table 4), caution – multiphase); NbScTiZr_Kitagawa (Kitagawa, 10.1016/j.mtcomm.2024.109690, Tc=8.11 K, TAB (verified), caution – multiphase eutectic); NbScTiZr_Seki (Seki, 10.1007/s10948-023-06643-z, Tc=7.9 K, TAB (verified), caution – multiphase eutectic)
throttle; run ScNbTiZr_nominal --element_labels Sc Ti Zr Nb --concentrations 25 25 25 25 &
## xlsx row 106: ScTaTiZr_nominal (Krnel, 10.3390/ma15031122, Tc=7.4 K, TAB (Table 4), caution – multiphase)
throttle; run ScTaTiZr_nominal --element_labels Sc Ti Zr Ta --concentrations 25 25 25 25 &
## xlsx row 109: Hf10Nb25Sc25Ti20Zr20 (Kubo, 10.1016/j.jallcom.2025.184531, Tc=6.83 K, TAB (verified), caution – multiphase eutectic)
throttle; run Hf10Nb25Sc25Ti20Zr20 --element_labels Sc Ti Zr Nb Hf --concentrations 25 20 20 25 10 &
## xlsx row 110: Hf5Nb45Sc20Ti15Zr15 (Kubo, 10.1016/j.jallcom.2025.184531, Tc=7.69 K, TAB (verified), caution – multiphase eutectic)
throttle; run Hf5Nb45Sc20Ti15Zr15 --element_labels Sc Ti Zr Nb Hf --concentrations 20 15 15 45 5 &
## xlsx row 111: Hf5Nb45Sc10Ti5Zr35 (Kubo, 10.1016/j.jallcom.2025.184531, Tc=7.73 K, TAB (verified), caution – multiphase eutectic)
throttle; run Hf5Nb45Sc10Ti5Zr35 --element_labels Sc Ti Zr Nb Hf --concentrations 10 5 35 45 5 &

wait
echo "[$(date +%F_%T)] all done"
