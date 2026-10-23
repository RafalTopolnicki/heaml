#!/bin/bash
# Literature benchmark extension: bcc alloys containing V (and V+Sc) built from the extended element set
# Ti Nb Zr Hf Ta Mo W Y Re Ru Sc + V.  Prepared 2026-10-23; literature values and sources in
# results/Tc_literature_V_additions.csv and results/literature_baseline_extension.md (section "V additions").
# Same settings as 0_compare_with_known_bz10-10-20.sh (production k-points: lattice 10 / Debye 10 / final SCF 20,
# consts.py defaults), same output directory, so the Tc_Baseline_Comparison notebooks pick the results up once
# the V column is added to results/Tc_literature.xlsx and to the notebook loaders.
# No --reuse_from: nothing with V was computed before -> full lattice + Debye + final SCF for every entry.
# Entries with an existing results.json are skipped -> safe to re-run.  ONLY=<name> runs a single entry.
# BINARIES=1 additionally runs the V binaries (V-Nb, V-Ta, Ti-V, V-Mo; Corsan 1970, Cheng 1962, Kuentzler 1986)
# into validation.ratios/binaries.sra.ew0.6.bz10-10-20/<line>/ (same layout as 3_run_validation_binaries_bz10-10-20.sh).
# Requires V in scripts/src/elements.py and consts.py (added 2026-10-23; sync code before running remotely).
# After the pure-V run finishes, copy E0 from results/comparison.sra.ew0.6.bz10-10-20/V/lattice/lattice_summary.txt
# into ELEMENTS['V'].ebulk (elements.py) so that dE/dG are meaningful for V alloys.

NJOBS=${NJOBS:-16}                              # parallel alloys; on SLURM set from cpus (see slurm/)
export OMP_THREAD_LIMIT=${OMP_THREAD_LIMIT:-12}  # OpenMP threads per specx run
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

ew=0.6
params="--overwrite_params --ew ${ew} --rel sra --debye_mode kkr"
outputdir="results/comparison.sra.ew${ew}.bz10-10-20"
bindir="validation.ratios/binaries.sra.ew${ew}.bz10-10-20"

run_in() {  # $1 = full workdir, rest = process_hea.py arguments
    local wd=$1; shift
    local name=${wd#${outputdir}/}; name=${name#${bindir}/}
    if [ -n "${ONLY:-}" ]; then case " ${ONLY} " in *" ${name} "*) ;; *) return ;; esac; fi   # ONLY="a b c": only these entries (slurm array)
    if [ -f "${wd}/results.json" ]; then [ "${LIST:-0}" = "1" ] || echo "[skip] ${name}"; return; fi
    if [ "${LIST:-0}" = "1" ]; then echo "${name}"; return; fi   # LIST=1: print pending entries, run nothing (slurm/)
    mkdir -p "${wd}"
    echo "[$(date +%F_%T)] start ${name}"
    python scripts/process_hea.py "$@" --workdir "${wd}" ${params} > "${wd}/process_hea.log" 2>&1 \
        || echo "[FAILED] ${name} (see ${wd}/process_hea.log)"
    echo "[$(date +%F_%T)] done  ${name}"
}
run() { local name=$1; shift; run_in "${outputdir}/${name}" "$@"; }

throttle() {  # keep at most NJOBS background jobs
    while [ "$(jobs -rp | wc -l)" -ge "${NJOBS}" ]; do wait -n; done
}

# ── pure V (bcc; exp Tc 5.4 K, θ_D 380 K) — also provides ebulk for elements.py ─────────────────────────────
throttle; run V --element_labels V --concentrations 100 &

# ── Hf-Nb-Ti-V-Zr (Kitagawa group) ────────────────────────────────────────────────────────────────────────────
# Ishizu & Kitagawa, Results Phys. 13, 102275 (2019): Tc 5.3 K (χac), bcc, annealed 800 C / 4 d
throttle; run Hf21Nb25Ti15V15Zr24 --element_labels Hf Nb Ti V Zr --concentrations 21 25 15 15 24 &
# Senga et al., Supercond. Sci. Technol. 2026 (arXiv 2602.19376): equimolar, Tc 3.73 K (Cp), θ_D 209 K, γ 6.24; as-cast single bcc
throttle; run HfNbTiVZr --element_labels Hf Nb Ti V Zr --concentrations 20 20 20 20 20 &

# ── equiatomic quinaries with V ───────────────────────────────────────────────────────────────────────────────
# Jangid et al., APL 124, 192602 (2024): ScVTiHfNb, Tc 4.17 K (χ) / 4.47 (ρ) / 3.72 (Cp), θ_D 231, γ 7.41, a 3.327 A
throttle; run ScVTiHfNb --element_labels Sc V Ti Hf Nb --concentrations 20 20 20 20 20 &
# Sharma et al., PRMater 9, 064801 (2025): Ta0.2Nb0.2V0.2Ti0.2Zr0.2, Tc 5.13 K (Cp), θ_D 215, γ 8.60
throttle; run NbTaTiVZr --element_labels Nb Ta Ti V Zr --concentrations 20 20 20 20 20 &
# Sharma 2025 (X = Hf): Tc 4.0 K (Cp), θ_D 209, γ 8.03; Sarkar et al., Intermetallics 144, 107503 (2022): Tc 5.0 K, a 3.32 A, Hc2 6.63 T
throttle; run HfNbTaTiV --element_labels Hf Nb Ta Ti V --concentrations 20 20 20 20 20 &
# Wu, Chen, Wu, Nat. Sci. 10, 110 (2018) via Kitagawa 2020 Table 1: EDS Nb21.5Ta18.1Ti15.9Zr14.4Hf16.6V13.5, Tc 5.1 K
throttle; run NbTaTiZrHfV_Wu2018 --element_labels Nb Ta Ti Zr Hf V --concentrations 21.5 18.1 15.9 14.4 16.6 13.5 &

# ── von Rohr et al., PNAS 113, E7144 (2016), V-substituted (TaNb)0.67(HfZrTi)0.33 family (VEC 4.67) ───────────
# (TaNbV)0.67(HfZrTi)0.33: Tc 4.3 K (Table I) / 4.11 K (Cp), θ_D 236, γ 7.97
throttle; run TaNbV67HfZrTi33 --element_labels Ta Nb V Hf Zr Ti --concentrations 22.33 22.33 22.34 11 11 11 &
# (NbV)0.67(HfZrTi)0.33: Tc 7.2 K
throttle; run NbV67HfZrTi33 --element_labels Nb V Hf Zr Ti --concentrations 33.5 33.5 11 11 11 &
# (TaV)0.67(HfZrTi)0.33: Tc 4.0 K
throttle; run TaV67HfZrTi33 --element_labels Ta V Hf Zr Ti --concentrations 33.5 33.5 11 11 11 &
# [Ta(Mo0.5Sc0.5)V]0.67(HfZrTi)0.33: NOT superconducting above 1.9 K (same VEC 4.67) — upper bound, V + Sc
throttle; run TaMoScV67HfZrTi33 --element_labels Ta Mo Sc V Hf Zr Ti --concentrations 22.33 11.17 11.17 22.33 11 11 11 &

# ── Wrocław group (Idczak / Sobota / Nowak) ───────────────────────────────────────────────────────────────────
# Sobota et al., Acta Mater. 285, 120666 (2025): Ti0.5(VNbHfTa)0.5, Tc 5.9(1) K, θ_D 205, γ 7.9, Hc2 12.8 T; two bcc zones (Ta-/Ti-rich)
throttle; run Ti50VNbHfTa50 --element_labels Ti V Nb Hf Ta --concentrations 50 12.5 12.5 12.5 12.5 &
# Nowak et al., Metall. Mater. Trans. A 55, 3789 (2024): (NbTi)0.67(MoHfV)0.33, Tc 4.8(2) K, γ 7.4, Hc2 6.8 T; minor Mo-rich bcc phase
throttle; run NbTi67MoHfV33 --element_labels Nb Ti Mo Hf V --concentrations 33.5 33.5 11 11 11 &
# Sobota, Olejnik, Szymanski, Gnida, APL 129, 111905 (2026): Ti0.475(ZrNbTaHf)0.5Sc0.05 — Tc TO BE READ FROM THE PAPER (abstract: close to parent 6 K)
throttle; run Ti47.5ZrNbTaHf47.5Sc5 --element_labels Ti Zr Nb Ta Hf Sc --concentrations 47.5 11.875 11.875 11.875 11.875 5 &

# ── medium-entropy Ti-V-Nb-Ta and V-Ti-W (Luo / Marik groups) ────────────────────────────────────────────────
# Li et al., J. Phys. Chem. C 127, 16211 (2023): TiVNbTa, Tc 4.65 K (ρ, χ) / 4.38 (Cp), θ_D 166.5, γ 3.02
throttle; run TiVNbTa --element_labels Ti V Nb Ta --concentrations 25 25 25 25 &
# Sharma et al., Mater. Today Commun. 44, 111968 (2025): (TiV)0.5Nb0.4Ta0.1, Tc 5.2 K (Kitagawa 2025 review) / 4.50 (Cp), θ_D 256, γ 10.1
throttle; run Ti25V25Nb40Ta10 --element_labels Ti V Nb Ta --concentrations 25 25 40 10 &
# Li et al., Mater. Today Commun. 38, 108444 (2024): V4Ti2W, Tc ~5.0 K (abstract; 5.57 K onset in text), θ_D 171, a 3.107 A, VEC 4.83
throttle; run V57Ti29W14 --element_labels V Ti W --concentrations 57.14 28.57 14.29 &

# ── Ta-Nb-V ternary solid solutions, Wang & Cappelletti, J. Low Temp. Phys. 32, 481 (1978) (Cp) ──────────────
throttle; run V9.5Nb29.8Ta60.7 --element_labels V Nb Ta --concentrations 9.5 29.8 60.7 &    # Tc 4.47 K, θ_D 259
throttle; run V21.2Nb42Ta36.8  --element_labels V Nb Ta --concentrations 21.2 42.0 36.8 &   # Tc 4.41 K, θ_D 267
throttle; run V25.1Nb57.8Ta17.1 --element_labels V Nb Ta --concentrations 25.1 57.8 17.1 &  # Tc 4.67 K, θ_D 268

# ── V binaries (optional): Corsan & Cook, pss 40, 657 (1970); Cheng et al. 1962 (Ti-V); Kuentzler & Khan 1986 (V-Mo)
if [ "${BINARIES}" = "1" ]; then
    binary() {  # $1 = line name, $2 = el A, $3 = el B, rest = at.% of A
        local line=$1 A=$2 B=$3; shift 3
        for a in "$@"; do
            local b=$(python -c "print(round(100-${a},1))")
            throttle
            run_in "${bindir}/${line}/${A}${a}${B}${b}" --element_labels ${A} ${B} --concentrations ${a} ${b} &
        done
    }
    binary VNb V Nb 85 70 50 31 20 10        # Tc 4.30 4.03 4.25 4.90 5.76 6.95 K
    binary VTa V Ta 90 76 50 30 10           # Tc 4.47 3.58 2.73 3.00 3.58 K
    binary TiV Ti V 70 50 25 15              # Tc 6.14 7.30 7.16 7.02 K
    binary VMo V Mo 95 90                    # Tc 3.73 2.50 K
    throttle; run_in "${bindir}/pure/V" --element_labels V --concentrations 100 &
fi

wait
echo "=== All done. Results in: ${outputdir} (and ${bindir} if BINARIES=1) ==="
