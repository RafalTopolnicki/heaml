#!/bin/bash
set -euo pipefail

outdir="thermodynamic"

# ── baseline ──────────────────────────────────────────────────────────────────
#python scripts/structure_compare.py \
#    --element_labels Ti Nb Zr Hf \
#    --concentrations 0.25 0.25 0.25 0.25 \
#    --outdir "${outdir}/baseline_TiNbZrHf"

# ── sra.TiNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2 ────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Y \
    --concentrations 0.593995 0.111303 0.085830 0.208872 \
    --outdir "${outdir}/sra.TiNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2"

# ── sra.TiNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta5 ────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Y \
    --concentrations 0.589198 0.117055 0.066820 0.226928 \
    --outdir "${outdir}/sra.TiNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta5"

# ── sra.TiScNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2 ──────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Sc Mo \
    --concentrations 0.392733 0.019476 0.464613 0.123178 \
    --outdir "${outdir}/sra.TiScNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2"

# ── sra.TiScNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2_mincomp6 ─────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Sc Mo Y \
    --concentrations 0.213245 0.071544 0.053557 0.509266 0.094935 0.057453 \
    --outdir "${outdir}/sra.TiScNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2_mincomp6"

# ── sra.TiScNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2_mincomp7 ─────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Hf Sc Mo Y \
    --concentrations 0.186766 0.058618 0.052603 0.052089 0.501175 0.098599 0.050151 \
    --outdir "${outdir}/sra.TiScNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2_mincomp7"

# ── sra.TiScNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2_mincomp8 ─────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Hf Ta Sc Mo Y \
    --concentrations 0.144159 0.077530 0.051743 0.052450 0.050892 0.502360 0.061676 0.059190 \
    --outdir "${outdir}/sra.TiScNbZrHfTaMoWYRe.kp10.ew0.6.acqbeta2_mincomp8"

# ── sra.kp10.ew0.6.noSc ──────────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Zr Y \
    --concentrations 0.599754 0.325764 0.074482 \
    --outdir "${outdir}/sra.kp10.ew0.6.noSc"

# ── sra.kp10.ew0.6.ubc10 ─────────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Zr Hf Sc Mo W Y La \
    --concentrations 0.320035 0.052619 0.067331 0.294481 0.062903 0.036401 0.157198 0.009033 \
    --outdir "${outdir}/sra.kp10.ew0.6.ubc10"

# ── sra.kp10.ew0.6.ubc5 ──────────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Ta Sc W Y \
    --concentrations 0.371226 0.059311 0.028866 0.371740 0.056679 0.112178 \
    --outdir "${outdir}/sra.kp10.ew0.6.ubc5"

# ── sra.kp10.ew0.6_Scmax0.15 ─────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Hf Sc \
    --concentrations 0.599981 0.080921 0.167569 0.004593 0.146936 \
    --outdir "${outdir}/sra.kp10.ew0.6_Scmax0.15"

# ── sra.kp10.ew0.6_mincomp5.0.05 ─────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Hf Ta Sc Mo W Y \
    --concentrations 0.147630 0.041741 0.037194 0.019630 0.011353 0.493059 0.050460 0.054316 0.144616 \
    --outdir "${outdir}/sra.kp10.ew0.6_mincomp5.0.05"

# ── sra.kp10.ew0.6_mincomp5 ──────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Hf Sc Mo Y \
    --concentrations 0.223575 0.149793 0.103131 0.001028 0.371295 0.041983 0.109195 \
    --outdir "${outdir}/sra.kp10.ew0.6_mincomp5"

# ── sra.kp10.ew0.6_mincomp6 ──────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Hf Sc Mo Y La \
    --concentrations 0.097556 0.072405 0.073626 0.009547 0.599768 0.039071 0.050932 0.057095 \
    --outdir "${outdir}/sra.kp10.ew0.6_mincomp6"

# ── sra.kp10.ew0.6_mincomp7 ──────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Sc Mo Y La \
    --concentrations 0.088117 0.055467 0.126377 0.566506 0.050684 0.058821 0.054028 \
    --outdir "${outdir}/sra.kp10.ew0.6_mincomp7"

# ── sra.kp10.ew0.6_mincomp8 ──────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Hf Ta Sc Mo W Y \
    --concentrations 0.153381 0.086240 0.062318 0.063300 0.050478 0.466909 0.054438 0.001229 0.061708 \
    --outdir "${outdir}/sra.kp10.ew0.6_mincomp8"

# ── sra.kp10.ew0.6_mincomp9 ──────────────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Ta Sc Mo W Y La \
    --concentrations 0.062902 0.035642 0.131382 0.034394 0.590020 0.030239 0.025131 0.048350 0.041940 \
    --outdir "${outdir}/sra.kp10.ew0.6_mincomp9"

# ── sra.kp10.ew0.6_nocompositionlimit ────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Sc Mo \
    --concentrations 0.786986 0.004801 0.140265 0.067949 \
    --outdir "${outdir}/sra.kp10.ew0.6_nocompositionlimit"

# ── sra.kp10.ew0.6_withSc_targetTc ───────────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Sc Mo \
    --concentrations 0.596534 0.288120 0.115346 \
    --outdir "${outdir}/sra.kp10.ew0.6_withSc_targetTc"

# ── sra.kp10.ew0.6_withSc_targetlambda ───────────────────────────────────────
python scripts/structure_compare.py \
    --element_labels Ti Nb Zr Hf Ta Sc Mo W Y La \
    --concentrations 0.018027 0.105931 0.091387 0.023019 0.109123 0.550017 0.029307 0.011030 0.049086 0.013074 \
    --outdir "${outdir}/sra.kp10.ew0.6_withSc_targetlambda"
