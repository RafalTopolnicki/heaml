#!/usr/bin/env python3
"""
Elastic constants and Debye temperature from AkaiKKR using one fixed PHYSICAL RMT
for all distortions and all delta values.

Main features:
  1. Accepts multiple positive deltas via --deltas.
  2. Uses one common physical RMT in bohr for all cells:
       reference, tetragonal +/-delta_i, C44 distortion +/-delta_i.
  3. Converts that fixed physical RMT to AkaiKKR input rmt for each cell:
       rmt_input = RMT_common_bohr / a_local(delta)
  4. Runs two-sided strains by default.
  5. Extracts C' and C44 from an energy-vs-delta fit:
       dE_even(delta) = A2 delta^2 + A4 delta^4

     tetragonal:
       C' = A2_tetra / (6 V)

     monoclinic C44 mode:
       C44 = A2_c44 / (2 V)

     simple_shear C44 mode:
       C44 = 2 A2_c44 / V

Use --fit-mode linear to fit only A2 delta^2 through the origin.
Use --one-sided only for debugging; production should use two-sided strains.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd

from src.write_akai_input import scf_input
from src.utils import (
    dist_to_si,
    energy_to_si,
    parse_energy,
    converged_info_in_string,
    gzip_file,
    cleanup_potential_files,
    cleanup_fortran_files,
)
from src.consts import AKAIBIN, ATOMS_PER_CELL


RY_BOHR3_TO_GPA = energy_to_si(1.0) / dist_to_si(1.0) ** 3 / 1.0e9


def setup_logger(log_file: str) -> logging.Logger:
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


# -----------------------------------------------------------------------------
# Distortion definitions
# -----------------------------------------------------------------------------

def get_tetragonal_distortion(delta: float, a0: float) -> Tuple[float, float, float, float]:
    """
    Volume-conserving tetragonal distortion:
        F = diag(1+d, 1+d, 1/(1+d)^2)

    Leading elastic energy:
        Delta E = 6 V C' d^2 + O(d^3)

    AkaiKKR trc parameters:
        a_local, b/a, c/a, gamma
    """
    if 1.0 + delta <= 0.0:
        raise ValueError(f"Invalid tetragonal delta={delta}: 1+delta must be positive")

    a_local = a0 * (1.0 + delta)
    ba = 1.0
    ca = 1.0 / (1.0 + delta) ** 3
    gamma = 90.0
    return a_local, ba, ca, gamma


def get_monoclinic_distortion(delta: float, a0: float) -> Tuple[float, float, float, float]:
    """
    Volume-conserving monoclinic/shear distortion:
        F = [[1, d, 0],
             [d, 1, 0],
             [0, 0, 1/(1-d^2)]]

    Green strain has E12 = d.

    Leading elastic energy:
        Delta E = 2 V C44 d^2 + O(d^4)

    AkaiKKR trc parameters:
        a_local, b/a, c/a, gamma
    """
    if abs(delta) >= 1.0:
        raise ValueError(f"Invalid monoclinic delta={delta}: |delta| must be < 1")

    a_local = a0 * np.sqrt(1.0 + delta ** 2)
    ba = 1.0
    ca = 1.0 / (1.0 - delta ** 2) / np.sqrt(1.0 + delta ** 2)
    gamma_rad = np.pi / 2.0 - 2.0 * np.arctan(delta)
    gamma = np.rad2deg(gamma_rad)
    return a_local, ba, ca, gamma


def get_simple_shear_distortion(delta: float, a0: float) -> Tuple[float, float, float, float]:
    """
    Simple shear distortion:
        F = [[1, d, 0],
             [0, 1, 0],
             [0, 0, 1]]

    This is volume-preserving because det(F)=1.

    Lattice vectors:
        A = a0 * (1, 0, 0)
        B = a0 * (d, 1, 0)
        C = a0 * (0, 0, 1)

    Green strain has:
        E12 = d/2
        E22 = d^2/2

    To leading order:
        Delta E = 0.5 V C44 d^2 + O(d^4)

    Therefore:
        C44 = 2 Delta E / (V d^2)

    AkaiKKR trc parameters:
        a_local = a0
        b/a = sqrt(1+d^2)
        c/a = 1
        gamma = angle(A, B)
    """
    a_local = a0
    ba = np.sqrt(1.0 + delta ** 2)
    ca = 1.0

    cos_gamma = delta / ba
    gamma = np.rad2deg(np.arccos(cos_gamma))

    return a_local, ba, ca, gamma


# -----------------------------------------------------------------------------
# Maximum non-overlapping RMT for 2-atom bcc-like cell in Akai input units
# -----------------------------------------------------------------------------

def rmt_max_tetragonal(ca: float, ba: float = 1.0) -> float:
    """
    Dimensionless max RMT in units of current AkaiKKR lattice constant a.

    For atoms at (0,0,0) and (1/2,1/2,1/2) in an orthogonal bct cell:
        rmt_max = 1/4 sqrt(1 + (b/a)^2 + (c/a)^2)
    """
    return 0.25 * np.sqrt(1.0 + ba ** 2 + ca ** 2)


def rmt_max_monoclinic(ca: float, gamma_deg: float, ba: float = 1.0) -> float:
    """
    Dimensionless max RMT in units of current AkaiKKR lattice constant a.

    For atoms at (0,0,0) and (1/2,1/2,1/2), alpha=beta=90 deg:
        rmt_max = 1/4 sqrt(1 + (b/a)^2 + (c/a)^2 - 2(b/a)|cos gamma|)
    """
    gamma_rad = np.deg2rad(gamma_deg)
    return 0.25 * np.sqrt(
        1.0 + ba ** 2 + ca ** 2 - 2.0 * ba * abs(np.cos(gamma_rad))
    )


def rmt_max_for_cell(
    distortion: str,
    delta: float,
    a0: float,
) -> Tuple[float, float, float, float, float, float]:
    """
    Return:
        a_local, b/a, c/a, gamma, rmt_max_dimensionless, rmt_max_physical_bohr
    """
    if distortion == "tetragonal":
        a_local, ba, ca, gamma = get_tetragonal_distortion(delta, a0)
        rmt_dimless = rmt_max_tetragonal(ca=ca, ba=ba)

    elif distortion == "monoclinic":
        a_local, ba, ca, gamma = get_monoclinic_distortion(delta, a0)
        rmt_dimless = rmt_max_monoclinic(ca=ca, gamma_deg=gamma, ba=ba)

    elif distortion == "simple_shear":
        a_local, ba, ca, gamma = get_simple_shear_distortion(delta, a0)
        rmt_dimless = rmt_max_monoclinic(ca=ca, gamma_deg=gamma, ba=ba)

    else:
        raise ValueError(f"Unknown distortion: {distortion}")

    rmt_physical = a_local * rmt_dimless
    return a_local, ba, ca, gamma, rmt_dimless, rmt_physical


def normalize_deltas(deltas: List[float]) -> List[float]:
    """Return sorted unique positive deltas."""
    out = sorted({round(abs(float(d)), 12) for d in deltas if abs(float(d)) > 1e-14})
    if not out:
        raise ValueError("At least one nonzero delta is required")
    return out


def signed_deltas_from_positive(deltas: List[float], two_sided: bool = True) -> List[float]:
    signed = []
    for d in deltas:
        if two_sided:
            signed.extend([-d, d])
        else:
            signed.append(d)
    return signed


def choose_common_physical_rmt(
    a0: float,
    deltas: List[float],
    safety: float = 0.995,
    two_sided: bool = True,
    c44_distortion: str = "monoclinic",
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Choose one fixed physical RMT in bohr for all cells used by this script.

    Cells included:
      - tetragonal 0
      - C44-distortion 0
      - tetragonal +/-delta_i, unless --one-sided is used
      - C44-distortion +/-delta_i, unless --one-sided is used
    """
    entries = []
    all_deltas = [0.0] + signed_deltas_from_positive(deltas, two_sided=two_sided)

    for distortion in ["tetragonal", c44_distortion]:
        for d in all_deltas:
            a_local, ba, ca, gamma, rmt_dimless, rmt_physical = rmt_max_for_cell(
                distortion, d, a0
            )
            entries.append(
                {
                    "distortion": distortion,
                    "delta": d,
                    "a_local_bohr": a_local,
                    "b_over_a": ba,
                    "c_over_a": ca,
                    "gamma_deg": gamma,
                    "rmt_max_input": rmt_dimless,
                    "rmt_max_physical_bohr": rmt_physical,
                }
            )

    min_physical = min(e["rmt_max_physical_bohr"] for e in entries)
    return safety * min_physical, entries


def lattice_params_for_distortion(distortion: str, delta: float, a0: float) -> Dict[str, float]:
    if distortion == "reference":
        a_local, ba, ca, gamma = get_tetragonal_distortion(0.0, a0)

    elif distortion == "tetragonal":
        a_local, ba, ca, gamma = get_tetragonal_distortion(delta, a0)

    elif distortion == "monoclinic":
        a_local, ba, ca, gamma = get_monoclinic_distortion(delta, a0)

    elif distortion == "simple_shear":
        a_local, ba, ca, gamma = get_simple_shear_distortion(delta, a0)

    else:
        raise ValueError(f"Unknown distortion: {distortion}")

    return {
        "symmetry": "trc",
        "lattice_constant": a_local,
        "b/a": ba,
        "c/a": ca,
        "gamma": gamma,
    }


# -----------------------------------------------------------------------------
# AkaiKKR execution
# -----------------------------------------------------------------------------

def delta_tag(delta: float) -> str:
    if abs(delta) < 1e-14:
        return "0.000000"
    prefix = "p" if delta > 0 else "m"
    return prefix + f"{abs(delta):.6f}"


def run_scf(
    filename: str,
    lattice_params: Dict[str, float],
    args: Dict[str, Any],
    logger: logging.Logger,
    rmt_input: float,
):
    inp = filename + ".inp"
    out = filename + ".out"

    scf_input(
        filename=inp,
        lattice_params=lattice_params,
        elements=args["elements"],
        concentrations=args["concentrations"],
        ew=args["ew"],
        xc=args["xc"],
        rel=args["rel"],
        bzqlty=args["bzqlty"],
        rmt=rmt_input,
        pmix=args.get("pmix", 0.01),
        edelt=args.get("edelt", 0.001),
        mxl=args.get("mxl", 3),
        magtype=args.get("magtype", "nmag"),
    )

    logger.info("Running SCF: %s", filename)
    logger.info("  lattice_params=%s", lattice_params)
    logger.info("  rmt_input=%.12f", rmt_input)

    with open(inp, "r") as fin, open(out, "w") as fout:
        subprocess.run([AKAIBIN], stdin=fin, stdout=fout, stderr=subprocess.STDOUT)

    with open(out, "r", errors="replace") as f:
        text = f.read()

    energy = parse_energy(text)
    conv = converged_info_in_string(text)

    if "***err" in text.lower() or "err in" in text.lower():
        logger.error("AkaiKKR reported an error in %s", filename)
        conv = False

    if "given rmt" in text.lower() or "rmt's conflict" in text.lower():
        logger.warning("AkaiKKR reported an RMT conflict/reduction in %s", filename)
        conv = False

    if energy is None or not np.isfinite(energy):
        conv = False

    logger.info("Finished SCF: %s | energy=%s | converged=%s", filename, energy, conv)

    gz_out = None
    if args.get("compress", True):
        gz_out = gzip_file(out)

    cleanup_potential_files(filename)
    cleanup_fortran_files(filename)

    return energy, conv, gz_out


def run_scf_distortion(
    distortion: str,
    delta: float,
    args: Dict[str, Any],
    logger: logging.Logger,
    rmt_common_physical: float,
):
    if distortion == "reference":
        filename = f"{args['output']}_reference_0.000000"
    else:
        filename = f"{args['output']}_{distortion}_{delta_tag(delta)}"

    lattice_params = lattice_params_for_distortion(distortion, delta, args["a0"])
    a_local = lattice_params["lattice_constant"]
    rmt_input = rmt_common_physical / a_local

    energy, conv, gz_out = run_scf(
        filename=filename,
        lattice_params=lattice_params,
        args=args,
        logger=logger,
        rmt_input=rmt_input,
    )

    row = {
        "distortion": distortion,
        "delta": delta,
        "abs_delta": abs(delta),
        "a_local_bohr": a_local,
        "b_over_a": lattice_params["b/a"],
        "c_over_a": lattice_params["c/a"],
        "gamma_deg": lattice_params["gamma"],
        "rmt_common_physical_bohr": rmt_common_physical,
        "rmt_input": rmt_input,
        "energy_Ry": energy,
        "converged": conv,
        "out_gz": gz_out if gz_out else "",
    }

    return row


# -----------------------------------------------------------------------------
# Fitting and Debye calculation
# -----------------------------------------------------------------------------

def fit_a2_from_even_energies(even_df: pd.DataFrame, fit_mode: str) -> Dict[str, Any]:
    """
    Fit dE_even(delta) = A2 delta^2 + A4 delta^4, or linear through origin.
    Returns coefficients in Ry.
    """
    x = even_df["abs_delta"].values.astype(float) ** 2
    y = even_df["dE_even_Ry"].values.astype(float)

    if len(x) == 0:
        raise ValueError("No even energy data to fit")

    if fit_mode == "single" or len(x) == 1:
        a2 = float(np.sum(x * y) / np.sum(x * x))
        a4 = 0.0
        method = "single_or_linear_origin"

    elif fit_mode == "linear":
        a2 = float(np.sum(x * y) / np.sum(x * x))
        a4 = 0.0
        method = "linear_origin"

    elif fit_mode == "quartic":
        X = np.column_stack([x, x ** 2])
        coeff, residuals, rank, s = np.linalg.lstsq(X, y, rcond=None)
        a2 = float(coeff[0])
        a4 = float(coeff[1])
        method = "quartic_origin"

    else:
        raise ValueError(f"Unknown fit_mode: {fit_mode}")

    y_fit = a2 * x + a4 * x ** 2
    rmse = float(np.sqrt(np.mean((y - y_fit) ** 2))) if len(y) else np.nan

    return {
        "A2_Ry": a2,
        "A4_Ry": a4,
        "fit_method": method,
        "fit_rmse_Ry": rmse,
        "n_fit_points": int(len(x)),
    }


def build_even_energy_table(
    rows_df: pd.DataFrame,
    e0: float,
    two_sided: bool,
    c44_distortion: str,
) -> pd.DataFrame:
    rows = []

    for distortion in ["tetragonal", c44_distortion]:
        df = rows_df[(rows_df["distortion"] == distortion) & (rows_df["abs_delta"] > 0)].copy()

        for abs_d in sorted(df["abs_delta"].unique()):
            sub = df[np.isclose(df["abs_delta"], abs_d)]
            e_plus = sub[sub["delta"] > 0]["energy_Ry"].values
            e_minus = sub[sub["delta"] < 0]["energy_Ry"].values

            if two_sided:
                if len(e_plus) != 1 or len(e_minus) != 1:
                    raise RuntimeError(f"Missing +/- energies for {distortion}, delta={abs_d}")
                dE_even = 0.5 * ((float(e_plus[0]) - e0) + (float(e_minus[0]) - e0))
                e_p = float(e_plus[0])
                e_m = float(e_minus[0])

            else:
                if len(e_plus) != 1:
                    raise RuntimeError(f"Missing + energy for {distortion}, delta={abs_d}")
                dE_even = float(e_plus[0]) - e0
                e_p = float(e_plus[0])
                e_m = np.nan

            rows.append(
                {
                    "distortion": distortion,
                    "abs_delta": float(abs_d),
                    "energy0_Ry": float(e0),
                    "energy_plus_Ry": e_p,
                    "energy_minus_Ry": e_m,
                    "dE_even_Ry": float(dE_even),
                    "dE_even_over_delta2_Ry": float(dE_even / (abs_d ** 2)),
                }
            )

    return pd.DataFrame(rows)


def compute_c44_from_fit(c44_fit: Dict[str, Any], vol: float, c44_distortion: str) -> float:
    """
    Return C44 in Ry/bohr^3 from the fitted A2 coefficient.

    monoclinic:
        dE = 2 V C44 d^2
        A2 = 2 V C44
        C44 = A2 / (2 V)

    simple_shear:
        dE = 0.5 V C44 d^2
        A2 = 0.5 V C44
        C44 = 2 A2 / V
    """
    if c44_distortion == "monoclinic":
        return c44_fit["A2_Ry"] / (2.0 * vol)

    if c44_distortion == "simple_shear":
        return 2.0 * c44_fit["A2_Ry"] / vol

    raise ValueError(f"Unknown C44 distortion: {c44_distortion}")


def compute_c44_single_from_row(row: pd.Series, vol: float, c44_distortion: str) -> float:
    """
    Return single-delta C44 in Ry/bohr^3 from one row of even_df.
    """
    dE = row["dE_even_Ry"]
    d = row["abs_delta"]

    if c44_distortion == "monoclinic":
        return dE / (2.0 * vol * d ** 2)

    if c44_distortion == "simple_shear":
        return 2.0 * dE / (vol * d ** 2)

    raise ValueError(f"Unknown C44 distortion: {c44_distortion}")


def run_kkr_elastic_debye(**kwargs):
    workdir = kwargs.get("workdir", ".")
    os.makedirs(workdir, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(workdir)

    try:
        logger = setup_logger(f"{kwargs['output']}_run.log")
        logger.info("Starting multi-delta elastic + Debye workflow with fixed physical RMT")

        if len(kwargs["elements"]) != len(kwargs["concentrations"]):
            raise ValueError("elements and concentrations must have the same length")

        deltas = normalize_deltas(kwargs["deltas"])
        two_sided = not kwargs.get("one_sided", False)
        c44_distortion = kwargs.get("c44_mode", "monoclinic")

        logger.info("Using positive deltas: %s", deltas)
        logger.info("Two-sided strains: %s", two_sided)
        logger.info("C44 distortion mode: %s", c44_distortion)

        rmt_auto_physical, rmt_candidates = choose_common_physical_rmt(
            a0=kwargs["a0"],
            deltas=deltas,
            safety=kwargs.get("rmt_safety", 0.995),
            two_sided=two_sided,
            c44_distortion=c44_distortion,
        )

        rmt_forced = kwargs.get("rmt_common_physical_bohr", None)

        if rmt_forced is not None:
            rmt_common_physical = float(rmt_forced)

            # Safety check: forced RMT must not exceed the smallest allowed RMT.
            min_allowed = min(e["rmt_max_physical_bohr"] for e in rmt_candidates)

            if rmt_common_physical >= min_allowed:
                raise ValueError(
                    f"Forced --rmt-common-physical-bohr={rmt_common_physical} is too large. "
                    f"Smallest geometry-limited RMT is {min_allowed}. "
                    f"Use a value below this, e.g. {0.999 * min_allowed}."
                )

            logger.info("RMT candidates: %s", rmt_candidates)
            logger.info("Auto-selected fixed physical RMT would be %.12f bohr", rmt_auto_physical)
            logger.info("Using user-forced fixed physical RMT = %.12f bohr", rmt_common_physical)

        else:
            rmt_common_physical = rmt_auto_physical
            logger.info("RMT candidates: %s", rmt_candidates)
            logger.info("Chosen fixed physical RMT = %.12f bohr", rmt_common_physical)

        logger.info("RMT candidates: %s", rmt_candidates)
        logger.info("Chosen fixed physical RMT = %.12f bohr", rmt_common_physical)

        vol = kwargs["a0"] ** 3
        volsi = dist_to_si(kwargs["a0"]) ** 3

        rows = []

        # One common reference calculation for both distortion families.
        row_ref = run_scf_distortion("reference", 0.0, kwargs, logger, rmt_common_physical)
        rows.append(row_ref)

        signed_deltas = signed_deltas_from_positive(deltas, two_sided=two_sided)

        for distortion in ["tetragonal", c44_distortion]:
            for d in signed_deltas:
                rows.append(run_scf_distortion(distortion, d, kwargs, logger, rmt_common_physical))

        # Validate energies/convergence before computing constants.
        for row in rows:
            if row["energy_Ry"] is None or not np.isfinite(row["energy_Ry"]):
                raise RuntimeError(f"Invalid energy for {row['distortion']} delta={row['delta']}")
            if not row["converged"]:
                raise RuntimeError(f"SCF failed or did not converge for {row['distortion']} delta={row['delta']}")

        rows_df = pd.DataFrame(rows)
        rows_csv = f"{kwargs['output']}_all_scf_results.csv"
        rows_df.to_csv(rows_csv, index=False)

        e0 = float(row_ref["energy_Ry"])

        even_df = build_even_energy_table(
            rows_df,
            e0=e0,
            two_sided=two_sided,
            c44_distortion=c44_distortion,
        )

        # Per-delta elastic constants, useful diagnostics.
        even_df["Cp_single_Ry_bohr3"] = np.nan
        even_df["C44_single_Ry_bohr3"] = np.nan

        for idx, row in even_df.iterrows():
            if row["distortion"] == "tetragonal":
                even_df.loc[idx, "Cp_single_Ry_bohr3"] = (
                    row["dE_even_Ry"] / (6.0 * vol * row["abs_delta"] ** 2)
                )

            elif row["distortion"] == c44_distortion:
                even_df.loc[idx, "C44_single_Ry_bohr3"] = compute_c44_single_from_row(
                    row,
                    vol=vol,
                    c44_distortion=c44_distortion,
                )

        even_df["Cp_single_GPa"] = even_df["Cp_single_Ry_bohr3"] * RY_BOHR3_TO_GPA
        even_df["C44_single_GPa"] = even_df["C44_single_Ry_bohr3"] * RY_BOHR3_TO_GPA

        even_csv = f"{kwargs['output']}_even_energy_fit_data.csv"
        even_df.to_csv(even_csv, index=False)

        tetra_fit = fit_a2_from_even_energies(
            even_df[even_df["distortion"] == "tetragonal"],
            kwargs["fit_mode"],
        )

        c44_fit = fit_a2_from_even_energies(
            even_df[even_df["distortion"] == c44_distortion],
            kwargs["fit_mode"],
        )

        # ------------------------------------------------------------------
        # Raw elastic constants from strain-energy fits
        # ------------------------------------------------------------------
        Cp_raw = tetra_fit["A2_Ry"] / (6.0 * vol)
        C44_raw = compute_c44_from_fit(c44_fit, vol=vol, c44_distortion=c44_distortion)

        # Also report the old/standard monoclinic-style normalization as a diagnostic.
        # This is useful when c44_mode=simple_shear.
        C44_A2_over_2V_raw = c44_fit["A2_Ry"] / (2.0 * vol)
        C44_A2_over_4V_raw = c44_fit["A2_Ry"] / (4.0 * vol)
        C44_2A2_over_V_raw = 2.0 * c44_fit["A2_Ry"] / vol

        # ------------------------------------------------------------------
        # Empirical elastic-constant scaling
        # ------------------------------------------------------------------
        cp_scale = float(kwargs.get("cp_scale", 1.0))
        c44_scale = float(kwargs.get("c44_scale", 1.0))
        b0_scale = float(kwargs.get("b0_scale", 1.0))

        Cp = cp_scale * Cp_raw
        C44 = c44_scale * C44_raw
        B0_used_GPa = b0_scale * kwargs["B0"]

        C44_A2_over_2V = c44_scale * C44_A2_over_2V_raw
        C44_A2_over_4V = c44_scale * C44_A2_over_4V_raw
        C44_2A2_over_V = c44_scale * C44_2A2_over_V_raw

        logger.info("Elastic scaling factors:")
        logger.info("  cp_scale   = %.8f", cp_scale)
        logger.info("  c44_scale  = %.8f", c44_scale)
        logger.info("  b0_scale   = %.8f", b0_scale)
        logger.info("Raw elastic constants:")
        logger.info("  Cp_raw   = %.8f GPa", Cp_raw * RY_BOHR3_TO_GPA)
        logger.info("  C44_raw  = %.8f GPa", C44_raw * RY_BOHR3_TO_GPA)
        logger.info("  B0_raw   = %.8f GPa", kwargs["B0"])
        logger.info("Scaled elastic constants used for Debye:")
        logger.info("  Cp_used  = %.8f GPa", Cp * RY_BOHR3_TO_GPA)
        logger.info("  C44_used = %.8f GPa", C44 * RY_BOHR3_TO_GPA)
        logger.info("  B0_used  = %.8f GPa", B0_used_GPa)

        # Shear moduli for cubic polycrystal, Voigt/Reuss/Hill.
        Gv = (2.0 * Cp + 3.0 * C44) / 5.0
        Gr = 10.0 * Cp * C44 / (4.0 * C44 + 6.0 * Cp)
        Gh = (Gv + Gr) / 2.0
        Ghsi = Gh * energy_to_si(1.0) / dist_to_si(1.0) ** 3

        if Cp <= 0 or C44 <= 0 or Gh <= 0:
            logger.warning("Non-positive elastic constant found: Cp=%s, C44=%s, Gh=%s", Cp, C44, Gh)

        # Sound velocities.
        B0si = B0_used_GPa * 1e9

        # Sound velocities.
        vt = np.sqrt(Ghsi / kwargs["density"]) if Ghsi > 0 else np.nan

        vl_arg = B0si + 4.0 / 3.0 * Ghsi
        vl = np.sqrt(vl_arg / kwargs["density"]) if vl_arg > 0 else np.nan

        vm = (
            3.0 / (2.0 / vt ** 3 + 1.0 / vl ** 3)
        ) ** (1.0 / 3.0) if np.isfinite(vt) and np.isfinite(vl) else np.nan

        # Debye temperature.
        thetaDB = (
            4.79924e-11
            * (3.0 * ATOMS_PER_CELL / (4.0 * np.pi * volsi)) ** (1.0 / 3.0)
            * vm
        ) if np.isfinite(vm) else np.nan

        summary = {
            "a0_bohr": kwargs["a0"],
            "B0_GPa_raw": float(kwargs["B0"]),
            "B0_GPa": float(B0_used_GPa),
            "density_kg_m3": kwargs["density"],
            "deltas": deltas,
            "two_sided": two_sided,
            "fit_mode": kwargs["fit_mode"],
            "c44_mode": c44_distortion,

            "cp_scale": float(cp_scale),
            "c44_scale": float(c44_scale),
            "b0_scale": float(b0_scale),

            "rmt_safety": kwargs.get("rmt_safety", 0.995),
            "rmt_common_physical_bohr_forced": kwargs.get("rmt_common_physical_bohr", None),
            "rmt_common_physical_bohr": float(rmt_common_physical),
            "rmt_reference_input": float(row_ref["rmt_input"]),
            "energy0_Ry": float(e0),
            "tetra_fit": tetra_fit,
            "c44_fit": c44_fit,

            "Cp_raw_Ry_bohr3": float(Cp_raw),
            "C44_raw_Ry_bohr3": float(C44_raw),
            "Cp_raw_GPa": float(Cp_raw * RY_BOHR3_TO_GPA),
            "C44_raw_GPa": float(C44_raw * RY_BOHR3_TO_GPA),

            "Cp_Ry_bohr3": float(Cp),
            "C44_Ry_bohr3": float(C44),
            "Gv_Ry_bohr3": float(Gv),
            "Gr_Ry_bohr3": float(Gr),
            "Gh_Ry_bohr3": float(Gh),

            "Cp_GPa": float(Cp * RY_BOHR3_TO_GPA),
            "C44_GPa": float(C44 * RY_BOHR3_TO_GPA),
            "Gv_GPa": float(Gv * RY_BOHR3_TO_GPA),
            "Gr_GPa": float(Gr * RY_BOHR3_TO_GPA),
            "Gh_GPa": float(Gh * RY_BOHR3_TO_GPA),

            "vt_m_s": float(vt) if np.isfinite(vt) else None,
            "vl_m_s": float(vl) if np.isfinite(vl) else None,
            "vm_m_s": float(vm) if np.isfinite(vm) else None,
            "thetaDB_K": float(thetaDB) if np.isfinite(thetaDB) else None,
            "all_scf_results_csv": rows_csv,
            "even_energy_fit_data_csv": even_csv,
            "rmt_candidates_json": f"{kwargs['output']}_rmt_candidates.json",
            "run_log": f"{kwargs['output']}_run.log",
        }

        with open(f"{kwargs['output']}_rmt_candidates.json", "w") as f:
            json.dump(rmt_candidates, f, indent=2)

        with open(f"{kwargs['output']}_summary.txt", "w") as f:
            for k, v in summary.items():
                f.write(f"{k} = {v}\n")

        with open(f"{kwargs['output']}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Workflow finished successfully")
        return summary

    finally:
        os.chdir(old_cwd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Core.
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument("--concentrations", nargs="+", required=True)
    parser.add_argument("--a0", type=float, required=True)
    parser.add_argument("--B0", type=float, required=True)
    parser.add_argument("--density", type=float, required=True)

    # Multiple deltas. --delta is kept for backward compatibility.
    parser.add_argument("--delta", type=float, default=None, help="single positive distortion value; legacy shortcut")
    parser.add_argument(
        "--deltas",
        nargs="+",
        type=float,
        default=None,
        help="positive distortion values, e.g. --deltas 0.003 0.005 0.007 0.010",
    )
    parser.add_argument("--one-sided", action="store_true", help="debug only: use +delta only instead of +/-delta")
    parser.add_argument(
        "--fit-mode",
        choices=["linear", "quartic", "single"],
        default="quartic",
        help="fit dE=A2*d^2 or dE=A2*d^2+A4*d^4",
    )

    parser.add_argument(
        "--c44-mode",
        choices=["monoclinic", "simple_shear"],
        default="monoclinic",
        help="Distortion used for C44 extraction",
    )
    parser.add_argument(
        "--cp-scale",
        type=float,
        default=1.0,
        help="Empirical scale factor applied to C' before computing Debye temperature.",
    )

    parser.add_argument(
        "--c44-scale",
        type=float,
        default=1.0,
        help="Empirical scale factor applied to C44 before computing Debye temperature.",
    )

    parser.add_argument(
        "--b0-scale",
        type=float,
        default=1.0,
        help="Empirical scale factor applied to B0 before computing Debye temperature.",
    )

    # SCF params.
    parser.add_argument("--ew", type=float, default=0.7)
    parser.add_argument("--ewidth", type=float, default=None, help="alias for --ew")
    parser.add_argument("--xc", type=str, default="pbe")
    parser.add_argument("--rel", type=str, default="sra", choices=["nrl", "sra", "srals"])
    parser.add_argument("--bzqlty", type=float, default=20)
    parser.add_argument("--pmix", type=float, default=0.01)
    parser.add_argument("--edelt", type=float, default=0.0001)
    parser.add_argument("--mxl", type=int, default=3)
    parser.add_argument("--magtype", default="nmag", choices=["nmag", "mag"])

    # RMT handling.
    parser.add_argument("--rmt-safety", type=float, default=0.995)
    parser.add_argument(
        "--rmt-common-physical-bohr",
        type=float,
        default=None,
        help="Optional fixed physical RMT in bohr. If provided, overrides automatic RMT selection.",
    )

    # Workflow.
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--no-gzip", action="store_true")

    args = vars(parser.parse_args())
    args["concentrations"] = [float(x) for x in args["concentrations"]]
    args["compress"] = not args.pop("no_gzip")

    if args.get("ewidth") is not None:
        args["ew"] = args["ewidth"]
    args.pop("ewidth", None)

    if args["deltas"] is None:
        if args["delta"] is None:
            raise ValueError("Provide either --delta or --deltas")
        args["deltas"] = [args["delta"]]
    args.pop("delta", None)

    result = run_kkr_elastic_debye(**args)
    print(result)