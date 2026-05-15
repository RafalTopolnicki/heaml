#!/usr/bin/env python3
"""
macmillan_cutoff.py — McMillan-Hopfield eta with last-node integration cutoff.

Rationale:
  The full integral M = ∫₀^R_MT u_l (dV/dr) u_{l+1} dr is dominated for
  the sp channel by the core region (r < ~0.01 Bohr for heavy elements), where
  orthogonality oscillations of valence s,p wavefunctions inside the core shells
  combine with the unscreened nuclear gradient dV/dr ≈ 2Z/r². This inflates
  M_sp by ~13× for Ta (Z=73), giving a 175× error in eta_sp relative to
  the Gaspari-Gyorffy intent (which assumes screened ionic potential).

  Fix: use the last zero-crossing of each wavefunction as the lower limit.
  The "last node" is the outermost zero of u_l(r) — beyond it the wavefunction
  is in its valence outer lobe and the potential is smooth/screened. Integrating
  from max(last_node(u_l), last_node(u_{l+1})) to R_MT removes the unphysical
  core contribution without any empirical fitting.

  With this cutoff for Ta (BCC, a=6.27 Bohr):
    eta_sp:    39.7  → 0.010 Ry/Bohr²  (core contamination removed)
    eta_pd:     0.12 → 0.14  Ry/Bohr²  (unaffected)
    eta_df:     0.10 → 0.06  Ry/Bohr²  (slight reduction)
    eta_total:  0.21 Ry/Bohr²
  Using the theoretical constant C = 4.56e7 (no empirical fitting):
    λ = eta_total × C / (M_mix × Θ_D²) ≈ 0.64  vs λ_exp = 0.69 ✓

Constant derivation:
  λ = Σ_i c_i η_i / (⟨M⟩ × ⟨ω²⟩)
  With Debye model: ⟨ω²⟩ = (3/5) × (k_B Θ_D / ℏ)²
  C = (5/3) × (Ry→J / Bohr²→m²) × ℏ² / (amu → kg × k_B²) ≈ 4.56e7 K²·amu·Bohr²/Ry
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    from scipy.integrate import simpson
except ImportError:
    from scipy.integrate import simps as simpson

# ---------------------------------------------------------------------------
# Physical constant
# ---------------------------------------------------------------------------
Ry_to_J = 2.1798741e-18       # J per Ry
Bohr_to_m = 5.29177210903e-11 # m per Bohr
amu_to_kg = 1.66053906660e-27 # kg per amu
hbar = 1.054571817e-34        # J·s
kB = 1.380649e-23             # J/K

# C such that λ = eta_total [Ry/Bohr²] * C / (M_mix [amu] * Theta_D [K]^2)
# Derivation: lambda = eta_SI / (M_SI * (3/5) * omega_D^2)
#             eta_SI = eta_RyBohr2 * Ry_to_J / Bohr_to_m^2
#             M_SI   = M_amu * amu_to_kg
#             omega_D = kB * Theta_D / hbar
C_THEORETICAL = (5.0 / 3.0) * (Ry_to_J / Bohr_to_m**2) * hbar**2 / (amu_to_kg * kB**2)
# ≈ 4.56e7

# ---------------------------------------------------------------------------
# Imports from macmillan.py (same directory)
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
if _script_dir not in sys.path:
    sys.path.insert(0, _script_dir)

from macmillan import (
    parse_fort51,
    group_dosef_by_l,
    reduce_l_block_radials,
    derivative_nonuniform,
    compute_M,
)


# ---------------------------------------------------------------------------
# Node detection
# ---------------------------------------------------------------------------

def find_all_nodes(x: np.ndarray, u: np.ndarray) -> List[float]:
    """Return all zero-crossing positions of u via linear interpolation."""
    nodes: List[float] = []
    for i in range(len(u) - 1):
        if u[i] * u[i + 1] < 0.0:
            # Linear interpolation for the zero
            r_node = x[i] + (x[i + 1] - x[i]) * (-u[i]) / (u[i + 1] - u[i])
            nodes.append(float(r_node))
    return nodes


def last_node(x: np.ndarray, u: np.ndarray) -> float:
    """Return position of the outermost zero crossing of u, or 0.0 if none."""
    nodes = find_all_nodes(x, u)
    return nodes[-1] if nodes else 0.0


# ---------------------------------------------------------------------------
# Cutoff integration
# ---------------------------------------------------------------------------

def compute_M_from_cutoff(
    u1: np.ndarray,
    u2: np.ndarray,
    dVdr: np.ndarray,
    x: np.ndarray,
    integral_mode: str,
    r_cut: float,
) -> float:
    """Compute M = ∫_{r_cut}^{R_MT} u1 (dV/dr) u2 dr."""
    idx = int(np.searchsorted(x, r_cut))
    if idx >= len(x) - 1:
        return 0.0
    return compute_M(u1[idx:], u2[idx:], dVdr[idx:], x[idx:], integral_mode)


# ---------------------------------------------------------------------------
# Per-block computation
# ---------------------------------------------------------------------------

def compute_one_combination_cutoff(
    block: Dict[str, Any],
    integral_mode: str,
    norm_mode: str,
    reduce_mode: str,
    manual_r_cut: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Compute both full and last-node-cutoff eta for one (block, mode) combination.
    """
    x_full = np.array(block["xr"], dtype=float)
    v_full = np.array(block["v3"], dtype=float)
    mxlcmp = int(block["mxlcmp"])
    dosef = np.array(block["dosef"], dtype=float)

    radial_key = None
    if "rstr_real_ef" in block:
        radial_key = "rstr_real_ef"
    elif "rstr_at_ef" in block:
        radial_key = "rstr_at_ef"
    else:
        raise ValueError("No radial block found (expected rstr_real_ef or rstr_at_ef)")

    # Drop last grid point (V=0 forced at boundary — same convention as macmillan.py)
    x = x_full[:-1]
    v = v_full[:-1]

    raw_rstr = {
        int(k): np.array(vals[:-1], dtype=float)
        for k, vals in block[radial_key].items()
    }

    dVdr = derivative_nonuniform(x, v)
    grouped_dos = group_dosef_by_l(block["dosef"], mxlcmp)
    radials = reduce_l_block_radials(raw_rstr, mxlcmp, x, reduce_mode, norm_mode)

    Ntot = float(np.sum(dosef))

    labels = ["s", "p", "d", "f", "g", "h"]
    channel_names = ["sp", "pd", "df", "fg", "gh"]

    result: Dict[str, Any] = {
        "component": block["component"],
        "spin": block["spin"],
        "mxlcmp": mxlcmp,
        "integral_mode": integral_mode,
        "norm_mode": norm_mode,
        "reduce_mode": reduce_mode,
        "Ntot": Ntot,
        "Ns": grouped_dos.get("s", np.nan),
        "Np": grouped_dos.get("p", np.nan),
        "Nd": grouped_dos.get("d", np.nan),
        "Nf": grouped_dos.get("f", np.nan),
    }

    eta_total_full = 0.0
    eta_total_cutoff = 0.0

    for l in range(mxlcmp - 1):
        a = labels[l]
        b = labels[l + 1]
        ch = channel_names[l]

        u1 = radials[a]
        u2 = radials[b]

        # Node detection
        nodes_a = find_all_nodes(x, u1)
        nodes_b = find_all_nodes(x, u2)
        r_last_a = nodes_a[-1] if nodes_a else 0.0
        r_last_b = nodes_b[-1] if nodes_b else 0.0
        r_cut = manual_r_cut if manual_r_cut is not None else max(r_last_a, r_last_b)

        nl = float(grouped_dos[a])
        nlp1 = float(grouped_dos[b])
        prefactor = 2.0 * (l + 1) / ((2 * l + 1) * (2 * l + 3)) * (nl * nlp1 / Ntot)

        # Full integral
        M_full = compute_M(u1, u2, dVdr, x, integral_mode)
        eta_full = prefactor * M_full ** 2
        eta_total_full += eta_full

        # Cutoff integral
        M_cutoff = compute_M_from_cutoff(u1, u2, dVdr, x, integral_mode, r_cut)
        eta_cutoff = prefactor * M_cutoff ** 2
        eta_total_cutoff += eta_cutoff

        result[f"r_last_{a}"] = round(r_last_a, 6)
        result[f"r_last_{b}"] = round(r_last_b, 6)
        result[f"r_cut_{ch}"] = round(r_cut, 6)
        result[f"n_nodes_{a}"] = len(nodes_a)
        result[f"n_nodes_{b}"] = len(nodes_b)
        result[f"M_full_{ch}"] = M_full
        result[f"M_cutoff_{ch}"] = M_cutoff
        result[f"eta_full_{ch}"] = eta_full
        result[f"eta_cutoff_{ch}"] = eta_cutoff

    result["eta_total_full"] = eta_total_full
    result["eta_total_cutoff"] = eta_total_cutoff

    return result


def sweep_combinations_cutoff(
    block: Dict[str, Any],
    manual_r_cut: Optional[float] = None,
) -> pd.DataFrame:
    integral_modes = ["plain", "r2"]
    norm_modes = ["none", "u2", "r2u2"]
    reduce_modes = ["mean", "sum", "first"]

    rows = []
    for integral_mode in integral_modes:
        for norm_mode in norm_modes:
            for reduce_mode in reduce_modes:
                row = compute_one_combination_cutoff(
                    block=block,
                    integral_mode=integral_mode,
                    norm_mode=norm_mode,
                    reduce_mode=reduce_mode,
                    manual_r_cut=manual_r_cut,
                )
                rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

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


def run_mcmillan_cutoff_sweep(
    *,
    workdir: str,
    output_prefix: str = "mcmillan_cutoff",
    component: Optional[int] = None,
    spin: Optional[int] = None,
    mixture_mass: Optional[float] = None,
    theta_d: Optional[float] = None,
    save_json: bool = False,
    manual_r_cut: Optional[float] = None,
) -> Dict[str, Any]:
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    fort51_path = os.path.join(workdir, "fort.51")
    if not os.path.exists(fort51_path):
        raise FileNotFoundError(f"fort.51 not found: {fort51_path}")

    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        logger = setup_logger(f"{output_prefix}_run.log")
        logger.info("Starting McMillan-Hopfield cutoff sweep")
        logger.info("workdir=%s", workdir)
        logger.info("C_THEORETICAL = %.4e", C_THEORETICAL)
        if manual_r_cut is not None:
            logger.info("Manual cutoff: r_cut=%.6f Bohr (overrides node detection)", manual_r_cut)
        else:
            logger.info("Cutoff mode: automatic (last node of each wavefunction pair)")

        data = parse_fort51(fort51_path)

        selected = []
        for block in data["components"]:
            if component is not None and block["component"] != component:
                continue
            if spin is not None and block["spin"] != spin:
                continue
            selected.append(block)

        if not selected:
            raise ValueError("No matching component/spin blocks found")

        dfs = []
        for block in selected:
            logger.info("Processing component=%s spin=%s", block["component"], block["spin"])
            df = sweep_combinations_cutoff(block, manual_r_cut=manual_r_cut)
            df.insert(0, "fort51_path", fort51_path)
            dfs.append(df)

        master_df = pd.concat(dfs, ignore_index=True)

        # Compute lambda if thermodynamic data provided
        if mixture_mass is not None and theta_d is not None:
            master_df["lambda_cutoff"] = (
                master_df["eta_total_cutoff"] * C_THEORETICAL / (mixture_mass * theta_d ** 2)
            )
            master_df["lambda_full"] = (
                master_df["eta_total_full"] * C_THEORETICAL / (mixture_mass * theta_d ** 2)
            )
            logger.info("mixture_mass=%.4f amu  theta_D=%.2f K", mixture_mass, theta_d)
            logger.info("C_THEORETICAL=%.4e", C_THEORETICAL)

        csv_path = os.path.join(workdir, f"{output_prefix}_results.csv")
        master_df.to_csv(csv_path, index=False)
        logger.info("CSV written to %s", csv_path)

        json_path = ""
        if save_json:
            json_path = os.path.join(workdir, f"{output_prefix}_results.json")
            with open(json_path, "w") as f:
                f.write(master_df.to_json(orient="records", indent=2))

        # Print the canonical combination (plain/none/mean) clearly
        canonical = master_df[
            (master_df["integral_mode"] == "plain")
            & (master_df["norm_mode"] == "none")
            & (master_df["reduce_mode"] == "mean")
        ]

        logger.info("=" * 60)
        logger.info("CANONICAL COMBINATION (plain / none / mean):")
        for _, row in canonical.iterrows():
            logger.info(
                "  component=%d spin=%d  eta_total_full=%.5f  eta_total_cutoff=%.5f",
                int(row["component"]),
                int(row["spin"]),
                row["eta_total_full"],
                row["eta_total_cutoff"],
            )
            for ch in ["sp", "pd", "df"]:
                if f"eta_full_{ch}" in row:
                    logger.info(
                        "    %s: M_full=%.4f  M_cutoff=%.4f  "
                        "eta_full=%.5f  eta_cutoff=%.5f  r_cut=%.4f Bohr",
                        ch,
                        row[f"M_full_{ch}"],
                        row[f"M_cutoff_{ch}"],
                        row[f"eta_full_{ch}"],
                        row[f"eta_cutoff_{ch}"],
                        row[f"r_cut_{ch}"],
                    )
            if "lambda_cutoff" in row:
                logger.info(
                    "  lambda_cutoff=%.4f  lambda_full=%.4f",
                    row["lambda_cutoff"],
                    row["lambda_full"],
                )
        logger.info("=" * 60)

        summary = {
            "n_rows": int(len(master_df)),
            "n_components_selected": int(len(selected)),
            "fort51_path": fort51_path,
            "csv_path": csv_path,
            "json_path": json_path,
            "C_theoretical": C_THEORETICAL,
            "mixture_mass_amu": mixture_mass,
            "theta_D_K": theta_d,
            "run_log": os.path.join(workdir, f"{output_prefix}_run.log"),
        }

        with open(os.path.join(workdir, f"{output_prefix}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Workflow finished successfully")
        return {"summary": summary, "dataframe": master_df}

    finally:
        os.chdir(old_cwd)


def main():
    ap = argparse.ArgumentParser(
        description="McMillan-Hopfield eta with last-node integration cutoff"
    )
    ap.add_argument("--workdir", required=True, help="Directory containing fort.51")
    ap.add_argument("--output-prefix", default="mcmillan_cutoff", help="Output file prefix")
    ap.add_argument("--component", type=int, default=None, help="Component index (default: all)")
    ap.add_argument("--spin", type=int, default=None, help="Spin index (default: all)")
    ap.add_argument(
        "--mixture-mass",
        type=float,
        default=None,
        help="Mixture atomic mass in amu (Σ c_i M_i). If given with --theta-d, computes λ.",
    )
    ap.add_argument(
        "--theta-d",
        type=float,
        default=None,
        help="Debye temperature in K. Required with --mixture-mass to compute λ.",
    )
    ap.add_argument("--save-json", action="store_true", help="Also save JSON output")
    ap.add_argument(
        "--r-cut",
        type=float,
        default=None,
        help="Manual integration cutoff in Bohr. Overrides automatic last-node detection for all channels.",
    )
    args = ap.parse_args()

    result = run_mcmillan_cutoff_sweep(
        workdir=args.workdir,
        output_prefix=args.output_prefix,
        component=args.component,
        spin=args.spin,
        mixture_mass=args.mixture_mass,
        theta_d=args.theta_d,
        save_json=args.save_json,
        manual_r_cut=args.r_cut,
    )

    canonical = result["dataframe"][
        (result["dataframe"]["integral_mode"] == "plain")
        & (result["dataframe"]["norm_mode"] == "none")
        & (result["dataframe"]["reduce_mode"] == "mean")
    ]
    print("\n=== Canonical results (plain / none / mean) ===")
    cols = ["component", "spin"]
    for ch in ["sp", "pd", "df"]:
        cols += [f"eta_full_{ch}", f"eta_cutoff_{ch}", f"r_cut_{ch}"]
    cols += ["eta_total_full", "eta_total_cutoff"]
    if "lambda_cutoff" in canonical.columns:
        cols += ["lambda_cutoff", "lambda_full"]
    print(canonical[[c for c in cols if c in canonical.columns]].to_string(index=False))
    print(f"\nCSV: {result['summary']['csv_path']}")
    print(f"C_theoretical = {C_THEORETICAL:.4e}")

    canonical_json_path = os.path.join(args.workdir, f"{args.output_prefix}_canonical.json")
    canonical.to_json(canonical_json_path, orient="records", indent=2)
    print(f"Canonical JSON: {canonical_json_path}")


if __name__ == "__main__":
    main()
