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
  is in its valence outer lobe and the potential is smooth/screened.

  Five cutoff modes are supported (--cutoff-mode):
    max     (default) r_cut = max(r_last_l, r_last_{l+1})
                      Removes core contamination from whichever wavefunction
                      extends furthest.  sp→r_last_p, pd→r_last_p, df→r_last_f
    min               r_cut = min(r_last_l, r_last_{l+1})
                      Uses the shallower of the two nodes.
    lower             r_cut = r_last_l   (lower-l wavefunction only)
                      sp→r_last_s, pd→r_last_p, df→r_last_d
    upper             r_cut = r_last_{l+1}  (higher-l wavefunction only)
                      sp→r_last_p, pd→r_last_d, df→r_last_f
                      For pd this removes only d-core contamination, not p-core.
                      Physically motivated if p valence wavefunctions are already
                      OPW-orthogonal to p core states (true in SRA KKR).
    valence           Apply per-wavefunction cutoff only when the actual node
                      count exceeds the expected number of genuine valence nodes
                      (stored in VALENCE_NODES_EXPECTED in src/consts.py).
                      If n_actual > n_expected, r_cut = r_last (contamination
                      present); otherwise r_cut = 0 (all nodes are physical).
                      r_cut for the channel = max(r_cut_l, r_cut_{l+1}).
                      Fixes the La df anomaly: La 5d has 2 genuine valence nodes
                      (n-l-1=2), so no cutoff is applied there, preserving the
                      inner negative lobe that cancels part of M_df.

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

try:
    from finalscf import run_kkr_finalscf as _run_kkr_finalscf
    _KKR_AVAILABLE = True
except ImportError:
    _KKR_AVAILABLE = False

try:
    from src.consts import VALENCE_NODES_EXPECTED as _VALENCE_NODES_EXPECTED
except ImportError:
    _VALENCE_NODES_EXPECTED: Dict[str, Dict[int, int]] = {}


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

_CUTOFF_MODES = ("max", "min", "lower", "upper", "valence")


def _resolve_r_cut(
    r_last_a: float,
    r_last_b: float,
    cutoff_mode: str,
    manual_r_cut: Optional[float],
    *,
    n_nodes_a: int = 0,
    n_nodes_b: int = 0,
    valence_expected_a: int = 0,
    valence_expected_b: int = 0,
) -> float:
    """Return the integration lower limit for one channel pair (a=lower-l, b=upper-l)."""
    if manual_r_cut is not None:
        return manual_r_cut
    if cutoff_mode == "max":
        return max(r_last_a, r_last_b)
    if cutoff_mode == "min":
        return min(r_last_a, r_last_b)
    if cutoff_mode == "lower":
        return r_last_a
    if cutoff_mode == "upper":
        return r_last_b
    if cutoff_mode == "valence":
        # Apply cutoff only when extra core-contamination nodes are present.
        effective_a = r_last_a if n_nodes_a > valence_expected_a else 0.0
        effective_b = r_last_b if n_nodes_b > valence_expected_b else 0.0
        return max(effective_a, effective_b)
    raise ValueError(f"Unknown cutoff_mode {cutoff_mode!r}; choose from {_CUTOFF_MODES}")


def compute_one_combination_cutoff(
    block: Dict[str, Any],
    integral_mode: str,
    norm_mode: str,
    reduce_mode: str,
    manual_r_cut: Optional[float] = None,
    cutoff_mode: str = "max",
    valence_nodes: Optional[Dict[int, int]] = None,
) -> Dict[str, Any]:
    """
    Compute both full and last-node-cutoff eta for one (block, mode) combination.

    valence_nodes: mapping {l: expected_node_count} for the element in this block,
                   used only when cutoff_mode="valence".  If None, treated as all zeros.
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
        "cutoff_mode": cutoff_mode,
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
        vn = valence_nodes or {}
        r_cut = _resolve_r_cut(
            r_last_a, r_last_b, cutoff_mode, manual_r_cut,
            n_nodes_a=len(nodes_a),
            n_nodes_b=len(nodes_b),
            valence_expected_a=vn.get(l, 0),
            valence_expected_b=vn.get(l + 1, 0),
        )

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


def plot_radial_functions(
    block: Dict[str, Any],
    workdir: str,
    output_prefix: str,
    manual_r_cut: Optional[float] = None,
    cutoff_mode: str = "max",
    logger: Optional[logging.Logger] = None,
    valence_nodes: Optional[Dict[int, int]] = None,
) -> None:
    """Save one PNG per orbital showing u_l, V, dV/dr and adjacent integrands."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if logger:
            logger.warning("matplotlib not available — skipping orbital plots")
        return

    x_full = np.array(block["xr"], dtype=float)
    v_full = np.array(block["v3"], dtype=float)
    mxlcmp = int(block["mxlcmp"])

    radial_key = "rstr_real_ef" if "rstr_real_ef" in block else "rstr_at_ef"
    x = x_full[:-1]
    v = v_full[:-1]
    raw_rstr = {
        int(k): np.array(vals[:-1], dtype=float)
        for k, vals in block[radial_key].items()
    }

    dVdr = derivative_nonuniform(x, v)
    # Use canonical reduce/norm for plotting
    radials = reduce_l_block_radials(raw_rstr, mxlcmp, x, "mean", "none")

    labels = ["s", "p", "d", "f", "g", "h"]
    channel_names = ["sp", "pd", "df", "fg", "gh"]
    comp = block["component"]
    spin = block["spin"]

    # Pre-compute r_cut per channel
    r_cuts: Dict[str, float] = {}
    vn = valence_nodes or {}
    for l in range(mxlcmp - 1):
        ch = channel_names[l]
        na = find_all_nodes(x, radials[labels[l]])
        nb = find_all_nodes(x, radials[labels[l + 1]])
        r_cuts[ch] = _resolve_r_cut(
            na[-1] if na else 0.0,
            nb[-1] if nb else 0.0,
            cutoff_mode,
            manual_r_cut,
            n_nodes_a=len(na),
            n_nodes_b=len(nb),
            valence_expected_a=vn.get(l, 0),
            valence_expected_b=vn.get(l + 1, 0),
        )

    for l in range(mxlcmp):
        lbl = labels[l]
        u = radials[lbl]

        _LOG_FLOOR = 10.0  # clip |dV/dr| below this before log

        # Collect panels: (y-data, y-label, panel-title, r_cut lines, log_scale)
        panels: List[tuple] = [
            (u,    f"u_{lbl}(r)",          f"wavefunction  u_{lbl}(r)",         [], False),
            (v,    "V(r)  [Ry]",           "potential  V(r)",                   [], False),
            (dVdr, "dV/dr  [Ry/Bohr]",     "potential gradient  dV/dr",         [], False),
            (np.abs(dVdr), f"|dV/dr|  (floor={_LOG_FLOOR:.0f})  [Ry/Bohr]",
             f"|dV/dr|  log scale  (clipped below {_LOG_FLOOR:.0f})",            [], True),
        ]
        # Adjacent channels — integrand panels
        if l < mxlcmp - 1:
            ch = channel_names[l]
            b = labels[l + 1]
            integrand = u * dVdr * radials[b]
            panels.append((
                integrand,
                f"u_{lbl}·(dV/dr)·u_{b}",
                f"integrand  u_{lbl}·(dV/dr)·u_{b}  [{ch} channel]",
                [(r_cuts[ch], ch)],
                False,
            ))
        if l > 0:
            ch = channel_names[l - 1]
            a = labels[l - 1]
            integrand = radials[a] * dVdr * u
            panels.append((
                integrand,
                f"u_{a}·(dV/dr)·u_{lbl}",
                f"integrand  u_{a}·(dV/dr)·u_{lbl}  [{ch} channel]",
                [(r_cuts[ch], ch)],
                False,
            ))

        n = len(panels)
        fig, axes = plt.subplots(n, 1, figsize=(9, 2.8 * n), sharex=True)
        if n == 1:
            axes = [axes]

        cutoff_label = (
            f"manual r_cut={manual_r_cut:.4f} Bohr"
            if manual_r_cut is not None
            else f"auto ({cutoff_mode})"
        )
        fig.suptitle(
            f"Orbital  {lbl.upper()}  —  component={comp}  spin={spin}\n"
            f"cutoff: {cutoff_label}",
            fontsize=11,
        )

        for ax, panel in zip(axes, panels):
            y, ylabel, title, rlines, log_scale = panel
            if log_scale:
                y_plot = np.clip(y, _LOG_FLOOR, None)
                ax.semilogy(x, y_plot, lw=1.0, color="steelblue")
                ax.set_ylim(bottom=_LOG_FLOOR)
            else:
                ax.plot(x, y, lw=1.0, color="steelblue")
                ax.axhline(0, color="k", lw=0.5, ls=":")
            for rc, ch in rlines:
                ax.axvline(rc, color="tomato", lw=1.0, ls="--",
                           label=f"r_cut_{ch} = {rc:.4f} Bohr")
                ax.legend(fontsize=7, loc="upper right")
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=8)
            ax.grid(True, alpha=0.25)

        axes[-1].set_xlabel("r  [Bohr]", fontsize=9)
        plt.tight_layout()

        fname = os.path.join(
            workdir,
            f"{output_prefix}_orbital_{lbl}_comp{comp}_spin{spin}.png",
        )
        fig.savefig(fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        if logger:
            logger.info("Orbital plot saved: %s", fname)


def sweep_combinations_cutoff(
    block: Dict[str, Any],
    manual_r_cut: Optional[float] = None,
    cutoff_mode: str = "max",
    valence_nodes: Optional[Dict[int, int]] = None,
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
                    cutoff_mode=cutoff_mode,
                    valence_nodes=valence_nodes,
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
    cutoff_mode: str = "max",
    # KKR invocation
    run_kkr: bool = True,
    elements: Optional[List[str]] = None,
    concentrations: Optional[List[float]] = None,
    a0: Optional[float] = None,
    sym: str = "bcc",
    ew: float = 0.7,
    xc: str = "pbe",
    rel: str = "nrl",
    bzqlty: float = 10,
    pmix: float = 0.01,
    edelt: float = 0.001,
    mxl: int = 3,
) -> Dict[str, Any]:
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

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
            logger.info("Cutoff mode: %s", cutoff_mode)

        if run_kkr:
            if not _KKR_AVAILABLE:
                raise RuntimeError("run_kkr=True but finalscf.py could not be imported")
            if elements is None or concentrations is None or a0 is None:
                raise ValueError("--elements, --concentrations and --a0 are required when --run-kkr is set")
            logger.info(
                "Running KKR finalscf: sym=%s a0=%.5f elements=%s conc=%s",
                sym, a0, elements, concentrations,
            )
            _run_kkr_finalscf(
                workdir=workdir,
                output=output_prefix,
                sym=sym,
                elements=elements,
                concentrations=concentrations,
                a0=a0,
                ew=ew,
                xc=xc,
                rel=rel,
                bzqlty=bzqlty,
                pmix=pmix,
                edelt=edelt,
                mxl=mxl,
            )
            logger.info("KKR finalscf finished")

        fort51_path = os.path.join(workdir, "fort.51")
        if not os.path.exists(fort51_path):
            raise FileNotFoundError(f"fort.51 not found: {fort51_path}")

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

            # Resolve element label and valence node table for this block.
            comp_idx = int(block["component"]) - 1  # fort.51 components are 1-indexed
            elem_label: Optional[str] = None
            if elements is not None and 0 <= comp_idx < len(elements):
                elem_label = elements[comp_idx]
            block_valence_nodes: Optional[Dict[int, int]] = None
            if cutoff_mode == "valence" and elem_label is not None:
                block_valence_nodes = _VALENCE_NODES_EXPECTED.get(elem_label)
                if block_valence_nodes is None:
                    logger.warning(
                        "cutoff_mode=valence: no VALENCE_NODES_EXPECTED entry for %s; "
                        "treating all expected nodes as 0 (same as max mode)",
                        elem_label,
                    )
                    block_valence_nodes = {}

            plot_radial_functions(
                block, workdir, output_prefix, manual_r_cut, cutoff_mode, logger,
                valence_nodes=block_valence_nodes,
            )
            df = sweep_combinations_cutoff(
                block,
                manual_r_cut=manual_r_cut,
                cutoff_mode=cutoff_mode,
                valence_nodes=block_valence_nodes,
            )
            df.insert(0, "fort51_path", fort51_path)
            if elem_label is not None:
                df.insert(1, "element", elem_label)
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
    ap.add_argument(
        "--cutoff-mode",
        default="max",
        choices=list(_CUTOFF_MODES),
        help=(
            "How to pick r_cut from the two last-node positions for each channel pair. "
            "'max' (default): max(r_last_l, r_last_{l+1}). "
            "'min': min of the two. "
            "'lower': r_last_l only (lower angular momentum wavefunction). "
            "'upper': r_last_{l+1} only (higher-l wavefunction). "
            "'valence': apply cutoff only when n_actual_nodes > expected valence nodes "
            "(from VALENCE_NODES_EXPECTED in src/consts.py); otherwise no cutoff. "
            "Requires --elements to identify each component. "
            "Fixes La df anomaly: La 5d has 2 genuine valence nodes, so r_cut_df=0."
        ),
    )

    # KKR invocation
    kkr = ap.add_argument_group("KKR invocation (required when --run-kkr, ignored otherwise)")
    kkr.add_argument(
        "--run-kkr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run KKR finalscf before computing eta (default: true). "
             "Set --no-run-kkr to read pre-computed fort.51/fort.50 from workdir.",
    )
    kkr.add_argument("--elements", nargs="+", default=None, help="Element symbols, e.g. Ta Nb Hf")
    kkr.add_argument("--concentrations", nargs="+", type=float, default=None,
                     help="Concentrations matching --elements (must sum to 1)")
    kkr.add_argument("--a0", type=float, default=None, help="Lattice constant in Bohr")
    kkr.add_argument("--sym", default="bcc", choices=["bcc", "fcc"], help="Crystal structure")
    kkr.add_argument("--ew", type=float, default=0.7)
    kkr.add_argument("--xc", default="pbe")
    kkr.add_argument("--rel", default="nrl", choices=["nrl", "sra", "srals"])
    kkr.add_argument("--bzqlty", type=float, default=10)
    kkr.add_argument("--pmix", type=float, default=0.01)
    kkr.add_argument("--edelt", type=float, default=0.001)
    kkr.add_argument("--mxl", type=int, default=3)

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
        cutoff_mode=args.cutoff_mode,
        run_kkr=args.run_kkr,
        elements=args.elements,
        concentrations=args.concentrations,
        a0=args.a0,
        sym=args.sym,
        ew=args.ew,
        xc=args.xc,
        rel=args.rel,
        bzqlty=args.bzqlty,
        pmix=args.pmix,
        edelt=args.edelt,
        mxl=args.mxl,
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
