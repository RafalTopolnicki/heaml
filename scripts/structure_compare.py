#!/usr/bin/env python3
"""
structure_compare.py

Compare BCC, FCC, and HCP structural stability for a given HEA composition
via AkaiKKR total energy calculations with Birch-Murnaghan EOS fitting.

For BCC/FCC: 1D lattice scan.
For HCP:     2D grid over (a, c/a); for each c/a a 1D EOS is fitted, then
             E0(c/a) is minimised with a parabolic fit.

Output layout:
  thermodynamic/{composition}/
    bcc/   bcc_*.inp, bcc_*.out.gz, bcc_results.csv, bcc_summary.txt
    fcc/   fcc_*.inp, fcc_*.out.gz, fcc_results.csv, fcc_summary.txt
    hcp/   hcp_{ca}_*.inp, hcp_{ca}_*.out.gz, hcp_results.csv,
           hcp_ca_summary.csv, hcp_summary.txt
    structure_summary.json
"""

import argparse
import os
import subprocess
import sys
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# ── allow running as  python scripts/structure_compare.py  from the repo root
_here = os.path.dirname(os.path.abspath(__file__))
if _here not in sys.path:
    sys.path.insert(0, _here)

from src.elements import HEAClass
from src.consts import AKAIBIN, KKR_PARAMS_LATTICE
from src.utils import (
    dist_to_si, dist_from_si, energy_to_si, energy_from_si,
    parse_energy, converged_info_in_string,
    gzip_file, cleanup_potential_files, cleanup_fortran_files,
    generate_dirname, save_dict_to_json,
)
from src.write_akai_input import scf_input_bcc, scf_input_hcp


# ── physical constants shared with lattice.py
def birch_murnaghan(v, e0, b0, bP, v0):
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + (9 * v0 * b0 / 16) * (
        (eta - 1) ** 3 * bP + (eta - 1) ** 2 * (6 - 4 * eta)
    )


# ── volume formulas (per atom, SI units) ─────────────────────────────────────

def vol_bcc(a_si):
    """Primitive BCC: 1 atom, V = a³/2."""
    return a_si ** 3 / 2.0


def vol_fcc(a_si):
    """Primitive FCC: 1 atom, V = a³/4."""
    return a_si ** 3 / 4.0


def vol_hcp_per_atom(a_si, ca):
    """HCP: 2 atoms per unit cell, V_cell = sqrt(3)/2 * a² * c → V/atom = sqrt(3)/4 * a³ * (c/a)."""
    return (np.sqrt(3) / 4.0) * a_si ** 3 * ca


def volume_to_a_bcc(v_si):
    return (v_si * 2.0) ** (1.0 / 3.0)


def volume_to_a_fcc(v_si):
    return (v_si * 4.0) ** (1.0 / 3.0)


def volume_to_a_hcp(v_si, ca):
    return (v_si * 4.0 / (np.sqrt(3) * ca)) ** (1.0 / 3.0)


# ── generic EOS fitter ────────────────────────────────────────────────────────

def fit_eos_generic(lattices_bohr, energies_ry, converged_mask, vol_fn, vol_to_a_fn):
    """
    Fit Birch-Murnaghan EOS.

    vol_fn(a_si)     → V_per_atom in SI
    vol_to_a_fn(v_si) → a in SI  (inverse of vol_fn)

    Returns (a0_bohr, E0_ry, B0_gpa) or (None, None, None) on failure.
    """
    mask = np.asarray(converged_mask, dtype=bool)
    latt = np.asarray(lattices_bohr)[mask]
    ener = np.asarray(energies_ry)[mask]

    if len(latt) < 4:
        return None, None, None

    e = energy_to_si(ener)
    l = dist_to_si(latt)
    v = vol_fn(l)

    a2, a1, a0 = np.polyfit(v, e, 2)
    v0 = -a1 / (2 * a2)
    e0 = a2 * v0 ** 2 + a1 * v0 + a0
    b0 = max(2 * a2 * v0, 1e9)

    try:
        popt, _ = curve_fit(birch_murnaghan, v, e, p0=[e0, b0, 4.0, v0], maxfev=10000)
    except RuntimeError:
        return None, None, None

    e0, b0, bP, v0 = popt
    a0_bohr = dist_from_si(vol_to_a_fn(v0))
    E0_ry = energy_from_si(e0)
    B0_gpa = b0 / 1e9
    return a0_bohr, E0_ry, B0_gpa


# ── single KKR run ────────────────────────────────────────────────────────────

def _run_single(workdir, filename_base, inp_writer_fn, kkr_params):
    """
    Write input, run AkaiKKR, parse output.
    inp_writer_fn(filename, **kkr_params) must create the .inp file.
    Returns (energy_ry, converged).
    """
    inp = os.path.join(workdir, filename_base + ".inp")
    out = os.path.join(workdir, filename_base + ".out")
    inp_writer_fn(filename=inp, **kkr_params)
    with open(inp, "r") as fin, open(out, "w") as fout:
        subprocess.run([AKAIBIN], stdin=fin, stdout=fout)
    text = open(out).read()
    energy = parse_energy(text)
    conv = converged_info_in_string(text)
    cleanup_potential_files(inp.replace(".inp", ""))
    cleanup_fortran_files(inp.replace(".inp", ""))
    gzip_file(out)
    return energy, conv


# ── BCC / FCC scan ───────────────────────────────────────────────────────────

def run_cubic_eos(structure, a_init_bohr, hea_cfg, kkr_p, workdir, n_steps=7):
    """
    Scan lattice constants for BCC or FCC, fit EOS.

    structure : 'bcc' or 'fcc'
    Returns dict with lattice_constant_bohr, energy_per_atom_ry, bulk_modulus_gpa.
    """
    os.makedirs(workdir, exist_ok=True)
    latt_min = a_init_bohr * kkr_p.get('min_lattice_prop', 0.95)
    latt_max = a_init_bohr * kkr_p.get('max_lattice_prop', 1.05)
    lattices = np.linspace(latt_min, latt_max, n_steps)

    base_params = dict(
        elements=hea_cfg['elements'],
        concentrations=hea_cfg['concentrations'],
        ew=kkr_p.get('ew', 0.6),
        xc=kkr_p.get('xc', 'pbe'),
        rel=kkr_p.get('rel', 'sra'),
        bzqlty=kkr_p.get('bzqlty', 10),
        pmix=kkr_p.get('pmix', 0.01),
        edelt=kkr_p.get('edelt', 0.001),
        mxl=kkr_p.get('mxl', 3),
        magtype=kkr_p.get('magtype', 'nmag'),
        sym=structure,
    )

    def writer(filename, **kw):
        scf_input_bcc(filename=filename, lattice_params={"lattice_constant": kw['lattice']},
                      elements=kw['elements'], concentrations=kw['concentrations'],
                      ew=kw['ew'], xc=kw['xc'], rel=kw['rel'], bzqlty=kw['bzqlty'],
                      pmix=kw['pmix'], edelt=kw['edelt'], mxl=kw['mxl'],
                      magtype=kw['magtype'], sym=kw['sym'])

    rows = []
    for a in lattices:
        fname = f"{structure}_{a:.6f}"
        run_params = dict(base_params, lattice=a)
        e, conv = _run_single(workdir, fname, writer, run_params)
        rows.append({"lattice": a, "energy": e, "converged": conv})
        print(f"  {structure.upper()} a={a:.4f} Bohr  E={e:.6f} Ry  conv={conv}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(workdir, f"{structure}_results.csv"), index=False)

    vol_fn = vol_bcc if structure == 'bcc' else vol_fcc
    vol_inv = volume_to_a_bcc if structure == 'bcc' else volume_to_a_fcc
    a0, E0, B0 = fit_eos_generic(df["lattice"].values, df["energy"].values,
                                  df["converged"].values, vol_fn, vol_inv)

    with open(os.path.join(workdir, f"{structure}_summary.txt"), "w") as f:
        f.write(f"structure = {structure}\n")
        f.write(f"a0        = {a0}\n")
        f.write(f"E0        = {E0}\n")
        f.write(f"B0        = {B0}\n")

    print(f"  → {structure.upper()} EOS: a0={a0:.4f} Bohr  E0={E0:.6f} Ry  B0={B0:.1f} GPa")
    return {"lattice_constant_bohr": a0, "energy_per_atom_ry": E0, "bulk_modulus_gpa": B0}


# ── HCP 2D scan ───────────────────────────────────────────────────────────────

def run_hcp_eos(a_init_bohr, hea_cfg, kkr_p, workdir, ca_values=None, n_a_steps=7):
    """
    2D grid search over (c/a, a) for HCP.
    For each c/a: scan a, fit 1D EOS → E0(c/a).
    Fit parabola over c/a → find optimal c/a* and E0_HCP.

    Returns dict with lattice_constant_a_bohr, c_over_a, energy_per_atom_ry, bulk_modulus_gpa.
    """
    if ca_values is None:
        ca_values = [1.50, 1.57, 1.633, 1.70, 1.77]

    os.makedirs(workdir, exist_ok=True)

    base_params = dict(
        elements=hea_cfg['elements'],
        concentrations=hea_cfg['concentrations'],
        ew=kkr_p.get('ew', 0.6),
        xc=kkr_p.get('xc', 'pbe'),
        rel=kkr_p.get('rel', 'sra'),
        bzqlty=kkr_p.get('bzqlty', 10),
        pmix=kkr_p.get('pmix', 0.01),
        edelt=kkr_p.get('edelt', 0.001),
        mxl=kkr_p.get('mxl', 3),
        magtype=kkr_p.get('magtype', 'nmag'),
    )

    def writer(filename, **kw):
        scf_input_hcp(filename=filename,
                      lattice_params={"lattice_constant": kw['lattice'], "c/a": kw['ca']},
                      elements=kw['elements'], concentrations=kw['concentrations'],
                      ew=kw['ew'], xc=kw['xc'], rel=kw['rel'], bzqlty=kw['bzqlty'],
                      pmix=kw['pmix'], edelt=kw['edelt'], mxl=kw['mxl'],
                      magtype=kw['magtype'])

    all_rows = []
    ca_results = []

    latt_min = a_init_bohr * kkr_p.get('min_lattice_prop', 0.95)
    latt_max = a_init_bohr * kkr_p.get('max_lattice_prop', 1.05)
    lattices = np.linspace(latt_min, latt_max, n_a_steps)

    for ca in ca_values:
        ca_label = f"{ca:.4f}".replace(".", "p")
        rows = []
        for a in lattices:
            fname = f"hcp_{ca_label}_{a:.6f}"
            run_params = dict(base_params, lattice=a, ca=ca)
            # HCP natm=2 → AkaiKKR energy is for 2 atoms; divide by 2 for per-atom
            e_total, conv = _run_single(workdir, fname, writer, run_params)
            e_per_atom = e_total / 2.0 if not np.isnan(e_total) else np.nan
            rows.append({"lattice": a, "ca": ca, "energy": e_per_atom, "converged": conv})
            print(f"  HCP c/a={ca:.4f} a={a:.4f} Bohr  E/atom={e_per_atom:.6f} Ry  conv={conv}")
        all_rows.extend(rows)

        # fit 1D EOS at this c/a
        this_df = pd.DataFrame(rows)
        a0, E0, B0 = fit_eos_generic(
            this_df["lattice"].values, this_df["energy"].values,
            this_df["converged"].values,
            lambda a_si, _ca=ca: vol_hcp_per_atom(a_si, _ca),
            lambda v_si, _ca=ca: volume_to_a_hcp(v_si, _ca),
        )
        ca_results.append({"ca": ca, "a0_bohr": a0, "E0_ry": E0, "B0_gpa": B0})
        print(f"  → HCP c/a={ca:.4f} EOS: a0={a0}  E0={E0}  B0={B0}")

    df_all = pd.DataFrame(all_rows)
    df_all.to_csv(os.path.join(workdir, "hcp_results.csv"), index=False)

    df_ca = pd.DataFrame(ca_results).dropna()
    df_ca.to_csv(os.path.join(workdir, "hcp_ca_summary.csv"), index=False)

    # parabolic fit over c/a to find minimum E0
    ca_opt, E0_hcp, a0_opt, B0_opt = _fit_ca(df_ca)

    with open(os.path.join(workdir, "hcp_summary.txt"), "w") as f:
        f.write(f"structure      = hcp\n")
        f.write(f"a0             = {a0_opt}\n")
        f.write(f"c/a            = {ca_opt}\n")
        f.write(f"E0_per_atom    = {E0_hcp}\n")
        f.write(f"B0             = {B0_opt}\n")

    print(f"  → HCP optimum: a0={a0_opt:.4f} Bohr  c/a={ca_opt:.4f}  E0={E0_hcp:.6f} Ry")
    return {
        "lattice_constant_a_bohr": a0_opt,
        "c_over_a": ca_opt,
        "energy_per_atom_ry": E0_hcp,
        "bulk_modulus_gpa": B0_opt,
    }


def _fit_ca(df_ca):
    """Fit parabola to E0(c/a) and return (ca_opt, E0_min, a0_at_opt, B0_at_opt)."""
    ca_arr = df_ca["ca"].values.astype(float)
    E0_arr = df_ca["E0_ry"].values.astype(float)

    if len(ca_arr) < 3:
        # not enough points — just pick the minimum
        idx = np.argmin(E0_arr)
        return float(ca_arr[idx]), float(E0_arr[idx]), float(df_ca["a0_bohr"].values[idx]), float(df_ca["B0_gpa"].values[idx])

    # quadratic fit
    c2, c1, c0 = np.polyfit(ca_arr, E0_arr, 2)
    ca_opt = -c1 / (2 * c2) if c2 > 0 else float(ca_arr[np.argmin(E0_arr)])
    # clamp to scanned range
    ca_opt = float(np.clip(ca_opt, ca_arr.min(), ca_arr.max()))
    E0_min = c2 * ca_opt ** 2 + c1 * ca_opt + c0

    # interpolate a0 and B0 at ca_opt
    a0_at_opt = float(np.interp(ca_opt, ca_arr, df_ca["a0_bohr"].values.astype(float)))
    B0_at_opt = float(np.interp(ca_opt, ca_arr, df_ca["B0_gpa"].values.astype(float)))
    return ca_opt, float(E0_min), a0_at_opt, B0_at_opt


# ── master function ───────────────────────────────────────────────────────────

def run_structure_compare(element_labels, concentrations, outdir="thermodynamic",
                          kkr_params=None, n_steps=7, ca_values=None):
    """
    Run BCC, FCC, HCP EOS calculations for a given composition.
    Writes all output under  outdir/{composition_dirname}/
    Returns the summary dict (also written as structure_summary.json).
    """
    hea = HEAClass(labels=element_labels, concentrations=concentrations)
    dirname = generate_dirname(element_labels, hea.concentrations)
    comp_dir = os.path.join(outdir, dirname)
    os.makedirs(comp_dir, exist_ok=True)

    kkr_p = KKR_PARAMS_LATTICE.copy()
    if kkr_params:
        kkr_p.update(kkr_params)

    hea_cfg = {
        "elements": hea.return_atomic_numbers(),
        "concentrations": hea.concentrations,
    }

    a_bcc = hea.mixture_lattice
    # FCC at same atomic volume: V_FCC = a³/4 = V_BCC = a³/2  → a_FCC = a_BCC × 2^(1/3)
    a_fcc = a_bcc * 2 ** (1.0 / 3.0)
    # HCP at same atomic volume with ideal c/a: a_bcc ≈ 1.12 × a_hcp
    a_hcp = a_bcc / 1.12

    print(f"\nComposition: {dirname}")
    print(f"Initial lattice guesses: BCC={a_bcc:.4f}  FCC={a_fcc:.4f}  HCP={a_hcp:.4f} Bohr\n")

    print("=== BCC ===")
    bcc_result = run_cubic_eos(
        structure='bcc',
        a_init_bohr=a_bcc,
        hea_cfg=hea_cfg,
        kkr_p=kkr_p,
        workdir=os.path.join(comp_dir, "bcc"),
        n_steps=n_steps,
    )

    print("\n=== FCC ===")
    fcc_result = run_cubic_eos(
        structure='fcc',
        a_init_bohr=a_fcc,
        hea_cfg=hea_cfg,
        kkr_p=kkr_p,
        workdir=os.path.join(comp_dir, "fcc"),
        n_steps=n_steps,
    )

    print("\n=== HCP ===")
    hcp_result = run_hcp_eos(
        a_init_bohr=a_hcp,
        hea_cfg=hea_cfg,
        kkr_p=kkr_p,
        workdir=os.path.join(comp_dir, "hcp"),
        ca_values=ca_values,
        n_a_steps=n_steps,
    )

    # build summary
    e_bcc = bcc_result.get("energy_per_atom_ry")
    e_fcc = fcc_result.get("energy_per_atom_ry")
    e_hcp = hcp_result.get("energy_per_atom_ry")

    def safe_diff(a, b):
        if a is None or b is None:
            return None
        return float(a) - float(b)

    energies = {k: v for k, v in [("bcc", e_bcc), ("fcc", e_fcc), ("hcp", e_hcp)] if v is not None}
    most_stable = min(energies, key=energies.get) if energies else None

    summary = {
        "composition": dict(zip(element_labels, [float(c) for c in hea.concentrations])),
        "bcc": bcc_result,
        "fcc": fcc_result,
        "hcp": hcp_result,
        "energy_difference_bcc_minus_fcc_ry": safe_diff(e_bcc, e_fcc),
        "energy_difference_bcc_minus_hcp_ry": safe_diff(e_bcc, e_hcp),
        "most_stable": most_stable,
    }

    json_path = os.path.join(comp_dir, "structure_summary.json")
    save_dict_to_json(summary, json_path)

    print("\n=== SUMMARY ===")
    for struct, res in [("BCC", bcc_result), ("FCC", fcc_result), ("HCP", hcp_result)]:
        e = res.get("energy_per_atom_ry")
        print(f"  {struct}: E/atom = {e:.6f} Ry" if e else f"  {struct}: EOS fit failed")
    if e_bcc and e_fcc:
        print(f"  ΔE(BCC-FCC) = {e_bcc - e_fcc:+.6f} Ry/atom")
    if e_bcc and e_hcp:
        print(f"  ΔE(BCC-HCP) = {e_bcc - e_hcp:+.6f} Ry/atom")
    print(f"  Most stable: {most_stable}")
    print(f"\nSummary written to {json_path}")
    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare BCC/FCC/HCP stability for an HEA via AkaiKKR EOS."
    )
    parser.add_argument("--element_labels", nargs="+", required=True,
                        help="Element symbols, e.g.  Ti Nb Zr")
    parser.add_argument("--concentrations", nargs="+", type=float, required=True,
                        help="Molar fractions (need not sum to 1; will be normalised)")
    parser.add_argument("--outdir", default="thermodynamic",
                        help="Root output directory (default: thermodynamic)")
    parser.add_argument("--n_steps", type=int, default=7,
                        help="Number of lattice points per 1D scan (default: 7)")
    parser.add_argument("--ca_values", nargs="+", type=float,
                        default=[1.50, 1.57, 1.633, 1.70, 1.77],
                        help="c/a values for HCP grid (default: 1.50 1.57 1.633 1.70 1.77)")

    # optional KKR parameter overrides
    parser.add_argument("--ew",     type=float, default=None)
    parser.add_argument("--xc",     type=str,   default=None)
    parser.add_argument("--rel",    type=str,   default=None, choices=["nrl", "sra", "srals"])
    parser.add_argument("--bzqlty", type=float, default=None)
    parser.add_argument("--pmix",   type=float, default=None)
    parser.add_argument("--edelt",  type=float, default=None)
    parser.add_argument("--mxl",    type=int,   default=None)
    parser.add_argument("--magtype", type=str,  default=None, choices=["nmag", "mag"])

    args = parser.parse_args()

    kkr_overrides = {k: v for k, v in vars(args).items()
                     if k in ("ew", "xc", "rel", "bzqlty", "pmix", "edelt", "mxl", "magtype")
                     and v is not None}

    run_structure_compare(
        element_labels=args.element_labels,
        concentrations=args.concentrations,
        outdir=args.outdir,
        kkr_params=kkr_overrides if kkr_overrides else None,
        n_steps=args.n_steps,
        ca_values=args.ca_values,
    )