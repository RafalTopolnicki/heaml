#!/usr/bin/env python3
"""
Elastic constants and Debye temperature from AkaiKKR using fixed PHYSICAL RMT.

This is a drop-in replacement for debye.py for the single-delta workflow.
The key difference is that the muffin-tin radius is fixed in bohr, not as a
fixed dimensionless AkaiKKR input value.  For each distorted cell the script
uses

    rmt_input(delta) = RMT_common_bohr / a_local(delta)

where RMT_common_bohr is chosen from the smallest non-overlapping RMT among
all cells used in this run: tetragonal delta=0, tetragonal delta=+delta,
monoclinic delta=0, monoclinic delta=+delta.

The elastic formulas are the same as in debye.py:

    Delta E_tetra = 6 V C' delta^2
    Delta E_mono  = 2 V C44 delta^2

For now this script keeps the original one-sided, single-positive-delta
workflow.  It only fixes the physical-RMT inconsistency and improves logging.
"""

import argparse
import json
import logging
import os
import subprocess

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



def setup_logger(log_file):
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

def get_tetragonal_distortion(delta, a0):
    """
    Volume-conserving tetragonal distortion:
        F = diag(1+d, 1+d, 1/(1+d)^2)

    AkaiKKR trc parameters are returned as:
        a_local, b/a, c/a, gamma
    """
    a_local = a0 * (1.0 + delta)
    ba = 1.0
    ca = 1.0 / (1.0 + delta) ** 3
    gamma = 90.0
    return a_local, ba, ca, gamma


def get_monoclinic_distortion(delta, a0):
    """
    Volume-conserving monoclinic/shear distortion:
        F = [[1, d, 0], [d, 1, 0], [0, 0, 1/(1-d^2)]]

    This gives:
        Delta E = 2 V C44 d^2 + O(d^4)

    AkaiKKR trc parameters are returned as:
        a_local, b/a, c/a, gamma
    with alpha=beta=90 deg handled by write_akai_input.py.
    """
    a_local = a0 * np.sqrt(1.0 + delta ** 2)
    ba = 1.0
    ca = 1.0 / (1.0 - delta ** 2) / np.sqrt(1.0 + delta ** 2)
    gamma_rad = np.pi / 2.0 - 2.0 * np.arctan(delta)
    gamma = np.rad2deg(gamma_rad)
    return a_local, ba, ca, gamma


# -----------------------------------------------------------------------------
# Maximum non-overlapping RMT for 2-atom bcc-like cell in Akai input units
# -----------------------------------------------------------------------------

def rmt_max_tetragonal(ca, ba=1.0):
    """
    Dimensionless max RMT in units of the current AkaiKKR lattice constant a.

    For atoms at (0,0,0) and (1/2,1/2,1/2) in an orthogonal bct cell:
        rmt_max = 1/4 sqrt(1 + (b/a)^2 + (c/a)^2)
    """
    return 0.25 * np.sqrt(1.0 + ba ** 2 + ca ** 2)


def rmt_max_monoclinic(ca, gamma_deg, ba=1.0):
    """
    Dimensionless max RMT in units of the current AkaiKKR lattice constant a.

    For atoms at (0,0,0) and (1/2,1/2,1/2), alpha=beta=90 deg:
        rmt_max = 1/4 sqrt(1 + (b/a)^2 + (c/a)^2 - 2(b/a)|cos gamma|)
    """
    gamma_rad = np.deg2rad(gamma_deg)
    return 0.25 * np.sqrt(
        1.0 + ba ** 2 + ca ** 2 - 2.0 * ba * abs(np.cos(gamma_rad))
    )


def rmt_max_for_cell(distortion, delta, a0):
    """
    Return (a_local, ba, ca, gamma, rmt_max_dimensionless, rmt_max_physical_bohr).
    """
    if distortion == "tetragonal":
        a_local, ba, ca, gamma = get_tetragonal_distortion(delta, a0)
        rmt_dimless = rmt_max_tetragonal(ca=ca, ba=ba)
    elif distortion == "monoclinic":
        a_local, ba, ca, gamma = get_monoclinic_distortion(delta, a0)
        rmt_dimless = rmt_max_monoclinic(ca=ca, gamma_deg=gamma, ba=ba)
    else:
        raise ValueError(f"Unknown distortion: {distortion}")

    rmt_physical = a_local * rmt_dimless
    return a_local, ba, ca, gamma, rmt_dimless, rmt_physical


def choose_common_physical_rmt(a0, delta, safety=0.995):
    """
    Choose one fixed physical RMT in bohr for all cells used by this script.

    For the current single-delta workflow, use cells:
      - tetragonal 0
      - tetragonal +delta
      - monoclinic 0
      - monoclinic +delta

    The returned value is physical RMT in bohr.
    """
    entries = []
    for distortion in ["tetragonal", "monoclinic"]:
        for d in [0.0, delta]:
            a_local, ba, ca, gamma, rmt_dimless, rmt_physical = rmt_max_for_cell(
                distortion, d, a0
            )
            entries.append(
                {
                    "distortion": distortion,
                    "delta": d,
                    "a_local": a_local,
                    "ba": ba,
                    "ca": ca,
                    "gamma": gamma,
                    "rmt_max_input": rmt_dimless,
                    "rmt_max_physical_bohr": rmt_physical,
                }
            )

    min_physical = min(e["rmt_max_physical_bohr"] for e in entries)
    return safety * min_physical, entries


def lattice_params_for_distortion(distortion, delta, a0):
    if distortion == "tetragonal":
        a_local, ba, ca, gamma = get_tetragonal_distortion(delta, a0)
    elif distortion == "monoclinic":
        a_local, ba, ca, gamma = get_monoclinic_distortion(delta, a0)
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

def run_scf(filename, lattice_params, args, logger, rmt_input):
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

    if "given rmt" in text.lower() or "rmt's conflict" in text.lower():
        logger.warning("AkaiKKR reported an RMT conflict/reduction in %s", filename)

    logger.info("Finished SCF: %s | energy=%s | converged=%s", filename, energy, conv)

    gz_out = None
    if args.get("compress", True):
        gz_out = gzip_file(out)

    cleanup_potential_files(filename)
    cleanup_fortran_files(filename)

    return energy, conv, gz_out


def run_scf_distortion(distortion, delta, args, logger, rmt_common_physical):
    filename = f"{args['output']}_{distortion}_{delta:.6f}"
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
# Main workflow
# -----------------------------------------------------------------------------

def run_kkr_elastic_debye(**kwargs):
    workdir = kwargs.get("workdir", ".")
    os.makedirs(workdir, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        logger = setup_logger(f"{kwargs['output']}_run.log")
        logger.info("Starting elastic + Debye workflow: fixed physical RMT, single +delta")

        if len(kwargs["elements"]) != len(kwargs["concentrations"]):
            raise ValueError("elements and concentrations must have the same length")

        delta = kwargs["delta"]

        rmt_common_physical, rmt_candidates = choose_common_physical_rmt(
            a0=kwargs["a0"],
            delta=delta,
            safety=kwargs.get("rmt_safety", 0.995),
        )
        logger.info("RMT candidates: %s", rmt_candidates)
        logger.info("Chosen fixed physical RMT = %.12f bohr", rmt_common_physical)

        vol = kwargs["a0"] ** 3
        volsi = dist_to_si(kwargs["a0"]) ** 3
        B0si = kwargs["B0"] * 1e9

        rows = []

        # Tetragonal distortion: reference and +delta
        row_t0 = run_scf_distortion("tetragonal", 0.0, kwargs, logger, rmt_common_physical)
        row_td = run_scf_distortion("tetragonal", delta, kwargs, logger, rmt_common_physical)
        rows.extend([row_t0, row_td])

        # Monoclinic distortion: reference and +delta
        row_m0 = run_scf_distortion("monoclinic", 0.0, kwargs, logger, rmt_common_physical)
        row_md = run_scf_distortion("monoclinic", delta, kwargs, logger, rmt_common_physical)
        rows.extend([row_m0, row_md])

        # Validate energies/convergence before computing constants
        for row in rows:
            if row["energy_Ry"] is None or not np.isfinite(row["energy_Ry"]):
                raise RuntimeError(f"Invalid energy for {row['distortion']} delta={row['delta']}")
            if not row["converged"]:
                raise RuntimeError(f"SCF did not converge for {row['distortion']} delta={row['delta']}")

        energy0_tetra = row_t0["energy_Ry"]
        energy_tetra = row_td["energy_Ry"]
        energy0_mono = row_m0["energy_Ry"]
        energy_mono = row_md["energy_Ry"]

        dE_tetra = energy_tetra - energy0_tetra
        dE_mono = energy_mono - energy0_mono

        Cp = dE_tetra / (6.0 * vol * delta ** 2)
        C44 = dE_mono / (2.0 * vol * delta ** 2)

        # Shear moduli for cubic polycrystal, Voigt/Reuss/Hill.
        Gv = (2.0 * Cp + 3.0 * C44) / 5.0
        Gr = 10.0 * Cp * C44 / (4.0 * C44 + 6.0 * Cp)
        Gh = (Gv + Gr) / 2.0
        Ghsi = Gh * energy_to_si(1.0) / dist_to_si(1.0) ** 3

        # Sound velocities.
        vt = np.sqrt(Ghsi / kwargs["density"])
        vl = np.sqrt((B0si + 4.0 / 3.0 * Ghsi) / kwargs["density"])
        vm = (3.0 / (2.0 / vt ** 3 + 1.0 / vl ** 3)) ** (1.0 / 3.0)

        # Debye temperature.
        thetaDB = (
            4.79924e-11
            * (3.0 * ATOMS_PER_CELL / (4.0 * np.pi * volsi)) ** (1.0 / 3.0)
            * vm
        )

        results_csv = f"{kwargs['output']}_results.csv"
        df = pd.DataFrame(rows)
        df.to_csv(results_csv, index=False)

        summary = {
            "a0_bohr": kwargs["a0"],
            "B0_GPa": kwargs["B0"],
            "density_kg_m3": kwargs["density"],
            "delta": delta,
            "rmt_safety": kwargs.get("rmt_safety", 0.995),
            "rmt_common_physical_bohr": float(rmt_common_physical),
            "rmt_tetragonal_0_input": float(row_t0["rmt_input"]),
            "rmt_tetragonal_delta_input": float(row_td["rmt_input"]),
            "rmt_monoclinic_0_input": float(row_m0["rmt_input"]),
            "rmt_monoclinic_delta_input": float(row_md["rmt_input"]),
            "energy0_tetra_Ry": float(energy0_tetra),
            "energy_tetra_Ry": float(energy_tetra),
            "dE_tetra_Ry": float(dE_tetra),
            "energy0_mono_Ry": float(energy0_mono),
            "energy_mono_Ry": float(energy_mono),
            "dE_mono_Ry": float(dE_mono),
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
            "vt_m_s": float(vt),
            "vl_m_s": float(vl),
            "vm_m_s": float(vm),
            "thetaDB_K": float(thetaDB),
            "results_csv": results_csv,
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
    parser.add_argument("--delta", type=float, required=True, help="single positive distortion value")

    # SCF params.
    parser.add_argument("--ew", type=float, default=0.7)
    parser.add_argument("--xc", type=str, default="pbe")
    parser.add_argument("--rel", type=str, default="sra", choices=["nrl", "sra", "srals"])
    parser.add_argument("--bzqlty", type=float, default=20)
    parser.add_argument("--pmix", type=float, default=0.01)
    parser.add_argument("--edelt", type=float, default=0.0001)
    parser.add_argument("--mxl", type=int, default=3)
    parser.add_argument("--magtype", default="nmag", choices=["nmag", "mag"])

    # RMT handling.
    # The old --rmt argument is intentionally not used.  RMT is chosen from geometry.
    parser.add_argument("--rmt-safety", type=float, default=0.995)

    # Workflow.
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--no-gzip", action="store_true")

    args = vars(parser.parse_args())
    args["concentrations"] = [float(x) for x in args["concentrations"]]
    args["compress"] = not args.pop("no_gzip")

    result = run_kkr_elastic_debye(**args)
    print(result)