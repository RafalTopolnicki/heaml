import argparse
import json
import logging
import os
import subprocess

import numpy as np
import pandas as pd

from write_akai_input import scf_input
from src.utils import dist_to_si, energy_to_si, parse_energy, converged_info_in_string, gzip_file, cleanup_potential_files
from src.consts import AKAIBIN, ATOMS_PER_CELL


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

def get_tetragonal_distortion(delta, a0):
    a_local = a0 * (1 + delta)
    ba = 1.0
    ca = 1.0 / (1.0 + delta) ** 3
    gamma = 90.0
    return a_local, ba, ca, gamma


def get_monoclinic_distortion(delta, a0):
    a_local = a0 * np.sqrt(1 + delta ** 2)
    ba = 1.0
    ca = 1.0 / (1 - delta ** 2) * (1 / np.sqrt(1 + delta ** 2))
    gamma_rad = np.pi / 2 - 2 * np.arctan(delta)
    gamma = gamma_rad / np.pi * 180.0
    return a_local, ba, ca, gamma


def run_scf(filename, lattice_params, args, logger):
    scf_input(
        filename=filename,
        lattice_params=lattice_params,
        elements=args["elements"],
        concentrations=args["concentrations"],
        ew=args["ew"],
        xc=args["xc"],
        rel=args["rel"],
        bzqlty=args["bzqlty"],
        pmix=args["pmix"],
        edelt=args["edelt"],
        mxl=args["mxl"],
        rmt=args["rmt"],
    )

    inp = filename + ".inp"
    out = filename + ".out"

    logger.info("Running SCF: %s", filename)
    with open(inp, "r") as fin, open(out, "w") as fout:
        subprocess.run([AKAIBIN], stdin=fin, stdout=fout, stderr=subprocess.STDOUT)

    with open(out, "r", errors="replace") as f:
        text = f.read()

    energy = parse_energy(text)
    conv = converged_info_in_string(text)

    logger.info("Finished SCF: %s | energy=%s | converged=%s", filename, energy, conv)

    gz_out = None
    if args["compress"]:
        gz_out = gzip_file(out)

    cleanup_potential_files(filename)

    return energy, conv, gz_out


def run_scf_tetragonal(delta, args, logger):
    filename = f"{args['output']}_tetragonal_{delta:.6f}"
    a_local, ba, ca, gamma = get_tetragonal_distortion(delta, args["a0"])
    lattice_params = {
        "symmetry": "trc",
        "lattice_constant": a_local,
        "b/a": ba,
        "c/a": ca,
        "gamma": gamma,
    }
    return run_scf(filename, lattice_params, args, logger)


def run_scf_monoclinic(delta, args, logger):
    filename = f"{args['output']}_monoclinic_{delta:.6f}"
    a_local, ba, ca, gamma = get_monoclinic_distortion(delta, args["a0"])
    lattice_params = {
        "symmetry": "trc",
        "lattice_constant": a_local,
        "b/a": ba,
        "c/a": ca,
        "gamma": gamma,
    }
    return run_scf(filename, lattice_params, args, logger)


def run_kkr_elastic_debye(**kwargs):
    workdir = kwargs.get("workdir", ".")
    os.makedirs(workdir, exist_ok=True)

    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        logger = setup_logger(f"{kwargs['output']}_run.log")
        logger.info("Starting elastic + Debye workflow (single delta)")

        if len(kwargs["elements"]) != len(kwargs["concentrations"]):
            raise ValueError("elements and concentrations must have the same length")

        delta = kwargs["delta"]

        vol = kwargs["a0"] ** 3
        volsi = dist_to_si(kwargs["a0"]) ** 3
        B0si = kwargs["B0"] * 1e9

        # reference (delta = 0)
        energy0, conv0, gz0 = run_scf_tetragonal(0.0, kwargs, logger)

        # tetragonal distortion
        energy_tetra, conv_tetra, gz_tetra = run_scf_tetragonal(delta, kwargs, logger)
        Cp = (energy_tetra - energy0) / (6.0 * vol * delta ** 2)

        # monoclinic distortion
        energy_mono, conv_mono, gz_mono = run_scf_monoclinic(delta, kwargs, logger)
        C44 = (energy_mono - energy0) / (2.0 * vol * delta ** 2)

        # shear moduli
        Gv = (2.0 * Cp + 3.0 * C44) / 5.0
        Gr = 10.0 * Cp * C44 / (4.0 * C44 + 6.0 * Cp)
        Gh = (Gv + Gr) / 2.0
        Ghsi = Gh * energy_to_si(1.0) / dist_to_si(1.0) ** 3

        # sound velocities
        vt = np.sqrt(Ghsi / kwargs["rho"])
        vl = np.sqrt((B0si + 4.0 / 3.0 * Ghsi) / kwargs["rho"])
        vm = (3.0 / (2.0 / vt ** 3 + 1.0 / vl ** 3)) ** (1.0 / 3.0)

        # Debye temperature
        thetaDB = 4.79924e-11 * (3.0 * ATOMS_PER_CELL / (4.0 * np.pi * volsi)) ** (1.0 / 3.0) * vm

        # save results table (single row + reference)
        df = pd.DataFrame([
            [kwargs["a0"], 0.0, energy0, conv0, gz0 if gz0 else ""],
            [kwargs["a0"], delta, energy_tetra, conv_tetra, gz_tetra if gz_tetra else ""],
        ], columns=["lattice", "delta", "energy", "converged", "out_gz"])

        results_csv = f"{kwargs['output']}_results.csv"
        df.to_csv(results_csv, index=False)

        summary = {
            "a0_bohr": kwargs["a0"],
            "delta": delta,
            "energy0_ev": float(energy0),
            "Cp": float(Cp),
            "C44": float(C44),
            "Gv": float(Gv),
            "Gr": float(Gr),
            "Gh": float(Gh),
            "thetaDB": float(thetaDB),
            "results_csv": results_csv,
            "run_log": f"{kwargs['output']}_run.log",
        }

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

    # core
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument("--concentrations", nargs="+", required=True)
    parser.add_argument("--a0", type=float, required=True)
    parser.add_argument("--B0", type=float, required=True)
    parser.add_argument("--rho", type=float, required=True)
    parser.add_argument("--delta", type=float, required=True, help="single distortion value")

    # SCF params
    parser.add_argument("--ew", type=float, default=0.7)
    parser.add_argument("--xc", type=str, default="pbe")
    parser.add_argument("--rel", type=str, default="nrl", choices=["nrl", "sra", "srals"])
    parser.add_argument("--bzqlty", type=float, default=10)
    parser.add_argument("--pmix", type=float, default=0.01)
    parser.add_argument("--edelt", type=float, default=0.001)
    parser.add_argument("--mxl", type=int, default=3)
    parser.add_argument("--rmt", type=float, default=0.43088)

    # workflow
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--no-gzip", action="store_true")

    args = vars(parser.parse_args())
    args["concentrations"] = [float(x) for x in args["concentrations"]]
    args["compress"] = not args.pop("no_gzip")

    result = run_kkr_elastic_debye(**args)
    print(result)