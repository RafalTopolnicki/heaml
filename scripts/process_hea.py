import argparse
import os
import gzip
import shutil
import subprocess
import re

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from write_akai_input import scf_input, scf_input_bcc
from src.consts import AKAIBIN
from src.utils import dist_from_si, dist_to_si, energy_from_si, energy_to_si


TOTAL_ENERGY_RE = re.compile(r"total energy\s*=\s*([-+0-9.Ee]+)")


def parse_energy(text):
    m = TOTAL_ENERGY_RE.findall(text)
    return float(m[-1]) if m else np.nan


def converged(text):
    return "itr=499" not in text


def gzip_file(path):
    if not os.path.exists(path):
        return None
    gz = path + ".gz"
    with open(path, "rb") as f_in, gzip.open(gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)
    return gz


def run_scf(lattice, args):
    base = f"{args['output']}_{lattice:.6f}"
    inp = base + ".inp"
    out = base + ".out"

    scf_input_bcc(
        filename=base,
        lattice_params={"symmetry": args['sym'], "lattice_constant": lattice},
        elements=args['elements'],
        concentrations=args['concentrations'],
        ew=args.get("ew", 0.7),
        xc=args.get("xc", "pbe"),
        rel=args.get("rel", "nrl"),
        bzqlty=args.get("bzqlty", 10),
        pmix=args.get("pmix", 0.01),
        edelt=args.get("edelt", 0.001),
        mxl=args.get("mxl", 3),
    )

    with open(inp, "r") as fin, open(out, "w") as fout:
        subprocess.run([AKAIBIN], stdin=fin, stdout=fout)

    text = open(out).read()

    energy = parse_energy(text)
    conv = converged(text)

    if args.get("compress", True):
        gzip_file(out)

    return lattice, energy, conv


def lattice_to_volume(latt_si):
    return (latt_si**3) / 2.0


def volume_to_lattice(vol_si):
    return (vol_si * 2.0) ** (1.0 / 3.0)


def birch_murnaghan(v, e0, b0, bP, v0):
    eta = (v0 / v) ** (2.0 / 3.0)
    return e0 + (9 * v0 * b0 / 16) * (
        (eta - 1) ** 3 * bP + (eta - 1) ** 2 * (6 - 4 * eta)
    )


def fit_eos(df):
    df = df[df["converged"] == True].dropna()

    e = energy_to_si(df["energy"])
    l = dist_to_si(df["lattice"])
    v = lattice_to_volume(l)

    # initial guess
    a2, a1, a0 = np.polyfit(v, e, 2)
    v0 = -a1 / (2 * a2)
    e0 = a2 * v0**2 + a1 * v0 + a0
    b0 = max(2 * a2 * v0, 1e9)

    popt, _ = curve_fit(birch_murnaghan, v, e, p0=[e0, b0, 4.0, v0])

    e0, b0, bP, v0 = popt

    a0 = dist_from_si(volume_to_lattice(v0))
    E0 = energy_from_si(e0)
    B0 = b0 / 1e9

    return a0, E0, B0


# =========================
# ⭐ MASTER FUNCTION
# =========================
def run_kkr_eos(**kwargs):
    os.makedirs(kwargs.get("workdir", "."), exist_ok=True)
    os.chdir(kwargs.get("workdir", "."))

    lattices = np.arange(kwargs["min_lattice"],
                         kwargs["max_lattice"] + kwargs["step"],
                         kwargs["step"])

    results = []
    for l in lattices:
        res = run_scf(l, kwargs)
        results.append(res)

    df = pd.DataFrame(results, columns=["lattice", "energy", "converged"])
    df.to_csv(f"{kwargs['output']}_results.csv", index=False)

    a0, E0, B0 = fit_eos(df)

    with open(f"{kwargs['output']}_summary.txt", "w") as f:
        f.write(f"a0 = {a0}\nE0 = {E0}\nB0 = {B0}\n")

    return {
        "lattice_constant_bohr": a0,
        "minimum_energy_ev": E0,
        "bulk_modulus_gpa": B0,
    }


# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # core
    parser.add_argument("--output", required=True)
    parser.add_argument("--sym", default="bcc", choices=["bcc", "fcc"])
    parser.add_argument("--elements", nargs="+", required=True)
    parser.add_argument("--concentrations", nargs="+", required=True)
    parser.add_argument("--min_lattice", type=float, required=True)
    parser.add_argument("--max_lattice", type=float, required=True)
    parser.add_argument("--step", type=float, required=True)

    # SCF parameters (previously hidden)
    parser.add_argument("--ew", type=float, default=0.7)
    parser.add_argument("--xc", type=str, default="pbe")
    parser.add_argument("--rel", type=str, default="nrl", choices=["nrl", "sra", "srals"])
    parser.add_argument("--bzqlty", type=float, default=10)
    parser.add_argument("--pmix", type=float, default=0.01)
    parser.add_argument("--edelt", type=float, default=0.001)
    parser.add_argument("--mxl", type=int, default=3)

    # workflow / IO
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--no-gzip", action="store_true")

    args = vars(parser.parse_args())

    # type fixes
    args["concentrations"] = [float(x) for x in args["concentrations"]]

    # normalize flags
    args["compress"] = not args.pop("no_gzip")

    result = run_kkr_eos(**args)
    print(result)