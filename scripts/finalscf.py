import argparse
import os
import subprocess
import pandas as pd

from write_akai_input import scf_input_bcc
from src.consts import AKAIBIN
from src.utils import gzip_file, parse_energy, converged_info_in_string, cleanup_potential_files


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
    conv = converged_info_in_string(text)

    cleanup_potential_files(base)

    if args.get("compress", True):
        gzip_file(out)

    return lattice, energy, conv

# =========================
# ⭐ MASTER FUNCTION
# =========================
def run_kkr_finalscf(**kwargs):
    os.makedirs(kwargs.get("workdir", "."), exist_ok=True)
    os.chdir(kwargs.get("workdir", "."))

    lattice = kwargs["a0"]
    results = run_scf(lattice, kwargs)
    print(results)
    df = pd.DataFrame([results], columns=["lattice", "energy", "converged"])
    df.to_csv(f"{kwargs['output']}_results.csv", index=False)


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
    parser.add_argument("--a0", type=float, required=True)

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

    result = run_kkr_finalscf(**args)
    print(result)