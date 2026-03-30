import numpy as np
import sys
import os
import argparse
import string
import random
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
from write_akai_input import scf_input
from src.cons import AKAIBIN, ATOMS_PER_CELL

def run_scf(lattice, args):
    tmpfilename = f'{args.output}_{lattice:.3f}'
    lattice_params = {'symmetry': args.sym, 'lattice_constant': lattice}
    scf_input(filename=f'{tmpfilename}', lattice_params=lattice_params, elements=args.elements, concentrations=args.concentrations,
              ew=args.ew, xc=args.xc, rel=args.rel,
              bzqlty=args.bzqlty, pmix=args.pmix, edelt=args.edelt, mxl=args.mxl)
    os.system(f"{AKAIBIN} < {tmpfilename}.inp > {tmpfilename}.out")
    try:
        energy = subprocess.check_output([f"grep \"total energy\" {tmpfilename}.out | tail -1"], shell=True)
        energy = str(energy)
        pos = energy.find('=') + 2
        energy = float(energy[pos:-3])
    except:
        energy = np.nan
    try:
        subprocess.check_output([f"grep \"itr=499\" {tmpfilename}.out"], shell=True)
        converged = False
    except:
        converged = True
    try:
        os.system(f"gzip -f {tmpfilename}.out")
        os.system(f"rm {tmpfilename}.pot {tmpfilename}.pot.info")
    except:
        print(f'WARNING: gzip and rm not sucessful')
    # parse outputed energy
    return energy, converged


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="prefix for the outputfile")
    parser.add_argument("--sym", type=str, default="bcc", choices=["bcc", "fcc"], help="lattice symmetry")
    parser.add_argument("--elements", nargs="+", help="list of atomic masses")
    parser.add_argument("--concentrations", nargs="+", help="list of atomic concentrations")
    parser.add_argument("--min", type=float, required=True, help="minimal lattice constant to consider")
    parser.add_argument("--max", type=float, required=True, help="maximal lattice constant to consider")
    parser.add_argument("--step", type=float, required=True, help="incrementation step for lattice constants")

    parser.add_argument("--ew", type=float, default=0.7, help="ewidth")
    parser.add_argument("--xc", type=str, default='pbe', help="XC potential")
    parser.add_argument("--rel", type=str, default="nrl", choices=["nrl", "sra", "srals"], help="relativity mode")

    parser.add_argument("--bzqlty", type=float, default=10, help="bzqlty parameter")
    parser.add_argument("--pmix", type=float, default=0.01, help="pmix parameter")
    parser.add_argument("--edelt", type=float, default=0.001, help="edelt parameter")
    parser.add_argument("--mxl", type=int, default=3, help="mxl parameter")

    args = parser.parse_args()

    concentration_str = [f'{float(c):.3f}' for c in args.concentrations]
    concentration_str = ' '.join(concentration_str)

    lattice = args.min
    results = []
    while lattice <= args.max:
        energy, converged = run_scf(lattice, args)
        results.append([lattice, energy, converged] + args.elements + args.concentrations)
        lattice += args.step

    # write output
    df = pd.DataFrame(results, columns=['lattice', 'energy', 'converged', 'elements', 'concentrations'])
    df.to_csv(f'{args.output}_results.csv', index=False)
    # write min energy to separate file
    min_energy_index = np.argmin(df.iloc[:, 1])
    min_energy = np.min(df.iloc[:, 1])
    min_lattice = df.iloc[min_energy_index, 0]
    with open(f'{args.output}_best_energy.txt', 'w') as f:
        f.write(f'{min_energy} {min_lattice} {concentration_str}')
    f.close()
    # make a plot
    plt.plot(df.iloc[:, 0], df.iloc[:, 1], 'o-')
    plt.title(concentration_str)
    plt.savefig(f'{args.output}_energy_lattice.png')
    # clear temp files
    try:
        os.system(f"gzip *.out")
        os.system(f"rm *.pot *.pot.info")
    except:
        pass
    
if __name__ == "__main__":
    main()
