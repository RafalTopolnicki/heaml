import numpy as np
import os
import argparse
import string
import random
import subprocess
import pandas as pd
from write_akai_input import scf_input
from src.utils import dist_to_si, energy_to_si
from src.consts import AKAIBIN, ATOMS_PER_CELL

def get_tetragonal_distortion(delta, args):
    a_local = args.a0*(1+delta)
    ba = 1.0
    ca = 1.0/(1.0+delta)**3
    gamma = 90
    return a_local, ba, ca, gamma

def get_monoclinic_distortion(delta, args):
    a_local = args.a0*np.sqrt(1+delta**2)
    ba = 1.0
    ca = 1.0/(1-delta**2)*(1/np.sqrt(1+delta**2))
    gamma_rad = np.pi/2 - 2*np.arctan(delta)
    gamma = gamma_rad/np.pi*180
    return a_local, ba, ca, gamma

def run_scf(filename, lattice_params, args):
    scf_input(filename=f'{filename}', lattice_params=lattice_params,
              elements=args.elements, concentrations=args.concentrations,
              ew=args.ew, xc=args.xc, rel=args.rel,
              bzqlty=args.bzqlty, pmix=args.pmix, edelt=args.edelt, mxl=args.mxl,
              rmt=0.43088)
    os.system(f"{AKAIBIN} < {filename}.inp > {filename}.out")
    try:
        energy = subprocess.check_output([f"grep \"total energy\" {filename}.out | tail -1"], shell=True)
        energy = str(energy)
        pos = energy.find('=') + 2
        energy = float(energy[pos:-3])
    except:
        energy = np.nan
    try:
        subprocess.check_output([f"grep \"itr=499\" {filename}.out"], shell=True)
        converged = False
    except:
        converged = True
    try:
        os.system(f"gzip -f {filename}.out")
        os.system(f"rm {filename}.pot {filename}.pot.info")
    except:
        print(f'WARNING: gzip and rm not sucessful')
    # parse outputed energy
    return energy, converged


def run_scf_tetragonal(delta, args):
    tmpfilename = f'{args.output}_tetragonal_{delta:.3f}'
    a_local, ba, ca, gamma = get_tetragonal_distortion(delta=delta, args=args)
    lattice_params = {'symmetry': 'trc', 'lattice_constant': a_local, 'b/a': ba, 'c/a': ca, 'gamma': gamma}
    energy, converged = run_scf(filename=tmpfilename, lattice_params=lattice_params, args=args)
    return energy, converged

def run_scf_monoclinic(delta, args):
    tmpfilename = f'{args.output}_monoclinic_{delta:.3f}'
    a_local, ba, ca, gamma = get_monoclinic_distortion(delta=delta, args=args)
    lattice_params = {'symmetry': 'trc', 'lattice_constant': a_local, 'b/a': ba, 'c/a': ca, 'gamma': gamma}
    energy, converged = run_scf(filename=tmpfilename, lattice_params=lattice_params, args=args)
    return energy, converged

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=str, required=True, help="prefix for the outputfile")
    parser.add_argument("--elements", nargs="+", help="list of atomic masses")
    parser.add_argument("--concentrations", nargs="+", help="list of atomic concentrations")
    parser.add_argument("--a0", type=float, required=True, help="EQN lattice constant")
    parser.add_argument("--B0", type=float, required=True, help="B from equation of state")
    parser.add_argument("--rho", type=float, required=True, help="Density")
    parser.add_argument("--min", type=float, required=True, help="minimal delta")
    parser.add_argument("--max", type=float, required=True, help="maximal delta")
    parser.add_argument("--step", type=float, required=True, help="incrementation step for deltas")

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

    delta = args.min
    results = []
    vol = args.a0 ** 3
    volsi = dist_to_si(args.a0)**3
    B0si = args.B0*1e9
    energy0, converged0 = run_scf_tetragonal(delta=0, args=args)
    results.append([args.a0, 0.0, energy0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] + args.elements + args.concentrations)
    while delta <= args.max:
        # tetragonal
        energy_tetra, converged_teta = run_scf_tetragonal(delta, args)
        Cp = (energy_tetra-energy0)/(6*vol*delta**2)
        # monoclinic
        energy_mono, converged_mono = run_scf_monoclinic(delta, args)
        C44 = (energy_mono-energy0)/(2*vol*delta**2)

        Gv = (2*Cp+3*C44)/5.0
        Gr = 10*Cp*C44/(4*C44+6*Cp)
        Gh = (Gv+Gr)/2.0
        Ghsi = Gh*energy_to_si(1)/dist_to_si(1)**3

        vt = np.sqrt(Ghsi/args.rho)
        vl = np.sqrt((B0si+4.0/3.0*Ghsi)/args.rho)
        vm = (3.0/(2.0/vt**3+1/vl**3))**(1.0/3.0)
        thetaDB = 4.79924e-11*(3*ATOMS_PER_CELL/(4*np.pi*volsi))**(1.0/3.0)*vm

        results.append([args.a0, delta, energy0, energy_tetra, converged_teta, energy_mono, converged_mono, Cp, C44, Gv, Gr, Gh, Ghsi, vt, vl, vm, thetaDB] + args.elements + args.concentrations)
        print(results)
        delta += args.step

    # write output
    df = pd.DataFrame(results, columns=['lattice', 'delta', 'energy0', 'energy_tetra', 'conv_tetra', 'energy_mono', 'conv_mono', 'Cp', 'C44', 'Gv', 'Gr', 'Gh', 'Ghsi', 'vt', 'vl', 'vm', 'thetaDB', 'elements', 'concentrations'])
    df.to_csv(f'{args.output}_results.csv', index=False)

if __name__ == "__main__":
    main()
