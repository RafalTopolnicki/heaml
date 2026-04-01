#!/bin/bash
## Ti, Nb, Zr, Hf, Ta, Sc, Y, La, Mo, W
# FIRST RUN TO KNOW WHERE TO SEARCH FOR LATTICE CONSTANTS OF ALL ELEMENTS
# BCC FORCED
# Ti: 6.15
#python scripts/lattice.py --workdir results/elements/lattice_search/Ti --output latt --sym bcc --elements 22 --concentrations 100 --min_lattice 6 --max_lattice 6.5 --step 0.05
# Nb: 6.25
#python scripts/lattice.py --workdir results/elements/lattice_search/Nb --output latt --sym bcc --elements 41 --concentrations 100 --min_lattice 6 --max_lattice 6.5 --step 0.05
# Zr: 6.73
#python scripts/lattice.py --workdir results/elements/lattice_search/Zr --output latt --sym bcc --elements 40 --concentrations 100 --min_lattice 6.5 --max_lattice 7 --step 0.05
# Hf: 6.83
#python scripts/lattice.py --workdir results/elements/lattice_search/Hf --output latt --sym bcc --elements 72 --concentrations 100 --min_lattice 6.0 --max_lattice 7 --step 0.1
# Ta: 6.37
#python scripts/lattice.py --workdir results/elements/lattice_search/Ta --output latt --sym bcc --elements 73 --concentrations 100 --min_lattice 6 --max_lattice 6.5 --step 0.05
# Sc: 6.93
#python scripts/lattice.py --workdir results/elements/lattice_search/Sc --output latt --sym bcc --elements 21 --concentrations 100 --min_lattice 6.5 --max_lattice 7.5 --step 0.1
# Y:
#python scripts/lattice.py --workdir results/elements/lattice_search/Y --output latt --sym bcc --elements 70 --concentrations 100 --min_lattice 8 --max_lattice 12 --step 1 --ew 0.7 --rel sra