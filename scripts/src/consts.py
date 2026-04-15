
AKAIMODBIN="/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx"
#AKAIMODBIN="/home/amber/HEAML/AkaiKKR/cpa2002v010.potential2026/specx"
AKAIBIN=AKAIMODBIN

ATOMS_PER_CELL = 2

composition_labels = ["Ti", "Nb", "Zr", "Hf", "Ta", "Sc", "Mo", "W", "Y", "La"]

CANDIDATE_COMPOSITIONS_N = 1000
ACQUISITION_ALPHA = 1.0
ACQUISITION_METRIC = 'cosine'
TARGET = 'Tc_mu0.2'

## KKR-PARAMS
KKR_PARAMS_LATTICE = {
    'ew': 0.6,
    'xc': 'pbe',
    'rel': 'sra',
    'bzqlty': 10,
    'mxl': 3,
    'magtype': 'nmag',
    'lattice_steps': 5,
    'min_lattice_prop': 0.95,
    'max_lattice_prop': 1.05,
    'pmix': 0.01,
    'edelt': 0.001,
    'subdir': 'lattice',
    'output': 'lattice'
}
KKR_PARAMS_FINALSCF = {
    'ew': 0.6,
    'xc': 'pbe',
    'rel': 'sra',
    'bzqlty': 20,
    'mxl': 3,
    'magtype': 'nmag',
    'delta': 0.005, # IS THIS NEEDED HERE?
    'pmix': 0.01,
    'edelt': 0.001,
    'subdir': 'finalscf',
    'output': 'finalscf',
}
# MONOCLINIC
# RMT: 0.42723 for delta=0.020
# RMT: 0.43012 for delta=0.010
# RMT: 0.43157 for delta=0.005
# RMT: 0.43272 for delta=0.001
# TETRAGONAL
# RMT: 0.43088 for all delta
KKR_PARAMS_DEBYE = {
    'ew': 0.6,
    'xc': 'pbe',
    'rel': 'sra',
    'bzqlty': 10,
    'mxl': 3,
    'magtype': 'nmag',
    'delta': 0.005,
    'rmt': 0.43157, # CHECK THIS LATER
    'pmix': 0.01,

    'edelt': 0.001,
    'subdir': 'debye',
    'output': 'debye',
}
