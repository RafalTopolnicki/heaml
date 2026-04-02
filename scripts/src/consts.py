
AKAIMODBIN="/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx"
#AKAIMODBIN="/home/amber/HEAML/AkaiKKR/cpa2002v010.potential2026/specx"
AKAIBIN=AKAIMODBIN

ATOMS_PER_CELL = 2

## KKR-PARAMS
KKR_PARAMS_LATTICE = {
    'ew': 0.7,
    'xc': 'pbe',
    'rel': 'nrl',
    'bzqlty': 10,
    'mxl': 3,
    'lattice_steps': 5,
    'min_lattice_prop': 0.95,
    'max_lattice_prop': 1.05,
    'subdir': 'lattice',
    'output': 'lattice'
}
KKR_PARAMS_FINALSCF = {
    'ew': 0.7,
    'xc': 'pbe',
    'rel': 'nrl',
    'bzqlty': 20,
    'mxl': 3,
    'delta': 0.005, # IS THIS NEEDED HERE?
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
    'ew': 0.7,
    'xc': 'pbe',
    'rel': 'nrl',
    'bzqlty': 10,
    'mxl': 3,
    'delta': 0.005,
    'rmt': 0.43157, # CHECK THIS LATER
    'pmix': 0.01,
    'edelt': 0.001,
    'subdir': 'debye',
    'output': 'debye',
}
