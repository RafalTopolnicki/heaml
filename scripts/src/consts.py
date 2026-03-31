
AKAIMODBIN="/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx"
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
    'delta': 0.005,
    'subdir': 'finalscf',
    'output': 'finalscf',
}
KKR_PARAMS_DEBYE = {
    'ew': 0.7,
    'xc': 'pbe',
    'rel': 'nrl',
    'bzqlty': 10,
    'mxl': 3,
    'delta': 0.01,
    'rmt': 0.43088, # CHECK THIS LATER
    'pmix': 0.01,
    'edelt': 0.001,
    'subdir': 'debye',
    'output': 'debye',
}
