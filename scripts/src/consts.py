
AKAIMODBIN="/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx"
AKAIBIN=AKAIMODBIN

ATOMS_PER_CELL = 2

## KKR-PARAMS
KKR_PARAMS_LATTICE = {
    'ew': 0.7,
    'xc': 'vbh',
    'rel': 'nrl',
    'bzqlty': 10,
    'mxl': 3
}
KKR_PARAMS_DEBYE = {
    'ew': 0.7,
    'xc': 'vbh',
    'rel': 'nrl',
    'bzqlty': 20,
    'mxl': 3,
    'delta': 0.005,
    'rmt': 0.43088, # CHECK THIS LATER
}
KKR_PARAMS_FINALSCF = {
    'ew': 0.7,
    'xc': 'vbh',
    'rel': 'nrl',
    'bzqlty': 20,
    'mxl': 3,
    'delta': 0.005,
    'rmt': 0.43088, # CHECK THIS LATER
}