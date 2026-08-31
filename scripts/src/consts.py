import os

paths = [
    "/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx",
    "/home/rto/HEAML/AkaiKKR/cpa2002v010.potential2026/specx",
    "/home/amber/HEAML/AkaiKKR/cpa2002v010.potential2026/specx",
]

AKAIMODBIN = None
for p in paths:
    if os.path.isfile(p) and os.access(p, os.X_OK):
        AKAIMODBIN = p
        break

if AKAIMODBIN is None:
    raise RuntimeError("specx binary not found in known locations")

print(f"Using AKAIMODBIN: {AKAIMODBIN}")

AKAIBIN=AKAIMODBIN

ATOMS_PER_CELL = 2

# Maximal set of elements considered across all experiments.
# Individual runs may use a subset via the --elements CLI argument.
composition_labels = ["Ti", "Nb", "Zr", "Hf", "Ta", "Sc", "Mo", "W", "Y", "La", "Re", "Ru"]

# Expected number of genuine radial nodes for the valence orbital of each element and l.
# Used by cutoff_mode="valence" in macmillan_cutoff.py: apply last-node cutoff only when
# n_actual_nodes > n_expected (extra core-contamination nodes present).
# s, p: 0 — inner oscillations are always orthogonality contamination.
# d:    0 for 3d, 1 for 4d, 2 for 5d  (n-l-1 genuine valence nodes).
# f:    0 for all — first f element La has 4f^1, n-l-1 = 0.
VALENCE_NODES_EXPECTED: dict = {
    "Sc": {0: 0, 1: 0, 2: 0, 3: 0},  # 3d
    "Ti": {0: 0, 1: 0, 2: 0, 3: 0},  # 3d
    "Y":  {0: 0, 1: 0, 2: 1, 3: 0},  # 4d
    "Zr": {0: 0, 1: 0, 2: 1, 3: 0},  # 4d
    "Nb": {0: 0, 1: 0, 2: 1, 3: 0},  # 4d
    "Mo": {0: 0, 1: 0, 2: 1, 3: 0},  # 4d
    "Hf": {0: 0, 1: 0, 2: 2, 3: 0},  # 5d
    "Ta": {0: 0, 1: 0, 2: 2, 3: 0},  # 5d
    "W":  {0: 0, 1: 0, 2: 2, 3: 0},  # 5d
    "La": {0: 0, 1: 0, 2: 2, 3: 0},  # 5d (4f: n-l-1=0, no genuine f nodes)
}

CANDIDATE_COMPOSITIONS_N = 100_000 # in each iteration new points are generated
ACQUISITION_METRIC = 'cosine'

TARGET = 'Tc_mu0.2'
TARGET_DG = 'dG_eV'
ACQUISITION_ALPHA = 1.0

# Thermodynamic stability
DG_T_K = 1000            # temperature used in dG = dE - T*dS_conf
DG_THRESHOLD_DEFAULT = 1000.0  # effectively no penalty; set to ~0.1 to enable thermodynamic constraint

#TARGET = 'lambda'
#ACQUISITION_ALPHA = 0.2 # smaller value due to different scale of lambda

MIN_NOVELTY_DIST = 0.02

# Physical validity thresholds for elastic stability.
# Compositions that violate either threshold are considered mechanically ill-defined:
# their KKR-computed Debye temperature collapses toward zero (Hill shear modulus ≈ 0),
# which drives lambda → ∞ via 1/ω². Such points are excluded from optimization by
# zeroing out the target; the raw computed value is preserved in 'lambda_raw_phys'.
PHYSICAL_CP_MIN_GPA =  0.0   # Cauchy pressure lower bound (GPa); more negative → cubic instability
PHYSICAL_THETA_MIN_K = 100.0 # KKR Debye temperature lower bound (K); below this the elastic model is unreliable

FRESH_FRACTION = 0.8      # initial: 0.8 — share of global (Sobol) candidates per iteration
MODEL_SUBSAMPLE_FRACTION = 0.5  # fraction of known data each model is trained on; lower = more diversity between ensemble members
# If True, use the early-stopping model directly for ensemble predictions instead of retraining on the full
# subsample. Retraining (False) causes train R²→1 and collapses ensemble σ, turning μ+2σ into pure
# exploitation. Early-stopping preserves genuine per-model variance and a meaningful uncertainty estimate.
MODEL_USE_EARLY_STOPPING = True
LOCAL_TOP_K = 20           # initial: 5  — number of top compositions used as local-search centers
LOCAL_NOISE_SCALE = 0.03  # initial: 0.03 — std of Gaussian perturbation for local candidates

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

    # McMillan-Hopfield integration cutoff.
    # How r_cut is derived from the two last-node positions for each channel (l, l+1):
    #   'none'    no cutoff — integrate from 0 (full muffin-tin integral, original behaviour).
    #   'max'     r_cut = max(r_last_l, r_last_{l+1})  — removes core contamination from
    #             whichever wavefunction extends furthest.
    #   'min'     r_cut = min of the two last nodes.
    #   'lower'   r_cut = r_last_l   (e.g. pd uses r_last_p)
    #   'upper'   r_cut = r_last_{l+1} (e.g. pd uses r_last_d) — gives unphysical λ≈23, do not use.
    #   'valence' per-wavefunction cutoff only when n_actual_nodes > expected valence nodes
    #             (see VALENCE_NODES_EXPECTED above). Fixes La df anomaly: La 5d has 2 genuine
    #             valence nodes so no cutoff is applied there, preserving the physical inner lobe.
    'mcmillan_cutoff_mode': 'valence',
}
# MONOCLINIC
# RMT: 0.42723 for delta=0.020
# RMT: 0.43012 for delta=0.010
# RMT: 0.43157 for delta=0.005
# RMT: 0.43272 for delta=0.001
# TETRAGONAL
# RMT: 0.43301 for delta=0.000
# RMT: 0.43088 for delta=0.005
KKR_PARAMS_DEBYE = {
    "ew": 0.6,
    "xc": "pbe",
    "rel": "sra",
    "bzqlty": 10,
    "mxl": 3,
    "magtype": "nmag",

    "deltas": [0.005],
    "fit_mode": "linear",
    "one_sided": True,
    "c44_mode": "monoclinic",
    "rmt_safety": 0.999,

    "cp_scale": 1.0,
    "c44_scale": 0.33,
    "b0_scale": 1.0,

    # If True, use composition-weighted elemental Debye temperatures (mixture_debye_temperature)
    # instead of the KKR-computed thetaDB_K for the McMillan lambda and Tc formulas.
    "use_mixture_debye": False,

    "pmix": 0.01,
    "edelt": 0.001,
    "subdir": "debye",
    "output": "debye",
}