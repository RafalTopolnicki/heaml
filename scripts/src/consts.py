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
    "Ru": {0: 0, 1: 0, 2: 1, 3: 0},  # 4d
    "Hf": {0: 0, 1: 0, 2: 2, 3: 0},  # 5d
    "Ta": {0: 0, 1: 0, 2: 2, 3: 0},  # 5d
    "W":  {0: 0, 1: 0, 2: 2, 3: 0},  # 5d
    "Re": {0: 0, 1: 0, 2: 2, 3: 0},  # 5d
    "La": {0: 0, 1: 0, 2: 2, 3: 0},  # 5d (4f: n-l-1=0, no genuine f nodes)
}

# ---------------------------------------------------------------------------
# Stoner spin-fluctuation correction to Tc (Berk-Schrieffer 1966)
# ---------------------------------------------------------------------------

# Stoner exchange integrals I (eV), Janak (1977) PRB 16, 255.
# I = -d²E_xc/dM²; computed within LDA from element band structure.
STONER_I_EV = {
    "Sc": 0.46, "Ti": 0.48, "Y":  0.44, "Zr": 0.39,
    "Nb": 0.35, "Mo": 0.41, "Ru": 0.50, "Hf": 0.39,
    "Ta": 0.33, "W":  0.39, "Re": 0.37, "La": 0.32,
}

# Paramagnon energy scale E_sf (meV) per element (Option B).
# Used to compute gamma = ln(E_sf_mix / omega_D) for each composition.
# Estimated from spin-fluctuation temperatures in literature (specific heat, neutron, DFT).
# Ordering: lighter 3d (Sc, Ti) < 4d (Ru, Zr, Nb) < 5d (Hf, Ta, W).
# Values chosen so that E_sf_mix ≈ 3×omega_D for Ti-rich alloys, giving gamma≈1.1
# consistent with experimental Tc < 5K for the Ti0.6Sc0.32Ru0.08 champion.
E_SF_MEV = {
    "Sc": 50.0, "Ti": 60.0, "La": 40.0,
    "Y":  80.0, "Ru": 120.0,
    "Zr": 150.0, "Nb": 200.0, "Mo": 300.0,
    "Hf": 200.0, "Ta": 250.0, "Re": 300.0, "W": 400.0,
}

# Approximate partial DOS at EF per spin per atom (states/eV) for each element.
# Used ONLY in Option C acquisition penalty (approx for candidate screening without KKR).
# Derived from KKR-CPA component Ntot values (Ntot/13.606 eV) for a representative
# Ti-rich alloy. These are composition-dependent in reality; treat as rough estimates.
STONER_N_EF_APPROX = {
    "Sc": 0.61, "Ti": 1.05, "Y":  0.49, "Zr": 0.63,
    "Nb": 0.63, "Mo": 0.58, "Ru": 0.45, "Hf": 0.60,
    "Ta": 0.59, "W":  0.53, "Re": 0.45, "La": 0.58,
}

# Stoner acquisition penalty (Option C).
# STONER_BETA = 0.0 disables the penalty entirely (no effect on optimization).
# When enabled: penalty = exp(-STONER_BETA * max(S_mix - STONER_S_THRESHOLD, 0))
# S_mix = 1 / (1 - I_mix * N_mix(EF)); STONER_S_THRESHOLD = 1.5 means penalty
# only kicks in when Stoner enhancement is moderate-to-strong.
STONER_BETA = 0.0        # set > 0 to enable; ~1.0 is a moderate penalty
STONER_S_THRESHOLD = 1.5

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