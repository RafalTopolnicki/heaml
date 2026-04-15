from src.elements import ATOMIC_FEATURES
import numpy as np

ADDITIONAL_FEATURES = ['VEC',
 'atomic_size_mismatch_delta',
 'atomic_size_mismatch_delta_percent',
 'chi_mean',
 'chi_std',
 'config_entropy_nat',
 'd_count_mean',
 'd_count_std',
 'max_fraction',
 'mean_atomic_radius_pm',
 'mean_d_electron_count',
 'mean_electronegativity_pauling',
 'mean_valence_electron_count',
 'mean_vdw_radius_pm',
 'mean_vec',
 'min_fraction_nonzero',
 'mixing_entropy_R',
 'n_elements_nonzero',
 'pair_absdiff_atomic_radius_pm',
 'pair_absdiff_atomic_radius_pm_norm',
 'pair_absdiff_d_electron_count',
 'pair_absdiff_d_electron_count_norm',
 'pair_absdiff_electronegativity_pauling',
 'pair_absdiff_electronegativity_pauling_norm',
 'pair_absdiff_valence_electron_count',
 'pair_absdiff_valence_electron_count_norm',
 'pair_absdiff_vdw_radius_pm',
 'pair_absdiff_vdw_radius_pm_norm',
 'pair_absdiff_vec',
 'pair_absdiff_vec_norm',
 'std_atomic_radius_pm',
 'std_d_electron_count',
 'std_electronegativity_pauling',
 'std_valence_electron_count',
 'std_vdw_radius_pm',
 'std_vec',
 'var_atomic_radius_pm',
 'var_d_electron_count',
 'var_electronegativity_pauling',
 'var_valence_electron_count',
 'var_vdw_radius_pm',
 'var_vec']

def compute_hea_features(
    comp_dict: dict,
    normalize_composition: bool = False,
    eps: float = 1e-12,
) -> dict:
    """
    Compute HEA-style features for a SINGLE composition.

    Parameters
    ----------
    comp_dict : dict
        Example: {"Ti": 0.2, "Nb": 0.3, ...}
    ATOMIC_FEATURES : dict
        Elemental properties dictionary
    normalize_composition : bool
        If True, normalize fractions to sum to 1
    eps : float
        Small number for numerical stability

    Returns
    -------
    dict
        Computed HEA features
    """

    if not ATOMIC_FEATURES:
        raise ValueError("ATOMIC_FEATURES is empty")

    elements = list(comp_dict.keys())

    # validate elements
    ref_elements = set(next(iter(ATOMIC_FEATURES.values())).keys())
    missing = [el for el in elements if el not in ref_elements]
    if missing:
        raise ValueError(f"Elements missing from ATOMIC_FEATURES: {missing}")

    # build composition array
    X = np.array([comp_dict[el] for el in elements], dtype=float)

    if normalize_composition:
        total = np.sum(X)
        if total <= 0:
            raise ValueError("Invalid composition sum")
        X = X / total

    out = {}

    # -----------------------
    # D. entropy features
    # -----------------------
    X_safe = np.where(X > 0, X, eps)

    entropy = -np.sum(np.where(X > 0, X * np.log(X_safe), 0.0))
    out["config_entropy_nat"] = entropy
    out["mixing_entropy_R"] = entropy

    out["n_elements_nonzero"] = int(np.sum(X > 0))
    out["max_fraction"] = float(np.max(X))

    nz = X[X > 0]
    out["min_fraction_nonzero"] = float(np.min(nz)) if len(nz) else np.nan

    # ----------------------------------------
    # A, B, C, E. property-derived features
    # ----------------------------------------
    for prop, prop_map in ATOMIC_FEATURES.items():
        p = np.array([prop_map[el] for el in elements], dtype=float)

        # A. mean
        mean_p = np.sum(X * p)
        out[f"mean_{prop}"] = float(mean_p)

        # B. variance / std
        centered = p - mean_p
        var_p = np.sum(X * centered**2)
        std_p = np.sqrt(var_p)

        out[f"var_{prop}"] = float(var_p)
        out[f"std_{prop}"] = float(std_p)

        # special: atomic size mismatch
        if prop == "atomic_radius_pm":
            delta = np.sqrt(np.sum(X * (1.0 - p / mean_p) ** 2))
            out["atomic_size_mismatch_delta"] = float(delta)
            out["atomic_size_mismatch_delta_percent"] = float(100.0 * delta)

        # C. pairwise mismatch
        D = np.abs(p[:, None] - p[None, :])
        pair_abs = np.sum(X[:, None] * X[None, :] * D)

        out[f"pair_absdiff_{prop}"] = float(pair_abs)
        out[f"pair_absdiff_{prop}_norm"] = float(pair_abs / (abs(mean_p) + eps))

    # -----------------------
    # E. electronic shortcuts
    # -----------------------
    if "vec" in ATOMIC_FEATURES:
        out["VEC"] = out["mean_vec"]
    elif "valence_electron_count" in ATOMIC_FEATURES:
        out["VEC"] = out["mean_valence_electron_count"]

    if "d_electron_count" in ATOMIC_FEATURES:
        out["d_count_mean"] = out["mean_d_electron_count"]
        out["d_count_std"] = out["std_d_electron_count"]

    if "f_electron_count" in ATOMIC_FEATURES:
        out["f_count_mean"] = out["mean_f_electron_count"]
        out["f_count_std"] = out["std_f_electron_count"]

    if "electronegativity_pauling" in ATOMIC_FEATURES:
        out["chi_mean"] = out["mean_electronegativity_pauling"]
        out["chi_std"] = out["std_electronegativity_pauling"]

    if "atomic_mass_amu" in ATOMIC_FEATURES:
        out["mass_mean"] = out["mean_atomic_mass_amu"]
        out["mass_std"] = out["std_atomic_mass_amu"]

    return out