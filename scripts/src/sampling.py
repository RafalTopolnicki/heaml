import pandas as pd
import numpy as np
import sobol_seq

from src.consts import composition_labels, CANDIDATE_COMPOSITIONS_N
from src.features import compute_hea_features


def generate_simplex_sobol(n_components, n_points, skip=0, eps=1e-12):
    """
    Generate quasi-uniform points on the simplex using Sobol points in [0,1]^d
    and the exponential-normalization transform.
    """
    u = sobol_seq.i4_sobol_generate(n_components, n_points, skip)
    u = np.clip(u, eps, 1.0)  # avoid log(0)
    x = -np.log(u)
    x = x / x.sum(axis=1, keepdims=True)
    return x


def is_valid_composition(comp_dict, min_comp=None, max_comp=None):
    """
    Optional composition filter.
    """
    if min_comp is not None:
        for el, vmin in min_comp.items():
            if comp_dict.get(el, 0.0) < vmin:
                return False

    if max_comp is not None:
        for el, vmax in max_comp.items():
            if comp_dict.get(el, 0.0) > vmax:
                return False

    return True


def generate_candidates_data(
    n_candidates=CANDIDATE_COMPOSITIONS_N,
    min_comp=None,
    max_comp=None,
    oversample_factor=2,
    max_rounds=50,
    round_decimals=10,
):
    """
    Generate exactly n_candidates candidate compositions.

    Strategy:
    - generate Sobol simplex points in batches
    - optionally filter them
    - deduplicate by rounded composition
    - stop once enough are collected
    """
    candidates = []
    seen = set()
    skip = 0
    n_components = len(composition_labels)

    for _ in range(max_rounds):
        batch_size = max(1000, oversample_factor * (n_candidates - len(candidates)))
        composition_grid = generate_simplex_sobol(
            n_components=n_components,
            n_points=batch_size,
            skip=skip,
        )
        skip += batch_size

        for comp in composition_grid:
            comp_d = dict(zip(composition_labels, comp))

            if not is_valid_composition(comp_d, min_comp=min_comp, max_comp=max_comp):
                continue

            # deduplicate by composition only
            key = tuple(round(float(comp_d[el]), round_decimals) for el in composition_labels)
            if key in seen:
                continue
            seen.add(key)

            full_d = {**comp_d, **compute_hea_features(comp_d)}
            candidates.append(full_d)

            if len(candidates) >= n_candidates:
                return pd.DataFrame(candidates)

    raise RuntimeError(
        f"Could only generate {len(candidates)} unique valid candidates "
        f"after {max_rounds} rounds. Relax constraints or increase max_rounds."
    )