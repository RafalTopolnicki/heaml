import pandas as pd
import numpy as np
import sobol_seq

from src.consts import composition_labels, CANDIDATE_COMPOSITIONS_N, TARGET
from src.features import compute_hea_features


def generate_simplex_sobol(n_components, n_points, skip=0, eps=1e-12):
    u = sobol_seq.i4_sobol_generate(n_components, n_points, skip)
    u = np.clip(u, eps, 1.0)
    x = -np.log(u)
    x = x / x.sum(axis=1, keepdims=True)
    return x


def is_valid_composition(comp_dict, min_comp=None, max_comp=None, min_components=1, min_components_ratio=0.0):
    if min_comp is not None:
        for el, vmin in min_comp.items():
            if comp_dict.get(el, 0.0) < vmin:
                return False

    if max_comp is not None:
        for el, vmax in max_comp.items():
            if comp_dict.get(el, 0.0) > vmax:
                return False

    if min_components > 1:
        n_above = sum(1 for v in comp_dict.values() if v > min_components_ratio)
        if n_above < min_components:
            return False

    return True


def generate_global_candidates(
    n_candidates,
    min_comp=None,
    max_comp=None,
    skip=0,
    oversample_factor=3,
    max_rounds=20,
    round_decimals=8,
    min_components=1,
    min_components_ratio=0.0,
    composition_labels=None,
):
    # Must use the run's element subset: falling back to the module-level list (all 12 elements)
    # silently produced Sc/La-containing, sub-normalized candidates in restricted-element runs.
    if composition_labels is None:
        from src.consts import composition_labels
    candidates = []
    seen = set()
    n_components = len(composition_labels)

    current_skip = skip
    for _ in range(max_rounds):
        batch_size = max(1000, oversample_factor * (n_candidates - len(candidates)))
        composition_grid = generate_simplex_sobol(
            n_components=n_components,
            n_points=batch_size,
            skip=current_skip,
        )
        current_skip += batch_size

        for comp in composition_grid:
            comp_d = dict(zip(composition_labels, comp))

            if not is_valid_composition(comp_d, min_comp=min_comp, max_comp=max_comp,
                                        min_components=min_components, min_components_ratio=min_components_ratio):
                continue

            key = tuple(round(float(comp_d[el]), round_decimals) for el in composition_labels)
            if key in seen:
                continue
            seen.add(key)

            comp_d = {**comp_d, **compute_hea_features(comp_d), "_source": "global"}
            candidates.append(comp_d)

            if len(candidates) >= n_candidates:
                return pd.DataFrame(candidates)

    raise RuntimeError(f"Could only generate {len(candidates)} global candidates")


def generate_local_candidates(
    known_data,
    n_candidates,
    composition_labels=None,
    top_k=5,
    noise_scale=0.03,
    min_comp=None,
    max_comp=None,
    round_decimals=8,
    seed=0,
    min_components=1,
    min_components_ratio=0.0,
):
    from src.consts import composition_labels as _default_composition_labels
    if composition_labels is None:
        composition_labels = _default_composition_labels

    rng = np.random.default_rng(seed)

    df_known = pd.DataFrame(known_data).copy()
    df_known = df_known[pd.notna(df_known[TARGET])]
    df_known = df_known.sort_values(TARGET, ascending=False).head(top_k)

    if len(df_known) == 0:
        return pd.DataFrame([])

    candidates = []
    seen = set()

    centers = df_known[composition_labels].values
    per_center = max(1, int(np.ceil(n_candidates / len(centers))))

    for center in centers:
        generated_here = 0
        tries = 0

        while generated_here < per_center and len(candidates) < n_candidates and tries < per_center * 100:
            tries += 1

            noise = rng.normal(loc=0.0, scale=noise_scale, size=len(center))
            x = center + noise
            x = np.clip(x, 0.0, None)

            s = x.sum()
            if s <= 0:
                continue
            x = x / s

            comp_d = dict(zip(composition_labels, x))

            if not is_valid_composition(comp_d, min_comp=min_comp, max_comp=max_comp,
                                        min_components=min_components, min_components_ratio=min_components_ratio):
                continue

            key = tuple(round(float(comp_d[el]), round_decimals) for el in composition_labels)
            if key in seen:
                continue
            seen.add(key)

            comp_d = {**comp_d, **compute_hea_features(comp_d), "_source": "local"}
            candidates.append(comp_d)
            generated_here += 1

            if len(candidates) >= n_candidates:
                break

    return pd.DataFrame(candidates)


def generate_face_candidates(
    n_candidates,
    composition_labels,
    min_comp=None,
    max_comp=None,
    k_span=3,
    max_tries_factor=50,
    round_decimals=8,
    seed=0,
    min_components=1,
    min_components_ratio=0.0,
):
    """Candidates lying exactly on k-faces of the simplex (k elements nonzero, rest = 0).

    Uniform Sobol/Dirichlet sampling of the full simplex almost never produces compositions
    with few components, so low-dimensional faces (binaries, ternaries, ...) are reached only
    by chance via local search. Here k is drawn uniformly from [k_min, k_min + k_span], with
    k_min = max(2, min_components), the subset of k elements uniformly at random, and the
    fractions from a flat Dirichlet on that subset.
    """
    rng = np.random.default_rng(seed)
    n_el = len(composition_labels)
    k_min = min(max(2, min_components), n_el)
    k_max = min(k_min + k_span, n_el)

    candidates = []
    seen = set()
    tries = 0
    while len(candidates) < n_candidates and tries < n_candidates * max_tries_factor:
        tries += 1
        k = rng.integers(k_min, k_max + 1)
        idx = rng.choice(n_el, size=k, replace=False)
        x = np.zeros(n_el)
        x[idx] = rng.dirichlet(np.ones(k))
        comp_d = dict(zip(composition_labels, x))

        if not is_valid_composition(comp_d, min_comp=min_comp, max_comp=max_comp,
                                    min_components=min_components, min_components_ratio=min_components_ratio):
            continue

        key = tuple(round(float(comp_d[el]), round_decimals) for el in composition_labels)
        if key in seen:
            continue
        seen.add(key)

        comp_d = {**comp_d, **compute_hea_features(comp_d), "_source": "face"}
        candidates.append(comp_d)

    return pd.DataFrame(candidates)


def generate_candidates_data(
    known_data=None,
    min_comp=None,
    max_comp=None,
    n_candidates=CANDIDATE_COMPOSITIONS_N,
    fresh_fraction=0.7,
    local_top_k=5,
    local_noise_scale=0.03,
    seed=0,
    composition_labels=None,
    min_components=1,
    min_components_ratio=0.0,
    face_fraction=0.0,
):
    from src.consts import composition_labels as _default_composition_labels
    if composition_labels is None:
        composition_labels = _default_composition_labels

    # face candidates are carved out of the global share, so the local share is unchanged
    n_fresh_total = int(round(n_candidates * fresh_fraction))
    n_face = int(round(n_candidates * face_fraction))
    n_fresh = max(0, n_fresh_total - n_face)
    n_local = n_candidates - n_fresh_total

    print(f"[candidates] generating {n_fresh} global + {n_face} face + {n_local} local (seed={seed})")
    df_fresh = generate_global_candidates(
        n_candidates=n_fresh,
        min_comp=min_comp,
        max_comp=max_comp,
        skip=seed * max(n_candidates * 3, 1000),
        min_components=min_components,
        min_components_ratio=min_components_ratio,
        composition_labels=composition_labels,
    )
    print(f"[candidates] global done: {len(df_fresh)}")

    if n_face > 0:
        df_face = generate_face_candidates(
            n_candidates=n_face,
            composition_labels=composition_labels,
            min_comp=min_comp,
            max_comp=max_comp,
            seed=seed + 7919,
            min_components=min_components,
            min_components_ratio=min_components_ratio,
        )
        print(f"[candidates] face done: {len(df_face)}")
        df_fresh = pd.concat([df_fresh, df_face], ignore_index=True)

    if known_data is None or len(known_data) == 0 or n_local == 0:
        print(f"[candidates] skipping local (no known data or n_local=0), total={len(df_fresh)}")
        return df_fresh.reset_index(drop=True)

    df_local = generate_local_candidates(
        known_data=known_data,
        n_candidates=n_local,
        composition_labels=composition_labels,
        top_k=local_top_k,
        noise_scale=local_noise_scale,
        min_comp=min_comp,
        max_comp=max_comp,
        seed=seed,
        min_components=min_components,
        min_components_ratio=min_components_ratio,
    )
    print(f"[candidates] local done: {len(df_local)} (top_k={local_top_k}, noise={local_noise_scale})")

    df_all = pd.concat([df_fresh, df_local], ignore_index=True)
    n_before = len(df_all)

    df_all = df_all.drop_duplicates(subset=composition_labels).reset_index(drop=True)
    print(f"[candidates] after dedup: {len(df_all)} (dropped {n_before - len(df_all)} duplicates)")

    return df_all