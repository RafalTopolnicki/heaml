import pandas as pd
import numpy as np
from src.consts import composition_labels, CANDIDATE_COMPOSITIONS_N
from src.features import compute_hea_features
import sobol_seq

def generate_simplex_sobol(n_components, n_points):
    u = sobol_seq.i4_sobol_generate(n_components, n_points)

    # project to simplex
    x = -np.log(u)
    x = x / x.sum(axis=1, keepdims=True)

    return x

def generate_candidates_data():
    composition_grid = generate_simplex_sobol(n_components=len(composition_labels), n_points=CANDIDATE_COMPOSITIONS_N)
    candidates = []
    for comp in composition_grid:
        comp_d = dict(zip(composition_labels, comp))
        comp_d = {**comp_d, **compute_hea_features(comp_d)}
        candidates.append(comp_d)
    return pd.DataFrame(candidates)
















# def generate_compositions(n_components=6, tol=1e-6):
#     """
#     Generate all n_components-tuples from composition_grid
#     that sum to 1 (within tolerance).
#
#     Returns:
#         numpy array of shape (N, n_components)
#     """
#     composition_grid = np.linspace(NEW_COMPOSITION_MIN, NEW_COMPOSITION_MAX, NEW_COMPOSITION_NUM)
#     grid = np.array(composition_grid)
#     results = []
#
#     def backtrack(current, remaining_sum, depth):
#         if depth == n_components - 1:
#             # last component determined by remainder
#             val = remaining_sum
#             # snap to nearest grid value
#             idx = np.abs(grid - val).argmin()
#             if abs(grid[idx] - val) < tol:
#                 results.append(current + [grid[idx]])
#             return
#
#         for g in grid:
#             if g > remaining_sum:
#                 break  # pruning
#             backtrack(current + [g], remaining_sum - g, depth + 1)
#
#     backtrack([], 1.0, 0)
#     return np.array(results)
#
