"""Flat CSV database of finished KKR compositions: one row per composition, holding the raw quantities read from the
KKR files (src.process_kkr.read_kkr_raw: composition, lattice/Debye/elastic data, per-element McMillan-Hopfield eta
and N(E_F), energy, KKR parameters) together with everything derived from them (eta_mix, lambda, Tc_mu*, Stoner,
features, dG), so the file can be read as a table on its own.

run_opt_compositions.py --init_db reads such files instead of scanning KKR directories (no need to keep the init
datasets on the cluster): the derived columns are dropped and recomputed from the raw ones with the current code
(derive_kkr), exactly as process_kkr does for a directory, so changes in consts (DG_T_K, Stoner, physical limits)
apply without rebuilding the database. The stored values stay available for comparison.

Extra columns written by write_database: element_labels ("Ti,Nb,..." as in run_params.json), source (dataset
directory), eta_variant (from results.json, if present). Dict/list values (KKR_PARAMS_*, deltas) are JSON strings.
"""
import json
import math
import os

import pandas as pd

from src.consts import composition_labels as ALL_ELEMENTS
from src.features import compute_hea_features
from src.process_kkr import read_kkr_raw, derive_kkr

META_COLUMNS = ['element_labels', 'source', 'eta_variant']
# keys derive_kkr() sets besides the features; dropped before recomputation so no stale value survives
DERIVED_KEYS = {'eta_mix', 'lambda', 'outside_range', 'lambda_unphysical', 'lambda_nocutoff', 'dG_eV'}
DERIVED_PREFIXES = ('Tc_', 'Stoner_')


def database_row(path, dirname):
    """Row for path/dirname (raw + derived + meta columns); rows with thetaDB=None are kept (Tc=0)."""
    raw, elements = read_kkr_raw(path, dirname)
    row = derive_kkr(dict(raw), elements, allow_missing_theta=True)
    row['element_labels'] = ','.join(elements)
    row['source'] = os.path.basename(os.path.normpath(path))
    try:
        with open(os.path.join(path, dirname, 'results.json')) as f:
            row['eta_variant'] = json.load(f).get('eta_variant')
    except (OSError, ValueError):
        row['eta_variant'] = None
    if row['eta_variant'] is None:
        fs = raw.get('KKR_PARAMS_FINALSCF') or {}
        if 'mcmillan_cutoff_mode' in fs:
            row['eta_variant'] = '+'.join([fs['mcmillan_cutoff_mode']] + ([fs['mcmillan_dos_normalization']]
                                          if 'mcmillan_dos_normalization' in fs else []))
    return row


def write_database(rows, out_path):
    """Write rows (list of dicts) to out_path; meta columns and the name come first."""
    df = pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()} for r in rows])
    first = [c for c in ['name'] + META_COLUMNS if c in df.columns]
    df = df[first + [c for c in df.columns if c not in first]]
    df.to_csv(out_path, index=False)
    return df


def _cell(v):
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if isinstance(v, str):
        if v in ('True', 'False'):
            return v == 'True'
        if v[:1] in ('{', '['):
            try:
                return json.loads(v)
            except ValueError:
                return v
    elif hasattr(v, 'item'):  # numpy scalar -> python
        return v.item()
    return v


def _derived_keys(row, elements):
    feats = compute_hea_features(comp_dict={e: row[e] for e in elements}, normalize_composition=True)
    return set(feats) | DERIVED_KEYS | {k for k in row if k.startswith(DERIVED_PREFIXES)}


def read_database(paths, allow_missing_theta=False):
    """Rows of one or more database CSVs as process_kkr-style dicts, derived quantities recomputed.
    allow_missing_theta=False (optimizer) drops rows without KKR thetaDB, like process_kkr does.
    Returns (rows, stored) with stored[name] = the row as written in the file (meta columns removed from rows)."""
    if isinstance(paths, str):
        paths = [paths]
    rows, stored = [], {}
    for path in paths:
        df = pd.read_csv(path, float_precision='round_trip', low_memory=False)
        for rec in df.to_dict(orient='records'):
            rec = {k: _cell(v) for k, v in rec.items()}
            elements = rec['element_labels'].split(',')
            # element columns of other datasets in the same file are not part of this composition
            for e in ALL_ELEMENTS:
                if e not in elements and rec.get(e) is None:
                    rec.pop(e, None)
            stored[rec['name']] = dict(rec)
            data = {k: v for k, v in rec.items() if k not in META_COLUMNS and k not in _derived_keys(rec, elements)}
            try:
                rows.append(derive_kkr(data, elements, allow_missing_theta=allow_missing_theta))
            except Exception as e:
                if allow_missing_theta or data.get('thetaDB') is not None:
                    print(f"Error in processing {rec['name']} from {path}: {e}")
    return rows, stored
