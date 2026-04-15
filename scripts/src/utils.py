import re
import numpy as np
import gzip
import shutil
import os
import json
from datetime import datetime

TOTAL_ENERGY_RE = re.compile(r"total energy\s*=\s*([-+0-9.Ee]+)")

def generate_dirname(composition_labels, composition_ratio):
    txt = ""
    for label, ratio in zip(composition_labels, composition_ratio):
        txt += f"{label}{ratio:.4f}"
    return txt

def append_errorlog(errorlog_path, workdirname):
    errorlog_dir = os.path.dirname(errorlog_path)
    if errorlog_dir:
        os.makedirs(errorlog_dir, exist_ok=True)
    timestamp = datetime.now().isoformat(timespec="seconds")
    with open(errorlog_path, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} {workdirname}\n")

def parse_energy(text):
    m = TOTAL_ENERGY_RE.findall(text)
    return float(m[-1]) if m else np.nan


def converged_info_in_string(text):
    return "itr=499" not in text

def gzip_file(path):
    if not os.path.exists(path):
        return None
    gz = path + ".gz"
    with open(path, "rb") as f_in, gzip.open(gz, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.remove(path)
    return gz

def cleanup_potential_files(base):
    for suffix in [".pot", ".pot.info"]:
        f = base + suffix
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

def cleanup_fortran_files(base):
    for suffix in [".50", ".51"]:
        f = base + suffix
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception:
                pass

def save_dict_to_json(data: dict, filepath: str):
    # ensure directory exists
    dirname = os.path.dirname(filepath)
    if dirname:
        os.makedirs(dirname, exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def dist_to_si(x):
	return x*5.29177e-11

def dist_from_si(x):
	return x/5.29177e-11


def energy_to_si(e):
    return e * 2.1798741e-18


def energy_from_si(e):
    return e / 2.1798741e-18


def format_composition(comp_dict, digits=4):
    parts = []
    for el, val in comp_dict.items():
        parts.append(f"{el}={val:.{digits}f}")
    return ", ".join(parts)


def log_iteration_summary(
    log_path,
    iteration,
    known_data,
    new_data,
    composition_labels,
    target_col="Tc_mu0.1",
):
    """
    Append one iteration summary to a text log file.

    Parameters
    ----------
    log_path : str
        Path to log file.
    iteration : int
        Iteration index.
    known_data : list[dict]
        Full known dataset after this iteration.
    new_data : list[dict]
        Only the structures added in this iteration.
    composition_labels : list[str]
        Element/composition column names.
    target_col : str
        Name of target column, default 'Tc_mu0.1'
    """
    def valid_rows(rows):
        out = []
        for row in rows:
            if target_col in row and row[target_col] is not None:
                out.append(row)
        return out

    known_valid = valid_rows(known_data)
    new_valid = valid_rows(new_data)

    best_known = max(known_valid, key=lambda r: r[target_col]) if known_valid else None
    best_new = max(new_valid, key=lambda r: r[target_col]) if new_valid else None

    with open(log_path, "a") as f:
        f.write("=" * 80 + "\n")
        f.write(f"timestamp: {datetime.now().isoformat()}\n")
        f.write(f"iteration: {iteration}\n")
        f.write(f"n_known_total: {len(known_data)}\n")
        f.write(f"n_new_this_iteration: {len(new_data)}\n")

        if best_known is not None:
            comp_known = {el: best_known.get(el, 0.0) for el in composition_labels}
            f.write(f"best_known_{target_col}: {best_known[target_col]:.8f}\n")
            f.write(f"best_known_composition: {format_composition(comp_known)}\n")
        else:
            f.write(f"best_known_{target_col}: NA\n")

        if best_new is not None:
            comp_new = {el: best_new.get(el, 0.0) for el in composition_labels}
            f.write(f"best_new_{target_col}: {best_new[target_col]:.8f}\n")
            f.write(f"best_new_composition: {format_composition(comp_new)}\n")
        else:
            f.write(f"best_new_{target_col}: NA\n")

        f.write("\nnew_structures:\n")
        if not new_valid:
            f.write("  none\n")
        else:
            for i, row in enumerate(new_valid, start=1):
                comp = {el: row.get(el, 0.0) for el in composition_labels}
                f.write(
                    f"  {i:03d} | {target_col}={row[target_col]:.8f} | "
                    f"{format_composition(comp)}\n"
                )

        f.write("\n")