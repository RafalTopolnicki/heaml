import re
import numpy as np
import gzip
import shutil
import os
import json
from datetime import datetime
from src.consts import FOUND_ITERATION_KEY

TOTAL_ENERGY_RE = re.compile(r"total energy\s*=\s*([-+0-9.Ee]+)")
AKAI_COMMAND_RE = re.compile(r"^\s*go=(\S+)\s+file=", re.MULTILINE)

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


def extract_akai_command_block(text, command):
    matches = list(AKAI_COMMAND_RE.finditer(text))
    if not matches:
        return text

    for index, match in enumerate(matches):
        if match.group(1).lower() == command.lower():
            end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
            return text[match.start():end]

    return ""


def parse_go_energy(text):
    return parse_energy(extract_akai_command_block(text, "go"))


def converged_go_in_string(text):
    go_text = extract_akai_command_block(text, "go")
    if not go_text:
        return False
    return converged_info_in_string(go_text) and "*** no convergence" not in go_text.lower()

def gzip_file(path):
    if not os.path.isfile(path):
        return None
    gz = path + ".gz"
    # write to a temp name and rename, so a crash never leaves a truncated .gz next to a deleted original
    tmp = gz + ".tmp"
    with open(path, "rb") as f_in, gzip.open(tmp, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    os.replace(tmp, gz)
    os.remove(path)
    return gz

# Per-composition archive of the raw KKR files (inputs, gzipped outputs, fort.5x, CSVs, logs). The SLURM home quota
# counts files (62k), and a finished composition used to hold ~40 of them (incl. subdirectories); after packing only
# results.json, run_params.json and kkr_raw.tar remain. process_kkr() reads debye_summary.json and
# mcmillan_cutoff_results.csv through read_kkr_file(), i.e. from disk when present, else from the archive.
# Plain (uncompressed) tar: the big members are already .gz, and 'a' mode can append to an existing archive.
KKR_ARCHIVE = "kkr_raw.tar"
KKR_PACK_SUBDIRS = ["lattice", "debye", "finalscf", "finalscf.copy"]
KKR_KEEP_FILES = set()   # paths relative to the composition dir that stay on disk (none by default)


def pack_kkr_files(compdir):
    """Move everything in compdir/{lattice,debye,finalscf} except KKR_KEEP_FILES into compdir/kkr_raw.tar and remove
    emptied subdirectories. Files are deleted only after the archive is complete; an existing archive is appended to
    (a re-run's members follow the old ones, so extraction keeps the newest). Returns the number of files packed."""
    import tarfile
    members = []
    for sub in KKR_PACK_SUBDIRS:
        subdir = os.path.join(compdir, sub)
        if not os.path.isdir(subdir) or os.path.islink(subdir):
            continue
        for dirpath, _, filenames in os.walk(subdir):
            for name in sorted(filenames):
                path = os.path.join(dirpath, name)
                rel = os.path.relpath(path, compdir).replace(os.sep, "/")
                if rel not in KKR_KEEP_FILES and os.path.isfile(path) and not os.path.islink(path):
                    members.append((path, rel))
    if not members:
        return 0
    archive = os.path.join(compdir, KKR_ARCHIVE)
    tmp = archive + ".tmp"
    if os.path.exists(archive):
        shutil.copy2(archive, tmp)
    with tarfile.open(tmp, "a") as tar:
        for path, rel in members:
            tar.add(path, arcname=rel)
    os.replace(tmp, archive)
    for path, _ in members:
        os.remove(path)
    for sub in KKR_PACK_SUBDIRS:
        subdir = os.path.join(compdir, sub)
        for dirpath, _, _ in os.walk(subdir, topdown=False):  # deepest first
            try:
                os.rmdir(dirpath)  # only succeeds when empty
            except OSError:
                pass
    return len(members)


def read_kkr_file(compdir, rel):
    """Text of compdir/<rel>, from disk or (when packed) from compdir/kkr_raw.tar; last archived copy wins."""
    import tarfile
    path = os.path.join(compdir, rel)
    if os.path.isfile(path):
        with open(path) as f:
            return f.read()
    archive = os.path.join(compdir, KKR_ARCHIVE)
    if os.path.isfile(archive):
        with tarfile.open(archive) as tar:
            hits = [m for m in tar.getmembers() if m.name == rel]
            if hits:
                return tar.extractfile(hits[-1]).read().decode()
    raise FileNotFoundError(f"{rel} neither in {compdir} nor in its {KKR_ARCHIVE}")


def cleanup_potential_files(base):
    # The input written for "<base>.inp" says "go <base>.inp.pot", so AkaiKKR names the potential
    # <base>.inp.pot(.info). Until 2026-10-21 only <base>.pot was tried, which never exists, and every
    # potential (~0.4 MB each, ~7 per composition) stayed on disk. Exact names only, no globbing.
    for suffix in [".inp.pot", ".inp.pot.info", ".pot", ".pot.info"]:
        f = base + suffix
        if os.path.isfile(f):
            try:
                os.remove(f)
            except Exception:
                pass

def cleanup_fortran_files(base):
    for f in [base + ".50", base + ".51", "fort.50", "fort.51"]:
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
    min_comp=None,
    max_comp=None,
    min_components=1,
    min_components_ratio=0.0,
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
    min_comp, max_comp : dict or None
        Composition bounds. When provided, best_known is taken only over rows
        that satisfy the bounds — out-of-bounds init points are excluded from
        the reported maximum (but remain in known_data for model training).
    min_components, min_components_ratio : int, float
        Rows with fewer than min_components elements above min_components_ratio are
        likewise excluded from best_known (they stay in known_data for training).
    """
    def in_bounds(row):
        for el, lo in (min_comp or {}).items():
            if row.get(el, 0.0) < lo - 1e-9:
                return False
        for el, hi in (max_comp or {}).items():
            if row.get(el, 0.0) > hi + 1e-9:
                return False
        if min_components > 1:
            if sum(1 for el in composition_labels if row.get(el, 0.0) > min_components_ratio) < min_components:
                return False
        return True

    def valid_rows(rows, apply_bounds=False):
        out = []
        for row in rows:
            if target_col in row and row[target_col] is not None:
                if not apply_bounds or in_bounds(row):
                    out.append(row)
        return out

    known_valid = valid_rows(known_data, apply_bounds=True)
    new_valid = valid_rows(new_data)  # new data always satisfies bounds

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
            found_it = best_known.get(FOUND_ITERATION_KEY)
            f.write(f"best_known_found_in_iteration: {'NA' if found_it is None else found_it}\n")
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
