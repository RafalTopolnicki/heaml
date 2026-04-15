import re
import numpy as np
import gzip
import shutil
import os
import json

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