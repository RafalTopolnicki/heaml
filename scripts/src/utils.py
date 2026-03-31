import re
import numpy as np
import gzip
import shutil
import os

TOTAL_ENERGY_RE = re.compile(r"total energy\s*=\s*([-+0-9.Ee]+)")

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

def dist_to_si(x):
	return x*5.29177e-11

def dist_from_si(x):
	return x/5.29177e-11


def energy_to_si(e):
    return e * 2.1798741e-18


def energy_from_si(e):
    return e / 2.1798741e-18