#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import math
import os
from typing import Dict, List, Any

import numpy as np
import pandas as pd
try:
    from scipy.integrate import simpson
except:
    from scipy.integrate import simps as simpson


SPECIAL_LABELS = {"rstr_real_ef", "rstr_at_ef"}


def setup_logger(log_file):
    logger = logging.getLogger(log_file)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def parse_float(s: str) -> float:
    return float(s.replace("D", "E"))


def tokenize_text(text: str) -> List[str]:
    return text.split()


def group_dosef_by_l(dosef: List[float], mxlcmp: int) -> Dict[str, float]:
    labels = ["s", "p", "d", "f", "g", "h"]
    out: Dict[str, float] = {}
    idx = 0
    for l in range(mxlcmp):
        n = 2 * l + 1
        out[labels[l]] = float(sum(dosef[idx:idx + n]))
        idx += n
    return out


def parse_fort51(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        tokens = tokenize_text(f.read())

    n = len(tokens)
    p = 0

    def require(expected: str):
        nonlocal p
        if p >= n or tokens[p] != expected:
            got = tokens[p] if p < n else "EOF"
            raise ValueError(f"Expected token '{expected}', got '{got}' at token {p}")
        p += 1

    def read_int() -> int:
        nonlocal p
        if p >= n:
            raise ValueError("Unexpected EOF while reading integer")
        val = int(tokens[p])
        p += 1
        return val

    def read_n_floats(count: int) -> List[float]:
        nonlocal p
        vals: List[float] = []
        while len(vals) < count:
            if p >= n:
                raise ValueError(f"Unexpected EOF while reading {count} floats")
            vals.append(parse_float(tokens[p]))
            p += 1
        return vals

    data: Dict[str, Any] = {"components": []}

    require("ncmpx")
    data["ncmpx"] = read_int()

    require("meshr")
    data["meshr"] = read_int()

    require("nspin")
    data["nspin"] = read_int()

    if p < n and tokens[p] == "mse":
        p += 1
        data["mse"] = read_int()

    if p < n and tokens[p] == "kkdump":
        p += 1
        data["kkdump"] = read_int()

    require("ef")
    data["ef"] = read_n_floats(data["nspin"])

    if p < n and tokens[p] == "elvl_kkdump":
        p += 1
        data["elvl_kkdump"] = parse_float(tokens[p])
        p += 1

    meshr = data["meshr"]

    while p < n:
        if tokens[p] != "component":
            p += 1
            continue

        p += 1
        comp_index = read_int()

        require("spin")
        spin_index = read_int()

        require("mxlcmp")
        mxlcmp = read_int()

        require("xr")
        xr = read_n_floats(meshr)

        require("dr")
        dr = read_n_floats(meshr)

        require("v3")
        v3 = read_n_floats(meshr)

        require("dosef")
        dosef = read_n_floats(mxlcmp ** 2)

        block: Dict[str, Any] = {
            "component": comp_index,
            "spin": spin_index,
            "mxlcmp": mxlcmp,
            "xr": xr,
            "dr": dr,
            "v3": v3,
            "dosef": dosef,
        }

        if p < n and tokens[p] in SPECIAL_LABELS:
            radial_label = tokens[p]
            p += 1
            rstr: Dict[int, List[float]] = {}

            for _ in range(mxlcmp ** 2):
                require("lm")
                lm_index = read_int()
                rstr[lm_index] = read_n_floats(meshr)

            block[radial_label] = rstr

        data["components"].append(block)

    return data


def parse_fort52_detl(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        tokens = tokenize_text(f.read())

    n = len(tokens)
    p = 0

    def require(expected: str):
        nonlocal p
        if p >= n or tokens[p] != expected:
            got = tokens[p] if p < n else "EOF"
            raise ValueError(f"Expected token '{expected}', got '{got}' at token {p}")
        p += 1

    def read_int() -> int:
        nonlocal p
        val = int(tokens[p])
        p += 1
        return val

    def read_float() -> float:
        nonlocal p
        val = parse_float(tokens[p])
        p += 1
        return val

    def is_float_token(tok: str) -> bool:
        try:
            parse_float(tok)
            return True
        except Exception:
            return False

    require("nspin")
    nspin = read_int()

    require("mse")
    mse = read_int()

    require("ef")

    ef = []
    while p < n and tokens[p] != "spin":
        if is_float_token(tokens[p]):
            ef.append(read_float())
        else:
            break

    if len(ef) == 0:
        raise ValueError("No ef values found in fort.52")

    if len(ef) > nspin:
        ef = ef[:nspin]
    elif len(ef) < nspin:
        ef.extend([ef[-1]] * (nspin - len(ef)))

    spins: Dict[int, Dict[str, np.ndarray]] = {}

    while p < n:
        require("spin")
        ispin = read_int()
        require("energy_detl")

        E = np.zeros(mse, dtype=np.complex128)
        detl = np.zeros(mse, dtype=np.complex128)

        for k in range(mse):
            ere = read_float()
            eim = read_float()
            dre = read_float()
            dim = read_float()

            E[k] = ere + 1j * eim
            detl[k] = dre + 1j * dim

        spins[ispin] = {"E": E, "detl": detl}

    return {
        "nspin": nspin,
        "mse": mse,
        "ef": ef,
        "spins": spins,
    }


def compute_lloyd_dos(E: np.ndarray, detl: np.ndarray, eps: float = 1e-12):
    Ere = np.real(E)
    absdet = np.abs(detl)

    mask = (absdet > eps) & np.isfinite(Ere) & np.isfinite(detl.real) & np.isfinite(detl.imag)

    Ere_valid = Ere[mask]
    detl_valid = detl[mask]

    if len(Ere_valid) < 3:
        raise ValueError("Not enough valid determinant points for Lloyd DOS")

    keep = np.ones(len(Ere_valid), dtype=bool)
    keep[1:] = np.diff(Ere_valid) != 0.0

    Ere_valid = Ere_valid[keep]
    detl_valid = detl_valid[keep]

    if len(Ere_valid) < 3:
        raise ValueError("Not enough unique energy points for Lloyd DOS")

    logdet = np.log(detl_valid)
    dlogdet_dE = np.gradient(logdet, Ere_valid)
    dos_valid = (1.0 / np.pi) * np.imag(dlogdet_dE)

    return Ere_valid, dos_valid


def lloyd_total_dos_at_ef(detl_data: Dict[str, Any], spin_index: int, ef_value: float) -> float:
    if spin_index not in detl_data["spins"]:
        raise ValueError(f"Spin {spin_index} not found in determinant data")

    E = detl_data["spins"][spin_index]["E"]
    detl = detl_data["spins"][spin_index]["detl"]

    Ere_valid, dos_valid = compute_lloyd_dos(E, detl)
    return float(np.interp(ef_value, Ere_valid, dos_valid))


def normalize_u(x: np.ndarray, u: np.ndarray, norm_mode: str) -> np.ndarray:
    if norm_mode == "none":
        return u.copy()

    if norm_mode == "u2":
        norm2 = float(simpson(u * u, x=x))
    elif norm_mode == "r2u2":
        norm2 = float(simpson(x * x * u * u, x=x))
    else:
        raise ValueError(f"Unknown norm_mode: {norm_mode}")

    if norm2 <= 0 or not np.isfinite(norm2):
        raise ValueError(f"Invalid norm encountered while normalizing radial function: {norm_mode}")

    return u / math.sqrt(norm2)


def reduce_l_block_radials(
    rstr_block: Dict[int, List[float]],
    mxlcmp: int,
    x: np.ndarray,
    reduce_mode: str,
    norm_mode: str,
) -> Dict[str, np.ndarray]:
    labels = ["s", "p", "d", "f", "g", "h"]
    out: Dict[str, np.ndarray] = {}

    lm_start = 1
    for l in range(mxlcmp):
        nchan = 2 * l + 1
        arrs = []
        for lm in range(lm_start, lm_start + nchan):
            if lm not in rstr_block:
                raise ValueError(f"Missing lm={lm} in radial block")
            arrs.append(np.array(rstr_block[lm], dtype=float))

        stack = np.vstack(arrs)

        if reduce_mode == "mean":
            u = np.mean(stack, axis=0)
        elif reduce_mode == "sum":
            u = np.sum(stack, axis=0)
        elif reduce_mode == "first":
            u = np.array(rstr_block[lm_start], dtype=float)
        else:
            raise ValueError(f"Unknown reduce_mode: {reduce_mode}")

        out[labels[l]] = normalize_u(x, u, norm_mode)
        lm_start += nchan

    return out


def derivative_nonuniform(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.gradient(y, x)


def compute_M(u1: np.ndarray, u2: np.ndarray, dVdr: np.ndarray, x: np.ndarray, integral_mode: str) -> float:
    if integral_mode == "plain":
        integrand = u1 * dVdr * u2
    elif integral_mode == "r2":
        integrand = x * x * u1 * dVdr * u2
    else:
        raise ValueError(f"Unknown integral_mode: {integral_mode}")

    return float(simpson(integrand, x=x))


def compute_one_combination(
    block: Dict[str, Any],
    integral_mode: str,
    norm_mode: str,
    reduce_mode: str,
    dos_mode: str,
    detl_data: Dict[str, Any] | None = None,
    ef_header: List[float] | None = None,
) -> Dict[str, Any]:
    x_full = np.array(block["xr"], dtype=float)
    v_full = np.array(block["v3"], dtype=float)
    mxlcmp = int(block["mxlcmp"])
    dosef = np.array(block["dosef"], dtype=float)

    radial_key = None
    if "rstr_real_ef" in block:
        radial_key = "rstr_real_ef"
    elif "rstr_at_ef" in block:
        radial_key = "rstr_at_ef"
    else:
        raise ValueError("No radial block found (expected rstr_real_ef or rstr_at_ef)")

    x = x_full[:-1]
    v = v_full[:-1]

    raw_rstr = {
        int(k): np.array(vals[:-1], dtype=float)
        for k, vals in block[radial_key].items()
    }

    dVdr = derivative_nonuniform(x, v)
    grouped_dos = group_dosef_by_l(block["dosef"], mxlcmp)
    radials = reduce_l_block_radials(raw_rstr, mxlcmp, x, reduce_mode, norm_mode)

    labels = ["s", "p", "d", "f", "g", "h"]

    if dos_mode == "fort51":
        Ntot = float(np.sum(dosef))
    elif dos_mode == "lloyd":
        if detl_data is None:
            raise ValueError("dos_mode='lloyd' requires determinant data")
        if ef_header is None:
            raise ValueError("ef header not available for Lloyd DOS interpolation")
        spin_index = int(block["spin"])
        ef_value = float(ef_header[spin_index - 1])
        Ntot = lloyd_total_dos_at_ef(detl_data, spin_index, ef_value)
    else:
        raise ValueError(f"Unknown dos_mode: {dos_mode}")

    result: Dict[str, Any] = {
        "component": block["component"],
        "spin": block["spin"],
        "mxlcmp": mxlcmp,
        "integral_mode": integral_mode,
        "norm_mode": norm_mode,
        "reduce_mode": reduce_mode,
        "dos_mode": dos_mode,
        "Ntot": Ntot,
        "Ns": grouped_dos.get("s", np.nan),
        "Np": grouped_dos.get("p", np.nan),
        "Nd": grouped_dos.get("d", np.nan),
        "Nf": grouped_dos.get("f", np.nan),
    }

    eta_total = 0.0
    channel_names = ["sp", "pd", "df", "fg", "gh"]

    for l in range(mxlcmp - 1):
        a = labels[l]
        b = labels[l + 1]
        ch = channel_names[l]

        u1 = radials[a]
        u2 = radials[b]

        M = compute_M(u1, u2, dVdr, x, integral_mode)
        beta = M * M

        nl = float(grouped_dos[a])
        nlp1 = float(grouped_dos[b])

        eta = float(
            2.0 * (l + 1) / ((2 * l + 1) * (2 * l + 3))
            * (nl * nlp1 / Ntot)
            * beta
        )
        eta_total += eta

        result[f"M_{ch}"] = M
        result[f"beta_{ch}"] = beta
        result[f"eta_{ch}"] = eta

    result["eta_total"] = eta_total
    return result


def sweep_combinations(
    block: Dict[str, Any],
    dos_mode: str,
    detl_data: Dict[str, Any] | None = None,
    ef_header: List[float] | None = None,
) -> pd.DataFrame:
    integral_modes = ["plain", "r2"]
    norm_modes = ["none", "u2", "r2u2"]
    reduce_modes = ["mean", "sum", "first"]

    rows = []
    for integral_mode in integral_modes:
        for norm_mode in norm_modes:
            for reduce_mode in reduce_modes:
                row = compute_one_combination(
                    block=block,
                    integral_mode=integral_mode,
                    norm_mode=norm_mode,
                    reduce_mode=reduce_mode,
                    dos_mode=dos_mode,
                    detl_data=detl_data,
                    ef_header=ef_header,
                )
                rows.append(row)
    return pd.DataFrame(rows)


def run_mcmillan_sweep(
    *,
    workdir: str,
    output_prefix: str = "mcmillan",
    component: int | None = None,
    spin: int | None = None,
    dos_mode: str = "fort51",
    detl_file: str | None = None,
    save_json: bool = False,
) -> Dict[str, Any]:
    workdir = os.path.abspath(workdir)
    os.makedirs(workdir, exist_ok=True)

    fort51_path = os.path.join(workdir, "fort.51")
    if not os.path.exists(fort51_path):
        raise FileNotFoundError(f"fort.51 not found in workdir: {fort51_path}")

    detl_path = None
    if detl_file is not None:
        detl_path = detl_file if os.path.isabs(detl_file) else os.path.join(workdir, detl_file)
    elif dos_mode == "lloyd":
        detl_path = os.path.join(workdir, "fort.52")

    old_cwd = os.getcwd()
    os.chdir(workdir)
    try:
        logger = setup_logger(f"{output_prefix}_run.log")
        logger.info("Starting McMillan-Hopfield sweep")
        logger.info("workdir=%s", workdir)
        logger.info("fort51=%s", fort51_path)
        logger.info("dos_mode=%s", dos_mode)

        if dos_mode == "lloyd" and not detl_path:
            raise ValueError("dos_mode='lloyd' requires determinant file")
        if dos_mode == "lloyd" and not os.path.exists(detl_path):
            raise FileNotFoundError(f"Determinant file not found: {detl_path}")

        data = parse_fort51(fort51_path)
        detl_data = parse_fort52_detl(detl_path) if detl_path else None

        selected = []
        for block in data["components"]:
            if component is not None and block["component"] != component:
                continue
            if spin is not None and block["spin"] != spin:
                continue
            selected.append(block)

        if not selected:
            raise ValueError("No matching component/spin blocks found")

        dfs = []
        for block in selected:
            logger.info("Processing component=%s spin=%s", block["component"], block["spin"])
            df = sweep_combinations(
                block=block,
                dos_mode=dos_mode,
                detl_data=detl_data,
                ef_header=data.get("ef"),
            )
            df.insert(0, "fort51_path", fort51_path)
            df.insert(1, "detl_file", detl_path if detl_path else "")
            dfs.append(df)

        master_df = pd.concat(dfs, ignore_index=True)

        preferred_order = [
            "fort51_path",
            "dos_mode",
            "detl_file",
            "component", "spin", "mxlcmp",
            "integral_mode", "norm_mode", "reduce_mode",
            "Ns", "Np", "Nd", "Nf", "Ntot",
            "beta_sp", "beta_pd", "beta_df",
            "eta_sp", "eta_pd", "eta_df",
            "M_sp", "M_pd", "M_df",
            "eta_total",
        ]
        existing = [c for c in preferred_order if c in master_df.columns]
        remaining = [c for c in master_df.columns if c not in existing]
        master_df = master_df[existing + remaining]

        csv_path = os.path.join(workdir, f"{output_prefix}_results.csv")
        master_df.to_csv(csv_path, index=False)
        logger.info("CSV written to %s", csv_path)

        json_path = ""
        if save_json:
            json_path = os.path.join(workdir, f"{output_prefix}_results.json")
            with open(json_path, "w") as f:
                f.write(master_df.to_json(orient="records", indent=2))
            logger.info("JSON written to %s", json_path)

        summary = {
            "n_rows": int(len(master_df)),
            "n_components_selected": int(len(selected)),
            "fort51_path": fort51_path,
            "detl_file": detl_path if detl_path else "",
            "csv_path": csv_path,
            "json_path": json_path,
            "run_log": os.path.join(workdir, f"{output_prefix}_run.log"),
        }

        with open(os.path.join(workdir, f"{output_prefix}_summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

        logger.info("Workflow finished successfully")
        return {
            "summary": summary,
            "dataframe": master_df,
        }

    finally:
        os.chdir(old_cwd)


def main():
    ap = argparse.ArgumentParser(description="Compute M, beta and eta from AkaiKKR fort.51 over all combinations")
    ap.add_argument("--workdir", type=str, required=True,
                    help="Working directory containing fort.51 and output files")
    ap.add_argument("--output-prefix", type=str, default="mcmillan",
                    help="Prefix for output files")
    ap.add_argument("--component", type=int, default=None,
                    help="Component index to process (default: all)")
    ap.add_argument("--spin", type=int, default=None,
                    help="Spin index to process (default: all)")
    ap.add_argument("--dos-mode", choices=["fort51", "lloyd"], default="fort51",
                    help="Use total DOS from fort.51 or Lloyd determinant formula")
    ap.add_argument("--detl-file", type=str, default=None,
                    help="Determinant filename inside workdir, default fort.52 for lloyd mode")
    ap.add_argument("--save-json", action="store_true",
                    help="Also save JSON output")
    args = ap.parse_args()

    result = run_mcmillan_sweep(
        workdir=args.workdir,
        output_prefix=args.output_prefix,
        component=args.component,
        spin=args.spin,
        dos_mode=args.dos_mode,
        detl_file=args.detl_file,
        save_json=args.save_json,
    )

    print(result["dataframe"].to_string(index=False))
    print(f"\nCSV written to: {result['summary']['csv_path']}")
    if result["summary"]["json_path"]:
        print(f"JSON written to: {result['summary']['json_path']}")


if __name__ == "__main__":
    main()