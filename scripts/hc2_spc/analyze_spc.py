"""
Hc2 estimate from the AkaiKKR-CPA Bloch spectral function A(k,E)  (DOCS/ROZMOWA_2026-11-03_Hc2.md).

Input: spc file in AkaiKKR format (c) (matrix nE x nk, rows = energies E-E_F; header with energy range and
symmetry-point indices), produced by hc2_spc/<sys>/run_fine.sh (go=spc31, path G-H-N-P-G-N, see make_spc_inputs.py).

Per Fermi-level crossing on the k-path (one "branch"):
  * at every energy |E-E_F| <= --de_max, fit a Lorentzian + constant in k around the peak
      A(k) = c + h / (1 + ((k-k0)/w)^2),  w = HWHM in k
  * v_F = |dE/dk0| from a linear fit k0(E)        (atomic Rydberg units: hbar = 1, E in Ry, k in 1/bohr)
  * w(E_F) from the same fit; disorder part w_dis = w - edelt / v_F  (edelt = imaginary part of the energy mesh)
  * l = 1 / (2 w_dis)  (mean free path),  tau = l / v_F
  * check: Lorentzian fit in E at fixed k = k0(E_F) -> Gamma_E (HWHM in Ry); expected Gamma_E ~ v_F * w
Averages over branches:
  * D_flat = <v l>/3                       (each crossing weight 1)
  * D_dos  = sum(l) / sum(1/v) / 3         (weight 1/v  ~ local DOS of a crossing)
Hc2 (dirty limit, WHH with alpha = lambda_so = 0 for the orbital part; Tarantini 2026 SuST 39 021001, App. C):
  * mu0 |dHc2/dT|_Tc = 4 Phi0 kB / (pi^2 hbar D)            [T/K]
  * mu0 Hc2_orb(0)   = 0.693 Tc mu0|dHc2/dT|
  * mu0 H_P(0)       = 1.86 T/K * Tc   (BCS Pauli limit)
  * alpha_M          = sqrt(2) Hc2_orb(0) / H_P(0)
  * mu0 Hc2(0)       ~ Hc2_orb(0) / sqrt(1 + alpha_M^2)       (Maki, lambda_so = 0)
Consistency: rho_0 = 1 / (e^2 N_V D) with N_V from the KKR site DOS N(E_F) (results.json <El>_Ntot, assumed
states/Ry/atom/spin, x2 for spin) and the bcc atomic volume a^3/2.

LIMITATIONS: only crossings on the high-symmetry path (no full Fermi-surface average), no vertex corrections
(quasiparticle lifetime, not transport tau), Lorentzian line shape assumed; very broad peaks (w ~ k_F) are flagged.

Usage:
  python scripts/hc2_spc/analyze_spc.py hc2_spc/Ta34Nb33Hf8Zr14Ti11 [--spc spc_fine_up.spc] [--tc 7.3] [--plot]
Writes <dir>/hc2_branches.csv and <dir>/hc2_summary.json (and hc2_spc.png with --plot).
"""
import argparse
import json
import os
import re
from fractions import Fraction

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

# --- constants (SI)
HBAR = 1.054571817e-34
KB = 1.380649e-23
E_CH = 1.602176634e-19
PHI0 = 2.067833848e-15
BOHR = 0.529177210903e-10
RY_J = 2.1798723611e-18
V_AU = RY_J * BOHR / HBAR            # 1 Ry*bohr/hbar in m/s (= 1.0938e6)
T_AU = HBAR / RY_J                   # Rydberg time unit in s
BASELINE = "results/comparison.sra.ew0.6.bz10-10-20.gl"


def load_source(sysdir):
    """results.json / run_params.json of the KKR run this system was built from (<sysdir>/source.json, written by
    make_spc_inputs.py; fallback: baseline dir of the same name). Returns (res, rp, x) with x = fractions summing to 1."""
    sj = os.path.join(sysdir, "source.json")
    sjd = json.load(open(sj)) if os.path.exists(sj) else {}
    if "results" in sjd:                      # self-contained (make_spc_inputs.py --bundle; Topola has no results/)
        res, rp = sjd["results"], sjd["run_params"]
        x = np.array(rp["concentrations"], float)
        return res, rp, x / x.sum()
    src = sjd["src"] if sjd else os.path.join(BASELINE, os.path.basename(os.path.normpath(sysdir)))
    res = json.load(open(os.path.join(src, "results.json")))
    rp = json.load(open(os.path.join(src, "run_params.json")))
    x = np.array(rp["concentrations"], float)
    return res, rp, x / x.sum()


# ---------------------------------------------------------------- reading
def read_spc_c(path):
    """AkaiKKR format (c): '# e_first e_last nE nlabels' and '# idx '(kx ky kz)' ...' header lines, then
    nE rows x nk columns. The header energies are the centres of the first/last interval (E-E_F, Ry)."""
    e_first = e_last = None
    labels = []
    with open(path) as f:
        for line in f:
            if not line.startswith("#"):
                break
            if line.startswith("###"):
                continue
            body = line[1:].strip()
            if not body:
                continue
            if "'" in body:
                for idx, vec in re.findall(r"(\d+)\s+'\(([^)]*)\)'", body):
                    try:
                        kv = tuple(float(Fraction(x)) for x in vec.split())
                    except ValueError:          # non-symmetry points are printed as '(x h z)': caller supplies them
                        kv = None
                    labels.append((int(idx) - 1, kv))
            elif e_first is None:
                tok = body.split()
                e_first, e_last = float(tok[0]), float(tok[1])
    # Fortran writes each energy row with format (500f14.7) -> rows longer than 500 k-points wrap over lines
    with open(path) as f:
        vals = np.array([float(t) for line in f if not line.startswith("#") for t in line.split()])
    nk = labels[-1][0] + 1
    A = vals.reshape(-1, nk)
    E = np.linspace(e_first, e_last, A.shape[0])
    return E, A, labels


def kpath_coords(labels, nk, a_bohr):
    """Cumulative path length (1/bohr) at every k index; points are equidistant inside each segment."""
    s = np.zeros(nk)
    seg = np.zeros(nk, dtype=int)
    for j in range(len(labels) - 1):
        i0, k0 = labels[j]
        i1, k1 = labels[j + 1]
        L = np.linalg.norm(np.subtract(k1, k0)) * 2 * np.pi / a_bohr
        idx = np.arange(i0, i1 + 1)
        s[idx] = s[i0] + L * (idx - i0) / (i1 - i0)
        seg[i0:i1 + 1] = j
    return s, seg


def seg_name(labels, j):
    names = {(0, 0, 0): "G", (0, 1, 0): "H", (0.5, 0.5, 0): "N", (0.5, 0.5, 0.5): "P"}
    a = names.get(tuple(labels[j][1]), "?")
    b = names.get(tuple(labels[j + 1][1]), "?")
    return f"{a}-{b}"


# ---------------------------------------------------------------- fitting
def lorentz(x, c, h, x0, w):
    return c + h / (1.0 + ((x - x0) / w) ** 2)


def multi_lorentz(x, c, *p):
    y = np.full_like(x, c)
    for h, x0, w in zip(p[0::3], p[1::3], p[2::3]):
        y += h / (1.0 + ((x - x0) / w) ** 2)
    return y


def fit_peak(x, y, i_peak, half_window):
    """Lorentzian + constant fit around index i_peak; returns (x0, |w|, h) or None."""
    lo, hi = max(0, i_peak - half_window), min(len(x), i_peak + half_window + 1)
    xs, ys = x[lo:hi], y[lo:hi]
    if len(xs) < 5:
        return None
    w0 = max((xs[-1] - xs[0]) / 8, 1e-6)
    p0 = [max(ys.min(), 0.0), y[i_peak] - ys.min(), x[i_peak], w0]
    try:
        p, _ = curve_fit(lorentz, xs, ys, p0=p0, maxfev=20000)
    except (RuntimeError, ValueError):
        return None
    c, h, x0, w = p
    if not (xs[0] <= x0 <= xs[-1]) or h <= 0:
        return None
    return x0, abs(w), h


def half_window_from_fwhm(y, i, dx, x_max_window):
    """Fitting half-window (points): ~3 FWHM estimated from the raw data, capped."""
    h = y[i] / 2.0
    l = r = i
    while l > 0 and y[l] > h:
        l -= 1
    while r < len(y) - 1 and y[r] > h:
        r += 1
    return int(min(max(3 * (r - l), 8), x_max_window / dx))


def fit_crossings(E, A, labels, s, edelt, de_max, prom, seg_ids=None):
    """Fermi-level crossings on every path segment between consecutive labels (or only seg_ids).
    Returns a list of dicts (one per crossing); adds seg_j and kF_s (absolute path coordinate, 1/bohr)."""
    nE, nk = A.shape
    dk = np.median(np.diff(s)[np.diff(s) > 0])
    iEF = int(np.argmin(abs(E)))
    sel_E = np.where(abs(E) <= de_max)[0]

    # peaks at E_F, separately inside each segment (no fit across symmetry points). All peaks of one segment are
    # fitted together (sum of Lorentzians + constant), so that close pairs (H-N, G-N) are not merged into one.
    row = A[iEF]
    branches = []
    for j in (range(len(labels) - 1) if seg_ids is None else seg_ids):
        i0, i1 = labels[j][0], labels[j + 1][0]
        xs = s[i0:i1 + 1]
        pk, _ = find_peaks(row[i0:i1 + 1], prominence=prom * row.max())
        pk = [p for p in pk if 0 < p < i1 - i0]          # skip peaks sitting on a symmetry point
        if not pk:
            continue
        n = len(pk)
        p_ef = np.concatenate([[min(row[i0:i1 + 1].min(), 5.0)]] +
                              [[row[i0 + p], xs[p], 2 * dk] for p in pk])
        track = {}                                          # ie -> fitted params
        for direction in (+1, -1):
            p_prev = p_ef.copy()
            order = sel_E[sel_E >= iEF] if direction > 0 else sel_E[sel_E < iEF][::-1]
            for ie in order:
                y = A[ie, i0:i1 + 1]
                try:
                    pp, _ = curve_fit(multi_lorentz, xs, y, p0=p_prev, maxfev=40000)
                except (RuntimeError, ValueError):
                    break
                x0 = pp[2::3]
                if np.any(abs(x0 - p_prev[2::3]) > 0.1 * (xs[-1] - xs[0])) or np.any(pp[1::3] <= 0):
                    break                                   # lost a peak
                pp[3::3] = abs(pp[3::3])
                track[ie] = pp
                p_prev = pp
        if len(track) < 4 or iEF not in track:
            continue
        ies = np.array(sorted(track))
        es = E[ies]
        P = np.array([track[i] for i in ies])
        for m in range(n):
            k0s, ws = P[:, 2 + 3 * m], P[:, 3 + 3 * m]
            slope = np.polyfit(es, k0s, 1)[0]               # dk/dE  [1/(bohr Ry)]
            if slope == 0:
                continue
            v = 1.0 / abs(slope)                            # Ry*bohr  (hbar = 1)
            kF = float(track[iEF][2 + 3 * m])
            w_EF = float(track[iEF][3 + 3 * m])
            w_dis = w_EF - edelt / v
            # energy-direction check at k nearest to kF (window limited to the spc energy range)
            ikF = int(np.argmin(abs(s - kF)))
            colE = A[:, ikF]
            rE = fit_peak(E, colE, int(np.argmax(colE * (abs(E) <= 2 * de_max))), nE // 2)
            gE = rE[1] if rE is not None else None
            edelt_limited = bool(w_dis <= 0.1 * w_EF)        # width ~ edelt: no measurable disorder broadening
            l_bohr = 1.0 / (2.0 * w_dis) if not edelt_limited else np.nan
            branches.append(dict(
                seg_j=j, kF_s=kF, dkdE_sign=float(np.sign(slope)),  # sign(v.u), u = segment direction
                segment=seg_name(labels, j), k_index=ikF, kF_from_seg_start_bohr_inv=kF - xs[0],
                n_energies=int(len(es)), n_peaks_in_segment=n, v_F_au=v, v_F_m_s=v * V_AU,
                hwhm_k_bohr_inv=w_EF, hwhm_k_dis_bohr_inv=w_dis, edelt_fraction=(edelt / v) / w_EF,
                gamma_E_fit_Ry=gE, gamma_E_from_k_Ry=v * w_EF,
                l_bohr=l_bohr, l_A=l_bohr * BOHR * 1e10, tau_s=(l_bohr / v) * T_AU, edelt_limited=edelt_limited,
                broad_flag=bool(w_EF * 2 > 0.25 * (xs[-1] - xs[0])),
            ))
    return branches


# ---------------------------------------------------------------- main analysis
def analyze(d, spc, a_bohr, edelt, de_max, prom, tc, n_states, plot):
    E, A, labels = read_spc_c(os.path.join(d, spc))
    nE, nk = A.shape
    s, seg = kpath_coords(labels, nk, a_bohr)
    dk = np.median(np.diff(s)[np.diff(s) > 0])
    iEF = int(np.argmin(abs(E)))
    sel_E = np.where(abs(E) <= de_max)[0]

    branches = fit_crossings(E, A, labels, s, edelt, de_max, prom)

    out = dict(dir=d, spc=spc, a_bohr=a_bohr, edelt_Ry=edelt, de_max_Ry=de_max, n_branches=len(branches))
    ok = [b for b in branches if np.isfinite(b["l_bohr"])]
    if branches:
        out.update(v_F_mean_all_m_s=float(np.mean([b["v_F_m_s"] for b in branches])),
                   n_edelt_limited=int(sum(b["edelt_limited"] for b in branches)))
    if ok:
        v = np.array([b["v_F_m_s"] for b in ok])
        l = np.array([b["l_A"] for b in ok]) * 1e-10
        D_flat = np.mean(v * l) / 3.0
        D_dos = np.sum(l) / np.sum(1.0 / v) / 3.0
        out.update(v_F_mean_used_m_s=float(v.mean()), n_used=len(ok), l_mean_A=float(l.mean() * 1e10),
                   D_flat_m2_s=float(D_flat), D_dos_m2_s=float(D_dos))
        for tag, D in (("flat", D_flat), ("dos", D_dos)):
            S = 4 * PHI0 * KB / (np.pi ** 2 * HBAR * D)           # T/K
            r = {"mu0_dHc2dT_T_per_K": float(S)}
            if n_states is not None:
                r["rho0_uOhm_cm"] = float(1.0 / (E_CH ** 2 * n_states * D) * 1e8)
            if tc is not None:
                h_orb = 0.693 * tc * S
                h_p = 1.86 * tc
                am = np.sqrt(2) * h_orb / h_p
                r.update(Tc_K=tc, mu0_Hc2_orb0_T=float(h_orb), mu0_HP0_T=float(h_p), alpha_M=float(am),
                         mu0_Hc2_0_Maki_T=float(h_orb / np.sqrt(1 + am ** 2)))
            out[f"hc2_{tag}"] = r

    if plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(np.log10(A + 1e-3), aspect="auto", origin="lower", extent=[s[0], s[-1], E[0], E[-1]], cmap="magma")
        for lb in labels:
            ax.axvline(s[lb[0]], color="w", lw=0.5)
        ax.axhline(0, color="c", lw=0.5)
        for b in branches:
            ax.plot(s[b["k_index"]], 0, "c+")
        ax.set_ylabel("E - E_F [Ry]")
        ax.set_xlabel("k-path [1/bohr]")
        fig.tight_layout()
        fig.savefig(os.path.join(d, "hc2_spc.png"), dpi=150)
    return branches, out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dir")
    ap.add_argument("--spc", default="spc_fine_up.spc")
    ap.add_argument("--de_max", type=float, default=0.008, help="energy window |E-E_F| for v_F fit [Ry]")
    ap.add_argument("--prom", type=float, default=0.05, help="peak prominence as fraction of max A(k,E_F)")
    ap.add_argument("--tc", type=float, default=None, help="Tc [K]; default: Tc_mu0.1 from baseline results.json")
    ap.add_argument("--plot", action="store_true")
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.dir))
    res, rp, x = load_source(args.dir)
    a = float(res["a0_bohr"])
    edelt = float(res["KKR_PARAMS_FINALSCF"]["edelt"])
    tc = args.tc if args.tc is not None else res.get("Tc_mu0.1")
    # N(E_F) per volume, both spins (assumes <El>_Ntot = states/Ry/atom/spin; site DOS, not Lloyd)
    n_at = sum(xi * res.get(f"{el}_Ntot", 0.0) for el, xi in zip(rp["element_labels"], x)) * 2.0  # states/Ry/atom
    n_states = n_at / RY_J / (a ** 3 / 2.0 * BOHR ** 3)                                        # states/(J m^3)

    branches, out = analyze(args.dir, args.spc, a, edelt, args.de_max, args.prom, tc, n_states, args.plot)
    out.update(name=name, N_EF_states_Ry_atom_both_spins=n_at, Tc_source="--tc" if args.tc else "Tc_mu0.1")
    import csv
    with open(os.path.join(args.dir, "hc2_branches.csv"), "w", newline="") as f:
        if branches:
            w = csv.DictWriter(f, fieldnames=list(branches[0].keys()))
            w.writeheader()
            w.writerows(branches)
    json.dump(out, open(os.path.join(args.dir, "hc2_summary.json"), "w"), indent=2)

    print(f"{name}: {len(branches)} crossings")
    for b in branches:
        print(f"  {b['segment']:4s} k#{b['k_index']:4d}  v_F={b['v_F_m_s'] / 1e5:5.2f}e5 m/s  "
              f"hwhm_k={b['hwhm_k_bohr_inv']:.4f} (edelt {100 * b['edelt_fraction']:.0f}%)  l={b['l_A']:6.1f} A  "
              f"Gamma_E fit/from-k = {b['gamma_E_fit_Ry'] or float('nan'):.4f}/{b['gamma_E_from_k_Ry']:.4f} Ry"
              f"{'  BROAD' if b['broad_flag'] else ''}{'  EDELT-LIMITED' if b['edelt_limited'] else ''}")
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("D_", "hc2", "v_F", "l_", "N_EF", "n_"))}, indent=2))


if __name__ == "__main__":
    main()
