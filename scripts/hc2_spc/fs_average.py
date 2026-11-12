"""
Fermi-surface average of D = <|v|^2 tau>/3 from A(k,E) on radial rays (make_fs_rays.py -> <sysdir>/fs/).

Per crossing on a ray with unit direction n (fit_crossings from analyze_spc.py, multi-Lorentz per segment):
  v_r   = |dE/dk| along the ray                       (Ry*bohr, hbar = 1)
  w     = HWHM in k along the ray at E_F; Gamma = v_r * w (energy HWHM, independent of direction)
  Gamma_dis = Gamma - edelt ;  tau = 1 / (2 Gamma_dis)
  k_F   = radial distance from Gamma
  DOS weight of the ray element: W = k_F^2 dOmega / v_r       (states between E and E+dE in the cone dOmega)
  |v|   = with --vstar: from Cartesian lines through the crossing (diag_star.py); otherwise
          v_r / cos(alpha), alpha = angle between n and the Fermi-surface normal; the normal is estimated by a
          plane fit through neighbouring crossings (all 48 O_h images of all rays, within a radius ~2.2 ray
          spacings); fall-back cos(alpha) = 1 (then |v| = v_r is a lower bound)
  D       = sum W |v|^2 tau / (3 sum W)      D_lower uses v_r instead of |v|
  N_FS    = 48 dOmega sum k_F^2 / v_r / (2 pi)^3  per spin per volume  -> compared with the KKR site DOS (coverage check)
Hc2 as in analyze_spc.py (dirty-limit WHH, Maki with lambda_so = 0).

Crossings with v_r > --v_max or Gamma > --gamma_max are rejected as fit artefacts (counted in n_rejected).
Usage: python scripts/hc2_spc/fs_average.py hc2_spc/Ta34Nb33Hf8Zr14Ti11 [--tc 7.3]
Writes <sysdir>/fs/fs_crossings.csv and <sysdir>/fs/fs_summary.json
"""
import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_spc import (load_source, BOHR, E_CH, HBAR, KB, PHI0, RY_J, T_AU, V_AU,  # noqa: E402
                         fit_crossings, kpath_coords, read_spc_c)

OH = [np.diag(sg)[list(pm)] for pm in itertools.permutations(range(3))
      for sg in itertools.product((1, -1), repeat=3)]                      # 48 matrices


def hc2_block(D, tc, n_states):
    S = 4 * PHI0 * KB / (np.pi ** 2 * HBAR * D)
    r = {"D_m2_s": float(D), "mu0_dHc2dT_T_per_K": float(S)}
    if n_states:
        r["rho0_uOhm_cm"] = float(1.0 / (E_CH ** 2 * n_states * D) * 1e8)
    if tc:
        h_orb = 0.693 * tc * S
        h_p = 1.86 * tc
        am = np.sqrt(2) * h_orb / h_p
        r.update(Tc_K=tc, mu0_Hc2_orb0_T=float(h_orb), mu0_HP0_T=float(h_p), alpha_M=float(am),
                 mu0_Hc2_0_Maki_T=float(h_orb / np.sqrt(1 + am ** 2)))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sysdir")
    ap.add_argument("--de_max", type=float, default=0.008)
    ap.add_argument("--prom", type=float, default=0.05)
    ap.add_argument("--tc", type=float, default=None)
    ap.add_argument("--v_max", type=float, default=3.0, help="reject crossings with v_r above this [Ry bohr] (3.3e6 m/s)")
    ap.add_argument("--gamma_max", type=float, default=0.1, help="reject crossings with energy HWHM above this [Ry]")
    ap.add_argument("--sub", default="fs", help="subdirectory with the ray runs")
    ap.add_argument("--vstar", action="store_true", help="use |v| from star/star_results.csv (diag_star.py) where available")
    ap.add_argument("--r_fac", type=float, default=2.2, help="neighbour radius in units of the local ray spacing")
    args = ap.parse_args()

    name = os.path.basename(os.path.normpath(args.sysdir))
    res, rp, x = load_source(args.sysdir)
    a = float(res["a0_bohr"])
    edelt = float(res["KKR_PARAMS_FINALSCF"]["edelt"])
    tc = args.tc if args.tc is not None else res.get("Tc_mu0.1")
    n_at_spin = sum(xi * res.get(f"{el}_Ntot", 0.0) for el, xi in zip(rp["element_labels"], x))  # states/Ry/atom/spin
    omega_at = a ** 3 / 2.0                                                                        # bohr^3
    fsdir = os.path.join(args.sysdir, args.sub)
    meta = json.load(open(os.path.join(fsdir, "fs_rays.json")))
    edelt = float(meta.get("edelt_spc", edelt))              # edelt actually used in the spc runs
    dirs = np.array(meta["dirs"])
    dOm = meta["domega"]
    spacing = np.sqrt(dOm)                                                   # rad

    cross = []
    a_int = []
    n_rejected = 0
    for run in meta["runs"]:
        f = os.path.join(fsdir, run["name"] + "_up.spc")
        if not os.path.exists(f):
            print(f"missing {f}")
            continue
        E, A, lab = read_spc_c(f)
        if len(lab) != len(run["points"]):
            raise RuntimeError(f"{f}: {len(lab)} labels vs {len(run['points'])} path points")
        labels = [(lab[i][0], tuple(run["points"][i])) for i in range(len(lab))]
        s, _ = kpath_coords(labels, A.shape[1], a)
        seg2ray = {r["seg"]: r for r in run["rays"]}
        # peak-independent check: N(E_F) ~ (Omega/(2pi)^3) * 48 dOmega * sum_rays int k^2 A(k,E_F) dk
        iEF = int(np.argmin(abs(E)))
        for sj, r in seg2ray.items():
            i0, i1 = labels[sj][0], labels[sj + 1][0]
            kk = s[i0:i1 + 1] - s[i0] if r["outward"] else s[i1] - s[i0:i1 + 1]
            a_int.append(abs(np.trapz(kk ** 2 * A[iEF, i0:i1 + 1], kk)))
        for c in fit_crossings(E, A, labels, s, edelt, args.de_max, args.prom, seg_ids=sorted(seg2ray)):
            r = seg2ray[c["seg_j"]]
            i0, i1 = labels[c["seg_j"]][0], labels[c["seg_j"] + 1][0]
            kF = c["kF_s"] - s[i0] if r["outward"] else s[i1] - c["kF_s"]
            v_r = c["v_F_au"]
            gam = v_r * c["hwhm_k_bohr_inv"]
            if v_r > args.v_max or gam > args.gamma_max:      # unphysical fit (Lorentzian spread over background)
                n_rejected += 1
                continue
            cross.append(dict(run=run["name"], ray=r["ray"], kF_bohr_inv=kF, v_r_au=v_r,
                              hwhm_k=c["hwhm_k_bohr_inv"], gamma_Ry=gam, gamma_dis_Ry=gam - edelt,
                              edelt_limited=c["edelt_limited"], broad=c["broad_flag"]))

    # Fermi-surface normals from neighbouring crossings (all O_h images)
    P = np.array([c["kF_bohr_inv"] * dirs[c["ray"]] for c in cross])
    imgs = np.concatenate([P @ R.T for R in OH])                             # (48 n, 3)
    n_fallback = 0
    for c, p in zip(cross, P):
        rad = args.r_fac * spacing * max(c["kF_bohr_inv"], 1e-6)
        d = np.linalg.norm(imgs - p, axis=1)
        nb = imgs[(d < rad) & (d > 1e-9)]
        cos_a = 1.0
        if len(nb) >= 2:
            pts = np.vstack([p, nb])
            u, sv, vt = np.linalg.svd(pts - pts.mean(axis=0))
            if sv[1] > 1e-3 * sv[0]:                                         # not collinear
                normal = vt[2]
                cos_a = abs(normal @ (p / np.linalg.norm(p)))
        if cos_a == 1.0:
            n_fallback += 1
        cos_a = max(cos_a, 0.15)
        c.update(cos_alpha=cos_a, n_neigh=int(len(nb)), v_abs_au=c["v_r_au"] / cos_a,
                 W=c["kF_bohr_inv"] ** 2 / c["v_r_au"])

    # total (Lloyd) DOS at E_F per spin per cell: 'def=' in the SCF output (spmain.f: d Im ln det / dE)
    n_lloyd = None
    try:
        tok = [l for l in open(os.path.join(args.sysdir, "scf.out")) if "def=" in l][-1].split()
        n_lloyd = float(tok[tok.index("def=") + 1])
    except (OSError, IndexError, ValueError):
        pass
    # full velocity vector from Cartesian "star" lines (diag_star.py, --n 0) where available
    n_star = 0
    star_csv = os.path.join(args.sysdir, "star", "cone_results.csv")
    if not os.path.exists(star_csv):
        star_csv = os.path.join(args.sysdir, "star", "grad_results.csv")
    if args.vstar and os.path.exists(star_csv):
        sm = json.load(open(os.path.join(args.sysdir, "star", "star.json")))
        if sm.get("sub") == args.sub:
            vst = {(r["run"], int(r["ray"]), round(float(r["kF"]), 6)): r for r in csv.DictReader(open(star_csv))}
            for c in cross:
                r = vst.get((c["run"], c["ray"], round(c["kF_bohr_inv"], 6)))
                if r:
                    c["v_abs_au"] = float(r.get("v_final") or r["v_grad"])   # cone LSQ / lower bound
                    c["cos_alpha"] = c["v_r_au"] / c["v_abs_au"]
                    n_star += 1
        else:
            print(f"star results belong to sub={sm.get('sub')}, not {args.sub}: ignored")
    out = dict(name=name, n_rays=len(dirs), n_vstar=n_star, n_crossings=len(cross), n_rejected=n_rejected, n_normal_fallback=n_fallback,
               edelt_Ry=edelt, de_max_Ry=args.de_max)
    W = np.array([c["W"] for c in cross])
    # DOS from the sampled Fermi surface: per spin, per atom, states/Ry
    n_fs = 48 * dOm * W.sum() / (2 * np.pi) ** 3 * omega_at
    out.update(N_EF_FS_states_Ry_atom_spin=float(n_fs), N_EF_KKR_site_states_Ry_atom_spin=float(n_at_spin),
               FS_coverage=float(n_fs / n_at_spin) if n_at_spin else None,
               N_EF_Lloyd_states_Ry_atom_spin=n_lloyd,
               FS_coverage_Lloyd=float(n_fs / n_lloyd) if n_lloyd else None,
               Aint_over_Lloyd=float(48 * dOm * np.sum(a_int) / (2 * np.pi) ** 3 * omega_at / n_lloyd) if n_lloyd else None,
               N_EF_Aint_raw=float(48 * dOm * np.sum(a_int) / (2 * np.pi) ** 3 * omega_at),
               v_abs_DOS_avg_m_s=float(np.sum(W * np.array([c["v_abs_au"] for c in cross])) / W.sum() * V_AU),
               v_r_DOS_avg_m_s=float(np.sum(W * np.array([c["v_r_au"] for c in cross])) / W.sum() * V_AU))
    # rms Fermi velocity and Drude plasma energy: omega_p^2 = e^2 N_V <v^2>/3 / eps0  (N_V both spins, per J m^3)
    va_all = np.array([c["v_abs_au"] for c in cross]) * V_AU
    v2 = float(np.sum(W * va_all ** 2) / W.sum())
    n_vol_fs = 2 * n_fs / RY_J / (omega_at * BOHR ** 3)
    eps0 = 8.8541878128e-12
    out.update(v_rms_m_s=float(np.sqrt(v2)),
               hbar_omega_p_eV=float(HBAR * np.sqrt(E_CH ** 2 * n_vol_fs * v2 / 3 / eps0) / E_CH))
    ok = [c for c in cross if not c["edelt_limited"] and c["gamma_dis_Ry"] > 0]
    out["n_used_for_D"] = len(ok)
    if ok:
        Wk = np.array([c["W"] for c in ok])
        tau = np.array([1.0 / (2 * c["gamma_dis_Ry"]) for c in ok])         # Ry^-1
        va = np.array([c["v_abs_au"] for c in ok])
        vr = np.array([c["v_r_au"] for c in ok])
        conv = V_AU * BOHR                                                   # (Ry*bohr)^2/Ry -> m^2/s
        D = np.sum(Wk * va ** 2 * tau) / (3 * Wk.sum()) * conv
        D_low = np.sum(Wk * vr ** 2 * tau) / (3 * Wk.sum()) * conv
        # transport-weighted mean free path and lifetime
        out.update(tau_DOS_avg_fs=float(np.sum(Wk * tau) / Wk.sum() * T_AU * 1e15),
                   l_DOS_avg_A=float(np.sum(Wk * va * tau) / Wk.sum() * BOHR * 1e10),
                   frac_DOS_used=float(Wk.sum() / W.sum()))
        n_vol = 2 * n_at_spin / RY_J / (omega_at * BOHR ** 3)                # states/(J m^3), both spins
        out["hc2"] = hc2_block(D, tc, n_vol)
        out["hc2_lower_D_vr"] = hc2_block(D_low, tc, n_vol)
    for c in cross:
        c.update(v_abs_m_s=c["v_abs_au"] * V_AU, l_A=(c["v_abs_au"] / (2 * c["gamma_dis_Ry"]) * BOHR * 1e10)
                 if c["gamma_dis_Ry"] > 0 and not c["edelt_limited"] else float("nan"))
    with open(os.path.join(fsdir, "fs_crossings.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(cross[0].keys()))
        w.writeheader()
        w.writerows(cross)
    json.dump(out, open(os.path.join(fsdir, "fs_summary.json"), "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
