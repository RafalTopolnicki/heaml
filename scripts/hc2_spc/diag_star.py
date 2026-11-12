"""
Diagnostic: full Fermi-velocity vector at Fermi-surface crossings from short Cartesian lines ("stars").

For crossings found on the radial rays (<sysdir>/<sub>/fs_crossings.csv) the point P = k_F n is surrounded by
three short lines P -/+ delta e_x, e_y, e_z. Along each line the peak shift with energy gives |v_i| = |dE/dk_i|.
Lines are chained into AkaiKKR format-(c) polylines (P-dx -> P+dx -> P-dy -> P+dy -> P-dz -> P+dz; the connecting
pieces are not used), 4 crossings per run (24 path points <= 25).

  make:     python scripts/hc2_spc/diag_star.py make    hc2_spc/Nb --sub fs_e003 [--n 48] [--delta 0.06]
  run:      hc2_spc/Nb/star/run_star.sh
  analyse:  python scripts/hc2_spc/diag_star.py analyse hc2_spc/Nb --sub fs_e003
  gradient: python scripts/hc2_spc/diag_star.py make_grad hc2_spc/Nb ; star/run_grad.sh ; ... analyse_grad hc2_spc/Nb
            one more line along the sign-resolved star vector; |v| = max(v_r, v.u) (lower bound, cannot overshoot);
            written to star/grad_results.csv and used by fs_average.py --vstar.
  cone:     ... make_cone ; star/run_cone.sh ; ... analyse_cone  — 6 lines (u0 + 5 on a 30 deg cone), signed
            v.u_i, least squares for the full vector with rejection of wrong-sheet lines; v_final = |v_LSQ| if
            >= 0.95 x lower bound, else the lower bound max(v_grad, |v.u_i|). -> star/cone_results.csv
Compares |v|_star with v_r (radial) and v_abs (normal-plane estimate used in fs_average.py), and recomputes the
DOS-weighted v_rms / plasma energy for the sampled crossings.
"""
import argparse
import csv
import itertools
import json
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_spc import V_AU, fit_crossings, kpath_coords, load_source, read_spc_c  # noqa: E402

PER_RUN = 4


def make(args):
    res, rp, x = load_source(args.sysdir)
    a = float(res["a0_bohr"])
    cross = list(csv.DictReader(open(os.path.join(args.sysdir, args.sub, "fs_crossings.csv"))))
    meta = json.load(open(os.path.join(args.sysdir, args.sub, "fs_rays.json")))
    dirs = np.array(meta["dirs"])
    rng = np.random.default_rng(0)
    W = np.array([float(c["W"]) for c in cross])
    if args.n and args.n < len(cross):
        pick = rng.choice(len(cross), size=args.n, replace=False, p=W / W.sum())   # DOS-weighted sample
    else:
        pick = np.arange(len(cross))                                              # --n 0: all crossings
    head = open(os.path.join(args.sysdir, args.sub, "fs_00.inp")).read().split("# k-points")[0]
    pot = re.search(r"^spc\S*\s+(\S+)", head, re.M).group(1).replace("../", "")
    head = head.replace(f"../{pot}", f"../{pot}")
    sd = os.path.join(args.sysdir, "star")
    os.makedirs(sd, exist_ok=True)
    runs = []
    for r0 in range(0, len(pick), PER_RUN):
        pts, items = [], []
        for ci in pick[r0:r0 + PER_RUN]:
            c = cross[ci]
            P = float(c["kF_bohr_inv"]) * dirs[int(c["ray"])] * a / (2 * np.pi)      # -> units 2pi/a
            segs = []
            for ax in range(3):
                e = np.zeros(3)
                e[ax] = args.delta
                pts += [P - e, P + e]
                segs.append(len(pts) - 2)                     # segment index between the two points
            items.append(dict(cross=int(ci), P=P.tolist(), segs=segs))
        pts = np.array(pts)
        L = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        nk = int(np.ceil(L / args.dk)) + 1
        name = f"star_{len(runs):02d}"
        with open(os.path.join(sd, name + ".inp"), "w") as f:
            f.write(head + "# k-points\n" + f" {nk}\n" + "".join(f" {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n" for p in pts) + "end\n")
        runs.append(dict(name=name, points=pts.tolist(), items=items))
    json.dump(dict(sub=args.sub, delta=args.delta, edelt_spc=meta.get("edelt_spc"), runs=runs),
              open(os.path.join(sd, "star.json"), "w"), indent=1)
    with open(os.path.join(sd, "run_star.sh"), "w") as f:
        f.write('#!/bin/bash\nset -e\ncd "$(dirname "$0")"\n')
        f.write('BIN=${AKAIMODBIN:-/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx}\n')
        for r in runs:
            f.write(f"$BIN < {r['name']}.inp > {r['name']}.out && mv ../{pot}_up.spc {r['name']}_up.spc\n")
        f.write(f"rm -f ../{pot}_dn.spc\n")
    os.chmod(os.path.join(sd, "run_star.sh"), 0o755)
    print(f"{sd}: {len(pick)} crossings, {len(runs)} runs")


def analyse(args):
    res, rp, x = load_source(args.sysdir)
    a = float(res["a0_bohr"])
    sd = os.path.join(args.sysdir, "star")
    meta = json.load(open(os.path.join(sd, "star.json")))
    edelt = float(meta["edelt_spc"] or res["KKR_PARAMS_FINALSCF"]["edelt"])
    cross = list(csv.DictReader(open(os.path.join(args.sysdir, meta["sub"], "fs_crossings.csv"))))
    rows = []
    for run in meta["runs"]:
        E, A, lab = read_spc_c(os.path.join(sd, run["name"] + "_up.spc"))
        labels = [(lab[i][0], tuple(run["points"][i])) for i in range(len(lab))]
        s, _ = kpath_coords(labels, A.shape[1], a)
        for it in run["items"]:
            comps, nfound = [], 0
            for sj in it["segs"]:
                i0, i1 = labels[sj][0], labels[sj + 1][0]
                mid = 0.5 * (s[i0] + s[i1])
                cs = fit_crossings(E, A, labels, s, edelt, args.de_max, args.prom, seg_ids=[sj])
                if cs:
                    best = min(cs, key=lambda c: abs(c["kF_s"] - mid))
                    if abs(best["kF_s"] - mid) < 0.5 * (s[i1] - s[i0]) and best["v_F_au"] < 3.0:
                        comps.append(best["v_F_au"])
                        nfound += 1
                        continue
                comps.append(0.0)                              # no crossing on this line: component ~ 0
            c = cross[it["cross"]]
            vs = float(np.sqrt(np.sum(np.square(comps))))
            rows.append(dict(cross=it["cross"], run=c["run"], ray=int(c["ray"]), kF=float(c["kF_bohr_inv"]), W=float(c["W"]),
                             v_r=float(c["v_r_au"]), v_abs_normals=float(c["v_abs_au"]), cos_alpha=float(c["cos_alpha"]),
                             vx=comps[0], vy=comps[1], vz=comps[2], n_lines_found=nfound, v_star=vs))
    W = np.array([r["W"] for r in rows])
    vr = np.array([r["v_r"] for r in rows])
    vn = np.array([r["v_abs_normals"] for r in rows])
    vs = np.array([r["v_star"] for r in rows])
    good = np.array([r["n_lines_found"] >= 2 for r in rows])
    print(f"{len(rows)} crossings, >=2 lines found: {good.sum()}")
    print(f"  v_star >= v_r (consistency) : {(vs[good] >= 0.95 * vr[good]).mean():.2f}")
    # sampled with probability ~W -> plain means; all crossings -> W-weighted means
    wts = np.ones(good.sum()) if (args.n and args.n < len(cross)) else W[good]
    for lbl, v in (("v_r", vr), ("v_abs(normals)", vn), ("v_star", vs)):
        m = np.sum(wts * v[good]) / wts.sum()
        rms = np.sqrt(np.sum(wts * v[good] ** 2) / wts.sum())
        print(f"  {lbl:15s} <v> = {m * V_AU / 1e5:.2f}e5   v_rms = {rms * V_AU / 1e5:.2f}e5 m/s (DOS-weighted)")
    ratio = vs[good] / vn[good]
    print(f"  v_star / v_abs(normals): median {np.median(ratio):.2f}, 10-90% {np.percentile(ratio, 10):.2f}-{np.percentile(ratio, 90):.2f}")
    with open(os.path.join(sd, "star_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def grad_direction(row, n, normal_cos):
    """Unit vector along the estimated Fermi velocity: sign-resolved star vector whose radial projection matches v_r
    best; fallback: radial direction n (then the line just repeats v_r)."""
    v = np.array([float(row["vx"]), float(row["vy"]), float(row["vz"])])
    best, err = None, np.inf
    for sg in itertools.product((1, -1), repeat=3):
        vv = np.array(sg) * v
        e = abs(vv @ n - float(row["v_r"])) / float(row["v_r"])
        if e < err:
            best, err = vv, e
    if best is not None and np.linalg.norm(best) > 0 and err < 0.5:
        return best / np.linalg.norm(best), err
    return n, np.nan


def make_grad(args):
    """One line through every crossing along the estimated velocity direction (12 lines per run)."""
    res, rp, x = load_source(args.sysdir)
    a = float(res["a0_bohr"])
    sd = os.path.join(args.sysdir, "star")
    meta = json.load(open(os.path.join(sd, "star.json")))
    rays = json.load(open(os.path.join(args.sysdir, meta["sub"], "fs_rays.json")))
    dirs = np.array(rays["dirs"])
    rows = list(csv.DictReader(open(os.path.join(sd, "star_results.csv"))))
    head = open(os.path.join(sd, "star_00.inp")).read().split("# k-points")[0]
    pot = re.search(r"^spc\S*\s+(\S+)", head, re.M).group(1).replace("../", "")
    runs = []
    for r0 in range(0, len(rows), 12):
        pts, items = [], []
        for r in rows[r0:r0 + 12]:
            n = dirs[int(r["ray"])]
            u, err = grad_direction(r, n, None)
            P = float(r["kF"]) * n * a / (2 * np.pi)
            pts += [P - args.delta * u, P + args.delta * u]
            items.append(dict(row=r0 + len(items), seg=len(pts) - 2, u=u.tolist(), sign_err=err))
        pts = np.array(pts)
        L = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        name = f"grad_{len(runs):02d}"
        with open(os.path.join(sd, name + ".inp"), "w") as f:
            f.write(head + "# k-points\n" + f" {int(np.ceil(L / args.dk)) + 1}\n" +
                    "".join(f" {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n" for p in pts) + "end\n")
        runs.append(dict(name=name, points=pts.tolist(), items=items))
    json.dump(dict(delta=args.delta, runs=runs), open(os.path.join(sd, "grad.json"), "w"), indent=1)
    with open(os.path.join(sd, "run_grad.sh"), "w") as f:
        f.write('#!/bin/bash\nset -e\ncd "$(dirname "$0")"\n')
        f.write('BIN=${AKAIMODBIN:-/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx}\n')
        for r in runs:
            f.write(f"$BIN < {r['name']}.inp > {r['name']}.out && mv ../{pot}_up.spc {r['name']}_up.spc\n")
        f.write(f"rm -f ../{pot}_dn.spc\n")
    os.chmod(os.path.join(sd, "run_grad.sh"), 0o755)
    print(f"{sd}: {len(rows)} gradient lines, {len(runs)} runs")


def analyse_grad(args):
    """v_u along the gradient line; |v| = max(v_r, v_u) is a lower bound that cannot overshoot (v.u <= |v|)."""
    res, rp, x = load_source(args.sysdir)
    a = float(res["a0_bohr"])
    sd = os.path.join(args.sysdir, "star")
    meta = json.load(open(os.path.join(sd, "star.json")))
    edelt = float(meta["edelt_spc"] or res["KKR_PARAMS_FINALSCF"]["edelt"])
    g = json.load(open(os.path.join(sd, "grad.json")))
    rows = list(csv.DictReader(open(os.path.join(sd, "star_results.csv"))))
    for run in g["runs"]:
        E, A, lab = read_spc_c(os.path.join(sd, run["name"] + "_up.spc"))
        labels = [(lab[i][0], tuple(run["points"][i])) for i in range(len(lab))]
        s, _ = kpath_coords(labels, A.shape[1], a)
        for it in run["items"]:
            sj = it["seg"]
            i0, i1 = labels[sj][0], labels[sj + 1][0]
            mid = 0.5 * (s[i0] + s[i1])
            cs = fit_crossings(E, A, labels, s, edelt, args.de_max, args.prom, seg_ids=[sj])
            vu = np.nan
            if cs:
                best = min(cs, key=lambda c: abs(c["kF_s"] - mid))
                if abs(best["kF_s"] - mid) < 0.25 * (s[i1] - s[i0]) and best["v_F_au"] < 3.0:
                    vu = best["v_F_au"]
            r = rows[it["row"]]
            r["v_u"] = vu
            r["sign_err"] = it["sign_err"]
            r["v_grad"] = max(float(r["v_r"]), vu) if np.isfinite(vu) else float(r["v_r"])
    W = np.array([float(r["W"]) for r in rows])
    for lbl in ("v_r", "v_abs_normals", "v_star", "v_grad"):
        v = np.array([float(r[lbl]) for r in rows])
        print(f"  {lbl:15s} <v> = {np.sum(W * v) / W.sum() * V_AU / 1e5:.2f}e5   "
              f"v_rms = {np.sqrt(np.sum(W * v ** 2) / W.sum()) * V_AU / 1e5:.2f}e5 m/s (DOS-weighted, all crossings)")
    found = np.array([np.isfinite(float(r["v_u"])) for r in rows])
    print(f"  gradient line found: {found.mean():.2f} of crossings ({W[found].sum() / W.sum():.2f} of W)")
    with open(os.path.join(sd, "grad_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def cone_dirs(u0, half_angle_deg, n_ring):
    """u0 plus n_ring unit vectors on a cone of the given half-angle around u0."""
    u0 = np.asarray(u0, float) / np.linalg.norm(u0)
    t = np.array([1.0, 0, 0]) if abs(u0[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(u0, t); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u0, e1)
    th = np.radians(half_angle_deg)
    out = [u0]
    for k in range(n_ring):
        ph = 2 * np.pi * k / n_ring
        out.append(np.cos(th) * u0 + np.sin(th) * (np.cos(ph) * e1 + np.sin(ph) * e2))
    return out


def make_cone(args):
    """6 lines per crossing (u0 = gradient-line direction from grad.json + 5 on a cone), 2 crossings per run."""
    res, rp, x = load_source(args.sysdir)
    a = float(res["a0_bohr"])
    sd = os.path.join(args.sysdir, "star")
    g = json.load(open(os.path.join(sd, "grad.json")))
    rows = list(csv.DictReader(open(os.path.join(sd, "grad_results.csv"))))
    rays = json.load(open(os.path.join(args.sysdir, json.load(open(os.path.join(sd, "star.json")))["sub"], "fs_rays.json")))
    dirs = np.array(rays["dirs"])
    u_of = {it["row"]: it["u"] for run in g["runs"] for it in run["items"]}
    head = open(os.path.join(sd, "star_00.inp")).read().split("# k-points")[0]
    pot = re.search(r"^spc\S*\s+(\S+)", head, re.M).group(1).replace("../", "")
    runs = []
    for r0 in range(0, len(rows), 2):
        pts, items = [], []
        for ri in range(r0, min(r0 + 2, len(rows))):
            r = rows[ri]
            n = dirs[int(r["ray"])]
            P = float(r["kF"]) * n * a / (2 * np.pi)
            us, segs = cone_dirs(u_of[ri], args.cone, 5), []
            for u in us:
                pts += [P - args.delta * u, P + args.delta * u]
                segs.append(len(pts) - 2)
            items.append(dict(row=ri, segs=segs, us=[u.tolist() for u in us]))
        pts = np.array(pts)
        L = np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
        name = f"cone_{len(runs):03d}"
        with open(os.path.join(sd, name + ".inp"), "w") as f:
            f.write(head + "# k-points\n" + f" {int(np.ceil(L / args.dk)) + 1}\n" +
                    "".join(f" {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}\n" for p in pts) + "end\n")
        runs.append(dict(name=name, points=pts.tolist(), items=items))
    json.dump(dict(delta=args.delta, cone_deg=args.cone, runs=runs), open(os.path.join(sd, "cone.json"), "w"), indent=1)
    with open(os.path.join(sd, "run_cone.sh"), "w") as f:
        f.write('#!/bin/bash\nset -e\ncd "$(dirname "$0")"\n')
        f.write('BIN=${AKAIMODBIN:-/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx}\n')
        for r in runs:
            f.write(f"$BIN < {r['name']}.inp > {r['name']}.out && mv ../{pot}_up.spc {r['name']}_up.spc\n")
        f.write(f"rm -f ../{pot}_dn.spc\n")
    os.chmod(os.path.join(sd, "run_cone.sh"), 0o755)
    print(f"{sd}: {len(rows)} crossings x 6 cone lines, {len(runs)} runs")


def solve_cone(U, m, tol):
    """Least squares u_i . v = m_i with iterative rejection of the worst line (wrong sheet); needs >= 3 lines
    and relative rms residual < tol. Returns (v_vec or None, n_used, rel_resid)."""
    U, m = np.array(U), np.array(m)
    keep = np.ones(len(m), bool)
    while keep.sum() >= 3:
        v, *_ = np.linalg.lstsq(U[keep], m[keep], rcond=None)
        r = U @ v - m
        nv = np.linalg.norm(v)
        rel = np.sqrt(np.mean(r[keep] ** 2)) / nv if nv > 0 else np.inf
        if rel < tol:
            return v, int(keep.sum()), float(rel)
        worst = np.argmax(np.where(keep, abs(r), -1))
        keep[worst] = False
    return None, int(keep.sum()), np.nan


def analyse_cone(args):
    res, rp, x = load_source(args.sysdir)
    a = float(res["a0_bohr"])
    sd = os.path.join(args.sysdir, "star")
    meta = json.load(open(os.path.join(sd, "star.json")))
    edelt = float(meta["edelt_spc"] or res["KKR_PARAMS_FINALSCF"]["edelt"])
    cm = json.load(open(os.path.join(sd, "cone.json")))
    rows = list(csv.DictReader(open(os.path.join(sd, "grad_results.csv"))))
    # energy HWHM Gamma of each crossing (direction independent) from the ray fit -> width check of every line
    fc = list(csv.DictReader(open(os.path.join(args.sysdir, meta["sub"], "fs_crossings.csv"))))
    gam = {(c["run"], int(c["ray"]), round(float(c["kF_bohr_inv"]), 6)): float(c["gamma_Ry"]) for c in fc}
    n_rej_width = 0
    for run in cm["runs"]:
        E, A, lab = read_spc_c(os.path.join(sd, run["name"] + "_up.spc"))
        labels = [(lab[i][0], tuple(run["points"][i])) for i in range(len(lab))]
        s, _ = kpath_coords(labels, A.shape[1], a)
        for it in run["items"]:
            U, m = [], []
            r = rows[it["row"]]
            G = gam.get((r["run"], int(r["ray"]), round(float(r["kF"]), 6)))
            for sj, u in zip(it["segs"], it["us"]):
                i0, i1 = labels[sj][0], labels[sj + 1][0]
                mid = 0.5 * (s[i0] + s[i1])
                cs = fit_crossings(E, A, labels, s, edelt, args.de_max, args.prom, seg_ids=[sj])
                if cs:
                    b = min(cs, key=lambda c: abs(c["kF_s"] - mid))
                    if abs(b["kF_s"] - mid) < 0.25 * (s[i1] - s[i0]) and b["v_F_au"] < 3.0:
                        # width check: Gamma = (v.u) * HWHM_u must hold for a clean single-band line
                        n_all = int(np.sum(abs(E) <= args.de_max))
                        ok_w = G is None or abs(b["v_F_au"] * b["hwhm_k_bohr_inv"] / G - 1) < args.width_tol
                        if ok_w and b["n_energies"] >= 0.8 * n_all and b["hwhm_k_bohr_inv"] < 0.25 * (s[i1] - s[i0]):
                            U.append(u)
                            m.append(b["dkdE_sign"] * b["v_F_au"])
                        else:
                            n_rej_width += 1
            v, nu, rel = solve_cone(U, m, args.tol) if len(m) >= 3 else (None, len(m), np.nan)
            lower = max([float(r["v_grad"])] + [abs(q) for q in m])
            r.update(cone_lines=len(m), cone_used=nu, cone_rel_resid=rel,
                     v_cone_lower=lower, v_cone=float(np.linalg.norm(v)) if v is not None else np.nan)
            # final: LSQ vector if it is consistent with the lower bound, else the lower bound
            r["v_final"] = r["v_cone"] if (v is not None and r["v_cone"] >= 0.95 * lower) else lower
    print(f"  lines rejected by width/tracking checks: {n_rej_width}")
    W = np.array([float(r["W"]) for r in rows])
    ok = np.array([np.isfinite(float(r["v_cone"])) for r in rows])
    print(f"  LSQ vector accepted: {ok.mean():.2f} of crossings ({W[ok].sum() / W.sum():.2f} of W)")
    for lbl in ("v_r", "v_abs_normals", "v_grad", "v_cone_lower", "v_final"):
        v = np.array([float(r[lbl]) for r in rows])
        print(f"  {lbl:15s} <v> = {np.sum(W * v) / W.sum() * V_AU / 1e5:.2f}e5   "
              f"v_rms = {np.sqrt(np.sum(W * v ** 2) / W.sum()) * V_AU / 1e5:.2f}e5 m/s (DOS-weighted)")
    with open(os.path.join(sd, "cone_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "analyse", "make_grad", "analyse_grad", "make_cone", "analyse_cone"])
    ap.add_argument("sysdir")
    ap.add_argument("--sub", default="fs_e003")
    ap.add_argument("--n", type=int, default=48, help="number of crossings (DOS-weighted sample); 0 = all")
    ap.add_argument("--delta", type=float, default=0.06, help="half-length of the Cartesian lines [2pi/a]")
    ap.add_argument("--dk", type=float, default=0.002, help="k spacing [2pi/a]")
    ap.add_argument("--de_max", type=float, default=0.008)
    ap.add_argument("--prom", type=float, default=0.05)
    ap.add_argument("--cone", type=float, default=30.0, help="cone half-angle [deg] (make_cone)")
    ap.add_argument("--width_tol", type=float, default=0.3, help="max |v.u * HWHM_u / Gamma - 1| for a cone line")
    ap.add_argument("--tol", type=float, default=0.15, help="max relative rms residual of the cone LSQ")
    args = ap.parse_args()
    {"make": make, "analyse": analyse, "make_grad": make_grad, "analyse_grad": analyse_grad,
     "make_cone": make_cone, "analyse_cone": analyse_cone}[args.mode](args)


if __name__ == "__main__":
    main()
