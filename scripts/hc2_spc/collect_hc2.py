"""
Collect the Hc2 baseline runs (run_hc2_pipeline.sh) into one table — WORKSTATION, after rsync from Topola.

Reads <runs>/<name>/{source.json, fs/fs_summary.json, star/cone_results.csv} and writes <runs>/hc2_table.csv:
  D (cone |v|), rho0 with site and Lloyd N(E_F), bare slope 4 Phi0 kB/(pi^2 hbar D), slope x (1+lambda)
  (electron-phonon renormalisation, D_eff = D/(1+lambda); lambda = baseline KKR lambda),
  Hc2_orb(0) = 0.693 Tc slope, alpha_M = sqrt2 Hc2_orb/H_P (H_P = 1.86 Tc), Hc2(0) Maki = Hc2_orb/sqrt(1+alpha^2),
  for Tc = Tc_exp (median of literature rows, if any) and Tc = Tc_mu0.1 (KKR).
  Quality: frac_W_lsq = DOS weight with accepted cone LSQ (rest = lower bound of |v| -> D low, slope high).
The output stays in the Hc2 tree; nothing is written into results/.

    rsync -av topola:/lu/topola/home/rtop/heaml/hc2_runs/ hc2_runs/
    python scripts/hc2_spc/collect_hc2.py hc2_runs/baseline.sra.ew0.6.bz10-10-20.gl
"""
import argparse
import csv
import json
import os

import numpy as np

HBAR, KB, E_CH = 1.054571817e-34, 1.380649e-23, 1.602176634e-19
PHI0 = 2.067833848e-15
BOHR, RY_J = 5.29177210903e-11, 2.1798723611e-18


def hc2_numbers(slope, tc):
    if not tc:
        return dict(Hc2_orb0_T=np.nan, alpha_M=np.nan, Hc2_Maki_T=np.nan)
    h_orb = 0.693 * tc * slope
    am = np.sqrt(2) * h_orb / (1.86 * tc)
    return dict(Hc2_orb0_T=h_orb, alpha_M=am, Hc2_Maki_T=h_orb / np.sqrt(1 + am ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows, missing = [], []
    for name in sorted(os.listdir(args.runs)):
        d = os.path.join(args.runs, name)
        fsj = os.path.join(d, "fs", "fs_summary.json")
        if not os.path.isdir(d) or not os.path.isfile(os.path.join(d, "source.json")):
            continue
        status = open(os.path.join(d, "hc2_status.txt")).read().strip() if \
            os.path.isfile(os.path.join(d, "hc2_status.txt")) else "NOT_STARTED"
        if status != "DONE" or not os.path.isfile(fsj):
            missing.append((name, status))
            continue
        src = json.load(open(os.path.join(d, "source.json")))
        res, rp = src["results"], src["run_params"]
        s = json.load(open(fsj))
        D = s["hc2"]["D_m2_s"]
        v_at = float(res["a0_bohr"]) ** 3 / 2 * BOHR ** 3                 # bcc, m^3 per atom
        n_v = lambda n_spin: 2 * n_spin / RY_J / v_at                     # noqa: E731  states / J / m^3
        rho = lambda n_spin: 1e8 / (E_CH ** 2 * n_v(n_spin) * D)          # noqa: E731  micro-Ohm cm
        slope = 4 * PHI0 * KB / (np.pi ** 2 * HBAR * D)
        lam = float(res["lambda"])
        # DOS weight with accepted cone LSQ
        frac_lsq = np.nan
        cpath = os.path.join(d, "star", "cone_results.csv")
        if os.path.isfile(cpath):
            c = list(csv.DictReader(open(cpath)))
            W = np.array([float(r["W"]) for r in c])
            ok = np.array([r["v_cone"] not in ("", "nan") and r["v_final"] == r["v_cone"] for r in c])
            frac_lsq = float(W[ok].sum() / W.sum()) if W.sum() > 0 else np.nan
        comp = " ".join(f"{e}{100 * c / sum(rp['concentrations']):.1f}" for e, c in
                        zip(rp["element_labels"], rp["concentrations"]) if c > 0)
        row = dict(name=name, composition=comp, a0_bohr=res["a0_bohr"], lambda_kkr=lam,
                   Tc_mu01_K=res.get("Tc_mu0.1"), Tc_exp_K=res.get("Tc_exp"),
                   N_site_states_Ry_spin=s["N_EF_KKR_site_states_Ry_atom_spin"],
                   N_Lloyd_states_Ry_spin=s["N_EF_Lloyd_states_Ry_atom_spin"],
                   FS_coverage_Lloyd=s["FS_coverage_Lloyd"], Aint_over_Lloyd=s["Aint_over_Lloyd"],
                   n_crossings=s["n_crossings"], n_rejected=s["n_rejected"], frac_W_lsq=frac_lsq,
                   v_rms_m_s=s["v_rms_m_s"], hbar_omega_p_eV=s["hbar_omega_p_eV"], l_A=s["l_DOS_avg_A"],
                   D_m2_s=D, rho0_site_uOhm_cm=rho(s["N_EF_KKR_site_states_Ry_atom_spin"]),
                   rho0_Lloyd_uOhm_cm=rho(s["N_EF_Lloyd_states_Ry_atom_spin"]),
                   slope_bare_T_K=slope, slope_1plam_T_K=slope * (1 + lam))
        for tag, tc in (("exp", res.get("Tc_exp")), ("mu01", res.get("Tc_mu0.1"))):
            for k, v in hc2_numbers(slope * (1 + lam), tc).items():
                row[f"{k}_Tc{tag}"] = v
        rows.append(row)

    out = args.out or os.path.join(args.runs, "hc2_table.csv")
    if rows:
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    print(f"{out}: {len(rows)} systems DONE; not done: {len(missing)}"
          + ("" if not missing else " — " + ", ".join(f"{n} ({st})" for n, st in missing[:20])
             + (" ..." if len(missing) > 20 else "")))


if __name__ == "__main__":
    main()
