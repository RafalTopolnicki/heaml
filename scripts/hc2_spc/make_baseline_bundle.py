"""
Hc2 (Maki/WHH) on the baseline set — step 0, WORKSTATION ONLY (needs results/, which is not in git).

Collects from the baseline KKR results (results/comparison.sra.ew0.6.bz10-10-20.gl/<name>/{results,run_params}.json)
only the fields the Hc2 scripts need, and the experimental Tc from results/Tc_literature.xlsx (composition match as in
scripts/baseline_corrections.py), into ONE small json that is committed to git and read on Topola by
make_spc_inputs.py --bundle.  Nothing is written into results/.

    python scripts/hc2_spc/make_baseline_bundle.py            # -> scripts/hc2_spc/baseline_bundle.json
    python scripts/hc2_spc/make_baseline_bundle.py --include_pure

Pure elements are skipped by default: without chemical disorder A(k,E) is limited by edelt only and the dirty-limit
formula does not apply (Nb pilot, DOCS/ROZMOWA_2026-11-03_Hc2.md §2).
"""
import argparse
import json
import os

import numpy as np

BASELINE = "results/comparison.sra.ew0.6.bz10-10-20.gl"
XLSX = "results/Tc_literature.xlsx"
XLSX_ELEMENTS = ["Sc", "Ti", "Y", "Zr", "Nb", "Mo", "Ru", "La", "Hf", "Ta", "W", "Re", "V"]   # xlsx columns B..N
COL_NAME, COL_EL0, COL_TC_EXP = 1, 2, 19
RES_KEYS = ["name", "a0_bohr", "KKR_PARAMS_FINALSCF", "lambda", "Tc_mu0.1", "Tc_mu0.2", "thetaDB_K",
            "mixture_debye_temperature", "eta_mix"]


def load_tc_exp():
    """[(fractions over XLSX_ELEMENTS, Tc_exp, xlsx name)] ; several rows may share a composition."""
    try:
        import openpyxl
    except ImportError:
        print("openpyxl missing: Tc_exp not added")
        return []
    ws = openpyxl.load_workbook(XLSX, data_only=True)["Tc_literature"]
    num = lambda c: float(c.value) if isinstance(c.value, (int, float)) else np.nan   # noqa: E731
    rows = []
    for r in range(4, ws.max_row + 1):
        name, tc = ws.cell(r, COL_NAME).value, num(ws.cell(r, COL_TC_EXP))
        x = np.nan_to_num([num(ws.cell(r, COL_EL0 + i)) for i in range(len(XLSX_ELEMENTS))])
        if not name or np.isnan(tc) or x.sum() == 0:
            continue
        rows.append((x / x.sum(), tc, str(name)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", default=BASELINE)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "baseline_bundle.json"))
    ap.add_argument("--include_pure", action="store_true")
    args = ap.parse_args()

    tc_rows = load_tc_exp()
    out, skipped = {}, []
    for d in sorted(os.listdir(args.baseline)):
        pj, pr = (os.path.join(args.baseline, d, f) for f in ("results.json", "run_params.json"))
        if not (os.path.isfile(pj) and os.path.isfile(pr)):
            continue
        res, rp = json.load(open(pj)), json.load(open(pr))
        if not all(k in res for k in ("a0_bohr", "KKR_PARAMS_FINALSCF", "lambda")):
            skipped.append((d, "incomplete results.json"))
            continue
        labels, conc = rp["element_labels"], [float(c) for c in rp["concentrations"]]
        nz = [e for e, c in zip(labels, conc) if c > 0]
        if len(nz) < 2 and not args.include_pure:
            skipped.append((d, "pure element"))
            continue
        r = {k: res[k] for k in RES_KEYS if k in res}
        r.update({f"{e}_Ntot": res.get(f"{e}_Ntot") for e in labels})
        r.update({e: res.get(e) for e in labels})
        # experimental Tc: all xlsx rows with the same composition (max |dx| <= 2e-3)
        x = np.array([dict(zip(labels, conc)).get(e, 0.0) for e in XLSX_ELEMENTS])
        x = x / x.sum()
        match = [(tc, nm) for xx, tc, nm in tc_rows if np.abs(xx - x).max() <= 2e-3]
        r["Tc_exp_all"] = [m[0] for m in match]
        r["Tc_exp_names"] = [m[1] for m in match]
        r["Tc_exp"] = float(np.median(r["Tc_exp_all"])) if match else None
        out[d] = {"src": os.path.join(args.baseline, d), "results": r,
                  "run_params": {"element_labels": labels, "concentrations": conc}}
    json.dump(out, open(args.out, "w"), indent=1)
    n_tc = sum(v["results"]["Tc_exp"] is not None for v in out.values())
    print(f"{args.out}: {len(out)} systems ({n_tc} with Tc_exp); skipped {len(skipped)}: "
          + ", ".join(f"{d} ({why})" for d, why in skipped))


if __name__ == "__main__":
    main()
