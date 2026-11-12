"""
Generate AkaiKKR inputs for the Bloch spectral function A(k,E) (go=spc) — Hc2 pilot (DOCS/ROZMOWA_2026-11-03_Hc2.md).

Per system it writes  hc2_spc/<name>/ :
    scf.inp       SCF with the production finalscf parameters (sra, pbe, ew 0.6, bzqlty 20, mxl 3, edelt 1e-3, rmt 0)
                  at the equilibrium a0 taken from the baseline GL results.json
    spc_wide.inp  A(k,E) along G-H-N-P-G-N, ewidth 1.0 Ry (E_F-0.8 .. E_F+0.2 Ry, step 5 mRy) — band overview
    spc_ef.inp    same path, ewidth 0.2 Ry (E_F-0.16 .. E_F+0.04 Ry, step 1 mRy) — k_F, v_F and widths near E_F
    spc_fine.inp  same path, --nk_fine points (default 1200), ewidth 0.05 Ry (E_F-0.04 .. E_F+0.01 Ry, step 0.25 mRy)
                  — input for scripts/hc2_spc/analyze_spc.py
    run.sh        runs scf -> spc_wide -> spc_ef
    run_fine.sh   runs only spc_fine (needs the converged potential from run.sh)
    spc output: <pot>_up.spc, renamed to spc_<tag>_up.spc; AkaiKKR reads 'spc31' as 'spc' -> format (c):
                matrix nE x nk (rows = energies), header with E-E_F range [Ry] and symmetry-point indices

Energy mesh in spc: msex=201 points, ~80 % of ewidth below E_F (specx.f / Run_AKAI-KKR.pdf §5).
'spc31' = k-input format (c) (total number of k-points + list of symmetry points), output format (a) (x y z columns).
k in units of 2pi/a of the conventional cubic cell; bcc: H=(0,1,0), N=(1/2,1/2,0), P=(1/2,1/2,1/2).

Usage:  python scripts/hc2_spc/make_spc_inputs.py [--outdir hc2_spc] [--nk 300]          # pilot systems (SYSTEMS)
        python scripts/hc2_spc/make_spc_inputs.py --bundle scripts/hc2_spc/baseline_bundle.json \
               --outdir hc2_runs/baseline.sra.ew0.6.bz10-10-20.gl [--only A B]          # baseline set (Topola)
With --bundle every system dir gets a self-contained source.json (results + run_params subset), so the analysis
scripts do not need results/ (not in git).  --outdir inside results/, *.ratios* is refused.
"""
import argparse
import json
import os

BASELINE = "results/comparison.sra.ew0.6.bz10-10-20.gl"
BINARIES = "validation.ratios/binaries.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl"
ZERO8 = "Zr0.0000Hf0.0000Ta0.0000Mo0.0000W0.0000Y0.0000V0.0000Sc0.0000"

SYSTEMS = {
    "Nb": {"elements": ["Nb"], "conc": [100.0]},
    "Ta34Nb33Hf8Zr14Ti11": {"elements": ["Ta", "Nb", "Hf", "Zr", "Ti"], "conc": [34.0, 33.0, 8.0, 14.0, 11.0]},
    # Nb-Ti validation (Muller PhD 1989): a0, lambda, N(E_F) from the GL binary lines
    "Nb50Ti50": {"elements": ["Nb", "Ti"], "conc": [50.0, 50.0], "src": BINARIES + "/Ti0.5000Nb0.5000" + ZERO8},
    "Nb40Ti60": {"elements": ["Nb", "Ti"], "conc": [40.0, 60.0], "src": BINARIES + "/Ti0.6000Nb0.4000" + ZERO8},
}

KPATH_BCC = [  # G-H-N-P-G-N  (as in AkaiKKR in/fe example)
    ("G", "0 0 0"),
    ("H", "0 1 0"),
    ("N", "1/2 1/2 0"),
    ("P", "1/2 1/2 1/2"),
    ("G", "0 0 0"),
    ("N", "1/2 1/2 0"),
]

C = "c-------------------------\n"
Z = {"Sc": 21, "Ti": 22, "V": 23, "Y": 39, "Zr": 40, "Nb": 41, "Mo": 42, "Ru": 44, "La": 57, "Hf": 72, "Ta": 73,
     "W": 74, "Re": 75}  # AkaiKKR needs anclr (Z)
# Hc2 runs must never land next to the Tc results (main result of the project)
FORBIDDEN_OUT = ("results", "validation.ratios", "opt.ratios", "opt.ratios.gl", "random.ratios", "random.ratios.gl")


def block(go, pot, a0, ew, record, outtyp, p, elements, conc):
    s = f"c--- elements {elements}\n{C}"
    s += f"{go} {pot}\n{C}"
    s += "c   brvtyp     a        c/a   b/a   alpha   beta   gamma\n"
    s += f"    bcc    {a0}  , , , , , , \n{C}"
    s += "c   edelt    ewidth    reltyp   sdftyp   magtyp   record\n"
    s += f"    {p['edelt']}    {ew}    {p['rel']}   {p['xc']}      {p['magtype']}      {record}\n{C}"
    s += "c   outtyp    bzqlty   maxitr   pmix\n"
    s += f"    {outtyp}     {int(p['bzqlty'])}    500 {p['pmix']}\n{C}"
    s += "c  ntyp\n   1\n" + C
    s += "c   type    ncmp    rmt    field   mxl  anclr   conc\n"
    s += f"   X   {len(elements)}  0  0.0   {p['mxl']}\n"
    for el, c in zip(elements, conc):
        s += f"       {Z[el]}   {c / 100.0:.6f}\n"
    s += C + "c natm\n   1\n" + C
    s += "c   atmicx                        atmtyp\n0a         0b         0c       X\n" + C
    return s


def kpath(nk):
    # NO "end" line here: readk stops at the first alphabetic token -> nk3=0 (tested 2026-11-03)
    s = "# k-points for spc: total number, then symmetry points (units 2pi/a)\n"
    s += f" {nk}\n"
    for lbl, k in KPATH_BCC:
        s += f" {k}\n"
    return s + "end\n"  # terminates the job list (otherwise EOF crash in cnsole.f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="hc2_spc")
    ap.add_argument("--nk", type=int, default=300)
    ap.add_argument("--nk_fine", type=int, default=1200)
    ap.add_argument("--only", nargs="*", help="generate only these systems")
    ap.add_argument("--bundle", default=None, help="json from make_baseline_bundle.py (instead of SYSTEMS)")
    ap.add_argument("--skip_existing", action="store_true", help="do not touch system dirs that already exist")
    args = ap.parse_args()

    top = os.path.normpath(args.outdir).split(os.sep)[0]
    if top in FORBIDDEN_OUT or os.path.normpath(args.outdir).startswith(("results", "validation.ratios")):
        raise SystemExit(f"refusing --outdir {args.outdir}: Hc2 runs must stay out of the Tc result trees {FORBIDDEN_OUT}")

    if args.bundle:
        bundle = json.load(open(args.bundle))
        systems = {}
        for name, b in bundle.items():
            rp = b["run_params"]
            el = [e for e, c in zip(rp["element_labels"], rp["concentrations"]) if float(c) > 0]
            c = [float(c) for c in rp["concentrations"] if float(c) > 0]
            tot = sum(c)
            systems[name] = {"elements": el, "conc": [100.0 * ci / tot for ci in c], "embedded": b}
    else:
        systems = SYSTEMS

    for name, sysd in systems.items():
        if args.only and name not in args.only:
            continue
        d = os.path.join(args.outdir, name)
        if args.skip_existing and os.path.isdir(d):
            continue
        if "embedded" in sysd:
            res = sysd["embedded"]["results"]
        else:
            src = sysd.get("src", os.path.join(BASELINE, name))
            res = json.load(open(os.path.join(src, "results.json")))
        p = res["KKR_PARAMS_FINALSCF"]
        a0 = f"{res['a0_bohr']:.6f}"
        pot = f"{name.lower()}.pot"
        os.makedirs(d, exist_ok=True)
        if "embedded" in sysd:                                                # self-contained (Topola)
            json.dump(sysd["embedded"], open(os.path.join(d, "source.json"), "w"), indent=1)
        else:
            json.dump({"src": src}, open(os.path.join(d, "source.json"), "w"))   # read by analyze_spc / fs_average
        el, c = sysd["elements"], sysd["conc"]

        with open(os.path.join(d, "scf.inp"), "w") as f:
            f.write(block("go", pot, a0, p["ew"], "init", "update", p, el, c))
        # spc: re-uses the converged potential (record 2nd, outtyp quit); ewidth sets the real-axis window
        with open(os.path.join(d, "spc_wide.inp"), "w") as f:
            f.write(block("spc31", pot, a0, 1.0, "2nd", "quit", p, el, c) + kpath(args.nk))
        with open(os.path.join(d, "spc_ef.inp"), "w") as f:
            f.write(block("spc31", pot, a0, 0.2, "2nd", "quit", p, el, c) + kpath(args.nk))
        with open(os.path.join(d, "spc_fine.inp"), "w") as f:
            f.write(block("spc31", pot, a0, 0.05, "2nd", "quit", p, el, c) + kpath(args.nk_fine))
        with open(os.path.join(d, "run_fine.sh"), "w") as f:
            f.write(f"""#!/bin/bash
# {name}: fine A(k,E) near E_F for analyze_spc.py (uses the potential from run.sh)
set -e
cd "$(dirname "$0")"
BIN=${{AKAIMODBIN:-/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx}}
[ -f {pot} ] || {{ echo "missing {pot}: run ./run.sh first"; exit 1; }}
$BIN < spc_fine.inp > spc_fine.out && mv {pot}_up.spc spc_fine_up.spc
""")
        os.chmod(os.path.join(d, "run_fine.sh"), 0o755)

        with open(os.path.join(d, "run.sh"), "w") as f:
            f.write(f"""#!/bin/bash
# {name}: SCF (production finalscf params, a0={a0} bohr) -> A(k,E) wide -> A(k,E) near E_F
set -e
cd "$(dirname "$0")"
BIN=${{AKAIMODBIN:-/home/rafal/WORK/HEA/RKKY/cpa2002v010.potential2026/specx}}
$BIN < scf.inp > scf.out
if grep -q "itr=499" scf.out || grep -qi "no convergence" scf.out; then echo "WARNING: SCF not converged (scf.out)"; fi
$BIN < spc_wide.inp > spc_wide.out && mv {name.lower()}.pot_up.spc spc_wide_up.spc
$BIN < spc_ef.inp   > spc_ef.out   && mv {name.lower()}.pot_up.spc spc_ef_up.spc
""")
        os.chmod(os.path.join(d, "run.sh"), 0o755)
        print(f"{d}: a0={a0} bohr, pot={pot}")


if __name__ == "__main__":
    main()
