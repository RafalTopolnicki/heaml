import argparse
import json
import os
import shutil
from src.elements import HEAClass
from src.utils import save_dict_to_json, gzip_file, pack_kkr_files, read_kkr_file
from src.consts import KKR_PARAMS_LATTICE, KKR_PARAMS_DEBYE, KKR_PARAMS_FINALSCF
from lattice import run_kkr_eos
from finalscf import run_kkr_finalscf
from debye import run_kkr_elastic_debye
from macmillan_cutoff import run_mcmillan_cutoff_sweep
from src.process_kkr import process_kkr


# =========================
# --reuse_from: take lattice (+ KKR Debye) from an earlier run, recompute only the final SCF
# =========================
def _read_lattice_summary(text):
    # lattice_summary.txt written by lattice.run_kkr_eos: "a0 = ...", "E0 = ...", "B0 = ..."
    vals = {}
    for line in text.splitlines():
        if '=' in line:
            k, v = line.split('=', 1)
            vals[k.strip()] = float(v)
    return {"lattice_constant_bohr": vals["a0"],
            "minimum_energy_ev": vals["E0"],
            "bulk_modulus_gpa": vals["B0"]}


def _same_params(old, new, skip=()):
    old = dict(old)
    # older runs stored the Debye source as use_mixture_debye (False = KKR distortions) instead of debye_mode
    if 'use_mixture_debye' in old:
        old.setdefault('debye_mode', 'mix' if old.pop('use_mixture_debye') else 'kkr')
    keys = (set(old) | set(new)) - set(skip)
    return {k: (old.get(k), new.get(k)) for k in keys if old.get(k) != new.get(k)}


def prepare_reuse(reuse_from, workdir, run_params, reuse_debye):
    """Check that `reuse_from` was computed with the same composition and lattice (and Debye) parameters,
    bring its lattice/ (and debye/) into `workdir` and move an existing workdir/finalscf to finalscf.copy,
    so the directory layout after the rerun is the same as after a full run.
    Must be called BEFORE run_params.json is written (in-place reuse overwrites it)."""
    old_params = json.load(open(os.path.join(reuse_from, 'run_params.json')))

    if old_params['element_labels'] != run_params['element_labels'] or \
            [float(x) for x in old_params['concentrations']] != [float(x) for x in run_params['concentrations']]:
        raise ValueError(f"--reuse_from {reuse_from}: different composition "
                         f"({old_params['element_labels']} {old_params['concentrations']})")
    steps = ['KKR_PARAMS_LATTICE'] + (['KKR_PARAMS_DEBYE'] if reuse_debye else [])
    for step in steps:
        diff = _same_params(old_params[step], run_params[step])
        if diff:
            raise ValueError(f"--reuse_from {reuse_from}: {step} differs (old, new): {diff}")

    subdirs = [run_params['KKR_PARAMS_LATTICE']['subdir']]
    if reuse_debye:
        subdirs.append(run_params['KKR_PARAMS_DEBYE']['subdir'])
    in_place = os.path.realpath(reuse_from) == os.path.realpath(workdir)
    # a packed run (kkr_raw.tar) has no lattice/, debye/ on disk: take the summaries from its archive
    lattice_txt = read_kkr_file(reuse_from, f"{subdirs[0]}/lattice_summary.txt")
    debye_txt = read_kkr_file(reuse_from, f"{subdirs[1]}/debye_summary.json") if reuse_debye else None
    for sub in subdirs:
        if not in_place and os.path.isdir(os.path.join(reuse_from, sub)):
            # .pot files are SCF restart data, not needed for anything downstream
            shutil.copytree(os.path.join(reuse_from, sub), os.path.join(workdir, sub), dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns('*.pot', '*.pot.info'))

    # keep the previous final SCF; if finalscf.copy already exists it holds the original one
    # (earlier reuse attempt), so the current finalscf is just a leftover of that attempt
    fs = os.path.join(workdir, run_params['KKR_PARAMS_FINALSCF']['subdir'])
    if os.path.isdir(fs):
        if os.path.exists(fs + '.copy'):
            shutil.rmtree(fs)
        else:
            os.rename(fs, fs + '.copy')
    if in_place:
        for name in ['run_params.json', 'results.json']:
            f = os.path.join(workdir, name)
            if os.path.exists(f) and not os.path.exists(f + '.copy'):
                shutil.copy2(f, f + '.copy')

    if not in_place:
        for sub, name, txt in [(subdirs[0], 'lattice_summary.txt', lattice_txt)] + \
                ([(subdirs[1], 'debye_summary.json', debye_txt)] if reuse_debye else []):
            os.makedirs(os.path.join(workdir, sub), exist_ok=True)
            with open(os.path.join(workdir, sub, name), 'w') as f:
                f.write(txt)
    eof_output = _read_lattice_summary(lattice_txt)
    debye_output = json.loads(debye_txt) if reuse_debye else None
    print(f"Reusing lattice{' + Debye' if reuse_debye else ''} from {reuse_from}: {eof_output}")
    return eof_output, debye_output


# =========================
# ⭐ MASTER FUNCTION
# =========================
def run_one_hea(**kwargs):
    workdir = kwargs.get("workdir", ".")
    overwrite_params = kwargs.get("overwrite_params", False)
    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()

    hea = HEAClass(labels=kwargs['element_labels'],
                   concentrations=kwargs['concentrations'])
    hea_configuration = {'elements': hea.return_atomic_numbers(),
                         'concentrations': hea.concentrations,
                         'density': hea.density}
    # overwrite def params
    KKR_PARAMS_LATTICE_PARAMS = KKR_PARAMS_LATTICE.copy()
    KKR_PARAMS_DEBYE_PARAMS = KKR_PARAMS_DEBYE.copy()
    KKR_PARAMS_FINALSCF_PARAMS = KKR_PARAMS_FINALSCF.copy()
    for param in ["ew", "xc", "rel", "bzqlty", "pmix", "magtype", "edelt", "mxl"]:
        if kwargs.get(param) is not None:
            KKR_PARAMS_LATTICE_PARAMS[param] = kwargs[param]
            KKR_PARAMS_DEBYE_PARAMS[param] = kwargs[param]
            KKR_PARAMS_FINALSCF_PARAMS[param] = kwargs[param]
    # final SCF (eta, N(EF)) only; production default is 20 while lattice/Debye use 10
    if kwargs.get("bzqlty_finalscf") is not None:
        KKR_PARAMS_FINALSCF_PARAMS["bzqlty"] = kwargs["bzqlty_finalscf"]

    debye_mode = kwargs.get('debye_mode', KKR_PARAMS_DEBYE_PARAMS.get('debye_mode', 'kkr'))
    KKR_PARAMS_DEBYE_PARAMS['debye_mode'] = debye_mode
    run_params = {
        'element_labels': kwargs['element_labels'],
        'concentrations': list(kwargs['concentrations']),
        'density': hea.density,
        'mixture_lattice': hea.mixture_lattice,
        'mixture_bulk_modulus': hea.mixture_bulk_modulus,
        'mixture_debye_temperature': hea.mixture_debye_temperature,
        'mixture_mass': hea.mass,
        'debye_mode': debye_mode,
        'KKR_PARAMS_LATTICE': KKR_PARAMS_LATTICE_PARAMS,
        'KKR_PARAMS_DEBYE': KKR_PARAMS_DEBYE_PARAMS,
        'KKR_PARAMS_FINALSCF': KKR_PARAMS_FINALSCF_PARAMS
                }
    reuse_from = kwargs.get('reuse_from')
    reuse_debye = debye_mode == 'kkr'  # mix / mjs_opt θ_D needs no KKR run, it is recomputed below
    reused_debye = None
    if reuse_from:
        eof_output, reused_debye = prepare_reuse(reuse_from, workdir, run_params, reuse_debye)
    save_dict_to_json(run_params, os.path.join(workdir, 'run_params.json'))
    if not reuse_from:
        # optimize lattice
        lattice_params = KKR_PARAMS_LATTICE_PARAMS
        lattice_params.update(hea_configuration)
        lattice_params['min_lattice'] = hea.mixture_lattice * lattice_params['min_lattice_prop']
        lattice_params['max_lattice'] = hea.mixture_lattice * lattice_params['max_lattice_prop']
        latt_step = (lattice_params['max_lattice'] - lattice_params['min_lattice'])/lattice_params['lattice_steps']
        lattice_params['step'] = latt_step
        lattice_params['workdir'] = os.path.join(workdir, lattice_params['subdir'])
        eof_output = run_kkr_eos(**lattice_params)
        print('Lattice computations DONE')
        print(eof_output)
        os.chdir(cwd)
    if kwargs['task'] == 'lattice':
        return

    # run final scf
    scf_params = KKR_PARAMS_FINALSCF_PARAMS
    scf_params.update(hea_configuration)
    scf_params['a0'] = eof_output['lattice_constant_bohr']
    scf_params['workdir'] = os.path.join(workdir, scf_params['subdir'])
    scf_params['sym'] = 'bcc'
    scf_output = run_kkr_finalscf(**scf_params)
    print('SCF computations DONE')
    print(scf_output)
    os.chdir(cwd)

    # run Debye
    if debye_mode in ('mix', 'mjs_opt'):
        debye_workdir = os.path.join(workdir, KKR_PARAMS_DEBYE_PARAMS["subdir"])
        os.makedirs(debye_workdir, exist_ok=True)
        if debye_mode == 'mix':
            theta = hea.mixture_debye_temperature
        else:  # mjs_opt
            import math
            _BOHR_M = 5.29177e-11
            _HBAR_KB = 7.63823e-12
            _C_MJS_OPT = 0.778
            _a0_m = eof_output['lattice_constant_bohr'] * _BOHR_M
            _Va = _a0_m**3 / 2.0
            _v_B = math.sqrt(eof_output['bulk_modulus_gpa'] * 1e9 / hea.density)
            theta = _C_MJS_OPT * _HBAR_KB * (6.0 * math.pi**2 / _Va)**(1.0/3.0) * _v_B
        debye_output = {"thetaDB_K": theta}
        save_dict_to_json(debye_output, os.path.join(debye_workdir, "debye_summary.json"))
        print(f"Debye skipped (debye_mode={debye_mode}): thetaDB_K={theta:.2f} K")
    elif reused_debye is not None:
        debye_output = reused_debye
        print(f"Debye reused: thetaDB_K={debye_output.get('thetaDB_K')}")
    else:
        debye_params = KKR_PARAMS_DEBYE_PARAMS.copy()
        debye_params.update(hea_configuration)

        debye_params["a0"] = eof_output["lattice_constant_bohr"]
        debye_params["B0"] = eof_output["bulk_modulus_gpa"]
        debye_params["workdir"] = os.path.join(workdir, debye_params["subdir"])

        # debye.py expects "deltas", not only "delta"
        if "deltas" not in debye_params:
            debye_params["deltas"] = [debye_params.get("delta", 0.005)]

        # You want one value only
        debye_params["deltas"] = [0.005]

        # Recommended default for one delta
        debye_params["fit_mode"] = debye_params.get("fit_mode", "linear")

        # Keep standard two-sided +/- delta
        debye_params["one_sided"] = True

        # Use the original monoclinic C44 mode unless you explicitly want simple_shear
        debye_params["c44_mode"] = debye_params.get("c44_mode", "monoclinic")

        debye_output = run_kkr_elastic_debye(**debye_params)

        print("Debye computations DONE")
        print(debye_output)
        os.chdir(cwd)

    # run McMillan-Hopfield with last-node cutoff (after Debye so theta_D is available)
    theta_d_for_log = debye_output.get('thetaDB_K')
    run_mcmillan_cutoff_sweep(
        workdir=scf_params['workdir'],
        run_kkr=False,
        mixture_mass=run_params['mixture_mass'],
        theta_d=theta_d_for_log,
        cutoff_mode=KKR_PARAMS_FINALSCF_PARAMS.get('mcmillan_cutoff_mode', 'max'),
        dos_normalization=KKR_PARAMS_FINALSCF_PARAMS.get('mcmillan_dos_normalization', 'site'),
        elements=kwargs['element_labels'],
        save_plots=kwargs.get('save_plots', False),
    )

    # fort.50/fort.51 (~3.7 MB) are needed only by the McMillan step above; gzip rather than delete
    # so eta can still be recomputed (other cutoff, macmillan_opw.py) after `gunzip fort.51.gz`.
    # Only reached when the sweep succeeded; on failure the files stay as they are for debugging.
    if kwargs.get('compress', True):
        for name in ['fort.50', 'fort.51', f"{scf_params['output']}_{scf_params['a0']:.6f}.out"]:
            gzip_file(os.path.join(scf_params['workdir'], name))
    # file quota on the cluster: everything except the summaries read below goes into <workdir>/kkr_raw.tar
    # (tar -xf kkr_raw.tar restores lattice/, debye/, finalscf/); only reached when the steps above succeeded
    if kwargs.get('pack', True):
        pack_kkr_files(workdir)

    # make all final computations
    allresults = process_kkr(path=workdir, dirname='', allow_missing_theta=True)
    save_dict_to_json(allresults, os.path.join(workdir, "results.json"))



# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels", nargs="+", required=True)
    parser.add_argument("--concentrations", nargs="+", required=True)
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--no-gzip", action="store_true")
    parser.add_argument("--no-pack", action="store_true",
                        help="keep lattice/, debye/, finalscf/ as files instead of packing them into kkr_raw.tar")
    parser.add_argument("--overwrite_params", action="store_true")


    parser.add_argument("--ew", type=float, default=KKR_PARAMS_LATTICE['ew'])
    parser.add_argument("--xc", type=str, default=KKR_PARAMS_LATTICE['xc'])
    parser.add_argument("--rel", type=str, choices=["nrl", "sra", "srals"], default=KKR_PARAMS_LATTICE['rel'])
    # None → per-step defaults from consts (lattice 10, Debye 10, final SCF 20), same as the
    # random/opt pipelines. Until 2026-10-22 the default was 10 and overrode ALL steps (final SCF too).
    parser.add_argument("--bzqlty", type=float, default=None,
                        help="k-point quality for ALL steps (lattice, Debye, final SCF); default: per-step consts")
    parser.add_argument("--bzqlty_finalscf", type=float, default=None,
                        help="k-point quality for the final SCF only (applied after --bzqlty)")
    parser.add_argument("--pmix", type=float, default=KKR_PARAMS_LATTICE['pmix'])
    parser.add_argument("--edelt", type=float, default=KKR_PARAMS_LATTICE['edelt'])
    parser.add_argument("--mxl", type=int, default=KKR_PARAMS_LATTICE['mxl'])
    parser.add_argument("--magtype", choices=["nmag", "mag"], default=KKR_PARAMS_LATTICE['magtype'])
    parser.add_argument("--task", type=str, default="all", choices=["lattice", "all"])
    parser.add_argument("--reuse_from", type=str, default=None,
                        help="Earlier workdir of the same composition: take its lattice/ (and, with --debye_mode kkr, "
                             "debye/) instead of recomputing them; only the final SCF + McMillan run. Lattice/Debye "
                             "KKR params must match. Can equal --workdir (in place). An existing workdir/finalscf "
                             "is moved to finalscf.copy (in place also run_params.json/results.json → *.copy).")
    parser.add_argument("--save_plots", action="store_true", help="Save McMillan orbital PNG plots (off by default)")
    parser.add_argument("--debye_mode", type=str, default="kkr", choices=["kkr", "mix", "mjs_opt"],
                        help="Debye temperature source: 'kkr' (full KKR distortions, default), "
                             "'mix' (composition-weighted elemental θ_D, no distortions), "
                             "'mjs_opt' (MJS bulk-modulus formula with C=0.778, no distortions).")

    args = vars(parser.parse_args())

    # type fixes
    args["concentrations"] = [float(x) for x in args["concentrations"]]

    # normalize flags
    args["compress"] = not args.pop("no_gzip")
    args["pack"] = not args.pop("no_pack")

    result = run_one_hea(**args)
    print(result)