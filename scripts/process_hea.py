import argparse
import os
from src.elements import HEAClass
from src.utils import save_dict_to_json
from src.consts import KKR_PARAMS_LATTICE, KKR_PARAMS_DEBYE, KKR_PARAMS_FINALSCF
from lattice import run_kkr_eos
from finalscf import run_kkr_finalscf
from debye import run_kkr_elastic_debye
from macmillan_cutoff import run_mcmillan_cutoff_sweep
from src.process_kkr import process_kkr


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
    if overwrite_params:
        for param in ["ew", "xc", "rel", "bzqlty", "pmix", "magtype", "edelt", "mxl"]:
            KKR_PARAMS_LATTICE_PARAMS[param] = kwargs.get(param, KKR_PARAMS_LATTICE[param])
            KKR_PARAMS_DEBYE_PARAMS[param] = kwargs.get(param, KKR_PARAMS_DEBYE[param])
            KKR_PARAMS_FINALSCF_PARAMS[param] = kwargs.get(param, KKR_PARAMS_FINALSCF[param])

    use_mixture_debye = bool(KKR_PARAMS_DEBYE_PARAMS.get('use_mixture_debye', False))
    run_params = {
        'element_labels': kwargs['element_labels'],
        'concentrations': list(kwargs['concentrations']),
        'density': hea.density,
        'mixture_lattice': hea.mixture_lattice,
        'mixture_bulk_modulus': hea.mixture_bulk_modulus,
        'mixture_debye_temperature': hea.mixture_debye_temperature,
        'mixture_mass': hea.mass,
        'use_mixture_debye': use_mixture_debye,
        'KKR_PARAMS_LATTICE': KKR_PARAMS_LATTICE_PARAMS,
        'KKR_PARAMS_DEBYE': KKR_PARAMS_DEBYE_PARAMS,
        'KKR_PARAMS_FINALSCF': KKR_PARAMS_FINALSCF_PARAMS
                }
    save_dict_to_json(run_params, os.path.join(workdir, 'run_params.json'))
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
    if use_mixture_debye:
        # Skip expensive KKR tetragonal/monoclinic distortions; use composition-weighted
        # elemental Debye temperatures directly.
        debye_workdir = os.path.join(workdir, KKR_PARAMS_DEBYE_PARAMS["subdir"])
        os.makedirs(debye_workdir, exist_ok=True)
        debye_output = {"thetaDB_K": hea.mixture_debye_temperature}
        save_dict_to_json(debye_output, os.path.join(debye_workdir, "debye_summary.json"))
        print(f"Debye skipped (use_mixture_debye=True): thetaDB_K={hea.mixture_debye_temperature:.2f} K")
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
        mixture_mass=run_params['mixture_mass'],
        theta_d=theta_d_for_log,
    )

    # make all final computations
    allresults = process_kkr(path=workdir, dirname='')
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
    parser.add_argument("--overwrite_params", action="store_true")


    parser.add_argument("--ew", type=float, default=KKR_PARAMS_LATTICE['ew'])
    parser.add_argument("--xc", type=str, default=KKR_PARAMS_LATTICE['xc'])
    parser.add_argument("--rel", type=str, choices=["nrl", "sra", "srals"], default=KKR_PARAMS_LATTICE['rel'])
    parser.add_argument("--bzqlty", type=float, default=KKR_PARAMS_LATTICE['bzqlty'])
    parser.add_argument("--pmix", type=float, default=KKR_PARAMS_LATTICE['pmix'])
    parser.add_argument("--edelt", type=float, default=KKR_PARAMS_LATTICE['edelt'])
    parser.add_argument("--mxl", type=int, default=KKR_PARAMS_LATTICE['mxl'])
    parser.add_argument("--magtype", choices=["nmag", "mag"], default=KKR_PARAMS_LATTICE['magtype'])
    parser.add_argument("--task", type=str, default="all", choices=["lattice", "all"])

    args = vars(parser.parse_args())

    # type fixes
    args["concentrations"] = [float(x) for x in args["concentrations"]]

    # normalize flags
    args["compress"] = not args.pop("no_gzip")

    result = run_one_hea(**args)
    print(result)