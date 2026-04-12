import argparse
import os
from src.elements import HEAClass
from src.utils import save_dict_to_json
from src.consts import KKR_PARAMS_LATTICE, KKR_PARAMS_DEBYE, KKR_PARAMS_FINALSCF
from lattice import run_kkr_eos
from finalscf import run_kkr_finalscf
from debye import run_kkr_elastic_debye
from macmillan import run_mcmillan_sweep


# =========================
# ⭐ MASTER FUNCTION
# =========================
def run_one_hea(**kwargs):
    workdir = kwargs.get("workdir", ".")
    os.makedirs(workdir, exist_ok=True)
    cwd = os.getcwd()

    hea = HEAClass(labels=kwargs['element_labels'],
                   concentrations=kwargs['concentrations'])
    hea_configuration = {'elements': hea.return_atomic_numbers(),
                         'concentrations': hea.concentrations,
                         'density': hea.density}
    run_params = {
        'element_labels': kwargs['element_labels'],
        'concentrations': list(kwargs['concentrations']),
        'density': hea.density,
        'mixture_lattice': hea.mixture_lattice,
        'mixture_bulk_modulus': hea.mixture_bulk_modulus,
        'mixture_debye_temperature': hea.mixture_debye_temperature,
        'mixture_mass': hea.mass,
        'KKR_PARAMS_LATTICE': KKR_PARAMS_LATTICE,
        'KKR_PARAMS_DEBYE': KKR_PARAMS_DEBYE,
        'KKR_PARAMS_FINALSCF': KKR_PARAMS_FINALSCF
                }
    save_dict_to_json(run_params, os.path.join(workdir, 'run_params.json'))
    # optimize lattice
    lattice_params = KKR_PARAMS_LATTICE
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
    scf_params = KKR_PARAMS_FINALSCF
    scf_params.update(hea_configuration)
    scf_params['a0'] = eof_output['lattice_constant_bohr']
    scf_params['workdir'] = os.path.join(workdir, scf_params['subdir'])
    scf_params['sym'] = 'bcc'
    scf_output = run_kkr_finalscf(**scf_params)
    print('SCF computations DONE')
    print(scf_output)
    os.chdir(cwd)

    # run MacMillan
    run_mcmillan_sweep(workdir=scf_params['workdir'])

    # run debye
    debye_params = KKR_PARAMS_DEBYE
    debye_params.update(hea_configuration)
    debye_params['a0'] = eof_output['lattice_constant_bohr']
    debye_params['B0'] = eof_output['bulk_modulus_gpa']
    debye_params['workdir'] = os.path.join(workdir, debye_params['subdir'])
    debye_output = run_kkr_elastic_debye(**debye_params)
    print('Debye computations DONE')
    print(debye_output)
    os.chdir(cwd)



# =========================
# CLI
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--element_labels", nargs="+", required=True)
    parser.add_argument("--concentrations", nargs="+", required=True)
    parser.add_argument("--workdir", type=str, default=".")
    parser.add_argument("--no-gzip", action="store_true")


    parser.add_argument("--ew", type=float, default=0.7)
    parser.add_argument("--xc", type=str, default="pbe")
    parser.add_argument("--rel", type=str, default="nrl", choices=["nrl", "sra", "srals"])
    parser.add_argument("--bzqlty", type=float, default=10)
    parser.add_argument("--pmix", type=float, default=0.01)
    parser.add_argument("--edelt", type=float, default=0.001)
    parser.add_argument("--mxl", type=int, default=3)
    parser.add_argument("--magtype", default="nmag", choices=["nmag", "mag"])
    parser.add_argument("--task", type=str, default="all", choices=["lattice", "all"])

    args = vars(parser.parse_args())

    # type fixes
    args["concentrations"] = [float(x) for x in args["concentrations"]]

    # normalize flags
    args["compress"] = not args.pop("no_gzip")

    result = run_one_hea(**args)
    print(result)