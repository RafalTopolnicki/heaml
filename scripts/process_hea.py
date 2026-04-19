import argparse
import os
from src.elements import HEAClass
from src.utils import save_dict_to_json
from src.consts import KKR_PARAMS_LATTICE, KKR_PARAMS_DEBYE, KKR_PARAMS_FINALSCF
from lattice import run_kkr_eos
from finalscf import run_kkr_finalscf
from debye import run_kkr_elastic_debye
from macmillan import run_mcmillan_sweep
from src.process_kkr import process_kkr


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
    # overwrite def params
    # for lattcie optimization only
    KKR_PARAMS_LATTICE_PARAMS = KKR_PARAMS_LATTICE.copy()
    KKR_PARAMS_LATTICE_PARAMS['ew'] = kwargs.get('ew', KKR_PARAMS_LATTICE['ew'])
    KKR_PARAMS_LATTICE_PARAMS['xc'] = kwargs.get('xc', KKR_PARAMS_LATTICE['xc'])
    KKR_PARAMS_LATTICE_PARAMS['rel'] = kwargs.get('rel', KKR_PARAMS_LATTICE['rel'])
    KKR_PARAMS_LATTICE_PARAMS['bzqlty'] = kwargs.get('bzqlty', KKR_PARAMS_LATTICE['bzqlty'])
    KKR_PARAMS_LATTICE_PARAMS['pmix'] = kwargs.get('pmix', KKR_PARAMS_LATTICE['pmix'])
    KKR_PARAMS_LATTICE_PARAMS['edelt'] = kwargs.get('edelt', KKR_PARAMS_LATTICE['edelt'])
    KKR_PARAMS_LATTICE_PARAMS['mxl'] = kwargs.get('mxl', KKR_PARAMS_LATTICE['mxl'])
    KKR_PARAMS_LATTICE_PARAMS['magtype'] = kwargs.get('magtype', KKR_PARAMS_LATTICE['magtype'])
    run_params = {
        'element_labels': kwargs['element_labels'],
        'concentrations': list(kwargs['concentrations']),
        'density': hea.density,
        'mixture_lattice': hea.mixture_lattice,
        'mixture_bulk_modulus': hea.mixture_bulk_modulus,
        'mixture_debye_temperature': hea.mixture_debye_temperature,
        'mixture_mass': hea.mass,
        'KKR_PARAMS_LATTICE': KKR_PARAMS_LATTICE_PARAMS,
        'KKR_PARAMS_DEBYE': KKR_PARAMS_DEBYE,
        'KKR_PARAMS_FINALSCF': KKR_PARAMS_FINALSCF
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