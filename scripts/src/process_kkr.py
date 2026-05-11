import numpy as np
import json
import os
import pandas as pd
from src.features import compute_hea_features
from src.consts import composition_labels


def normalize_composition(composition):
    composition = np.array(composition, dtype=float)
    total = np.sum(composition)
    if total <= 0:
        raise ValueError("Invalid composition sum")
    return (composition / total).tolist()


def compute_lambda(row):
    used_labels = [e for e in composition_labels if e in row.keys()]
    nominator = np.sum([row[e]*row[f'{e}_eta_total'] for e in used_labels])
    mixture_mass = row['mixture_mass']
    #composition_sum = np.sum([row[e] for e in composition_labels])
    denominator = mixture_mass*row['thetaDB']**2
    # Empirically calibrated from elemental Ta (lambda_exp=0.69, eta_mean mode):
    # C = lambda_exp * M_Ta * ThetaD^2 / eta_total = 0.69*180.9*288.19^2/39.91 ~ 2.60e5
    # CHECK THIS LATER
    return nominator/denominator*2.60e5

def read_params(path, dirname):
    data = json.load(open(os.path.join(path, dirname, 'run_params.json'), 'r'))
    concentrations = normalize_composition(data['concentrations'])
    for l, c in zip(data['element_labels'], concentrations):
        data[l] = c
    del data['element_labels']
    del data['concentrations']
    return data

def read_debye(path, dirname):
    data = json.load(open(os.path.join(path, dirname, "debye", "debye_summary.json"), "r"))

    # Backward compatibility:
    # old code downstream expects "thetaDB"
    if "thetaDB_K" in data and "thetaDB" not in data:
        data["thetaDB"] = data["thetaDB_K"]

    # Remove file/log metadata if present
    for key in [
        "run_log",
        "results_csv",
        "all_scf_results_csv",
        "even_energy_fit_data_csv",
        "rmt_candidates_json",
    ]:
        data.pop(key, None)

    # Remove large nested fit diagnostics if you do not want them as ML columns
    for key in [
        "tetra_fit",
        "c44_fit",
        "C44_diagnostics_Ry_bohr3",
        "C44_diagnostics_GPa",
    ]:
        data.pop(key, None)

    # Remove old single-run energy keys if present
    for key in [
        "delta",
        "energy0_mono_ev",
        "energy0_tetra_ev",
        "energy_tetra",
        "energy_mono",
    ]:
        data.pop(key, None)

    return data

def get_composition(path, dirname):
    data = json.load(open(os.path.join(path, dirname, 'run_params.json'), 'r'))
    return normalize_composition(data['concentrations']), data['element_labels']

def read_macmillan(path, dirname):
    composition_dict, elements = get_composition(path, dirname)
    df = pd.read_csv(open(os.path.join(path, dirname, 'finalscf', 'mcmillan_results.csv')))
    df = df[(df['reduce_mode'] == 'mean') & (df['integral_mode'] == 'plain') & (df['norm_mode'] == 'none')]
    assert len(df) == len(composition_dict)
    df['component_label'] = [elements[x-1] for x in df['component']]
    df.reset_index(drop=True, inplace=True)
    results = {}
    for _, row in df.iterrows():
        cmp_label = row['component_label']
        for key in ['eta_sp', 'eta_pd', 'eta_df', 'M_sp', 'M_pd', 'M_df', 'eta_total']:
            results[f'{cmp_label}_{key}'] = row[key]
    return results

def tc_from_data(data, mu):
    return data['thetaDB']/1.45*np.exp(-1.04*(1+data['lambda'])/(data['lambda']-mu*(1+0.62*data['lambda'])))

def process_kkr(path, dirname):
    try:
        composition, elements = get_composition(path, dirname)
        comp_dict = dict(zip(elements, composition))
        # Mind: dict merging old-style as this has to run on old python
        #data = {'name': dirname} | read_params(path, dirname) | read_debye(path, dirname) | read_macmillan(path, dirname)
        data = {'name': dirname}
        data.update(read_params(path, dirname))
        data.update(read_debye(path, dirname))
        data.update(read_macmillan(path, dirname))
        data['lambda'] = compute_lambda(data)
        data['Tc_mu0.1'] = tc_from_data(data, mu=0.1)
        data['Tc_mu0.2'] = tc_from_data(data, mu=0.2)
        data['Tc_mu0.3'] = tc_from_data(data, mu=0.3)
        # add features
        data = {**data, **compute_hea_features(comp_dict=comp_dict, normalize_composition=True)}
        return data
    except Exception as e:
         print(f'Error in processing {dirname}: {e}')
         return None

#out = process_kkr(path='/home/rafal/WORK/HEA/ML/random.ratios/sra.kp10.ew0.6/', dirname='Ti0.0008Nb0.3225Zr0.0191Hf0.4401Ta0.0272Sc0.0316Mo0.0243W0.0594Y0.0584La0.0165')
out = process_kkr(path='/home/rafal/WORK/HEA/', dirname='TEMP')
