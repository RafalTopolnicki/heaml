import numpy as np
import json
import os
import pandas as pd
from src.features import compute_hea_features
from src.consts import composition_labels


def compute_lambda(row):
    used_labels = [e for e in composition_labels if e in row.keys()]
    nominator = np.sum([row[e]*row[f'{e}_eta_total'] for e in used_labels])
    mixture_mass = row['mixture_mass']
    #composition_sum = np.sum([row[e] for e in composition_labels])
    denominator = mixture_mass*row['thetaDB']**2
    return nominator/denominator*5.47e4

def read_params(path, dirname):
    data = json.load(open(os.path.join(path, dirname, 'run_params.json'), 'r'))
    for l, c in zip(data['element_labels'], data['concentrations']):
        data[l] = c
    del data['element_labels']
    del data['concentrations']
    return data

def read_debye(path, dirname):
    data = json.load(open(os.path.join(path, dirname, 'debye', 'debye_summary.json'), 'r'))
    del data['run_log']
    del data['results_csv']
    del data['delta']
    del data['energy0_mono_ev']
    del data['energy0_tetra_ev']
    del data['energy_tetra']
    del data['energy_mono']
    return data

def get_composition(path, dirname):
    data = json.load(open(os.path.join(path, dirname, 'run_params.json'), 'r'))
    return data['concentrations'], data['element_labels']

def read_macmillan(path, dirname):
    composition_dict, elements = get_composition(path, dirname)
    df = pd.read_csv(open(os.path.join(path, dirname, 'finalscf', 'mcmillan_results.csv')))
    df = df[(df['reduce_mode'] == 'sum') & (df['integral_mode'] == 'plain') & (df['norm_mode'] == 'none')]
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
        data = {**data, **compute_hea_features(comp_dict=comp_dict)}
        return data
    except Exception as e:
         print(f'Error in processing {dirname}: {e}')
         return None

#out = process_kkr(path='/home/rafal/WORK/HEA/ML/random.ratios/sra.kp10.ew0.6/', dirname='Ti0.0008Nb0.3225Zr0.0191Hf0.4401Ta0.0272Sc0.0316Mo0.0243W0.0594Y0.0584La0.0165')
