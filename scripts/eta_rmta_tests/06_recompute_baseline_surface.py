"""Recompute eta/lambda for every process_hea workdir under a results directory with cutoff_mode='surface'
(Gaspari-Gyorffy boundary term, MT normalization), from the stored finalscf/fort.51(.gz) — no KKR run.
Writes <results_dir>/eta_surface.csv with per-element channels, eta_mix, lambda_surface and the production
(last-node 'valence' cutoff) lambda from results.json for comparison.
Usage: python scripts/eta_rmta_tests/06_recompute_baseline_surface.py results/comparison.sra.ew0.6.bz10-10-20
"""
import sys, os, json, numpy as np, pandas as pd
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from common import fort51_path
from macmillan import parse_fort51
from macmillan_cutoff import compute_one_combination_cutoff, C_THEORETICAL

MODE = sys.argv[2] if len(sys.argv) > 2 else 'surface'

def one(wd):
    try:
        rp = json.load(open(os.path.join(wd, 'run_params.json')))
        rj = json.load(open(os.path.join(wd, 'results.json')))
        d = parse_fort51(fort51_path(wd))
        ef = float(np.atleast_1d(d['ef'])[0])
        labels = rp['element_labels']; conc = np.asarray(rp['concentrations'], float); conc /= conc.sum()
        row = dict(name=os.path.basename(wd), ef_Ry=ef, thetaDB_K=rj['thetaDB_K'], mixture_mass=rj['mixture_mass'],
                   mixture_debye_temperature=rj['mixture_debye_temperature'],
                   eta_mix_cutoff=rj['eta_mix'], lambda_cutoff=rj['lambda'])
        eta_mix = 0.0
        for b in d['components']:
            if b['spin'] != 1: continue
            b['ef'] = ef
            r = compute_one_combination_cutoff(b, 'plain', 'none', 'mean', cutoff_mode=MODE)
            el = labels[b['component'] - 1]
            for ch in ('sp', 'pd', 'df'):
                row[f'{el}_eta_{ch}_{MODE}'] = r[f'eta_cutoff_{ch}']
            row[f'{el}_eta_total_{MODE}'] = r['eta_total_cutoff']
            row[f'{el}_Ntot'] = r['Ntot']
            eta_mix += conc[b['component'] - 1] * r['eta_total_cutoff']
        row[f'eta_mix_{MODE}'] = eta_mix
        row[f'lambda_{MODE}'] = eta_mix * C_THEORETICAL / (rj['mixture_mass'] * rj['thetaDB_K']**2)
        return row
    except Exception as e:
        return dict(name=os.path.basename(wd), error=repr(e))

if __name__ == '__main__':
    base = sys.argv[1]
    dirs = sorted(os.path.join(base, n) for n in os.listdir(base) if os.path.isfile(os.path.join(base, n, 'results.json')))
    with Pool(12) as p:
        rows = p.map(one, dirs)
    df = pd.DataFrame(rows)
    out = os.path.join(base, f'eta_{MODE}.csv'); df.to_csv(out, index=False)
    bad = df[df.get('error', pd.Series(dtype=object)).notna()] if 'error' in df else df.iloc[0:0]
    print(f'{len(df) - len(bad)} ok, {len(bad)} failed -> {out}'); 
    if len(bad): print(bad[['name', 'error']].to_string())
    ok = df[df[f'lambda_{MODE}'].notna()]
    print((ok[f'lambda_{MODE}'] / ok.lambda_cutoff).describe().round(3))
