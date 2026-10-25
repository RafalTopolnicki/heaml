"""Recompute eta (modes valence=A, surface=B, gg=G) from fort.51 of final-SCF reruns with another reltyp (nrl / srals),
using composition, theta_D and masses from the matching SRA baseline workdir (same name).
Usage: python scripts/eta_rmta_tests/13_eta_from_other_rel.py <rerun_base_dir> <sra_base_dir> <tag>
e.g.   python scripts/eta_rmta_tests/13_eta_from_other_rel.py results/test.nrl.ew0.6/V_set_nrl results/comparison.sra.ew0.6.bz10-10-20 nrl
Writes <sra_base_dir>/eta_{valence,surface,gg}_<tag>.csv (same columns as 06) + per-element channel table on stdout."""
import sys, os, json, numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from common import fort51_path
from macmillan import parse_fort51
from macmillan_cutoff import compute_one_combination_cutoff, C_THEORETICAL
from src.consts import VALENCE_NODES_EXPECTED
rerun, sra, tag = sys.argv[1:4]
MODES = {'valence': 'valence', 'surface': 'surface', 'gg': 'gg'}
rows = {m: [] for m in MODES}; chan = []
for name in sorted(os.listdir(rerun)):
    wd = os.path.join(rerun, name)
    if not os.path.isdir(wd): continue
    try: f51 = fort51_path(wd)
    except FileNotFoundError: print('no fort.51:', name); continue
    rp = json.load(open(os.path.join(sra, name, 'run_params.json'))); rj = json.load(open(os.path.join(sra, name, 'results.json')))
    d = parse_fort51(f51); ef = float(np.atleast_1d(d['ef'])[0])
    labels = rp['element_labels']; conc = np.asarray(rp['concentrations'], float); conc /= conc.sum()
    for m in MODES:
        row = dict(name=name, ef_Ry=ef, thetaDB_K=rj['thetaDB_K'], mixture_mass=rj['mixture_mass'],
                   mixture_debye_temperature=rj['mixture_debye_temperature'], eta_mix_cutoff=rj['eta_mix'], lambda_cutoff=rj['lambda'])
        eta_mix = 0.0
        for b in d['components']:
            if b['spin'] != 1: continue
            b['ef'] = ef; el = labels[b['component'] - 1]
            r = compute_one_combination_cutoff(b, 'plain', 'none', 'mean', cutoff_mode=m, valence_nodes=VALENCE_NODES_EXPECTED.get(el))
            for ch in ('sp', 'pd', 'df'): row[f'{el}_eta_{ch}_{m}'] = r[f'eta_cutoff_{ch}']
            row[f'{el}_eta_total_{m}'] = r['eta_total_cutoff']; row[f'{el}_Ntot'] = r['Ntot']
            eta_mix += conc[b['component'] - 1] * r['eta_total_cutoff']
            chan.append(dict(name=name, el=el, x=conc[b['component'] - 1], mode=m, N2=2 * r['Ntot'],
                             sp=1e3 * r['eta_cutoff_sp'], pd=1e3 * r['eta_cutoff_pd'], df=1e3 * r['eta_cutoff_df'], tot=1e3 * r['eta_total_cutoff']))
        row[f'eta_mix_{m}'] = eta_mix
        row[f'lambda_{m}'] = eta_mix * C_THEORETICAL / (rj['mixture_mass'] * rj['thetaDB_K']**2) if rj['thetaDB_K'] else np.nan
        rows[m].append(row)
for m in MODES:
    out = os.path.join(sra, f'eta_{m}_{tag}.csv'); pd.DataFrame(rows[m]).to_csv(out, index=False); print('->', out, len(rows[m]), 'rows')
c = pd.DataFrame(chan); c.to_csv(os.path.join(sra, f'eta_channels_{tag}.csv'), index=False)
pd.set_option('display.width', 200); print(c.round(1).to_string(index=False))
