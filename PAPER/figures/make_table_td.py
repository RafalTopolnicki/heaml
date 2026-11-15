"""Table: bcc/fcc/hcp energies and mixing enthalpy of the champions (set A, set B corr2) -> figures/table_td.tex,
numbers/td.json.  dG(T) = dH_bcc - T k_B S_conf with T = DG_T_K of consts.py (1500 K)."""
import numpy as np
import pandas as pd
import common as C
from src.consts import DG_T_K  # noqa: E402  (common.py put scripts/ on sys.path)

import json
KB = 8.617333262e-5 * 1000   # meV/K
# linear model of the corresponding Fig. 1 panel (T_exp = slope * T_calc + intercept), from make_fig_validation.py
_val = json.load(open(C.ROOT / 'PAPER/numbers/validation.json'))['metrics']
FIT = {'A': _val['A_mu0.1'], 'B': _val['B_corr2']}
rows_tex, out = [], {}
for name, prefix, elements in [('A', 'A_mu0.1', C.SET_A), ('B', 'B_corr2', C.SET_B)]:
    td, REF = C.champions_td(prefix)
    td['dG_T'] = td.dH_bcc - DG_T_K * KB * td.S_conf_kB
    td['Tc_resc'] = FIT[name]['slope'] * td.Tc + FIT[name]['intercept']
    out[name] = td.drop(columns='comp').to_dict('index')
    out[name + '_refs'] = REF.to_dict('index')
    for mc, r in td.iterrows():
        comp = C.comp_tex(pd.Series(r.comp), elements)
        f = lambda v: '--' if pd.isna(v) else f'{v:+.0f}'
        rows_tex.append(f"{name} & {mc} & {comp} & {r.Tc:.1f} & {r.Tc_resc:.1f} & {r.dE_fcc:.0f} & {r.dE_hcp:.0f} & {f(r.dH_bcc)} & {f(r.dH_gs)} & "
                        f"{r.S_conf_kB:.2f} & {f(r.dG_T)} \\\\")
    print(name, td[['Tc', 'Tc_resc', 'most_stable', 'dE_fcc', 'dE_hcp', 'dH_bcc', 'dH_gs', 'S_conf_kB', 'dG_T']].round(1))
out['T_K'] = DG_T_K
C.save_numbers('td', out)
(C.FIG / 'table_td.tex').write_text('\n'.join(rows_tex) + '\n')
print(REF.round(1))
