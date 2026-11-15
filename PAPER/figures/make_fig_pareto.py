"""Fig. 2: best Tc versus number of constituent elements (> 5 %) for set A and set B (corr2); all evaluated points.
Writes numbers/pareto.json (champion per mincomp run, best at >= k, best init, gains, element entry order)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import style
import common as C

import json
series = {'A': C.load_series('A'), 'B': C.load_series('B_corr2')}
# linear model of the corresponding Fig. 1 panel (T_exp = slope * T_calc + intercept), from make_fig_validation.py
_val = json.load(open(C.ROOT / 'PAPER/numbers/validation.json'))['metrics']
FIT = {'A': _val['A_mu0.1'], 'B': _val['B_corr2']}
C_RESC = style.C_FIT
rng = np.random.default_rng(0)
fig, ax = plt.subplots(1, 2, figsize=(style.DOUBLE, 3.15), sharey=True)
out = {}
for a, (name, runs) in zip(ax, series.items()):
    U, exact, ge = C.pareto(runs)
    init, opt = U[U.iteration == 0], U[U.iteration > 0]
    a.scatter(init.n_comp + rng.uniform(-.18, .18, len(init)), init.Tc, s=3, color='0.75', lw=0, label='initial set')
    a.scatter(opt.n_comp + rng.uniform(-.18, .18, len(opt)), opt.Tc, s=3, color=style.C_A, alpha=.5, lw=0,
              label='active learning')
    a.plot(exact.index, exact.values, 'o-', color=style.C_B, ms=3.5, label='best at exactly $k$ elements')
    a.plot(ge.index, ge.values, 's--', color='k', ms=3, label='best at $\\geq k$ elements')
    resc = FIT[name]['slope'] * ge + FIT[name]['intercept']
    a.plot(resc.index, resc.values, '--', color=C_RESC, lw=1.2,
           label='best at $\\geq k$, rescaled by the Fig. 1 fit')
    a.set(xlabel='number of elements $k$ (fraction > 5%)', xticks=range(1, 11),
          title=f'({"ab"[list(series).index(name)]}) set {name}' + (' ($T_c^{\\rm corr}$)' if name == 'B' else ' ($T_c$, $\\mu^*=0.1$)'))
    a.grid(alpha=.3)
    if name == 'A':
        a.set_ylabel('calculated $T_c$ (K)')
    champs = {}
    for r in runs:
        ch = C.champion(r)
        v0 = r['df'][(r['df'].iteration == 0) & r['df'].valid]
        champs[r['mc']] = dict(Tc=float(ch.Tc), comp=ch.comp, comp_dict={el: float(ch[el]) for el in (C.SET_A if name == 'A' else C.SET_B) if ch[el] > 0},
                               n_el=int(ch.n_comp), iteration=int(ch.iteration), source=ch.source,
                               lam=float(ch.lam) if pd.notna(ch.lam) else None, theta=float(ch.theta) if pd.notna(ch.theta) else None,
                               VEC=float(ch.VEC) if pd.notna(ch.VEC) else None,
                               init_best_Tc=float(v0.Tc.max()) if len(v0) else None, init_best=v0.loc[v0.Tc.idxmax()].comp if len(v0) else None,
                               n_init_valid=int(len(v0)), iterations=r['n_done'], early_stop=r['early_stop'],
                               n_opt_points=int((r['df'].iteration > 0).sum()),
                               frac_local=float((r['df'][r['df'].iteration > 0].source == 'local').mean()))
    # which elements sit at the 5 % floor in the champions
    floor = {mc: [el for el, x in c['comp_dict'].items() if 0.05 <= x < 0.09] for mc, c in champs.items()}
    out[name] = dict(best_exact=exact.to_dict(), best_ge=ge.to_dict(), champions=champs, floor_elements=floor,
                     n_points=int(len(U)), n_opt=int(len(opt)), n_init=int(len(init)),
                     drop_per_element=float((ge[2] - ge[10]) / 8),
                     corr_lambda_Tc=float(opt[['lam', 'Tc']].corr().iloc[0, 1]),
                     rescale=dict(slope=FIT[name]['slope'], intercept=FIT[name]['intercept']),
                     best_ge_rescaled={k: float(v) for k, v in resc.items()})
h, l = ax[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=5, bbox_to_anchor=(0.5, -0.01), markerscale=1.5)
plt.tight_layout(rect=(0, 0.07, 1, 1))
style.save(fig, 'fig_pareto')
C.save_numbers('pareto', out)
for name, o in out.items():
    print(name, {k: round(v, 2) for k, v in o['best_ge'].items()})
    for mc, c in o['champions'].items():
        print(f"  mc{mc}: {c['Tc']:.2f} K  {c['comp']}  it {c['iteration']}/{c['iterations']} (init best {c['init_best_Tc']}) local {c['frac_local']:.2f}")
