"""Fig. S2 (SI): KKR Tc along the Nb-Zr, Nb-Ti and Nb-Hf lines (GL, mu*=0.1) compared with experiment.
Experimental points: Nb0.85X0.15 (Kushwaha et al., arXiv:2512.18821, resistivity) and pure Nb from Tc_literature.xlsx
(if present).  Writes numbers/binaries.json."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import style
import common as C

bins = C.load_binaries()
bins = bins[bins.ok]
# experimental values typed here come from the sources named in the paper; keep them in one place
EXP = {'Zr': [(0.15, 11.06, 'Kushwaha2025')], 'Ti': [(0.15, 9.68, 'Kushwaha2025')], 'Hf': [(0.15, 9.78, 'Kushwaha2025')]}
# Nb-Ti: resistive Tc of recrystallised bcc alloys, H. J. Muller, PhD thesis (Univ. Wisconsin-Madison 1989), Table 2;
# only the compositions that are bcc throughout (>= 30 at.% Nb)
EXP['Ti'] += [(0.30, 10.00, 'Muller1989'), (0.40, 9.94, 'Muller1989'), (0.50, 9.71, 'Muller1989'),
              (0.60, 9.47, 'Muller1989'), (0.65, 9.16, 'Muller1989'), (0.70, 8.91, 'Muller1989')]
MARK = {'Kushwaha2025': ('*', 8, 'experiment, as-cast [Kushwaha]'), 'Muller1989': ('D', 3.5, 'experiment, annealed [Muller]'),
        'xlsx pure Nb': ('*', 8, None)}
B = C.load_baseline()
nb_rows = B[(B.Nb > 0.999)]
if len(nb_rows):
    for X in EXP:
        EXP[X].append((0.0, float(nb_rows.Tc_exp.iloc[0]), 'xlsx pure Nb'))

# linear model of Fig. 1(a): T_exp = slope * T_calc + intercept, fitted on the set-A literature alloys
mA = C.metrics(B.loc[B.in_A, 'Tc_mu0.1'], B.loc[B.in_A, 'Tc_exp'])
fig, ax = plt.subplots(1, 3, figsize=(style.DOUBLE, 2.5), sharey=True)
out = {'rescale': dict(slope=mA['slope'], intercept=mA['intercept'])}
for a, X in zip(ax, ['Zr', 'Ti', 'Hf']):
    d = bins[(bins.Nb > 0) & (bins[X] > 0)].sort_values(X)
    resc = mA['slope'] * d['Tc_mu0.1'] + mA['intercept']
    a.plot(d[X], d['Tc_mu0.1'], 'o-', color=style.C_A, ms=3, label='calculated')
    a.plot(d[X], resc, 's--', color='k', ms=2.5, label='calculated, rescaled with the fit of Fig. 1(a)')
    for x, t, src in EXP[X]:
        mk, ms, lab = MARK[src]
        a.plot(x, t, mk, color=style.C_B, ms=ms, label=lab, zorder=5)
    a.set(xlabel=f'$x$ in Nb$_{{1-x}}${X}$_x$', title=f'Nb–{X}', xlim=(-0.02, 1.0))
    a.grid(alpha=.3)
    best = d.loc[d['Tc_mu0.1'].idxmax()]
    d = d.assign(Tc_rescaled=resc)
    out[X] = dict(points=d[[X, 'Tc_mu0.1', 'Tc_rescaled', 'lambda_mix', 'theta_mix']].round(3).to_dict('records'),
                  rescaled_max=float(resc.max()),
                  kkr_max=float(best['Tc_mu0.1']), kkr_max_x=float(best[X]), exp=EXP[X])
ax[0].set_ylabel('$T_c$ (K)')
hl = {}
for a in ax:
    for hh, ll in zip(*a.get_legend_handles_labels()):
        hl.setdefault(ll, hh)
h, l = list(hl.values()), list(hl)
fig.legend(h, l, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.14))
plt.tight_layout()
style.save(fig, 'figS_binaries')
C.save_numbers('binaries', out)
for X, o in out.items():
    if X == 'rescale':
        continue
    print(X, 'rescaled:', [(p[X], round(p['Tc_rescaled'], 2)) for p in o['points']], '\n ', 'KKR max', round(o['kkr_max'], 2), 'at x =', o['kkr_max_x'], '| points:', [(p[X], round(p['Tc_mu0.1'], 2)) for p in o['points']])
