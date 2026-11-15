"""SI: sensitivity to the Coulomb pseudopotential mu*.
figS_mustar_baseline : agreement with the literature set as a function of mu* (set A plain, set B corr2).
figS_mustar_champions: Tc of the champion compositions (found with mu* = 0.1) re-evaluated at mu* = 0.05, 0.1, 0.2, 0.3.
Writes numbers/mustar.json and figures/table_mustar_champions.tex."""
import numpy as np
import matplotlib.pyplot as plt
import style
import common as C


def tc(theta, lam, mu, les=0.0):
    """McMillan with the electron-paramagnon term of the corrected estimator (les = 0: plain Eq. (3))."""
    d = lam - les - mu * (1 + 0.62 * (lam + les))
    return theta / 1.45 * np.exp(-1.04 * (1 + lam + les) / d) if d > 0 else 0.0


b = C.load_baseline()
A, B = b[b.in_A], b[b.in_B]
assert np.allclose([tc(r.theta_mix, r.lambda_mix, 0.1) for r in A.itertuples()], A['Tc_mu0.1'])
assert np.allclose([tc(r.theta_corr, r.lambda_corr, 0.1, r.lambda_es) for r in B.itertuples()], B['Tc_corr2'])

MUS = np.round(np.arange(0.04, 0.3001, 0.01), 2)
scan = {'A': {}, 'B': {}}
for mu in MUS:
    scan['A'][float(mu)] = C.metrics([tc(r.theta_mix, r.lambda_mix, mu) for r in A.itertuples()], A.Tc_exp)
    scan['B'][float(mu)] = C.metrics([tc(r.theta_corr, r.lambda_corr, mu, r.lambda_es) for r in B.itertuples()], B.Tc_exp)

fig, ax = plt.subplots(1, 4, figsize=(style.DOUBLE, 2.0))
panels = [('median_ratio', 'median $T_c^{\\rm calc}/T_c^{\\rm exp}$'), ('MAE', 'MAE (K)'), ('tau', "Kendall $\\tau$"),
          ('pair_acc_1K', 'pairs ordered ($|\\Delta T_c^{\\rm exp}|>1$ K)')]
for a, (k, lab), let in zip(ax, panels, 'abcd'):
    for name, col in (('A', style.C_A), ('B', style.C_B)):
        a.plot(MUS, [scan[name][float(m)][k] for m in MUS], color=col, label=f'set {name}' + (' (corr.)' if name == 'B' else ''))
    a.axvline(0.1, color='k', ls=':', lw=.7)
    if k == 'median_ratio':
        a.axhline(1, color='0.6', lw=.6)
        a.legend()
    a.set(xlabel='$\\mu^*$', title=f'({let}) {lab}')
    a.grid(alpha=.3)
plt.tight_layout(w_pad=.6)
style.save(fig, 'figS_mustar_baseline')

best = {n: {k: min(MUS, key=lambda m: f(scan[n][float(m)][k])) for k, f in
            (('MAE', lambda v: v), ('median_ratio', lambda v: abs(v - 1)), ('tau', lambda v: -v))} for n in scan}

# champions
MU_CH = [0.05, 0.1, 0.2, 0.3]
series = {'A': C.load_series('A'), 'B': C.load_series('B_corr2')}
champs = {}
fig, ax = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.6), sharey=True)
for a, (name, runs), let in zip(ax, series.items(), 'ab'):
    rows = {}
    for r in runs:
        ch = C.champion(r)
        if name == 'A':
            th, lam, les = ch.theta, ch.lam, 0.0
        else:
            th, lam, les = ch.theta_corr, ch.lam_corr, ch.lam_es
        assert abs(tc(th, lam, 0.1, les) - ch.Tc) < 1e-6, (name, r['mc'])
        rows[r['mc']] = dict(comp=ch.comp, lam=float(lam), theta=float(th), lam_sf=float(les),
                             Tc={str(mu): float(tc(th, lam, mu, les)) for mu in MU_CH})
    champs[name] = rows
    for mu, mk in zip(MU_CH, 'vosd'):
        a.plot(list(rows), [rows[k]['Tc'][str(mu)] for k in rows], mk + '-', ms=3.5, label=f'$\\mu^*={mu}$',
               color='k' if mu == 0.1 else None, lw=1.4 if mu == 0.1 else 1.0)
    a.set(xlabel='minimum number of elements $k$', xticks=range(2, 11), title=f'({let}) set {name}' + (' (corrected)' if name == 'B' else ''))
    a.grid(alpha=.3)
ax[0].set_ylabel('$T_c$ of the champion (K)')
ax[0].legend()
plt.tight_layout()
style.save(fig, 'figS_mustar_champions')

lines = []
for name in champs:
    for k, r in champs[name].items():
        lines.append(f"{name} & {k} & {r['lam']:.2f} & {r['theta']:.0f} & " + ' & '.join(f"{r['Tc'][str(mu)]:.1f}" for mu in MU_CH) + r' \\')
    if name == 'A':
        lines.append(r'\hline')
(C.FIG / 'table_mustar_champions.tex').write_text('\n'.join(lines) + '\n')

C.save_numbers('mustar', dict(scan=scan, best_mu={n: {k: float(v) for k, v in d.items()} for n, d in best.items()}, champions=champs))
for n in scan:
    print(n, 'best mu', best[n], {m: {k: round(scan[n][m][k], 3) for k in ('median_ratio', 'MAE', 'tau', 'pair_acc_all', 'pair_acc_1K', 'r')} for m in (0.05, 0.1, 0.13, 0.15, 0.2, 0.3)})
for n in champs:
    for k, r in champs[n].items():
        print(n, k, r['comp'][:40], round(r['lam'], 3), round(r['theta']), {m: round(v, 2) for m, v in r['Tc'].items()})
