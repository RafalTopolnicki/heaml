"""SI figures: (a) acquisition beta scan at mincomp 7, (b) interior-only vs full initial set (mincomp 2, 3),
(c) mu*=0.2 runs vs the main mu*=0.1 runs (best Tc at >= k).
figS_robust_champions: champion compositions of the same runs (stacked bars).  Also lambda vs theta_D coloured by Tc (set A, B).
Writes numbers/robustness.json."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import style
import common as C

A = C.load_series('A')
beta_runs = [C.load_run(C.OPT_GL / C.RUN_A.format(beta=b, mc=7), 'Tc_mu0.1', C.SET_A, f'A_mc7_b{b}') for b in [0, 1, 2, 5, 10]]
int_runs = [C.load_run(C.OPT_GL / C.RUN_A_INT.format(mc=mc), 'Tc_mu0.1', C.SET_A, f'A_int_mc{mc}') for mc in (2, 3)]
old = C.load_series('A_old', mcs=range(2, 9))
Bc = C.load_series('B_corr2')
Bp = C.load_series('B_plain')


def best_so_far(r):
    v = r['df'][r['df'].valid]
    its = np.arange(0, r['n_iter'] + 1)
    return its, np.array([v[v.iteration <= i].Tc.max() for i in its])


fig, ax = plt.subplots(1, 3, figsize=(style.DOUBLE, 2.5))
cm = plt.cm.plasma(np.linspace(0, .85, len(beta_runs)))
out = {'beta_scan': {}, 'interior': {}, 'old_vs_new': {}, 'B_plain_vs_corr2': {}}
for r, c in zip(beta_runs, cm):
    its, b = best_so_far(r)
    ax[0].step(its, b, where='post', color=c, label=f"$\\beta$={r['beta']:g}")
    ch = C.champion(r)
    out['beta_scan'][r['beta']] = dict(Tc=float(ch.Tc), comp=ch.comp, iteration=int(ch.iteration), iterations=r['n_done'])
ax[0].set(xlabel='iteration', ylabel='best valid $T_c$ (K)', title='(a) set A, $k\\geq7$: acquisition $\\beta$')
ax[0].legend()
for r in A[:2] + int_runs:
    its, b = best_so_far(r)
    interior = 'int' in r['key']
    ax[1].step(its, b, where='post', ls='--' if interior else '-', color=style.C_A if r['mc'] == 2 else style.C_B,
               label=f"$k\\geq{r['mc']}$, {'interior' if interior else 'full'} init")
    ch = C.champion(r)
    out['interior'][r['key']] = dict(Tc=float(ch.Tc), comp=ch.comp, iteration=int(ch.iteration), iterations=r['n_done'])
ax[1].set(xlabel='iteration', title='(b) set A: initial set without face points')
ax[1].legend()
_, _, ge_new = C.pareto(A)
_, _, ge_old = C.pareto(old)
_, _, ge_Bc = C.pareto(Bc)
_, _, ge_Bp = C.pareto(Bp)
ax[2].plot(ge_new.index, ge_new.values, 's-', color='k', ms=3, label='set A')
ax[2].plot(ge_old.index, ge_old.values, 'o--', color='0.5', ms=3, label='set A, $\\mu^*=0.2$')
ax[2].plot(ge_Bc.index, ge_Bc.values, 'd-', color=style.C_B, ms=3, label='set B, $T_c^{\\rm corr}$')
ax[2].plot(ge_Bp.index, ge_Bp.values, 'v:', color=style.C_B, ms=3, label='set B, no correction')
ax[2].set(xlabel='$k$', title='(c) best $T_c$ at $\\geq k$ elements', xticks=range(2, 11), xlim=(1.5, 10.5))
ax[2].legend()
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout()
style.save(fig, 'figS_robustness')
out['old_vs_new'] = dict(new=ge_new.to_dict(), old=ge_old.to_dict(),
                         old_champions={r['mc']: dict(Tc=float(C.champion(r).Tc), comp=C.champion(r).comp) for r in old})
out['B_plain_vs_corr2'] = dict(corr2=ge_Bc.to_dict(), plain=ge_Bp.to_dict(),
                               plain_champions={r['mc']: dict(Tc=float(C.champion(r).Tc), comp=C.champion(r).comp) for r in Bp})
# champion compositions of the test runs
def bars(a, items, title):
    """items: list of (label, champion row)."""
    champs = [ch for _, ch in items]
    labels = [f"{lab}: {ch.Tc:.1f} K" for lab, ch in items]
    order = sorted(C.SET_A, key=lambda el: -np.mean([ch[el] for ch in champs]))
    left = np.zeros(len(champs))
    y = np.arange(len(champs))
    for el in order:
        vals = np.array([ch[el] for ch in champs])
        a.barh(y, vals, left=left, color=style.EL_COLORS[el], edgecolor='white', lw=0.3)
        for i, (v, l) in enumerate(zip(vals, left)):
            if v > 0.045:
                a.text(l + v / 2, i, el, ha='center', va='center', fontsize=5,
                       color='white' if el in ('Nb', 'Ta', 'Hf', 'Re', 'V', 'Mo') else 'k')
        left += vals
    a.set_yticks(y)
    a.set_yticklabels(labels, fontsize=6)
    a.invert_yaxis()
    a.set(xlim=(0, 1), xlabel='atomic fraction', title=title)
    a.tick_params(axis='y', length=0)


fig = plt.figure(figsize=(style.DOUBLE, 4.6))
gs = fig.add_gridspec(2, 2, height_ratios=[5, 4])
a1, a2, a3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
bars(a1, [(f"$\\beta={r['beta']:g}$", C.champion(r)) for r in beta_runs], '(a) $k\\geq7$: acquisition weight $\\beta$')
bars(a2, [(f"$k\\geq{r['mc']}$, {'interior' if 'int' in r['key'] else 'full'}", C.champion(r))
          for r in [A[0], int_runs[0], A[1], int_runs[1]]], '(b) initial set: full / interior points only')
items = []
for rn, ro in zip(A[:7], old):
    assert rn['mc'] == ro['mc']
    items += [(f"$k\\geq{rn['mc']}$, $\\mu^*=0.1$", C.champion(rn)), (f"$k\\geq{ro['mc']}$, $\\mu^*=0.2$", C.champion(ro))]
bars(a3, items, '(c) $\\mu^*=0.1$ / $\\mu^*=0.2$')
plt.tight_layout()
style.save(fig, 'figS_robust_champions')
out['old_lambda'] = {r['mc']: (float(C.champion(r).lam) if pd.notna(C.champion(r).lam) else None) for r in old}

C.save_numbers('robustness', out)

# lambda vs theta_D coloured by Tc
fig, ax = plt.subplots(1, 2, figsize=(style.DOUBLE, 2.7))
for a, (name, runs, tcol) in zip(ax, [('A', A, '$T_c$'), ('B', Bc, '$T_c^{\\rm corr}$')]):
    U, _, _ = C.pareto(runs)
    O = U[U.iteration > 0]
    lam, th = ('lam_corr', 'theta_corr') if name == 'B' else ('lam', 'theta')
    sc = a.scatter(O[th], O[lam], c=O.Tc, s=4, cmap='plasma', lw=0)
    for r in runs:
        ch = C.champion(r)
        a.scatter(ch[th], ch[lam], s=40, facecolors='none', edgecolors='k', lw=0.7)
        a.annotate(f"{r['mc']}", (ch[th], ch[lam]), fontsize=5.5, xytext=(3, 3), textcoords='offset points')
    plt.colorbar(sc, ax=a, label=f'{tcol} (K)')
    a.set(xlabel='$\\theta_D^{\\rm mix}$ (K)', ylabel='$\\lambda$', title=f'({"ab"[0 if name == "A" else 1]}) set {name}')
    a.grid(alpha=.3)
plt.tight_layout()
style.save(fig, 'figS_lambda_theta')
print(out['beta_scan'])
print(out['interior'])
print('old', {k: round(v, 2) for k, v in ge_old.items()})
print('Bplain', {k: round(v, 2) for k, v in ge_Bp.items()})
