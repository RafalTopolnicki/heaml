"""Fig. 1: pipeline Tc vs experimental Tc for the literature alloys of set A (Tc_mu0.1) and set B (Tc_corr2).
Also the SI versions: set B without the correction, and KKR-theta_D vs mixture-theta_D.  Writes numbers/validation.json."""
import numpy as np
import matplotlib.pyplot as plt
import style
import common as C

B = C.load_baseline()


def panel(ax, df, col, title, color_vsc=False):
    m = C.metrics(df[col], df.Tc_exp)
    lim = (0, max(df.Tc_exp.max(), df[col].max()) * 1.08)
    for caution, mk, lab in [(False, 'o', 'single-phase bcc'), (True, '^', 'multiphase / caution')]:
        g = df[df.caution == caution]
        if color_vsc:
            for vsc, c in [(False, '0.35'), (True, style.C_B)]:
                gg = g[g.has_VSc == vsc]
                ax.scatter(gg[col], gg.Tc_exp, marker=mk, s=16, facecolors=c if vsc else 'none', edgecolors=c, lw=0.7,
                           label=f'{lab}{", with V or Sc" if vsc else ""}', zorder=3)
        else:
            ax.scatter(g[col], g.Tc_exp, marker=mk, s=16, facecolors='none', edgecolors='0.35', lw=0.7,
                       label=f'{lab} ($n$={len(g)})', zorder=3)
    x = np.linspace(*lim, 50)
    ax.plot(x, m['slope'] * x + m['intercept'], '-', color=style.C_FIT, lw=1.2, label='linear fit (this panel)')
    ax.text(0.03, 0.97, f"fit: $T_c^{{\\rm exp}} = {m['slope']:.2f}\\,T_c^{{\\rm calc}} {'+' if m['intercept'] >= 0 else '-'} {abs(m['intercept']):.2f}$ K\n"
                        f"$\\tau$ = {m['tau']:.2f}, $r$ = {m['r']:.2f}, MAE = {m['MAE']:.2f} K\n"
                        f"pairs ordered: {100*m['pair_acc_all']:.0f}% (all), {100*m['pair_acc_1K']:.0f}% ($|\\Delta T_c^{{\\rm exp}}|>1$ K)",
            transform=ax.transAxes, va='top', fontsize=6.5)
    ax.set(xlim=lim, ylim=lim, xlabel='calculated $T_c$ (K)', title=title)
    return m


# --- main figure -----------------------------------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(style.DOUBLE, 3.1))
mA = panel(ax[0], B[B.in_A], 'Tc_mu0.1', '(a) set A: $T_c(\\mu^*=0.1)$')
mB = panel(ax[1], B[B.in_B], 'Tc_corr2', '(b) set B: $T_c^{\\rm corr}$')
ax[0].set_ylabel('experimental $T_c$ (K)')
for a in ax:
    a.legend(loc='lower right')
plt.tight_layout()
style.save(fig, 'fig_validation')

# --- SI: set B with / without the correction, and KKR theta_D --------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(style.DOUBLE, 2.7))
mAk = panel(ax[0], B[B.in_A], 'Tc_kkr_theta_mu0.1', '(a) set A, $\\theta_D^{\\rm KKR}$')
mBp = panel(ax[1], B[B.in_B], 'Tc_mu0.1', '(b) set B, no correction', color_vsc=True)
mBc = panel(ax[2], B[B.in_B], 'Tc_corr2', '(c) set B, $T_c^{\\rm corr}$', color_vsc=True)
ax[0].set_ylabel('experimental $T_c$ (K)')
h, l = ax[1].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.1))
plt.tight_layout()
style.save(fig, 'figS_validation_variants')

d = B.drop_duplicates('kkr_dir')

# --- numbers --------------------------------------------------------------------------------------------------------
sub = {}
for name, sel, col in [('A_mu0.1', B.in_A, 'Tc_mu0.1'), ('A_mu0.2', B.in_A, 'Tc_mu0.2'),
                       ('A_single', B.in_A & ~B.caution, 'Tc_mu0.1'), ('A_kkr_theta', B.in_A, 'Tc_kkr_theta_mu0.1'),
                       ('B_corr2', B.in_B, 'Tc_corr2'), ('B_plain', B.in_B, 'Tc_mu0.1'),
                       ('B_single_corr2', B.in_B & ~B.caution, 'Tc_corr2'),
                       ('B_VSc_corr2', B.in_B & B.has_VSc, 'Tc_corr2'), ('B_VSc_plain', B.in_B & B.has_VSc, 'Tc_mu0.1'),
                       ('AB_corr2', B.in_A | B.in_B, 'Tc_corr2')]:
    sub[name] = C.metrics(B.loc[sel, col], B.loc[sel, 'Tc_exp'])
topA = B[B.in_A].nlargest(5, 'Tc_exp')[['name', 'Tc_exp', 'Tc_mu0.1', 'author', 'caution']]
topB = B[B.in_B].nlargest(5, 'Tc_exp')[['name', 'Tc_exp', 'Tc_corr2', 'author', 'caution']]
cnt = dict(rows_total=len(B), A_rows=int(B.in_A.sum()), A_comps=int(B[B.in_A].kkr_dir.nunique()),
           A_caution=int((B.in_A & B.caution).sum()), B_rows=int(B.in_B.sum()), B_comps=int(B[B.in_B].kkr_dir.nunique()),
           B_caution=int((B.in_B & B.caution).sum()), both=int((B.in_A & B.in_B).sum()),
           B_with_VSc=int((B.in_B & B.has_VSc).sum()), B_with_V=int((B.in_B & (B.V > 0)).sum()),
           B_with_Sc=int((B.in_B & (B.Sc > 0)).sum()), A_papers=int(B[B.in_A].doi.nunique()), B_papers=int(B[B.in_B].doi.nunique()),
           theta_kkr_range=[float(d.theta_kkr.min()), float(d.theta_kkr.max())],
           theta_mix_range=[float(d.theta_mix.min()), float(d.theta_mix.max())],
           n_theta_kkr_below_200=int((d.theta_kkr < 200).sum()), n_comps=len(d))
C.save_numbers('validation', dict(metrics=sub, counts=cnt, top_exp_A=topA.to_dict('records'), top_exp_B=topB.to_dict('records')))
print(cnt)
for k, v in sub.items():
    print(k, {kk: round(vv, 3) for kk, vv in v.items()})
