"""Figures + LaTeX table fragments for the step-by-step report on the McMillan-Hopfield matrix element
M_{l,l+1} = int u_l (dV/dr) u_{l+1} dr for one element (default Ta) in Ta34Nb33Hf8Zr14Ti11, SRA (production) vs NRL rerun.
Output: results/eta_rmta_report/fig/*.pdf and results/eta_rmta_report/tab_*.tex
Usage: python scripts/eta_rmta_tests/14_report_figures.py [element]"""
import sys, os, json, numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.special import spherical_jn, spherical_yn
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
from common import fort51_path
from macmillan import parse_fort51, reduce_l_block_radials, group_dosef_by_l, derivative_nonuniform
from macmillan_cutoff import compute_M_surface, compute_M_gg, find_all_nodes
from src.consts import VALENCE_NODES_EXPECTED

EL = sys.argv[1] if len(sys.argv) > 1 else 'Ta'
LABELS = ['Ta', 'Nb', 'Hf', 'Zr', 'Ti']; CONC = [.34, .33, .08, .14, .11]
Z = {'Ta': 73, 'Nb': 41, 'Hf': 72, 'Zr': 40, 'Ti': 22}[EL]
JASK = {'Ta': (0.9, 50.7, 102.4), 'Nb': (4.7, 55.3, 102.0), 'Hf': (1.7, 71.4, 87.5), 'Zr': (7.1, 79.3, 89.7), 'Ti': (4.9, 45.7, 75.6)}
JASK_N = {'Ta': 17.2, 'Nb': 18.8, 'Hf': 16.2, 'Zr': 17.2, 'Ti': 26.6}
RUNS = {'SRA': 'results/comparison.sra.ew0.6.bz10-10-20/Ta34Nb33Hf8Zr14Ti11', 'NRL': 'results/test.nrl.ew0.6/TNHZT_a3.36_nrl'}
OUT = 'results/eta_rmta_report'; FIG = os.path.join(OUT, 'fig'); os.makedirs(FIG, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.grid': True, 'grid.alpha': 0.3, 'legend.fontsize': 8, 'figure.dpi': 120})
COL = {'SRA': 'C3', 'NRL': 'C0'}; CH = [('s', 'p', 'sp', 0), ('p', 'd', 'pd', 1), ('d', 'f', 'df', 2)]

def load(wd, comp):
    d = parse_fort51(fort51_path(wd)); ef = float(np.atleast_1d(d['ef'])[0]); b = d['components'][comp]
    x = np.array(b['xr'], float)[:-1]; v = np.array(b['v3'], float)[:-1]
    raw = {int(k): np.array(vv[:-1], float) for k, vv in b['rstr_real_ef'].items()}
    U = reduce_l_block_radials(raw, int(b['mxlcmp']), x, 'mean', 'none'); dos = group_dosef_by_l(b['dosef'], int(b['mxlcmp']))
    return dict(x=x, v=v, U=U, dos=dos, N=float(sum(b['dosef'])), ef=ef, R=x[-1], vR=v[-1])

def phase_shift(u, x, ef, l, vmtz):
    lnx = np.log(x); R = x[-1]; k = np.sqrt(ef - vmtz); up = CubicSpline(lnx, u)(lnx[-1], 1) / R
    j, y = spherical_jn(l, k * R), spherical_yn(l, k * R); jp, yp = spherical_jn(l, k * R, derivative=True), spherical_yn(l, k * R, derivative=True)
    A, B = np.linalg.solve([[R * j, R * y], [j + k * R * jp, y + k * R * yp]], [u[-1], up])
    return np.arctan2(-B, A), A, B, k

D = {k: load(wd, LABELS.index(EL)) for k, wd in RUNS.items()}
res = {}
for k, d in D.items():
    x, v, U, ef, R = d['x'], d['v'], d['U'], d['ef'], d['R']
    dv_grad = derivative_nonuniform(x, v); dv_spl = CubicSpline(np.log(x), v)(np.log(x), 1) / x
    r = dict(N=d['N'], ef=ef, R=R, vR=d['vR'], dos=d['dos'], dv_grad=dv_grad, dv_spl=dv_spl, chan={})
    for a, b, ch, l in CH:
        u1, u2 = U[a], U[b]; I_spl = u1 * dv_spl * u2; I_grad = u1 * dv_grad * u2
        # cumulative from the outside: M(r0) = int_{r0}^{R} I dr
        cum = lambda I: np.concatenate([[0.0], np.cumsum(0.5 * (I[1:] + I[:-1]) * np.diff(x))])  # from 0 to x_i
        Mtot_spl = cum(I_spl); Mtot_grad = cum(I_grad)
        M_r0_spl = Mtot_spl[-1] - Mtot_spl; M_r0_grad = Mtot_grad[-1] - Mtot_grad
        n1, n2 = find_all_nodes(x, u1), find_all_nodes(x, u2)
        vn = VALENCE_NODES_EXPECTED.get(EL, {})
        rc1 = n1[-1] if (n1 and len(n1) > vn.get(l, 0)) else 0.0; rc2 = n2[-1] if (n2 and len(n2) > vn.get(l + 1, 0)) else 0.0
        rcut = max(rc1, rc2); icut = int(np.searchsorted(x, rcut))
        M_A = float(np.trapz(I_grad[icut:], x[icut:])) if rcut > 0 else float(np.trapz(I_grad, x))
        M_B = compute_M_surface(u1, u2, x, v, ef, l); M_G = compute_M_gg(u1, u2, x, v, ef, l)
        pref = 2 * (l + 1) / ((2 * l + 1) * (2 * l + 3)) * d['dos'][a] * d['dos'][b] / d['N'] * 1e3
        r['chan'][ch] = dict(l=l, I_spl=I_spl, I_grad=I_grad, M_r0_spl=M_r0_spl, M_r0_grad=M_r0_grad, nodes1=n1, nodes2=n2, rcut=rcut,
                             M_full_spl=float(np.trapz(I_spl, x)), M_full_grad=float(np.trapz(I_grad, x)), M_A=M_A, M_B=M_B, M_G=M_G, pref=pref,
                             eta=dict(full=pref * np.trapz(I_spl, x)**2, A=pref * M_A**2, B=pref * M_B**2, G=pref * M_G**2))
    # phase shifts, both conventions
    r['delta_G'] = {l: phase_shift(U['spdf'[l]], x, ef, l, 0.0) for l in range(4)}
    r['delta_B'] = {l: phase_shift(U['spdf'[l]], x, ef, l, d['vR']) for l in range(4)}
    res[k] = r

# ---------------- Fig 1: potential and its derivative ----------------
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
for k, d in D.items():
    ax[0].loglog(d['x'], -d['v'], color=COL[k], label=f'$-V(r)$ {k}')
    ax[1].loglog(d['x'], np.abs(res[k]['dv_spl']), color=COL[k], label=f'$|dV/dr|$ {k} (spline)')
x = D['SRA']['x']; ax[0].loglog(x, 2 * Z / x, 'k:', label=f'$2Z/r$, Z={Z}'); ax[1].loglog(x, 2 * Z / x**2, 'k:', label='$2Z/r^2$')
ax[1].loglog(x, np.abs(res['SRA']['dv_grad']), color='C2', lw=0.8, ls='--', label='np.gradient (produkcja)')
for a in ax: a.axvline(D['SRA']['R'], color='gray', lw=0.8); a.set_xlabel('$r$ [$a_0$]')
ax[0].set_ylabel('$-V$ [Ry]'); ax[1].set_ylabel('$|dV/dr|$ [Ry/$a_0$]'); ax[0].legend(); ax[1].legend()
ax[0].set_title(f'{EL}: potencjał MT (zero = poziom MT)'); ax[1].set_title('pochodna radialna')
fig.tight_layout(); fig.savefig(f'{FIG}/f1_potential_{EL}.pdf'); plt.close(fig)

# ---------------- Fig 2: radial functions, linear and log r ----------------
fig, ax = plt.subplots(2, 4, figsize=(9.6, 4.6), sharex='row')
for i, l in enumerate('spdf'):
    for k, d in D.items():
        ax[0, i].plot(d['x'], d['U'][l], color=COL[k], label=k); ax[1, i].semilogx(d['x'], d['U'][l], color=COL[k])
    for n in res['SRA']['chan'][('sp', 'pd', 'df')[min(i, 2)]]['nodes1' if i < 3 else 'nodes2']:
        pass
    nodes = find_all_nodes(D['SRA']['x'], D['SRA']['U'][l])
    for n in nodes: ax[1, i].axvline(n, color='C3', lw=0.6, ls=':')
    ax[0, i].set_title(f'$u_{l}(r)$, {len(nodes)} węzłów (SRA)'); ax[0, i].axhline(0, color='k', lw=0.5); ax[1, i].axhline(0, color='k', lw=0.5)
    ax[1, i].set_xlabel('$r$ [$a_0$] (log)'); ax[0, i].set_xlabel('$r$ [$a_0$]')
ax[0, 0].legend(); ax[0, 0].set_ylabel('$u_l = rR_l$, $\\int u^2 dr = 1$'); ax[1, 0].set_ylabel('$u_l$')
fig.suptitle(f'{EL} w Ta34Nb33Hf8Zr14Ti11: funkcje radialne przy $E_F$; kropkowane = węzły (ostatni węzeł → odcięcie A)', y=1.0)
fig.tight_layout(); fig.savefig(f'{FIG}/f2_radial_{EL}.pdf'); plt.close(fig)

# ---------------- Fig 3: small-r power law ----------------
fig, ax = plt.subplots(1, 4, figsize=(9.6, 2.6))
for i, l in enumerate('spdf'):
    for k, d in D.items():
        x, u = d['x'], np.abs(d['U'][l]); m = (x < 2e-2) & (u > 0)
        ax[i].loglog(x[m], u[m], color=COL[k], label=k)
        mm = (x < 2e-3) & (u > 0); s = np.polyfit(np.log(x[mm]), np.log(u[mm]), 1)[0]
        ax[i].text(0.03, 0.9 - 0.1 * list(D).index(k), f'{k}: $u\\propto r^{{{s:.2f}}}$', transform=ax[i].transAxes, color=COL[k], fontsize=8)
    x = D['SRA']['x']; m = x < 2e-2; ref = np.abs(D['NRL']['U'][l][m][-1]) * (x[m] / x[m][-1])**(i + 1)
    ax[i].loglog(x[m], ref, 'k:', label=f'$r^{{{i+1}}}$'); ax[i].set_title(f'$|u_{l}|$, małe $r$'); ax[i].set_xlabel('$r$ [$a_0$]')
ax[0].legend(loc='lower right'); fig.tight_layout(); fig.savefig(f'{FIG}/f3_smallr_{EL}.pdf'); plt.close(fig)

# ---------------- Fig 4: integrands ----------------
fig, ax = plt.subplots(2, 3, figsize=(9.6, 5.0))
for j, (a, b, ch, l) in enumerate(CH):
    for k in D:
        x = D[k]['x']; I = res[k]['chan'][ch]['I_spl']
        ax[0, j].plot(x, I, color=COL[k], label=k); ax[1, j].plot(x, x * I, color=COL[k], label=k)
    ax[0, j].set_xscale('log'); ax[0, j].set_yscale('symlog', linthresh=1.0); ax[1, j].set_xscale('log'); ax[1, j].set_yscale('symlog', linthresh=0.1)
    ax[0, j].set_title(f'{ch}: $I(r)=u_{a}\\,V\'\\,u_{b}$'); ax[1, j].set_title(f'{ch}: $r\\,I(r)$ (pole pod krzywą w log $r$ = $M$)')
    ax[1, j].set_xlabel('$r$ [$a_0$]'); ax[0, j].axhline(0, color='k', lw=0.5); ax[1, j].axhline(0, color='k', lw=0.5)
    rc = res['SRA']['chan'][ch]['rcut']
    if rc > 0:
        for aa in ax[:, j]: aa.axvline(rc, color='C3', ls='--', lw=0.8)
ax[0, 0].set_ylabel('$I$ [Ry/$a_0^2$] (symlog)'); ax[1, 0].set_ylabel('$r\\,I$ [Ry/$a_0$] (symlog)'); ax[0, 0].legend()
fig.suptitle(f'{EL}: funkcje podcałkowe (przerywana czerwona = odcięcie A, ostatni węzeł)', y=1.0); fig.tight_layout(); fig.savefig(f'{FIG}/f4_integrand_{EL}.pdf'); plt.close(fig)

# ---------------- Fig 5: cumulative integral from the outside ----------------
fig, ax = plt.subplots(1, 3, figsize=(9.6, 3.0))
for j, (a, b, ch, l) in enumerate(CH):
    for k in D:
        x = D[k]['x']; c = res[k]['chan'][ch]
        ax[j].plot(x, c['M_r0_spl'], color=COL[k], label=f'{k}: $\\int_{{r_0}}^{{R}} I\\,dr$')
        ax[j].axhline(c['M_B'], color=COL[k], ls=':', lw=1.0); ax[j].axhline(c['M_G'], color=COL[k], ls='-.', lw=1.0)
    ax[j].plot(D['SRA']['x'], res['SRA']['chan'][ch]['M_r0_grad'], color='C2', lw=0.8, ls='--', label='SRA, np.gradient')
    c = res['SRA']['chan'][ch]
    if c['rcut'] > 0: ax[j].axvline(c['rcut'], color='C3', ls='--', lw=0.8, label='odcięcie A (SRA)')
    ax[j].set_xscale('log'); ax[j].set_yscale('symlog', linthresh=0.5); ax[j].set_xlabel('$r_0$ [$a_0$]'); ax[j].set_title(f'{ch}: $M(r_0)=\\int_{{r_0}}^{{R}} I\\,dr$')
    ax[j].axhline(0, color='k', lw=0.5)
ax[0].set_ylabel('$M(r_0)$ [Ry/$a_0$] (symlog)'); ax[0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(f'{FIG}/f5_cumulative_{EL}.pdf'); plt.close(fig)

# ---------------- Fig 6: phase-shift matching at R_MT ----------------
fig, ax = plt.subplots(1, 4, figsize=(9.6, 2.7))
for l in range(4):
    for k, d in D.items():
        x, u, R = d['x'], d['U']['spdf'[l]], d['R']; dl, A, B, kk = res[k]['delta_G'][l]
        m = x > 0.3 * R; ax[l].plot(x[m], u[m], color=COL[k], label=f'{k} wewnątrz')
        ro = np.linspace(R, 1.8 * R, 100); ax[l].plot(ro, A * ro * spherical_jn(l, kk * ro) + B * ro * spherical_yn(l, kk * ro), color=COL[k], ls='--', label=f'{k} swobodna, $\\delta_{l}$={dl:.2f}')
    ax[l].axvline(D['SRA']['R'], color='gray', lw=0.8); ax[l].set_title(f'$l$={l}'); ax[l].set_xlabel('$r$ [$a_0$]'); ax[l].legend(fontsize=6.5)
ax[0].set_ylabel('$u_l$'); fig.suptitle(f'{EL}: dopasowanie do fali swobodnej $A\\,r j_l(kr)+B\\,r y_l(kr)$ przy $R_{{MT}}$, $k^2=E_F$ (zero MT)', y=1.0)
fig.tight_layout(); fig.savefig(f'{FIG}/f6_phaseshift_{EL}.pdf'); plt.close(fig)

# ---------------- Fig 7: eta per channel, variants vs Jaskiewicz ----------------
fig, ax = plt.subplots(figsize=(7.0, 3.0))
variants = [('Jaśkiewicz 2016', JASK[EL]), ('A: odcięcie (prod., SRA)', [res['SRA']['chan'][c]['eta']['A'] for c in ('sp', 'pd', 'df')]),
            ('pełna całka SRA', [res['SRA']['chan'][c]['eta']['full'] for c in ('sp', 'pd', 'df')]), ('pełna całka NRL', [res['NRL']['chan'][c]['eta']['full'] for c in ('sp', 'pd', 'df')]),
            ('B: powierzchnia SRA', [res['SRA']['chan'][c]['eta']['B'] for c in ('sp', 'pd', 'df')]), ('G: GG+skok MT, SRA', [res['SRA']['chan'][c]['eta']['G'] for c in ('sp', 'pd', 'df')]),
            ('G, NRL', [res['NRL']['chan'][c]['eta']['G'] for c in ('sp', 'pd', 'df')])]
w = 0.25; xs = np.arange(len(variants))
for i, ch in enumerate(('sp', 'pd', 'df')):
    vals = [np.clip(v[i], 0, 400) for _, v in variants]; ax.bar(xs + (i - 1) * w, vals, w, label=ch)
    for xx, v in zip(xs, [v[i] for _, v in variants]):
        if v > 400: ax.text(xx + (i - 1) * w, 395, f'{v:.0f}', ha='center', va='top', fontsize=6, rotation=90)
ax.set_xticks(xs); ax.set_xticklabels([n for n, _ in variants], rotation=20, ha='right'); ax.set_ylabel('$\\eta_l$ [mRy/$a_0^2$]'); ax.set_ylim(0, 400); ax.legend()
ax.set_title(f'{EL}: udziały kanałów w η (słupki >400 ucięte, wartość podpisana)'); fig.tight_layout(); fig.savefig(f'{FIG}/f7_eta_bars_{EL}.pdf'); plt.close(fig)

# ---------------- LaTeX tables ----------------
def tex_tab_phase():
    L = ['\\begin{tabular}{l rrrr rrr}', '\\toprule', 'bieg & $\\delta_s$ & $\\delta_p$ & $\\delta_d$ & $\\delta_f$ & $\\sin^2(\\delta_p-\\delta_s)$ & $\\sin^2(\\delta_d-\\delta_p)$ & $\\sin^2(\\delta_f-\\delta_d)$ \\\\', '\\midrule']
    for k in D:
        for conv, key in (('zero MT ($k^2=E_F$, G)', 'delta_G'), ('$k^2=E_F-v(R^-)$ (B)', 'delta_B')):
            dl = [res[k][key][l][0] for l in range(4)]
            L.append(f'{k}, {conv} & ' + ' & '.join(f'{v:.3f}' for v in dl) + ' & ' + ' & '.join(f'{np.sin(dl[l+1]-dl[l])**2:.3f}' for l in range(3)) + ' \\\\')
    L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)
def tex_tab_M():
    L = ['\\begin{tabular}{l l rrrrr}', '\\toprule', 'bieg & kanał & $M$ pełna (spline $V\'$) & $M$ pełna (np.gradient) & $M_A$ odcięcie & $M_B$ powierzchnia & $M_G$ GG \\\\', '\\midrule']
    for k in D:
        for a, b, ch, l in CH:
            c = res[k]['chan'][ch]
            L.append(f'{k} & {ch} & {c["M_full_spl"]:.4f} & {c["M_full_grad"]:.4f} & {c["M_A"]:.4f} & {c["M_B"]:.4f} & {c["M_G"]:.4f} \\\\')
    L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)
def tex_tab_eta():
    L = ['\\begin{tabular}{l rrrr}', '\\toprule', 'wariant & $\\eta_{sp}$ & $\\eta_{pd}$ & $\\eta_{df}$ & $\\eta$ \\\\', '\\midrule']
    def row(n, v): return f'{n} & ' + ' & '.join(f'{x:.1f}' for x in v) + f' & {sum(v):.1f} \\\\'
    L.append(row('Jaśkiewicz 2016, Tab.~1', JASK[EL]))
    for k in D:
        for vn, lab in (('full', 'pełna całka'), ('A', 'A: odcięcie'), ('B', 'B: powierzchnia'), ('G', 'G: GG + skok MT')):
            L.append(row(f'{k}, {lab}', [res[k]['chan'][c]['eta'][vn] for c in ('sp', 'pd', 'df')]))
    L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)
def tex_tab_dos():
    L = ['\\begin{tabular}{l rrrr r r}', '\\toprule', 'bieg & $n_s$ & $n_p$ & $n_d$ & $n_f$ & $N$ (2 spiny) & $E_F$ [Ry] \\\\', '\\midrule']
    for k in D:
        d = D[k]; L.append(f'{k} & ' + ' & '.join(f'{d["dos"][l]:.3f}' for l in 'spdf') + f' & {2*d["N"]:.2f} & {d["ef"]:.4f} \\\\')
    L.append(f'Jaśkiewicz 2016 & & & & & {JASK_N[EL]:.1f} & \\\\'); L += ['\\bottomrule', '\\end{tabular}']; return '\n'.join(L)
for n, f in (('phase', tex_tab_phase), ('M', tex_tab_M), ('eta', tex_tab_eta), ('dos', tex_tab_dos)):
    open(f'{OUT}/tab_{n}_{EL}.tex', 'w').write(f())
summary = {k: dict(N2=2 * D[k]['N'], ef=D[k]['ef'], R=D[k]['R'], vR=D[k]['vR'],
                   nodes={ch: (res[k]['chan'][ch]['nodes1'], res[k]['chan'][ch]['nodes2'], res[k]['chan'][ch]['rcut']) for ch in ('sp', 'pd', 'df')},
                   M={ch: {m: res[k]['chan'][ch][m] for m in ('M_full_spl', 'M_full_grad', 'M_A', 'M_B', 'M_G')} for ch in ('sp', 'pd', 'df')},
                   eta={ch: res[k]['chan'][ch]['eta'] for ch in ('sp', 'pd', 'df')}, delta_G=[res[k]['delta_G'][l][0] for l in range(4)]) for k in D}
json.dump(summary, open(f'{OUT}/summary_{EL}.json', 'w'), indent=1, default=float)
print(json.dumps({k: dict(N2=round(s['N2'], 2), R=round(s['R'], 4), vR=round(s['vR'], 4), nodes={c: (len(v[0]), len(v[1]), round(v[2], 3)) for c, v in s['nodes'].items()},
                          eta={c: {m: round(x, 1) for m, x in e.items()} for c, e in s['eta'].items()}, delta=[round(x, 3) for x in s['delta_G']]) for k, s in summary.items()}, indent=1))
