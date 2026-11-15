"""Builder for notebooks/OptGL_B_corr2_Runs.ipynb — set B (TiNbZrHfTaMoWYVSc) GL optimisations, target Tc_corr (corr2),
mincomp 2–10. Layout follows OptGL_mu0.1_Runs.ipynb (set A) sections 1–9.
python3 build_OptGL_B_corr2_Runs.py OptGL_B_corr2_Runs.ipynb && jupyter nbconvert --to notebook --execute --inplace OptGL_B_corr2_Runs.ipynb"""
import sys
import nbformat as nbf

OUT = sys.argv[1]
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip('\n')))

# bcc/fcc/hcp + ΔH of the champions (same cell in build_OptGL_mu0.1_Runs.py)
TD_CODE = r"""
import sys
sys.path.insert(0, '../scripts')
from src.elements import ELEMENTS as ELEMENT_DATA

TD_LIST = Path('../slurm/compositions/td_champions_gl.csv')
EL_TD = Path('../thermodynamic/elements.sra.ew0.6.topola')
RY_MEV = 13.605693122994 * 1000
STRUCT = ('bcc', 'fcc', 'hcp')
Z_TO_EL = {22: 'Ti', 41: 'Nb', 40: 'Zr', 72: 'Hf', 73: 'Ta', 42: 'Mo', 74: 'W', 39: 'Y', 75: 'Re', 44: 'Ru', 23: 'V', 21: 'Sc'}
EL_TD_WIDE = Path('../thermodynamic.ew1.0/elements.sra.topola')
REF_WIDE_EW = {'W', 'Re'}    # pure W and Re are not converged at ew 0.6 (valence band below the contour) -> ew 1.0 scans


def kkr_input_composition(inp):
    # concentrations exactly as AkaiKKR used them: the rounded values of the .inp, normalised as in specx.f.  Total
    # energies are ~1e4 Ry, so ΔH must use these and not the unrounded composition (5e-7 × 3e4 Ry ≈ 200 meV).
    comp, block = {}, False
    for line in Path(inp).read_text().splitlines():
        if 'anclr' in line and line.lstrip().startswith('c'):
            block = True
            continue
        if block:
            if line.lstrip().startswith('c'):
                break
            f = line.split()
            if len(f) == 2:
                comp[Z_TO_EL[int(round(float(f[0])))]] = float(f[1])
    norm = sum(comp.values())
    return {el: x / norm for el, x in comp.items()}


def read_td(d):
    s = json.loads((d / 'structure_summary.json').read_text())
    t = {f'E_{st}': s[st]['energy_per_atom_ry'] for st in STRUCT}
    a_scan = [float(re.findall(r'_(\d+\.\d+)', f.name)[-1]) for f in (d / 'bcc').glob('*.inp')]
    t.update(most_stable=s['most_stable'], a_bcc=s['bcc']['lattice_constant_bohr'], B_bcc=s['bcc']['bulk_modulus_gpa'],
             dE_fcc=(t['E_fcc'] - t['E_bcc']) * RY_MEV, dE_hcp=(t['E_hcp'] - t['E_bcc']) * RY_MEV,
             comp=kkr_input_composition(sorted((d / 'bcc').glob('*.inp'))[0]))
    t['a_in_scan'] = min(a_scan) < t['a_bcc'] < max(a_scan)
    return t


def element_td(el):
    return read_td((EL_TD_WIDE if el in REF_WIDE_EW else EL_TD) / f'{el}1.0000')


def e_bcc_ref(el):
    return element_td(el)['E_bcc'] if el in REF_WIDE_EW else ELEMENT_DATA[el].ebulk


def element_refs(elements):
    rows = []
    for el in elements:
        t = element_td(el)
        rows.append({'el': el, 'gs_KKR': min(STRUCT, key=lambda st: t[f'E_{st}']),
                     'E_fcc−E_bcc': t['dE_fcc'], 'E_hcp−E_bcc': t['dE_hcp'], 'a_bcc': t['a_bcc'], 'B_bcc': t['B_bcc'],
                     'a in scan': t['a_in_scan'], 'ew': 1.0 if el in REF_WIDE_EW else 0.6, 'E_bcc−ref': (t['E_bcc'] - e_bcc_ref(el)) * RY_MEV,
                     'E_gs−E_bcc': min(0.0, t['dE_fcc'], t['dE_hcp'])})
    return pd.DataFrame(rows).set_index('el')


def champions_td(prefix, champ=None):
    # rows <prefix>_mc<m> of TD_LIST; champ (index mincomp) = the notebook's champions, checked against the TD composition
    lst = pd.read_csv(TD_LIST)
    lst = lst[lst.tag.str.startswith(prefix + '_mc')]
    rows = []
    for _, r in lst.iterrows():
        mc = int(r.tag.rsplit('_mc', 1)[1])
        t = read_td(Path('..') / r.outdir / r.dirname)
        c = {el: x for el, x in t['comp'].items() if x > 0}
        if champ is not None:
            dev = max(abs(c.get(el, 0.0) - champ.loc[mc, el]) for el in ELEMENTS)
            assert dev < 1e-3, f'{r.tag}: TD composition differs from the notebook champion by {dev:.4f}'
        e_ref = sum(x * e_bcc_ref(el) for el, x in c.items())
        e_min = min(t[f'E_{st}'] for st in STRUCT)
        rows.append({'mincomp': mc, 'Tc': r.Tc, 'champion': comp_str(pd.Series(c)), 'most_stable': t['most_stable'],
                     'E_fcc−E_bcc': t['dE_fcc'], 'E_hcp−E_bcc': t['dE_hcp'], 'a_bcc': t['a_bcc'], 'B_bcc': t['B_bcc'],
                     'a in scan': t['a_in_scan'],
                     'ΔH_bcc': (t['E_bcc'] - e_ref) * RY_MEV,
                     'ΔH_gs': (e_min - e_ref) * RY_MEV - sum(x * REF.loc[el, 'E_gs−E_bcc'] for el, x in c.items())})
    return pd.DataFrame(rows).set_index('mincomp').sort_index()


def td_plot(series):
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
    for (name, t), m in zip(series.items(), 'osd^'):
        ax[0].plot(t.index, t['E_fcc−E_bcc'], m + '-', label=f'{name}: fcc')
        ax[0].plot(t.index, t['E_hcp−E_bcc'], m + '--', label=f'{name}: hcp')
        ax[1].plot(t.index, t['ΔH_bcc'], m + '-', label=f'{name}: ΔH_bcc')
        ax[1].plot(t.index, t['ΔH_gs'], m + ':', label=f'{name}: ΔH_gs')
    ax[0].set(xlabel='mincomp', ylabel='E − E_bcc [meV/atom]', title='Structure energy above bcc (champions)')
    ax[1].set(xlabel='mincomp', ylabel='ΔH [meV/atom]', title='Mixing enthalpy of the champions')
    ax[1].axhline(0, c='k', lw=.8)
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.show()


REF = element_refs(['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru', 'V', 'Sc'])
REF_FMT = {'E_fcc−E_bcc': '{:.1f}', 'E_hcp−E_bcc': '{:.1f}', 'a_bcc': '{:.3f}', 'B_bcc': '{:.0f}', 'ew': '{:.1f}', 'E_bcc−ref': '{:.1f}',
           'E_gs−E_bcc': '{:.1f}'}
TD_FMT = {'Tc': '{:.2f}', 'E_fcc−E_bcc': '{:.1f}', 'E_hcp−E_bcc': '{:.1f}', 'a_bcc': '{:.3f}', 'B_bcc': '{:.0f}',
          'ΔH_bcc': '{:+.1f}', 'ΔH_gs': '{:+.1f}'}
"""

md(r"""
# Skład B (TiNbZrHfTaMoWYVSc): optymalizacje GL z celem `Tc_corr` (corr2), mincomp 2–10

- **Runy:** `opt.ratios.gl/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.corr2.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp<m>.r0.05.max0.7.face0.1`
  (Topola, 2026-11-02/03).
- **Baza startowa:** `databases/sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.csv`, 1800 losowych składów (1712 w granicach
  optymalizatora).
- **Ustawienia:** wariant GL (G + Lloyd), θ_D z reguły mieszaniny, kara Stonera wyłączona, β = 2, `max_composition = 0.7`.
- **Cel `Tc_corr` (corr2, `consts.py` od commita 7aa080e):** McMillan z μ\* = 0.1 i dwiema poprawkami:
  - θ_D fazy bcc dla Sc (215 K) w regule mieszaniny, λ przeskalowane o (θ_mix/θ_corr)²;
  - sprzężenie elektron–paramagnon tylko na V, α_V = 0.4.

  Dla składów bez V i Sc `Tc_corr2 == Tc_mu0.1`.
- **Liczenie składników:** skład liczy się dla runu tylko wtedy, gdy ma co najmniej mincomp pierwiastków o udziale > 0.05.

Układ notebooka jest taki sam jak w `OptGL_mu0.1_Runs.ipynb` (skład A, sekcje 1–9). Dla każdego punktu liczone są ponownie
`Tc_corr2`, λ_corr, λ_es i θ_corr (przez `process_kkr.compute_tc_corrected`, z assertem zgodności z wartością zapisaną
w runie). Porównanie corr2 z wariantami plain/corr (tylko mc 2/3/7) jest w `OptGL_B_corr_Runs.ipynb`.
Notebook nie zapisuje plików graficznych.
""")

code(r"""
from pathlib import Path
import json, re, sys, tarfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

BASE = Path('../opt.ratios.gl')
TARGET = 'Tc_corr2'
RATIO = 0.05
ELEMENTS = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'V', 'Sc']
EL_COLORS = dict(zip(ELEMENTS, plt.cm.tab10.colors))
MCS = list(range(2, 11))
pd.set_option('display.width', 250, 'display.max_columns', 40, 'display.max_rows', 200)

sys.path.insert(0, '../scripts')
import src.process_kkr as pk
assert (dict(pk.TC_CORR_THETA_BCC_K), dict(pk.TC_CORR_SF_ALPHA)) == ({'Sc': 215.0}, {'V': 0.4}), \
    'consts.py zmienione od runów corr2'


def run_dir(mc, tag='corr2.mu0.1'):
    return BASE / f'sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.{tag}.acqbeta2.stonerbeta0.mixdebye.mincomp{mc}.r0.05.max0.7.face0.1'
""")

code(r"""
def parse_name(name):
    return {el: float(x) for el, x in re.findall(r'([A-Z][a-z]?)(\d+\.\d+)', name)}


def n_comp(row):
    return int(sum(row.get(el, 0.0) > RATIO for el in ELEMENTS))


def comp_str(row, thr=0.005):
    parts = [(el, row[el]) for el in ELEMENTS if row.get(el, 0.0) > thr]
    return ' '.join(f'{el}{100*v:.0f}' for el, v in sorted(parts, key=lambda p: -p[1]))


def iter_results(it_dir):
    # (results, selection_info) z computation.tar albo z rozpakowanego computation/
    tf = it_dir / 'computation.tar'
    if tf.exists():
        with tarfile.open(tf) as t:
            sel, res = {}, {}
            for m in t.getmembers():
                key = m.name.rstrip('/').split('/')[-2] if '/' in m.name.rstrip('/') else None
                if m.name.endswith('results.json'):
                    res[key] = json.load(t.extractfile(m))
                elif m.name.endswith('selection_info.json'):
                    sel[key] = json.load(t.extractfile(m))
            for k, r in res.items():
                yield r, sel.get(k, {})
        return
    for c in (it_dir / 'computation').glob('*'):
        rf, sf = c / 'results.json', c / 'selection_info.json'
        if rf.exists():
            yield json.loads(rf.read_text()), (json.loads(sf.read_text()) if sf.exists() else {})


# punkty startowe (te same we wszystkich runach): Tc_corr2 z runu corr2, Tc_mu0.1 z runu plain mc2
_init = pd.concat([
    pd.read_csv(run_dir(2) / 'init_tc_comparison.csv').set_index('name')['Tc_corr_computed'].rename('Tc_corr2'),
    pd.read_csv(run_dir(2, 'mu0.1') / 'init_tc_comparison.csv').set_index('name')['Tc_mu0.1_computed'].rename('Tc_mu0.1')],
    axis=1)
assert _init.notna().all().all()
_init = _init.reset_index()
INIT = pd.concat([pd.DataFrame([parse_name(n) for n in _init['name']]).reindex(columns=ELEMENTS).fillna(0.0),
                  _init[['Tc_mu0.1', 'Tc_corr2']]], axis=1)
INIT['Tc'] = INIT[TARGET]
INIT['iteration'] = 0
INIT['source'] = 'init'


def load_run(mc):
    d = run_dir(mc)
    params = json.loads((d / 'parameters.json').read_text())
    assert params['min_components'] == mc
    rows = []
    for it_dir in d.glob('iteration_*'):
        it = int(it_dir.name.split('_')[1])
        for r, s in iter_results(it_dir):
            c = pk.compute_tc_corrected(r, {el: r.get(el, 0.0) for el in ELEMENTS})
            assert abs(c['Tc_corr'] - r['Tc_corr']) < 1e-8, (mc, it)
            row = {el: r.get(el, 0.0) for el in ELEMENTS}
            row.update(iteration=it, source=s.get('source', '?'), Tc=c['Tc_corr'], Tc_corr2=c['Tc_corr'],
                       **{'Tc_mu0.1': r.get('Tc_mu0.1')},
                       lam=r.get('lambda'), lam_corr=c['lambda_corr'], lam_es=c['lambda_es'],
                       theta=r.get('thetaDB_K'), theta_corr=c['theta_corr'], eta=r.get('eta_mix'), VEC=r.get('VEC'),
                       pred=s.get('pred_target'), pred_std=s.get('pred_target_std'), dist=s.get('composition_distance'))
            rows.append(row)
    opt = pd.DataFrame(rows).sort_values('iteration').reset_index(drop=True)
    df = pd.concat([INIT, opt], ignore_index=True)
    df['n_comp'] = df.apply(n_comp, axis=1)
    df['valid'] = df['n_comp'] >= mc
    df['comp'] = df.apply(comp_str, axis=1)
    df['mincomp'] = mc
    df['run'] = f'mc{mc}'
    log = (d / 'optimization_log.txt').read_text()
    m = re.search(r'EARLY STOP after iteration (\d+)', log)
    status = f'early stop @ {m.group(1)}' if m else 'bez early stop'
    gaps = sorted(set(range(1, int(opt.iteration.max()) + 1)) - set(opt.iteration.unique()))
    if gaps:
        status += f', brak iteracji {gaps}'
    return dict(mc=mc, label=f'mc{mc}', params=params, df=df, status=status,
                n_iter=int(opt.iteration.max()), n_done=int(opt.iteration.nunique()))


runs = [load_run(mc) for mc in MCS]
ALL = pd.concat([r['df'] for r in runs], ignore_index=True)
for r in runs:
    d = r['df']
    print(f"{r['label']:>5}: init={int((d.iteration == 0).sum())} (valid {int(((d.iteration == 0) & d.valid).sum())}), "
          f"opt={int((d.iteration > 0).sum())}, iteracje={r['n_done']} (ostatnia {r['n_iter']}), {r['status']}")
""")

md("## 1. Tabela zbiorcza: najlepszy poprawny skład każdego runu")
code(r"""
def summary(rs):
    out = []
    for r in rs:
        d = r['df']; v = d[d.valid]
        i0, io = v[v.iteration == 0], v[v.iteration > 0]
        best = v.loc[v.Tc.idxmax()]
        out.append({'mincomp': r['mc'], 'iters': r['n_done'], 'status': r['status'],
                    'n_init_valid': len(i0), 'init_best': i0.Tc.max(), 'opt_best': io.Tc.max(),
                    'gain_K': io.Tc.max() - i0.Tc.max(), 'best_iter': int(best.iteration),
                    'frac_new_valid': d[d.iteration > 0].valid.mean(),
                    'champion': best.comp, 'n_el': int(best.n_comp), 'Tc_mu0.1': best['Tc_mu0.1'],
                    'lam': best.lam, 'lam_corr': best.lam_corr, 'lam_es': best.lam_es,
                    'theta': best.theta, 'theta_corr': best.theta_corr, 'VEC': best.VEC})
    return pd.DataFrame(out).set_index('mincomp')

FMT = {'init_best': '{:.2f}', 'opt_best': '{:.2f}', 'gain_K': '{:+.2f}', 'frac_new_valid': '{:.0%}', 'Tc_mu0.1': '{:.2f}',
       'lam': '{:.2f}', 'lam_corr': '{:.2f}', 'lam_es': '{:.3f}', 'theta': '{:.0f}', 'theta_corr': '{:.0f}', 'VEC': '{:.2f}'}
summ = summary(runs)
summ.style.format(FMT, na_rep='—')
""")

md("## 2. Zbieżność: najlepszy dotąd (tylko poprawne) i wszystkie nowe punkty w iteracji")
code(r"""
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
cmap = plt.cm.viridis(np.linspace(0, 0.9, len(runs)))
for r, c in zip(runs, cmap):
    d = r['df']; v = d[d.valid]
    its = np.arange(0, r['n_iter'] + 1)
    ax[0].step(its, [v[v.iteration <= i].Tc.max() for i in its], where='post', color=c, lw=2, label=r['label'])
    io = d[d.iteration > 0]
    ax[1].plot(io.groupby('iteration').Tc.max(), color=c, lw=1.5, label=r['label'])
ax[0].set(xlabel='iteracja', ylabel=f'najlepsze dotąd {TARGET} [K]', title='Najlepszy poprawny skład (iter 0 = start)')
ax[1].set(xlabel='iteracja', ylabel=f'max {TARGET} w partii [K]', title='Najlepszy punkt każdej partii (także niepoprawne)')
for a in ax:
    a.grid(alpha=.3); a.legend(fontsize=8)
plt.tight_layout(); plt.show()
""")
code(r"""
def scatter_runs(rs):
    n = len(rs)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(15, 3.2 * ((n + 1) // 2)), sharey=True, squeeze=False)
    for a, r in zip(axes.flat, rs):
        d = r['df']; io = d[d.iteration > 0]
        a.axhline(d[(d.iteration == 0) & d.valid].Tc.max(), color='gray', ls='--', lw=1, label='najlepszy poprawny start')
        loc = io.source == 'local'
        a.scatter(io[io.valid & loc].iteration, io[io.valid & loc].Tc, s=12, c='C0', label='poprawne, lokalne')
        a.scatter(io[io.valid & ~loc].iteration, io[io.valid & ~loc].Tc, s=28, c='C2', marker='D', label='poprawne, nielokalne')
        a.scatter(io[~io.valid].iteration, io[~io.valid].Tc, s=12, c='C3', marker='x', label=f"< {r['mc']} el.")
        a.set_title(f"{r['label']}: {r['status']}", fontsize=10); a.grid(alpha=.3); a.set_ylabel(TARGET)
    axes.flat[0].legend(fontsize=8)
    for a in list(axes.flat)[n:]:
        a.axis('off')
    plt.tight_layout(); plt.show()

scatter_runs(runs)
""")

md("## 3. Skład ocenianych partii: średni udział pierwiastka w iteracji")
code(r"""
def stack_runs(rs):
    n = len(rs)
    fig, axes = plt.subplots((n + 1) // 2, 2, figsize=(15, 3.2 * ((n + 1) // 2)), sharey=True, squeeze=False)
    for a, r in zip(axes.flat, rs):
        io = r['df'][r['df'].iteration > 0]
        m = io.groupby('iteration')[ELEMENTS].mean()
        a.stackplot(m.index, m.T.values, labels=ELEMENTS, colors=[EL_COLORS[e] for e in ELEMENTS])
        a.set(xlim=(m.index.min(), m.index.max()), ylim=(0, 1), title=r['label'])
    axes.flat[min(1, n - 1)].legend(ncol=5, fontsize=8, loc='upper right')
    for a in list(axes.flat)[n:]:
        a.axis('off')
    plt.tight_layout(); plt.show()

stack_runs(runs)
""")

md("## 4. Mistrzowie: top 5 poprawnych składów każdego runu")
code(r"""
def champ_bar(champ, labels, title):
    fig, ax = plt.subplots(figsize=(10, 0.45 * len(champ) + 1.5))
    left = np.zeros(len(champ))
    for el in ELEMENTS:
        ax.barh(labels, champ[el], left=left, color=EL_COLORS[el], label=el)
        left += champ[el].values
    ax.set(xlabel='udział', title=title); ax.invert_yaxis()
    ax.legend(ncol=10, fontsize=8, loc='lower center', bbox_to_anchor=(.5, 1.06))
    plt.tight_layout(); plt.show()

top = (ALL[ALL.valid].sort_values('Tc', ascending=False)
       .groupby('mincomp').head(5).sort_values(['mincomp', 'Tc'], ascending=[True, False]))
display(top[['mincomp', 'iteration', 'source', 'Tc', 'Tc_mu0.1', 'lam_corr', 'lam_es', 'theta_corr', 'VEC', 'n_comp', 'comp']]
        .style.format({'Tc': '{:.2f}', 'Tc_mu0.1': '{:.2f}', 'lam_corr': '{:.2f}', 'lam_es': '{:.3f}',
                       'theta_corr': '{:.0f}', 'VEC': '{:.2f}'}, na_rep='—').hide(axis='index'))
champ = ALL[ALL.valid].loc[lambda x: x.groupby('mincomp').Tc.idxmax()].set_index('mincomp')
champ_bar(champ, [f'mc{i}  {t:.1f} K' for i, t in zip(champ.index, champ.Tc)], 'Najlepszy poprawny skład każdego runu (GL, corr2)')
""")

md(r"""
## 5. Front Pareto: najlepsze Tc vs liczba pierwiastków

Wszystkie ocenione składy serii mincomp razem (ten sam pipeline i cel); liczone są pierwiastki o udziale > 0.05.
""")
code(r"""
U = ALL.drop_duplicates(subset=ELEMENTS)
front = U.groupby('n_comp').Tc.max()
front_ge = pd.Series({k: U[U.n_comp >= k].Tc.max() for k in range(1, 11)})
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.scatter(U.n_comp + np.random.default_rng(0).uniform(-.15, .15, len(U)), U.Tc, s=5, alpha=.3,
           c=np.where(U.iteration > 0, 'C0', 'gray'))
ax.plot(front.index, front.values, 'o-', c='C1', label='max przy dokładnie k pierwiastkach')
ax.plot(front_ge.index, front_ge.values, 's--', c='C3', label='max przy ≥ k pierwiastkach')
ax.set(xlabel='liczba pierwiastków (> 5 %)', ylabel=f'{TARGET} [K]', title='Tc vs złożoność składu (szare = start, niebieskie = opt)')
ax.grid(alpha=.3); ax.legend(); plt.tight_layout(); plt.show()
front_ge.rename('najlepsze Tc przy ≥ k pierwiastkach').to_frame().T.round(2)
""")

md(r"""
## 6. Fizyka mistrzów: λ vs θ_D, λ vs VEC

`Tc_corr2` zależy od θ_corr (θ_mix z θ_bcc dla Sc) i λ_corr = λ·(θ_mix/θ_corr)², pomniejszonego o λ_es dla V, więc na osiach
są te wielkości. Bez V i Sc θ_corr = θ_mix i λ_corr = λ.
""")
code(r"""
O = ALL[ALL.iteration > 0]
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
sc = ax[0].scatter(O.theta_corr, O.lam_corr, c=O.Tc, s=10, cmap='plasma')
ax[0].scatter(champ.theta_corr, champ.lam_corr, s=120, facecolor='none', edgecolor='k', lw=1.5)
for mc, row in champ.iterrows():
    ax[0].annotate(f'mc{mc}', (row.theta_corr, row.lam_corr), fontsize=8, xytext=(4, 4), textcoords='offset points')
plt.colorbar(sc, ax=ax[0], label=TARGET)
ax[0].set(xlabel='θ_corr [K]', ylabel='λ_corr', title='Wszystkie punkty optymalizacji (kółka = mistrzowie runów)')
sc = ax[1].scatter(O.VEC, O.lam_corr, c=O.Tc, s=10, cmap='plasma')
plt.colorbar(sc, ax=ax[1], label=TARGET)
ax[1].set(xlabel='VEC', ylabel='λ_corr', title='λ_corr vs VEC')
for a in ax:
    a.grid(alpha=.3)
plt.tight_layout(); plt.show()
print(f"corr(λ_corr, Tc) = {O[['lam_corr', 'Tc']].corr().iloc[0, 1]:.3f},  "
      f"corr(θ_corr, Tc) = {O[['theta_corr', 'Tc']].corr().iloc[0, 1]:.3f}")
""")

md("## 7. Jakość surogatu: predykcja CatBoost (przy wyborze) vs wynik KKR")
code(r"""
def surrogate(rs, key='label'):
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    out = []
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(rs)))
    for r, c in zip(rs, cm):
        io = r['df'][(r['df'].iteration > 0) & r['df'].pred.notna()]
        err = io.Tc - io.pred; z = err / io.pred_std
        ax[0].scatter(io.pred, io.Tc, s=8, color=c, label=r[key])
        out.append({key: r[key], 'n': len(io), 'MAE': err.abs().mean(), 'bias': err.mean(),
                    'mean_std': io.pred_std.mean(), 'within_1sigma': (z.abs() < 1).mean(), 'within_2sigma': (z.abs() < 2).mean()})
        ax[1].plot(io.assign(ae=err.abs()).groupby('iteration').ae.mean(), color=c, label=r[key])
    lim = [0, max(r['df'].Tc.max() for r in rs) + 1]
    ax[0].plot(lim, lim, 'k--', lw=1)
    ax[0].set(xlabel='predykcja (średnia zespołu)', ylabel=f'KKR {TARGET}', title='Predykcja vs KKR dla wybranych punktów')
    ax[1].set(xlabel='iteracja', ylabel='MAE partii [K]', title='Błąd surogatu na nowej partii')
    for a in ax:
        a.grid(alpha=.3); a.legend(fontsize=8)
    plt.tight_layout(); plt.show()
    return (pd.DataFrame(out).set_index(key).style.format('{:.2f}')
            .format({'n': '{:d}', 'within_1sigma': '{:.0%}', 'within_2sigma': '{:.0%}'}))

surrogate(runs)
""")

md("## 8. Skąd pochodzą wybrane punkty (źródło akwizycji)")
code(r"""
src = ALL[ALL.iteration > 0].groupby(['mincomp', 'source']).size().unstack(fill_value=0)
src.assign(champion_source=ALL[ALL.valid & (ALL.iteration > 0)].loc[lambda x: x.groupby('mincomp').Tc.idxmax()]
           .set_index('mincomp').source)
""")

md(r"""
## 9. Odległość mistrza od najlepszego punktu startowego

Odległość euklidesowa w przestrzeni składów między mistrzem runu a jego najlepszym poprawnym punktem startowym.
""")
code(r"""
def travel(rs):
    out = []
    for r in rs:
        v = r['df'][r['df'].valid]
        if not (v.iteration == 0).any():
            out.append(dict(run=r['label'], init_best='— (brak poprawnych punktów startowych)'))
            continue
        b0 = v[v.iteration == 0].loc[lambda x: x.Tc.idxmax()]
        b = v.loc[v.Tc.idxmax()]
        out.append(dict(run=r['label'], init_best=b0.comp, init_Tc=b0.Tc, champion=b.comp, champ_Tc=b.Tc,
                        distance=float(np.linalg.norm(b[ELEMENTS].values.astype(float) - b0[ELEMENTS].values.astype(float)))))
    return pd.DataFrame(out).set_index('run').round(3)

travel(runs)
""")

md(r"""
## 10. Specyfika corr2: gdzie działa poprawka i ile jest V i Sc

Lewy panel: `Tc_corr2` vs `Tc_mu0.1` (szare = start, kolor = mincomp, gwiazdki = mistrzowie). Pod przekątną leżą składy z V
(λ_es obniża Tc), nad przekątną składy ze Sc (niższe θ_D, więc wyższe λ); składy bez V i Sc leżą dokładnie na przekątnej.
""")
code(r"""
fig, ax = plt.subplots(figsize=(7.5, 7))
lim = [0, max(ALL[['Tc_mu0.1', 'Tc_corr2']].max()) + 1]
ax.scatter(INIT['Tc_mu0.1'], INIT['Tc_corr2'], s=5, c='0.75', label='start')
for r, c in zip(runs, cmap):
    io = r['df'][r['df'].iteration > 0]
    ax.scatter(io['Tc_mu0.1'], io['Tc_corr2'], s=10, color=c, alpha=.7, label=r['label'])
    b = champ.loc[r['mc']]
    ax.scatter(b['Tc_mu0.1'], b['Tc_corr2'], s=180, marker='*', color=c, edgecolor='k', zorder=5)
ax.plot(lim, lim, 'k:', lw=1)
ax.set(xlim=lim, ylim=lim, xlabel='Tc_mu0.1 [K]', ylabel='Tc_corr2 [K]'); ax.grid(alpha=.3); ax.legend(fontsize=8, ncol=2)
plt.tight_layout(); plt.show()

rows = []
for r in runs:
    io = r['df'][r['df'].iteration > 0]
    rows.append({'mincomp': r['mc'], 'n': len(io), 'z V > 5 % [%]': 100 * (io.V > RATIO).mean(),
                 'ze Sc > 5 % [%]': 100 * (io.Sc > RATIO).mean(), 'śr. V [%]': 100 * io.V.mean(),
                 'śr. Sc [%]': 100 * io.Sc.mean(), 'mistrz V [%]': 100 * champ.loc[r['mc']].V,
                 'mistrz Sc [%]': 100 * champ.loc[r['mc']].Sc,
                 'mistrz Tc_corr2 − Tc_mu0.1 [K]': champ.loc[r['mc']].Tc_corr2 - champ.loc[r['mc']]['Tc_mu0.1']})
pd.DataFrame(rows).set_index('mincomp').round(2)
""")

md(r"""
## 11. Termodynamika mistrzów: bcc / fcc / hcp i ΔH

**Co policzono.** Dla mistrza każdego runu na Topoli policzono `scripts/structure_compare.py`. Skrypt to
`slurm/topola_td_array.sbatch`, a lista składów to `slurm/compositions/td_champions_gl.csv` (2026-11-06/09). Ustawienia KKR są takie
same jak w etapie lattice optymalizacji (sra, PBE, ew 0.6, bzqlty 10, mxl 3, nmag): EOS dla bcc i fcc oraz hcp dla 5
wartości c/a. Wyniki są w `thermodynamic/<run>/<skład>/structure_summary.json`, a pierwiastki w
`thermodynamic/elements.sra.ew0.6.topola/`.

**Dwie serie mistrzów:**
- **B corr2:** mistrzowie tego notebooka, cel `Tc_corr2`;
- **B mu0.1:** mistrzowie runów składu B bez poprawki, cel `Tc_mu0.1`. Są to runy mc2–10
  `...gl.mu0.1.acqbeta2...`, z czego mc 2/3/7 opisuje `OptGL_B_corr_Runs.ipynb`, a mc 4–6 i 8–10 nie mają jeszcze
  osobnego notebooka. Kolumna `Tc` to `Tc_mu0.1` mistrza.

**Definicje:**
- **E_fcc − E_bcc, E_hcp − E_bcc** [meV/atom]: wartość > 0 oznacza, że bcc ma niższą energię.
- **ΔH_bcc** = E_bcc(stop) − Σ c_i E_bcc(i), z referencjami bcc. Referencje to `ebulk` z `src/elements.py` (ew 0.6),
  z wyjątkiem W, który przy ew 0.6 nie jest zbieżny względem okna energii (dno pasma pod konturem; a0 6.26 zamiast
  6.01 bohr) i pochodzi ze skanu ew 1.0 (`thermodynamic.ew1.0/elements.sra.topola/`), jak w Table I pracy.
- **Stężenia c_i** są te z wejścia KKR (6 cyfr, znormalizowane przez AkaiKKR), a nie nieobcięty skład: energie
  całkowite są rzędu 1e4 Ry, więc różnica 5e-7 w c_i między ciężkim (W, Hf, Ta) a lekkim (Ti, Sc, Y) pierwiastkiem
  zmienia ΔH nawet o ~200 meV.
- **ΔH_gs** = min po strukturach E(stop) − Σ c_i E_gs(i), z referencją w stanie podstawowym KKR każdego
  pierwiastka (hcp dla Ti, Zr, Hf, Y, Sc).
""")
code(TD_CODE)
code(r"""
TD_C2 = champions_td('B_corr2', champ)
TD_MU = champions_td('B_mu0.1')
display(REF.loc[ELEMENTS].style.format(REF_FMT, na_rep='—').set_caption('Referencje pierwiastków (Topola, te same ustawienia)'))
display(TD_C2.style.format(TD_FMT, na_rep='—').set_caption('Mistrzowie B corr2 (Tc = Tc_corr2)'))
display(TD_MU.style.format(TD_FMT, na_rep='—').set_caption('Mistrzowie B mu0.1 (Tc = Tc_mu0.1)'))
td_plot({'B corr2': TD_C2, 'B mu0.1': TD_MU})
pd.DataFrame({'ΔH_bcc corr2': TD_C2['ΔH_bcc'], 'ΔH_bcc mu0.1': TD_MU['ΔH_bcc'],
              'różnica mu0.1 − corr2': TD_MU['ΔH_bcc'] - TD_C2['ΔH_bcc'],
              'V mistrza mu0.1 [%]': [100 * parse_name(n).get('V', 0) for n in
                                      pd.read_csv(TD_LIST).set_index('tag').loc[[f'B_mu0.1_mc{m}' for m in TD_MU.index], 'dirname']],
              'Sc mistrza corr2 [%]': 100 * champ.Sc}).round(1)
""")
md(r"""
**Obserwacje.**
- **Struktura.** Wszyscy mistrzowie obu serii są bcc. fcc leży 190–236, a hcp 218–266 meV/atom wyżej.
- **ΔH_bcc jest dodatnie i rośnie z mincomp.** Dla B corr2 wynosi od +72 meV (Nb61Zr39) do +261 meV (mc10).
  Na mc2–3 jest takie samo jak w składzie A (+73, +77 meV). Dla mc4–8 jest wyższe niż w A o 9–24 meV
  (A: +79, +103, +105, +152, +160 meV).
- **Bez poprawki (B mu0.1) dodatkiem jest V:** w mc4–7 zamiast Sc, w mc8 obok Sc. Dla mc4–8 daje to ΔH_bcc wyższe o 35–55 meV niż w corr2
  przy podobnym Tc (każda seria liczona swoim celem, różnica ≤ 0.4 K). Dla mc9–10, gdzie oba warianty zawierają V i Sc,
  różnica spada do −6…−4 meV.
- **Wyjątek: mistrz mc2 bez poprawki (V68Ti32).** To inny basen, z a = 5.8 bohr i ΔH_bcc +66 meV.
- **Referencje.** Wybór stanu podstawowego zamiast bcc dodaje tylko 2–17 meV.
""")

md(r"""
## 12. Wnioski (2026-11-06)

**Przebieg.** Wszystkie 9 runów zakończyło się early stopem (16–37 iteracji). Każdy nowy punkt spełnia mincomp.
Wyszukiwanie jest lokalne: punkty z płaszczyzn sympleksu pojawiają się tylko na początku mc2–5, punkty globalne tylko
w mc9–10. Zysk względem najlepszego poprawnego punktu startowego wynosi +0.3 K dla mc2 (Nb68Zr32 był już w bazie)
i +3.6–6.0 K dla mc3–9. Dla mc10 w bazie nie było żadnego poprawnego punktu startowego.

**Ten sam basen co w składzie A: Nb + Zr, VEC ≈ 4.6.** Mistrzowie mają 47–67 % Nb i 7–40 % Zr, a pozostałe pierwiastki
siedzą na progu ok. 5 %. Dla mc2 i mc3 mistrzowie są praktycznie tacy sami jak w składzie A (Nb61Zr39, Nb54Zr40Ta5).

**Front Pareto** (najlepsze Tc_corr2 przy ≥ k pierwiastkach, k = 2…10): 18.97, 17.88, 16.69, 15.53, 14.31, 12.87,
11.20, 9.37, 7.24 K. Spadek wynosi ok. 1.2–1.9 K na każdy dodatkowy pierwiastek. W składzie A (`OptGL_mu0.1_Runs.ipynb`,
Tc_mu0.1) było to 18.99, 17.92, 16.60, 14.99, 13.52, 11.92, 9.73, 7.22, 4.36 K. Dla k ≥ 4 skład B jest więc wyżej o
0.1–1.5 K, a dla k = 10 o 2.9 K. Dla k = 4–8 różnica pochodzi z dodatku Sc, który z poprawką corr2 podnosi Tc o ok. 0.9 K. Dla k = 9–10 dochodzi to, że w A trzeba było dodać Re i Ru, a w B ich miejsce zajmują V i Sc.

**Fizyka.** Tak jak w A, Tc zależy prawie wyłącznie od λ: corr(λ_corr, Tc) = 0.995. λ_corr mistrza spada z 1.14 (mc2)
do 0.68 (mc10), a θ_corr rośnie z 273 do 296 K, czyli działa przeciwnie i nie kompensuje spadku λ.

**V i Sc.**
- **Sc jest pierwszym dodatkiem** od mc4 i występuje w każdym mistrzu mc4–10 (5–6 %). Daje ok. +0.9 K względem
  Tc_mu0.1: θ_bcc Sc obniża θ_corr o ok. 8 K, a λ_corr rośnie o ok. 5 %.
- **V wchodzi dopiero wtedy, gdy jest wymuszony** (mc9 i mc10, 5–6 %). Przy λ_es ≈ 0.01 kosztuje to tylko 0.05–0.1 K.
- Uwaga: przewaga składu B nad A przy dużym k wynika głównie z poprawki dla Sc. Na 6 jednofazowych stopach ze Sc ta poprawka
  pogarsza zgodność z eksperymentem (`Tc_Baseline_B_corr2.ipynb`), więc tę przewagę trzeba traktować ostrożnie.

**Surogat.** Bias jest dodatni wszędzie (KKR > predykcja), największy dla mc2 (+0.8 K) i maleje do +0.03 K dla mc10.
MAE spada z 0.88 do 0.08 K. Kalibracja σ: 53–76 % punktów mieści się w 1σ.
""")

nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, OUT)
print('written', OUT)
