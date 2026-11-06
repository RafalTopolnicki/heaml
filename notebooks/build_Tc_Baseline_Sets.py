"""Builder for the per-set baseline notebooks (GL baseline vs experiment):
  python3 build_Tc_Baseline_Sets.py A  Tc_Baseline_A.ipynb         (μ* = 0.1 and 0.2; Tc_corr2 == Tc_mu0.1 on A)
  python3 build_Tc_Baseline_Sets.py B  Tc_Baseline_B_corr2.ipynb   (Tc_mu0.1 vs Tc_corr2)
  python3 build_Tc_Baseline_Sets.py AB Tc_Baseline_AB_corr2.ipynb  (Tc_mu0.1 vs Tc_corr2)
then: jupyter nbconvert --to notebook --execute --inplace <notebook>"""
import sys
import nbformat as nbf

SET, OUT = sys.argv[1], sys.argv[2]
assert SET in ('A', 'B', 'AB')
_VSC = r"""
**Gdzie corr2 pomaga, a gdzie nie.** Zmienia się tylko 47 wierszy z V lub Sc; pozostałe (65 wspólnych z A, 9 z Re/Ru) są
identyczne.
- **Wiersze z V lub Sc razem** (n = 47): τ 0.16 → 0.64, źle uporządkowanych par 48 % → 11 %, r 0.16 → 0.75. To jest
  główne źródło poprawy całego zbioru.
- **Ze Sc** (n = 30): τ 0.53 → 0.67, mediana KKR/exp 0.71 → 1.09. Zysk pochodzi głównie z próbek „caution”
  z 20–25 % Sc (Krnel, Kitagawa, Seki), które przed poprawką były zaniżone ok. 1.5×. **6 jednofazowych stopów ze Sc**
  (von Rohr 2018, Jangid 2024) pogarsza się: τ 0.97 → 0.83, MAE 0.56 → 1.08 K, mediana 0.94 → 1.10.
- **Z V** (n = 18): poprawka przestrzeliwuje. Bias zmienia znak z +1.4 na −0.9 K (mediana 1.24 → 0.80), MAE spada
  1.72 → 1.42 K, ale kolejność się nie poprawia (τ 0.29 → 0.25). Najmocniej obniżone są stopy z dużym V: NbV67HfZrTi33
  (V 34 %) 8.9 → 4.8 K przy 7.2 K w eksperymencie, V57Ti29W14 5.6 → 2.4 K przy 5.0 K. α_V = 0.4 dopasowano na liniach
  binarnych z warunkiem kolejności Nb–Zr > Nb–Ti > Ti–V, a dla wieloskładnikowych stopów z V jest za duże.
- **Z Ti ≥ 20 %** (n = 52 w B, 57 w A ∪ B): τ 0.29 → 0.66. Zmieniają się tu jednak tylko te stopy, które zawierają
  również V lub Sc; poprawka nie dotyka Ti.
"""

CONCL = {
    'A': r"""
## Wnioski (2026-11-05)

- **corr2 nie ma wpływu na skład A:** `Tc_corr2 == Tc_mu0.1` dla wszystkich 74 wierszy (assert w sekcji 0). Ten notebook
  jest więc jednocześnie porównaniem dla celu `Tc_corr` na składzie A.
- **μ\* = 0.1:** τ 0.66 (single-phase 0.69), r 0.84, MAE 1.40 K, mediana KKR/exp 1.05; źle uporządkowanych 6 % par
  różniących się o ponad 1 K.
- **μ\* = 0.2** daje prawie tę samą kolejność (τ 0.65), ale zaniża Tc ok. 3× (mediana 0.30, MAE 4.3 K).
- **Re/Ru** (n = 9): τ 0.09, za mało punktów do wniosków.
- **Ti ≥ 20 %** (n = 33): τ 0.61, Tc lekko zaniżone (mediana 0.87).
- Wyniki są identyczne z sekcją 1 starszego notebooka `Tc_Baseline_Comparison.ipynb`.
""",
    'B': r"""
## Wnioski (2026-11-05)

| podzbiór | n | τ: Tc_mu0.1 → Tc_corr2 | MAE [K] | mediana KKR/exp |
|---|---|---|---|---|
| skład B | 112 | 0.46 → 0.66 | 1.67 → 1.46 | 1.03 → 1.05 |
| single-phase bcc | 80 | 0.62 → 0.68 | 1.35 → 1.31 | 1.09 → 1.06 |
| z V lub Sc | 47 | 0.16 → 0.64 | 2.03 → 1.54 | 0.90 → 0.96 |
| ze Sc | 30 | 0.53 → 0.67 | 2.18 → 1.60 | 0.71 → 1.09 |
| ze Sc, single-phase | 6 | 0.97 → 0.83 | 0.56 → 1.08 | 0.94 → 1.10 |
| z V | 18 | 0.29 → 0.25 | 1.72 → 1.42 | 1.24 → 0.80 |
| bez V i Sc | 65 | 0.67 (bez zmian) | 1.40 | 1.06 |

Z poprawką corr2 skład B jest odtwarzany niemal tak dobrze jak skład A: τ 0.66 wobec 0.66 dla A.
""" + _VSC,
    'AB': r"""
## Wnioski (2026-11-05)

**Którą wersję wybrać dla A ∪ B?** corr2 nie zmienia niczego w stopach bez V i Sc, więc w A ∪ B różnią się tylko wiersze
z B. Pokazane są oba warianty. `Tc_corr2` jest właściwym porównaniem, bo jest celem optymalizacji na składzie B, a na
składzie A daje ten sam wynik co `Tc_mu0.1`.

| podzbiór | n | τ: Tc_mu0.1 → Tc_corr2 | MAE [K] | mediana KKR/exp |
|---|---|---|---|---|
| A ∪ B | 121 | 0.47 → 0.65 | 1.65 → 1.45 | 1.02 → 1.04 |
| single-phase bcc | 85 | 0.61 → 0.67 | 1.39 → 1.35 | 1.09 → 1.05 |
| tylko A (z Re/Ru) | 9 | 0.09 (bez zmian) | 1.38 | 0.89 |
| A ∩ B | 65 | 0.67 (bez zmian) | 1.40 | 1.06 |
| tylko B (z V/Sc) | 47 | 0.16 → 0.64 | 2.03 → 1.54 | 0.90 → 0.96 |
""" + _VSC,
}

C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip('\n')))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip('\n')))

TITLE = {'A': 'skład A (Ti, Nb, Zr, Hf, Ta, Mo, W, Y, Re, Ru)',
         'B': 'skład B (Ti, Nb, Zr, Hf, Ta, Mo, W, Y, V, Sc), z poprawką corr2',
         'AB': 'A ∪ B, z poprawką corr2'}[SET]
VARIANTS = {'A': "[('Tc_mu0.1', 'μ* = 0.1'), ('Tc_mu0.2', 'μ* = 0.2')]",
            'B': "[('Tc_mu0.1', 'Tc_mu0.1 (bez poprawki)'), ('Tc_corr2', 'Tc_corr2')]",
            'AB': "[('Tc_mu0.1', 'Tc_mu0.1 (bez poprawki)'), ('Tc_corr2', 'Tc_corr2')]"}[SET]

md(rf"""
# Baseline: KKR (GL) vs eksperymentalne $T_c$, {TITLE}

- **Eksperyment:** `results/Tc_literature.xlsx`.
- **KKR:** `results/comparison.sra.ew0.6.bz10-10-20.gl` (wariant GL: wzór powierzchniowy Gaspari–Gyorffy'ego i całkowita
  DOS Lloyda w 1/N(E_F)).
- **θ_D:** reguła mieszaniny, jak w optymalizacji: $\lambda_{{\rm mix}} = \lambda_{{\rm KKR}}(\theta_{{\rm KKR}}/\theta_{{\rm mix}})^2$,
  potem McMillan.
- **Zbiory pierwiastków:** stop należy do zbioru, jeśli wszystkie jego pierwiastki są w zbiorze.
  - skład A = Ti, Nb, Zr, Hf, Ta, Mo, W, Y, Re, Ru;
  - skład B = Ti, Nb, Zr, Hf, Ta, Mo, W, Y, V, Sc;
  - A ∪ B = stopy należące do A lub do B.

**Poprawka corr2** (`Tc_corr2`, `process_kkr.compute_tc_corrected`, parametry `TC_CORR_*` w `consts.py`, cel optymalizacji
`--target Tc_corr` od 2026-11-02): McMillan z μ\* = 0.1 i dwiema zmianami:
- θ_D fazy bcc dla **Sc** (215 K zamiast 355 K) w regule mieszaniny, λ przeskalowane o $(\theta_{{\rm mix}}/\theta_{{\rm corr}})^2$;
- sprzężenie elektron–paramagnon tylko na **V** (Bose 2009): $\lambda_{{es}} = x_V\,\alpha_V\,s_V\ln\frac{{1}}{{1-s_V}}$,
  $s_V = I_V N_V(E_F)$, α_V = 0.4; $T_c = \frac{{\theta}}{{1.45}}\exp\left[-\frac{{1.04(1+\lambda+\lambda_{{es}})}}{{\lambda-\lambda_{{es}}-\mu^*(1+0.62(\lambda+\lambda_{{es}}))}}\right]$.

Poprawka dotyczy więc tylko stopów zawierających V lub Sc. **Dla składu A `Tc_corr2 == Tc_mu0.1` dokładnie** (sprawdzane
assertem poniżej), a w A ∪ B zmieniają się tylko wiersze z V/Sc.

Porównywane warianty: {VARIANTS.replace("'", "`").replace('[', '').replace(']', '')}. Notebook nie zapisuje plików graficznych.
""")

code(rf"""
import json, pathlib, sys
from itertools import combinations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import openpyxl
from scipy import stats
from IPython.display import display
sys.path.insert(0, '../scripts')
import src.process_kkr as pk
RY_EV = 13.605693123
assert (dict(pk.TC_CORR_THETA_BCC_K), dict(pk.TC_CORR_SF_ALPHA), pk.TC_CORR_MU) == ({{'Sc': 215.0}}, {{'V': 0.4}}, 0.1), \
    'consts.py nie odpowiada wersji corr2'

SET = '{SET}'
VARIANTS = {VARIANTS}
XLSX    = pathlib.Path('../results/Tc_literature.xlsx')
RESULTS = pathlib.Path('../results/comparison.sra.ew0.6.bz10-10-20.gl')
EXCLUDE = {{'NbTa67MoHfW33'}}
SET_A = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru']
SET_B = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'V', 'Sc']

# kolumny xlsx (arkusz 'Tc_literature'): A nazwa, B–N skład, S Tc_exp, Z autor, AE status
ELEMENTS = ['Sc', 'Ti', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'La', 'Hf', 'Ta', 'W', 'Re', 'V']
COL_NAME, COL_EL0, COL_TC_EXP, COL_AUTHOR, COL_STATUS = 1, 2, 19, 26, 31
COMP_TOL = 2e-3
MARKER = {{'single-phase bcc': 'o', 'caution (multiphase etc.)': '^'}}
pd.set_option('display.max_rows', 200, 'display.width', 250)
""")

md("## 0. Dane: eksperyment + KKR (dopasowanie po składzie)")
code(r"""
def _val(cell):
    try:
        return float(cell.value)
    except (TypeError, ValueError):
        return np.nan

kkr = {}
for d in sorted(RESULTS.iterdir()):
    p = d / 'results.json'
    if p.is_file():
        j = json.loads(p.read_text())
        x = np.array([float(j.get(e) or 0.0) for e in ELEMENTS])
        kkr[d.name] = (x / x.sum(), j)

def mcmillan(theta, lam, mu):
    denom = lam - mu * (1 + 0.62 * lam)
    return theta / 1.45 * np.exp(-1.04 * (1 + lam) / denom) if denom > 0 else 0.0

ws = openpyxl.load_workbook(XLSX, data_only=True)['Tc_literature']
rows, unmatched = [], []
for r in range(4, ws.max_row + 1):
    name, tc = ws.cell(r, COL_NAME).value, _val(ws.cell(r, COL_TC_EXP))
    x = np.nan_to_num([_val(ws.cell(r, COL_EL0 + i)) for i in range(len(ELEMENTS))])
    if not name or np.isnan(tc) or x.sum() == 0 or name in EXCLUDE:
        continue
    x = x / x.sum()
    k = min(kkr, key=lambda k: np.abs(kkr[k][0] - x).max())
    if np.abs(kkr[k][0] - x).max() > COMP_TOL:
        unmatched.append(name); continue
    j = kkr[k][1]
    els = {e for e, xi in zip(ELEMENTS, x) if xi > 0}
    comp = dict(zip(ELEMENTS, x))
    lam_mix = j['lambda'] * (j['thetaDB_K'] / j['mixture_debye_temperature']) ** 2
    # Tc_corr2: funkcja produkcyjna, na danych w postaci jak w optymalizacji (θ_mix, λ_mix, debye_mode 'mix')
    c = pk.compute_tc_corrected({'thetaDB': j['mixture_debye_temperature'], 'lambda': lam_mix, 'debye_mode': 'mix',
                                 **{f'{e}_Ntot': j.get(f'{e}_Ntot') for e in ELEMENTS}},
                                dict(zip(ELEMENTS, kkr[k][0])))     # skład KKR (θ_mix w results.json liczone z niego)
    status = ws.cell(r, COL_STATUS).value or ''
    rows.append(dict(name=name, author=ws.cell(r, COL_AUTHOR).value or '', kkr_dir=k, Tc_exp=tc,
                     group='caution (multiphase etc.)' if status.startswith('caution') else 'single-phase bcc',
                     in_A=els <= set(SET_A), in_B=els <= set(SET_B), has_VSc=bool(els & {'V', 'Sc'}), **comp,
                     theta_mix=j['mixture_debye_temperature'], lambda_mix=lam_mix,
                     **{'Tc_mu0.1': mcmillan(j['mixture_debye_temperature'], lam_mix, 0.1),
                        'Tc_mu0.2': mcmillan(j['mixture_debye_temperature'], lam_mix, 0.2),
                        'Tc_corr2': c['Tc_corr']},
                     theta_corr=c['theta_corr'], lambda_corr=c['lambda_corr'], lambda_es=c['lambda_es']))
ALL = pd.DataFrame(rows)

# corr2 nie zmienia niczego bez V i Sc
noVSc = ~ALL.has_VSc
assert np.allclose(ALL.loc[noVSc, 'Tc_corr2'], ALL.loc[noVSc, 'Tc_mu0.1'], atol=1e-12, rtol=0)
print(f"Tc_corr2 == Tc_mu0.1 dla wszystkich {noVSc.sum()} wierszy bez V i Sc (w tym całego składu A: {ALL.in_A.sum()})")

DF = {'A': ALL[ALL.in_A], 'B': ALL[ALL.in_B], 'AB': ALL[ALL.in_A | ALL.in_B]}[SET].reset_index(drop=True)
ELSET = {'A': SET_A, 'B': SET_B, 'AB': sorted(set(SET_A) | set(SET_B), key=ELEMENTS.index)}[SET]
print(f"wiersze z Tc_exp i wynikiem KKR (wszystkie): {len(ALL)} (bez wyniku KKR: {unmatched or 'brak'})")
print(f"zbiór {SET}: {len(DF)} wierszy ({(DF.group != 'single-phase bcc').sum()} 'caution'), {DF.kkr_dir.nunique()} składów KKR, "
      f"z V: {(DF.V > 0).sum()}, ze Sc: {(DF.Sc > 0).sum()}, z Re/Ru: {((DF.Re > 0) | (DF.Ru > 0)).sum()}")
""")

md(r"""
**Metryki:**
- **kolejność** (najważniejsza dla optymalizacji): Kendall τ, Spearman ρ oraz odsetek par źle uporządkowanych spośród par,
  których eksperymentalne Tc różnią się o ponad 1 K;
- **dopasowanie liniowe** $T_c^{\rm exp} = a\,T_c^{\rm KKR} + b$ (R², Pearson r) oraz prosta przez zero $T_c^{\rm exp} = s\,T_c^{\rm KKR}$;
- **wartości bezwzględne:** MAE, bias (średnia KKR − exp) i mediana stosunku KKR/exp.
""")
code(r"""
def misordered(x, y, dT=1.0):
    good = bad = 0
    for i, j in combinations(range(len(y)), 2):
        if abs(y[i] - y[j]) > dT:
            s = np.sign(y[i] - y[j]) * np.sign(x[i] - x[j])
            good += s > 0; bad += s < 0
    return bad / (good + bad) if good + bad else np.nan

def metrics(x, y):
    lr = stats.linregress(x, y)
    return {'n': len(y), 'Kendall τ': stats.kendalltau(x, y)[0], 'Spearman ρ': stats.spearmanr(x, y)[0],
            'źle uporządk. pary (ΔTc>1K)': misordered(x, y),
            'Pearson r': lr.rvalue, 'R² (LR)': lr.rvalue ** 2, 'a (LR)': lr.slope, 'b (LR) [K]': lr.intercept,
            's (LR przez 0)': np.sum(x * y) / np.sum(x * x),
            'MAE [K]': np.mean(np.abs(x - y)), 'bias [K]': np.mean(x - y), 'mediana KKR/exp': np.median(x / y)}


def summary_table(subsets):
    return pd.DataFrame([{'podzbiór': sn, 'wariant': col, **metrics(s[col].values, s.Tc_exp.values)}
                         for sn, s in subsets.items() if len(s) > 2 for col, _ in VARIANTS]
                        ).set_index(['podzbiór', 'wariant']).round(3)


def linear_plot(df, fit_mask=None, fit_label='wszystkie punkty'):
    # fit_mask: wiersze użyte do dopasowania i metryk; na wykresie zawsze wszystkie punkty
    fit = df if fit_mask is None else df[fit_mask]
    cols = [c for c, _ in VARIANTS]
    lim = (0, max(df.Tc_exp.max(), df[cols].values.max()) * 1.05)
    fig, axes = plt.subplots(1, len(VARIANTS), figsize=(7 * len(VARIANTS), 6.2), sharey=True)
    for ax, (col, lab) in zip(axes, VARIANTS):
        m = metrics(fit[col].values, fit.Tc_exp.values)
        for grp, g in df.groupby('group'):
            for vsc, gg in g.groupby('has_VSc'):
                ax.scatter(gg[col], gg.Tc_exp, marker=MARKER[grp], s=50, edgecolors='k', linewidths=0.4, zorder=3,
                           color='C1' if vsc else 'C0', label=f"{grp}{', z V/Sc' if vsc else ''} (n={len(gg)})")
        xl = np.linspace(*lim, 100)
        ax.plot(xl, xl, ':', color='gray', label='y = x')
        ax.plot(xl, m['a (LR)'] * xl + m['b (LR) [K]'], color='C3', lw=1.8,
                label=f"LR: y = {m['a (LR)']:.2f}x {m['b (LR) [K]']:+.2f}")
        ax.plot(xl, m['s (LR przez 0)'] * xl, '--', color='C2', lw=1.4, label=f"LR przez 0: y = {m['s (LR przez 0)']:.2f}x")
        ax.text(0.03, 0.97, f"dopasowanie i metryki: {fit_label} (n={len(fit)})\n"
                            f"Kendall τ = {m['Kendall τ']:.3f}\nSpearman ρ = {m['Spearman ρ']:.3f}\n"
                            f"Pearson r = {m['Pearson r']:.3f}\nR² = {m['R² (LR)']:.3f}\n"
                            f"MAE = {m['MAE [K]']:.2f} K\nbias = {m['bias [K]']:+.2f} K",
                transform=ax.transAxes, va='top', fontsize=10, bbox=dict(boxstyle='round', fc='white', ec='#cccccc', alpha=0.9))
        ax.set(xlim=lim, ylim=lim, xlabel=f'KKR {col} [K]', title=lab)
        ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower right')
    axes[0].set_ylabel('eksperymentalne $T_c$ [K]')
    fig.suptitle(f'Baseline, zbiór {SET} (n={len(df)} na wykresie, dopasowanie: {fit_label}), KKR GL, θ_D mix '
                 f'(pomarańczowe = stopy z V lub Sc)', fontsize=13)
    plt.tight_layout(); plt.show()


def element_panels(df, col, lab):
    present = [e for e in ELSET if (df[e] > 0).any()]
    ncol = 4; nrow = int(np.ceil(len(present) / ncol))
    lim = (0, max(df[col].max(), df.Tc_exp.max()) * 1.05)
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 3.9 * nrow), sharex=True, sharey=True, squeeze=False)
    for ax, el in zip(axes.flat, present):
        has = df[el] > 0
        ax.scatter(df.loc[~has, col], df.loc[~has, 'Tc_exp'], s=16, color='#d9d9d9', zorder=2)
        vmax = df.loc[has, el].max()
        for grp, g in df[has].groupby('group'):
            ax.scatter(g[col], g.Tc_exp, c=g[el], cmap='viridis', vmin=0, vmax=vmax, marker=MARKER[grp],
                       s=40, edgecolors='k', linewidths=0.3, zorder=3)
        ax.plot(lim, lim, ':', color='gray')
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=plt.Normalize(0, vmax), cmap='viridis'), ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(f'$x_{{\\rm {el}}}$')
        ax.set(xlim=lim, ylim=lim, title=f'{el} (n={has.sum()})')
    for ax in axes.flat[len(present):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel(f'KKR {col} [K]')
    for ax in axes[:, 0]:
        ax.set_ylabel('eksp. $T_c$ [K]')
    fig.suptitle(f'Zbiór {SET}: eksperyment vs KKR ({lab}), kolor = udział pierwiastka, ● single-phase ▲ caution', fontsize=12)
    plt.tight_layout(); plt.show()


def ordering_plot(df):
    fig, axes = plt.subplots(1, len(VARIANTS) + 1, figsize=(6.3 * (len(VARIANTS) + 1), 5.6))
    n = len(df)
    rk_exp = df.Tc_exp.rank(ascending=False)
    for ax, (col, lab) in zip(axes, VARIANTS):
        rk_kkr = df[col].rank(ascending=False)
        for grp, g in df.groupby('group'):
            ax.scatter(rk_kkr[g.index], rk_exp[g.index], marker=MARKER[grp], s=40, edgecolors='k', linewidths=0.3, label=grp)
        ax.plot([1, n], [1, n], ':', color='gray')
        ax.set(xlabel='ranga KKR', ylabel='ranga eksperymentalna', xlim=(0, n + 1), ylim=(0, n + 1),
               title=f"{lab}:  Spearman ρ = {stats.spearmanr(rk_kkr, rk_exp)[0]:.3f}")
        ax.invert_xaxis(); ax.invert_yaxis(); ax.grid(alpha=0.3); ax.legend(fontsize=8, loc='lower left')
    DTS = np.arange(0, 4.01, 0.25)
    y = df.Tc_exp.values
    for col, lab in VARIANTS:
        x = df[col].values
        axes[-1].plot(DTS, [1 - misordered(x, y, dT) for dT in DTS], marker='o', label=lab)
    axes[-1].set(xlabel='próg |ΔTc_exp| [K]', ylabel='odsetek par dobrze uporządkowanych',
                 title='Poprawna kolejność par vs różnica Tc', ylim=(0.5, 1.0))
    axes[-1].grid(alpha=0.3); axes[-1].legend()
    plt.tight_layout(); plt.show()
    (c1, _), (c2, _) = VARIANTS
    print(f"Kendall τ między rankingami KKR {c1} i {c2}: {stats.kendalltau(df[c1], df[c2])[0]:.3f}")


def alloy_table(df):
    cols = [c for c, _ in VARIANTS]
    extra = ['theta_corr', 'lambda_es'] if 'Tc_corr2' in cols else []
    tab = df[['name', 'author', 'group', 'Tc_exp', 'theta_mix', 'lambda_mix', *extra, *cols]].copy()
    for c in cols:
        tab[f'KKR/exp {c}'] = tab[c] / tab.Tc_exp
    return tab.sort_values('Tc_exp', ascending=False).reset_index(drop=True).round(3)
""")

SUBSETS = {
    'A': "{'skład A, wszystkie': DF, 'tylko single-phase bcc': DF[DF.group == 'single-phase bcc'],\n"
         "    'z Re/Ru': DF[(DF.Re > 0) | (DF.Ru > 0)], 'bez Re/Ru (wspólne z B)': DF[(DF.Re == 0) & (DF.Ru == 0)],\n"
         "    'z Ti ≥ 20 %': DF[DF.Ti >= 0.2]}",
    'B': "{'skład B, wszystkie': DF, 'tylko single-phase bcc': DF[DF.group == 'single-phase bcc'],\n"
         "    'z V': DF[DF.V > 0], 'ze Sc': DF[DF.Sc > 0], 'ze Sc, single-phase': DF[(DF.Sc > 0) & (DF.group == 'single-phase bcc')],\n"
         "    'z V lub Sc': DF[DF.has_VSc], 'bez V i Sc (wspólne z A)': DF[~DF.has_VSc], 'z Ti ≥ 20 %': DF[DF.Ti >= 0.2]}",
    'AB': "{'A ∪ B, wszystkie': DF, 'tylko single-phase bcc': DF[DF.group == 'single-phase bcc'],\n"
          "    'tylko A (z Re/Ru)': DF[DF.in_A & ~DF.in_B], 'A ∩ B': DF[DF.in_A & DF.in_B],\n"
          "    'tylko B (z V/Sc)': DF[DF.in_B & ~DF.in_A], 'z V': DF[DF.V > 0], 'ze Sc': DF[DF.Sc > 0],\n"
          "    'z Ti ≥ 20 %': DF[DF.Ti >= 0.2]}",
}[SET]

md("## 1. Metryki")
code(f"SUBSETS = {SUBSETS}\nsummary_table(SUBSETS)")
md("## 2. KKR vs eksperyment, dopasowanie liniowe")
code("linear_plot(DF)")
if SET == 'B':
    md(r"""
### 2b. Dopasowanie tylko na próbkach single-phase bcc

Na wykresie są wszystkie punkty (▲ = caution, wielofazowe itp.), ale proste i metryki w ramce są liczone wyłącznie na
próbkach jednofazowych (●). Te same metryki są w tabeli w sekcji 1, w wierszu „tylko single-phase bcc”.
""")
    code("linear_plot(DF, fit_mask=DF.group == 'single-phase bcc', fit_label='tylko single-phase bcc')")
md("## 3. Wykresy kolorowane udziałem pierwiastka")
code("for col, lab in VARIANTS:\n    element_panels(DF, col, lab)")
md("## 4. Zachowanie kolejności")
code("ordering_plot(DF)")
md("## 5. Tabela stopów (posortowana po Tc_exp)")
code("alloy_table(DF)")
if SET != 'A':
    md(r"""
## 6. Stopy, które poprawka zmienia najbardziej

`ratio` = KKR/exp przed (`Tc_mu0.1`) i po (`Tc_corr2`). Stopy ze Sc zyskują (niższe θ_corr, wyższe λ), stopy z V tracą (λ_es).
""")
    code(r"""
ch = DF[['name', 'group', 'Tc_exp', 'Tc_mu0.1', 'Tc_corr2', 'theta_mix', 'theta_corr', 'lambda_mix', 'lambda_corr',
         'lambda_es', 'Ti', 'Sc', 'V']].copy()
ch['ratio przed'] = ch['Tc_mu0.1'] / ch.Tc_exp
ch['ratio po'] = ch.Tc_corr2 / ch.Tc_exp
ch = ch[(ch.Tc_corr2 - ch['Tc_mu0.1']).abs() > 1e-9]
print(f'zmienionych wierszy: {len(ch)} z {len(DF)}')
ch.reindex((ch.Tc_corr2 - ch['Tc_mu0.1']).abs().sort_values(ascending=False).index).round(3)
""")

md(CONCL[SET])

nb = nbf.v4.new_notebook()
nb.cells = C
nb.metadata['kernelspec'] = {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'}
nbf.write(nb, OUT)
print('written', OUT)
