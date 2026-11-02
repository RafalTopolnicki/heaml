"""Offline test of two physical Tc corrections on the GL literature baseline (no KKR runs).

Correction 1 (bcc θ_D): the mixture Debye temperature θ_mix = Σ x_i θ_i uses the experimental θ_D of the elements'
ground-state phase; for Ti and Sc (hcp) the bcc lattice is much softer.  Here θ_i of selected elements is replaced by a
bcc value and λ_mix = λ_KKR (θ_KKR/θ_mix)², Tc = McMillan(θ_mix, λ_mix, μ*) are recomputed.

Correction 2 (spin fluctuations, Bose 2009 J. Phys.: Condens. Matter 21 025602, Eqs. 26-27): electron-paramagnon coupling
per site λ_es,i = α s_i ln(1/(1 - s_i)), s_i = I_i N_i(E_F) (KKR site DOS per spin, Stoner I from consts.STONER_I_EV),
λ_es = Σ x_i λ_es,i over the chosen elements, and
    Tc = θ/1.45 exp(-1.04 (1 + λ + λ_es) / (λ - λ_es - μ* (1 + 0.62 (λ + λ_es)))).

Usage (from the repo root):  python scripts/baseline_corrections.py [--out results/baseline_corrections.csv]
Parameters fitted to the data are cross-validated leave-one-author-out.
"""
import argparse, json, os, sys
from itertools import product
import numpy as np
import pandas as pd
import openpyxl
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.consts import STONER_I_EV, TC_CORR_THETA_BCC_K, TC_CORR_SF_ALPHA, TC_CORR_MU  # noqa: E402
from src.elements import ELEMENTS as ELEMENT_DATA  # noqa: E402

RY_EV = 13.605693123
XLSX = 'results/Tc_literature.xlsx'
RESULTS = 'results/comparison.sra.ew0.6.bz10-10-20.gl'
EXCLUDE = {'NbTa67MoHfW33'}
SET_A = {'Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru'}
SET_B = {'Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'V', 'Sc'}
ELEMENTS = ['Sc', 'Ti', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'La', 'Hf', 'Ta', 'W', 'Re', 'V']   # xlsx columns B..N
COL_NAME, COL_EL0, COL_TC_EXP, COL_AUTHOR, COL_STATUS = 1, 2, 19, 26, 31
THETA_EXP = {e: ELEMENT_DATA[e].debye_temperature for e in ELEMENTS}


def load():
    kkr = {}
    for d in sorted(os.listdir(RESULTS)):
        p = os.path.join(RESULTS, d, 'results.json')
        if os.path.isfile(p):
            j = json.load(open(p))
            x = np.array([float(j.get(e) or 0.0) for e in ELEMENTS])
            kkr[d] = (x / x.sum(), j)
    ws = openpyxl.load_workbook(XLSX, data_only=True)['Tc_literature']
    num = lambda c: float(c.value) if isinstance(c.value, (int, float)) else np.nan
    rows = []
    for r in range(4, ws.max_row + 1):
        name, tc = ws.cell(r, COL_NAME).value, num(ws.cell(r, COL_TC_EXP))
        x = np.nan_to_num([num(ws.cell(r, COL_EL0 + i)) for i in range(len(ELEMENTS))])
        if not name or np.isnan(tc) or x.sum() == 0 or name in EXCLUDE:
            continue
        x = x / x.sum()
        k = min(kkr, key=lambda k: np.abs(kkr[k][0] - x).max())
        if np.abs(kkr[k][0] - x).max() > 2e-3:
            continue
        x, j = kkr[k]                                   # KKR composition (xlsx values are rounded)
        els = {e for e, xi in zip(ELEMENTS, x) if xi > 0}
        status = ws.cell(r, COL_STATUS).value or ''
        row = dict(name=name, author=ws.cell(r, COL_AUTHOR).value or '?', Tc_exp=tc,
                   single=not status.startswith('caution'), in_A=els <= SET_A, in_B=els <= SET_B,
                   lam_kkr=j['lambda'], theta_kkr=j['thetaDB_K'], theta_mix_file=j['mixture_debye_temperature'])
        for e, xi in zip(ELEMENTS, x):
            row[e] = xi
            row[f'N_{e}'] = (j.get(f'{e}_Ntot') or 0.0) / RY_EV if xi > 0 else 0.0     # states/eV/atom/spin
        rows.append(row)
    df = pd.DataFrame(rows)
    df = df[df.in_A | df.in_B].reset_index(drop=True)
    th = sum(df[e] * THETA_EXP[e] for e in ELEMENTS)
    assert np.allclose(th, df.theta_mix_file, rtol=1e-6), 'θ_mix in results.json differs from Σ x θ_exp'
    return df


def tc_model(df, mu=0.1, theta_bcc=None, alpha=0.0, sf_elements=()):
    theta = {**THETA_EXP, **(theta_bcc or {})}
    th_mix = sum(df[e] * theta[e] for e in ELEMENTS).values
    lam = (df.lam_kkr * (df.theta_kkr / th_mix) ** 2).values
    les = np.zeros(len(df))
    for e in sf_elements:
        s = np.clip(STONER_I_EV[e] * df[f'N_{e}'].values, 0, 0.95)
        a = alpha.get(e, 0.0) if isinstance(alpha, dict) else alpha
        les += df[e].values * a * s * np.log(1.0 / (1.0 - s))
    denom = lam - les - mu * (1 + 0.62 * (lam + les))
    with np.errstate(over='ignore', invalid='ignore'):
        tc = np.where(denom > 0, th_mix / 1.45 * np.exp(-1.04 * (1 + lam + les) / np.where(denom > 0, denom, 1)), 0.0)
    return tc, lam, les


def misordered(x, y, dT=1.0):
    i, j = np.triu_indices(len(y), 1)
    m = np.abs(y[i] - y[j]) > dT
    s = np.sign(y[i] - y[j])[m] * np.sign(x[i] - x[j])[m]
    return (s < 0).sum() / max((s != 0).sum(), 1)


def metrics(tc, y):
    return dict(n=len(y), tau=stats.kendalltau(tc, y)[0], rho=stats.spearmanr(tc, y)[0], r=np.corrcoef(tc, y)[0, 1],
                bad_pairs=misordered(tc, y), MAE=np.mean(np.abs(tc - y)), bias=np.mean(tc - y),
                med_ratio=np.median(tc / y), rms_log=np.sqrt(np.mean(np.log(np.clip(tc, 0.05, None) / y) ** 2)))


def subsets(df):
    return {'A': df.in_A, 'B': df.in_B, 'A∪B': df.in_A | df.in_B, 'A∪B single': (df.in_A | df.in_B) & df.single,
            'A∩B': df.in_A & df.in_B, 'z V': df.V > 0, 'z Sc': df.Sc > 0, 'z Sc single': (df.Sc > 0) & df.single,
            'z Ti≥20%': df.Ti >= 0.2}


def evaluate(df, tc, label):
    y = df.Tc_exp.values
    return [dict(variant=label, subset=k, **metrics(tc[m.values], y[m.values])) for k, m in subsets(df).items() if m.sum() > 2]


def loss(tc, y):
    return np.mean(np.log(np.clip(tc, 0.05, None) / y) ** 2)


def fit(df, idx, variant):
    """grid search of the variant's free parameters on rows idx (rms log ratio, A ∪ B)."""
    y = df.Tc_exp.values[idx]
    best = None
    for params in variant['grid']:
        tc = tc_model(df, **variant['make'](*params))[0][idx]
        l = loss(tc, y)
        if best is None or l < best[0]:
            best = (l, params)
    return best[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/baseline_corrections.csv')
    args = ap.parse_args()
    df = load()
    print(f'{len(df)} rows (A {df.in_A.sum()}, B {df.in_B.sum()}, single-phase {df.single.sum()})')

    TH = np.arange(120, 421, 10)
    TI2, SC2, AL2 = np.arange(200, 421, 20), np.arange(140, 356, 15), np.round(np.arange(0, 1.01, 0.1), 2)
    AL = np.round(np.arange(0, 2.01, 0.1), 2)
    MU = np.round(np.arange(0.06, 0.16, 0.01), 2)
    variants = {
        '0 baseline μ*0.1': dict(grid=[()], make=lambda: dict()),
        '1a θ_bcc Ti, Sc (fit)': dict(grid=list(product(TH, TH)), make=lambda t, s: dict(theta_bcc={'Ti': t, 'Sc': s})),
        '1b θ_bcc Ti, Sc, Zr, Hf, Y (one scale, fit)': dict(
            grid=[(f,) for f in np.round(np.arange(0.4, 1.01, 0.05), 2)],
            make=lambda f: dict(theta_bcc={e: f * THETA_EXP[e] for e in ('Ti', 'Sc', 'Zr', 'Hf', 'Y')})),
        '2a λ_es all elements (α fit)': dict(grid=[(a,) for a in AL],
                                            make=lambda a: dict(alpha=a, sf_elements=tuple(ELEMENTS))),
        '2b λ_es V only (α fit)': dict(grid=[(a,) for a in AL], make=lambda a: dict(alpha=a, sf_elements=('V',))),
        '2c λ_es V only, α = 1 (Bose upper limit)': dict(grid=[()], make=lambda: dict(alpha=1.0, sf_elements=('V',))),
        '1a+2b (fit θ_Ti, θ_Sc, α)': dict(grid=list(product(TH[::2], TH[::2], AL[::2])),
                                         make=lambda t, s, a: dict(theta_bcc={'Ti': t, 'Sc': s}, alpha=a, sf_elements=('V',))),
        '1a+2a (fit θ_Ti, θ_Sc, α)': dict(grid=list(product(TI2, SC2, AL2)),
                                         make=lambda t, s, a: dict(theta_bcc={'Ti': t, 'Sc': s}, alpha=a,
                                                                   sf_elements=tuple(ELEMENTS))),
        'consts TC_CORR_* (= Tc_corr in process_kkr)': dict(grid=[()], make=lambda: dict(
            mu=TC_CORR_MU, theta_bcc=TC_CORR_THETA_BCC_K, alpha=TC_CORR_SF_ALPHA, sf_elements=tuple(ELEMENTS))),
    }
    rows, params_out = [], []
    all_idx = np.arange(len(df))
    authors = df.author.values
    for name, v in variants.items():
        p = fit(df, all_idx, v)
        kw = v['make'](*p)
        tc = tc_model(df, **kw)[0]
        rows += evaluate(df, tc, f'{name} [in-sample]')
        # μ* re-optimised on top of the variant (one more global parameter)
        kw_nomu = {k: v for k, v in kw.items() if k != 'mu'}
        best_mu = min(MU, key=lambda m: loss(tc_model(df, mu=m, **kw_nomu)[0], df.Tc_exp.values))
        # leave-one-author-out CV
        tc_cv = np.zeros(len(df))
        for a in np.unique(authors):
            test = authors == a
            pa = fit(df, np.where(~test)[0], v)
            tc_cv[test] = tc_model(df, **v['make'](*pa))[0][test]
        if len(v['grid']) > 1:
            rows += evaluate(df, tc_cv, f'{name} [CV]')
        params_out.append(dict(variant=name, params=p, best_mu_on_top=best_mu))
        print(f'{name:45} params {p}  (μ* refit on top: {best_mu})')
    res = pd.DataFrame(rows)
    res.to_csv(args.out, index=False)
    pd.set_option('display.width', 250, 'display.max_rows', 500)
    for sub in ['A', 'B', 'A∪B', 'A∪B single', 'z V', 'z Sc', 'z Sc single', 'z Ti≥20%']:
        print(f'\n=== {sub}')
        print(res[res.subset == sub].drop(columns='subset').set_index('variant').round(3).to_string())
    print(f'\nwritten {args.out}')


if __name__ == '__main__':
    main()
