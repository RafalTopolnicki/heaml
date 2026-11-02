"""Refit of the electron-paramagnon coupling λ_es of Tc_corr with per-element α for V and Ti (DOCS/ROZMOWA_2026-11-02_optB_VTi.md).

Results 2026-11-02: --theta_bcc Ti=300 Sc=215 (old consts) needs α_V = α_Ti = 0.6; --theta_bcc Sc=215 gives α_V 0.4, α = α_Ti = 0
(adopted in consts).  Data: the GL literature baseline of scripts/baseline_corrections.py (sets A ∪ B) + the GL binaries of
validation.ratios/binaries.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl with experimental Tc (Ti–V Cheng 1962, V–Nb / V–Ta Corsan 1970,
V–Mo Kuentzler 1986; results/Tc_literature_V_additions.csv).  θ_D mix, θ_bcc Ti/Sc from consts (fixed), μ* 0.1;
λ_es = Σ x_i α_i s_i ln(1/(1 - s_i)), α_i = α_V, α_Ti or α (all other elements).  Loss: rms log ratio, leave-one-author-out CV.
Ranking check on the binary lines Nb–Zr / Nb–Ti / Hf–Nb / Ti–V (max Tc of each line; exp. maxima ≈ 11 / 10 / 9.5 / 7.3 K,
Hulm & Blaugher 1961 values NOT verified, used only for the ordering).

Usage (from the repo root):  python scripts/refit_lambda_es.py [--out results/refit_lambda_es.csv]
"""
import argparse, glob, json, os, sys
from itertools import product
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baseline_corrections import load, ELEMENTS, THETA_EXP, metrics, subsets  # noqa: E402
from run_opt_compositions import recompute_debye_consistent  # noqa: E402
from src.consts import STONER_I_EV, TC_CORR_THETA_BCC_K, TC_CORR_SF_ALPHA, TC_CORR_MU  # noqa: E402

RY_EV = 13.605693123
BIN_DIR = 'validation.ratios/binaries.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl'
V_CSV = 'results/Tc_literature_V_additions.csv'
BIN_EL = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'V', 'Sc']
LINES = ['Nb-Zr', 'Nb-Ti', 'Hf-Nb', 'Ti-V']
EXP_MAX = {'Nb-Zr': 11.0, 'Nb-Ti': 10.0, 'Hf-Nb': 9.5, 'Ti-V': 7.3}


def load_binaries():
    rows = []
    for p in sorted(glob.glob(os.path.join(BIN_DIR, '*', 'results.json'))):
        rows.append(json.load(open(p)))
    recompute_debye_consistent(rows, 'mix', BIN_EL)
    exp = pd.read_csv(V_CSV)
    exp = exp[exp.status == 'binary']
    out = []
    for j in rows:
        x = np.array([j.get(e, 0.0) or 0.0 for e in ELEMENTS])
        x = x / x.sum()
        els = sorted(e for e, xi in zip(ELEMENTS, x) if xi > 0)
        tc_exp, author = np.nan, '?'
        for _, r in exp.iterrows():
            xe = np.array([float(r[e]) for e in ELEMENTS])
            if np.abs(xe / xe.sum() - x).max() < 2e-3:
                tc_exp, author = r.Tc_K, r.first_author
        row = dict(name=''.join(f'{e}{round(100 * xi)}' for e, xi in sorted(zip(ELEMENTS, x), key=lambda t: -t[1]) if xi > 0),
                   line='-'.join(els), author=author, Tc_exp=tc_exp, single=True, in_A=False, in_B=True,
                   lam_kkr=j['lambda'], theta_kkr=j['thetaDB'], theta_mix_file=j['thetaDB'])   # λ, θ already mix
        for e, xi in zip(ELEMENTS, x):
            row[e] = xi
            row[f'N_{e}'] = (j.get(f'{e}_Ntot') or 0.0) / RY_EV if xi > 0 else 0.0
        out.append(row)
    return pd.DataFrame(out)


def precompute(df, theta_bcc=TC_CORR_THETA_BCC_K):
    """θ_mix (bcc Ti/Sc), λ_mix and x_i s_i ln(1/(1-s_i)) summed over V, Ti and the rest, as arrays."""
    theta = {**THETA_EXP, **theta_bcc}
    th = sum(df[e] * theta[e] for e in ELEMENTS).values
    lam = (df.lam_kkr * (df.theta_kkr / th) ** 2).values
    f = {'V': 0.0, 'Ti': 0.0, 'rest': 0.0}
    for e in ELEMENTS:
        s = np.clip(STONER_I_EV.get(e, 0.40) * df[f'N_{e}'].values, 0, 0.95)
        f[e if e in ('V', 'Ti') else 'rest'] = f[e if e in ('V', 'Ti') else 'rest'] + df[e].values * s * np.log(1.0 / (1.0 - s))
    return dict(th=th, lam=lam, fV=f['V'], fTi=f['Ti'], frest=f['rest'])


def tc_grid(pre, grid, mu=TC_CORR_MU):
    """Tc and λ_es for every (α, α_V, α_Ti) in grid: arrays (len(grid), n_rows)."""
    g = np.asarray(grid, dtype=float).reshape(-1, 3)
    les = g[:, :1] * pre['frest'] + g[:, 1:2] * pre['fV'] + g[:, 2:3] * pre['fTi']
    lam, th = pre['lam'][None, :], pre['th'][None, :]
    d = lam - les - mu * (1 + 0.62 * (lam + les))
    ok = (d > 0) & (lam > 0)
    with np.errstate(over='ignore', invalid='ignore'):
        tc = np.where(ok, th / 1.45 * np.exp(-1.04 * (1 + lam + les) / np.where(ok, d, 1)), 0.0)
    return tc, les


def loss(tc, y):
    return np.mean(np.log(np.clip(tc, 0.05, None) / y) ** 2, axis=-1)


def line_max(bins, tc):
    out = {}
    for l in LINES:
        m = (bins.line == l).values
        out[l] = (tc[m].max(), bins['name'].values[m][np.argmax(tc[m])])
    return out


def ranking_mask(bins, tcg):
    """per grid point: max Nb–Zr > max Nb–Ti > max Ti–V and max Hf–Nb > max Ti–V."""
    mx = {l: tcg[:, (bins.line == l).values].max(axis=1) for l in LINES}
    return (mx['Nb-Zr'] > mx['Nb-Ti']) & (mx['Nb-Ti'] > mx['Ti-V']) & (mx['Hf-Nb'] > mx['Ti-V'])


def ranking_ok(lm):
    return lm['Nb-Zr'][0] > lm['Nb-Ti'][0] > lm['Ti-V'][0] and lm['Hf-Nb'][0] > lm['Ti-V'][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='results/refit_lambda_es.csv')
    ap.add_argument('--theta_bcc', nargs='*', default=None,
                    help="bcc θ_D overrides, e.g. Sc=215 (default: consts TC_CORR_THETA_BCC_K; no value = none)")
    args = ap.parse_args()
    theta_bcc = TC_CORR_THETA_BCC_K if args.theta_bcc is None else {k: float(v) for k, v in (a.split('=') for a in args.theta_bcc)}
    print(f'θ_bcc overrides: {theta_bcc}')
    base, bins = load(), load_binaries()
    fitb = bins[bins.Tc_exp.notna()]
    data = pd.concat([base, fitb], ignore_index=True).fillna({'line': ''})
    y = data.Tc_exp.values
    print(f'baseline {len(base)} rows + {len(fitb)} binaries with exp. Tc; {len(bins)} binaries in total')

    A = np.round(np.arange(0, 1.01, 0.1), 2)
    AV = np.round(np.arange(0, 3.01, 0.1), 2)
    variants = {
        'no λ_es (θ_bcc only)': dict(grid=[(0, 0, 0)]),
        'consts (= Tc_corr)': dict(grid=[(0.0, TC_CORR_SF_ALPHA.get('V', 0.0), TC_CORR_SF_ALPHA.get('Ti', 0.0))]),
        'α 0.3 all (Tc_corr of 2026-11-01)': dict(grid=[(0.3,) * 3]),
        'α common (fit)': dict(grid=[(a, a, a) for a in AV]),
        'α_V fit, others 0.3': dict(grid=[(0.3, v, 0.3) for v in AV]),
        'α_V, α_Ti fit, others 0.3': dict(grid=[(0.3, v, t) for v, t in product(AV, AV)]),
        'α, α_V, α_Ti fit': dict(grid=list(product(A, AV, AV))),
    }

    pre_d, pre_b = precompute(data, theta_bcc), precompute(bins, theta_bcc)
    rows, summary = [], []
    authors = data.author.values
    for name, v in variants.items():
        grid = np.asarray(v['grid'], dtype=float)
        tcg = tc_grid(pre_d, grid)[0]
        rank = ranking_mask(bins, tc_grid(pre_b, grid)[0])
        for need_rank in ([False, True] if len(grid) > 1 else [False]):
            allowed = rank if need_rank else np.ones(len(grid), bool)
            label = name + (' [ranking constraint]' if need_rank else '')
            if not allowed.any():
                summary.append(dict(variant=label, rank_ok=False))
                continue

            def best(idx):
                l = loss(tcg[:, idx], y[idx])
                return int(np.argmin(np.where(allowed, l, np.inf)))

            k = best(np.arange(len(data)))
            p = grid[k]
            tc = tcg[k]
            tc_cv = tc.copy()
            if len(grid) > 1:
                for a in np.unique(authors):
                    test = authors == a
                    tc_cv[test] = tcg[best(np.where(~test)[0]), test]
            tb, lb = (z[0] for z in tc_grid(pre_b, p))
            lm = line_max(bins, tb)
            s = dict(variant=label, alpha=p[0], alpha_V=p[1], alpha_Ti=p[2], rank_ok=ranking_ok(lm))
            for sub, m in {**subsets(data), 'V binaries': data.line != ''}.items():
                m = m.values
                if m.sum() > 2:
                    s[f'tau {sub}'] = stats.kendalltau(tc[m], y[m])[0]
                    s[f'tauCV {sub}'] = stats.kendalltau(tc_cv[m], y[m])[0]
                    s[f'med {sub}'] = np.median(tc[m] / y[m])
            s['rms_log'] = np.sqrt(loss(tc, y))
            s['rms_log CV'] = np.sqrt(loss(tc_cv, y))
            for l in LINES:
                s[f'max {l}'] = f'{lm[l][0]:.2f} ({lm[l][1]})'
            summary.append(s)
            for i, r in bins.iterrows():
                rows.append(dict(variant=label, name=r['name'], line=r.line, Tc_exp=r.Tc_exp, Tc=tb[i], lambda_es=lb[i]))
    pd.DataFrame(rows).to_csv(args.out, index=False)
    sm = pd.DataFrame(summary)
    sm.to_csv(args.out.replace('.csv', '_summary.csv'), index=False)
    pd.set_option('display.width', 250, 'display.max_columns', 100, 'display.max_colwidth', 60)
    cols = ['variant', 'alpha', 'alpha_V', 'alpha_Ti', 'rank_ok', 'rms_log', 'rms_log CV']
    print(sm[cols].round(3).to_string())
    print(sm[['variant'] + [c for c in sm if c.startswith('tau') and any(k in c for k in (' A', ' B', 'V bin', 'z V', 'Ti≥'))]]
          .round(2).to_string())
    print(sm[['variant'] + [c for c in sm if c.startswith('med ')]].round(2).to_string())
    print(sm[['variant'] + [f'max {l}' for l in LINES]].to_string())
    print(f'\nwritten {args.out} and {args.out.replace(".csv", "_summary.csv")}')


if __name__ == '__main__':
    main()
