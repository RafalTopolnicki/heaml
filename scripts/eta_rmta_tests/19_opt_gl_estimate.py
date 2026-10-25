"""Estimate how the optimisation results (opt.ratios.mix, slim dirs WITHOUT fort.51/.out -> no exact GL recompute) change
between the production variant A (cutoff 'valence', site DOS, target Tc_mu0.2) and variant GL (gg + Lloyd DOS, mu* 0.10).
The ratio eta_GL/eta_A is modelled as linear in composition, fitted on the baseline (results/comparison.../eta_gl.csv,
n=126, R^2 0.965, rms 0.021); lambda_GL = lambda_A * ratio, Tc from McMillan with the stored theta (mix mode: theta_mix).
Usage (repo root): python scripts/eta_rmta_tests/19_opt_gl_estimate.py [opt.ratios.mix]  -> results/opt_gl_estimate.csv"""
import sys, os, json, re, glob, numpy as np, pandas as pd
from scipy import stats
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
OPT = sys.argv[1] if len(sys.argv) > 1 else 'opt.ratios.mix'
B = 'results/comparison.sra.ew0.6.bz10-10-20'
ELS = ['Sc', 'Ti', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'La', 'Hf', 'Ta', 'W', 'Re', 'V']
MU_A, MU_GL = 0.2, 0.10

def mc(th, lam, mu):
    den = lam - mu * (1 + 0.62 * lam)
    with np.errstate(all='ignore'): t = th / 1.45 * np.exp(-1.04 * (1 + lam) / den)
    return np.where(den > 0, t, 0.0)

# --- ratio model from the baseline
d = pd.read_csv(f'{B}/eta_gl.csv'); X = []
for n in d.name:
    j = json.load(open(f'{B}/{n}/results.json')); x = np.array([float(j.get(e) or 0) for e in ELS]); X.append(x / x.sum())
X = np.array(X); ratio = (d.eta_mix_gl / d.eta_mix_A).values; m = np.isfinite(ratio) & (d.eta_mix_A.values > 0)
coef = np.linalg.lstsq(X[m], ratio[m], rcond=None)[0]; pred = X[m] @ coef
print('ratio model eta_GL/eta_A =', {e: round(c, 3) for e, c in zip(ELS, coef)})
print('  baseline fit: n=%d R2=%.3f rms=%.3f' % (m.sum(), 1 - ((ratio[m] - pred)**2).sum() / ((ratio[m] - ratio[m].mean())**2).sum(), np.sqrt(((ratio[m] - pred)**2).mean())))

# --- collect optimisation points
rows = []
for run in sorted(os.listdir(OPT)):
    tag = re.sub(r'.*stonerbeta0\.mixdebye\.', '', run).replace('.r0.05.max0.75.face0.1', '')
    tag = re.search(r'acqbeta(\d+)', run).group(0) + '/' + tag
    for rj in glob.glob(f'{OPT}/{run}/iteration_*/computation/*/results.json'):
        try: j = json.load(open(rj))
        except Exception: continue
        if not j or j.get('lambda') is None or not j.get('thetaDB_K'): continue
        x = np.array([float(j.get(e) or 0) for e in ELS]); x /= x.sum()
        r = float(x @ coef); lamA = j['lambda']; th = j['thetaDB_K']
        it = int(re.search(r'iteration_(\d+)', rj).group(1))
        comp = ''.join(f'{e}{100*v:.0f}' for e, v in zip(ELS, x) if v >= 0.005)
        rows.append(dict(run=tag, iteration=it, comp=comp, **{e: x[i] for i, e in enumerate(ELS) if e in 'Ti Nb Zr Hf Ta Mo W Y Re Ru'.split()},
                         theta=th, lambda_A=lamA, ratio=r, lambda_GL=lamA * r, Tc_A=float(mc(th, lamA, MU_A)),
                         Tc_GL=float(mc(th, lamA * r, MU_GL)), Tc_GL011=float(mc(th, lamA * r, 0.11))))
df = pd.DataFrame(rows); df.to_csv('results/opt_gl_estimate.csv', index=False)
print(f'\n{len(df)} optimisation points in {df.run.nunique()} runs -> results/opt_gl_estimate.csv')
print('ratio eta_GL/eta_A over all points: median %.3f, 5-95%% %.3f-%.3f' % (df.ratio.median(), df.ratio.quantile(.05), df.ratio.quantile(.95)))
print('Tc_GL(0.10)/Tc_A(0.20): median %.3f, 5-95%% %.3f-%.3f;  Spearman(Tc_A, Tc_GL) all points = %.3f' % (
    (df.Tc_GL / df.Tc_A).median(), (df.Tc_GL / df.Tc_A).quantile(.05), (df.Tc_GL / df.Tc_A).quantile(.95), stats.spearmanr(df.Tc_A, df.Tc_GL)[0]))
pd.set_option('display.width', 250)
print('\n=== per run: champion by Tc_A (production) and by Tc_GL (estimate); rank = position of the A-champion in the GL ranking')
out = []
for run, g in df.groupby('run'):
    a = g.loc[g.Tc_A.idxmax()]; b = g.loc[g.Tc_GL.idxmax()]
    rank = int((g.Tc_GL > a.Tc_GL).sum()) + 1
    out.append(dict(run=run, n=len(g), champion_A=a.comp, Tc_A=a.Tc_A, Tc_GL_of_A=a.Tc_GL, rank_A_in_GL=rank,
                    champion_GL=b.comp, Tc_GL=b.Tc_GL, Tc_A_of_GL=b.Tc_A, same=(a.comp == b.comp), spearman=stats.spearmanr(g.Tc_A, g.Tc_GL)[0],
                    top10_overlap=len(set(g.nlargest(10, 'Tc_A').comp) & set(g.nlargest(10, 'Tc_GL').comp))))
print(pd.DataFrame(out).round(2).to_string(index=False))
print('\n=== top 10 over all runs by Tc_A vs by Tc_GL')
u = df.sort_values('Tc_A', ascending=False).drop_duplicates('comp')
print(u.nlargest(10, 'Tc_A')[['run', 'comp', 'theta', 'lambda_A', 'ratio', 'lambda_GL', 'Tc_A', 'Tc_GL', 'Tc_GL011']].round(3).to_string(index=False))
print(u.nlargest(10, 'Tc_GL')[['run', 'comp', 'theta', 'lambda_A', 'ratio', 'lambda_GL', 'Tc_A', 'Tc_GL', 'Tc_GL011']].round(3).to_string(index=False))
print('\n=== which elements gain/lose under GL (Spearman of x_el with Tc_GL/Tc_A over all points)')
print({e: round(stats.spearmanr(df[e], df.Tc_GL / df.Tc_A)[0], 2) for e in 'Ti Nb Zr Hf Ta Mo W Y Re Ru'.split() if (df[e] > 0).sum() > 20})
