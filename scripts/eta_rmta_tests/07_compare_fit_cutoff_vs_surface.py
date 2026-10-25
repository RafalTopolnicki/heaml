"""Fit-to-experiment metrics (set B, Tc_literature.xlsx) for eta from the last-node cutoff (A, results.json)
vs the surface formula (B, eta_surface.csv from 06). Run from notebooks/ (loader uses ../results paths)."""
import sys, os, numpy as np, pandas as pd
from itertools import combinations
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
RES = '../results/comparison.sra.ew0.6.bz10-10-20'
g = {'RES': RES}; exec(open(os.path.join(HERE, 'baseline_loader.py')).read(), g)
df = g['df']; mcm = g['mcmillan']
es = pd.read_csv(f'{RES}/eta_surface.csv').set_index('name'); eg = pd.read_csv(f'{RES}/eta_gg.csv').set_index('name')
df['lambda_A'] = df['lambda']
df['lambda_B'] = df.kkr_dir.map(es.lambda_surface)
df['lambda_G'] = df.kkr_dir.map(eg.lambda_gg)
VERS = 'ABG'
for v in VERS:
    df[f'lambda_mix_{v}'] = df[f'lambda_{v}'] * (df.thetaDB_K / df.theta_mix)**2
def mc(th, lam, mu):
    den = lam - mu*(1+0.62*lam)
    with np.errstate(all='ignore'): t = th/1.45*np.exp(-1.04*(1+lam)/den)
    return np.where(den > 0, t, 0.0)
def rmslog(t, y): return np.sqrt(np.mean(np.log(np.clip(t, 1e-3, None)/y)**2))
def misorder(t, y, thr=1.0):
    n = bad = 0
    for i, j in combinations(range(len(t)), 2):
        if abs(y[i]-y[j]) > thr: n += 1; bad += (t[i]-t[j])*(y[i]-y[j]) < 0
    return bad/n
mus = np.round(np.arange(0.05, 0.4001, 0.005), 3)
for sub, m in [('set B all (n=74)', np.ones(len(df), bool)), ('single-phase bcc (n=65)', (df.group == 'single-phase bcc').values)]:
    d = df[m]; y = d.Tc_exp.values; print(f'\n=== {sub}')
    rows = []
    for v in VERS:
        lam = d[f'lambda_mix_{v}'].values; th = d.theta_mix.values
        T = [mc(th, lam, mu) for mu in mus]
        taus = [stats.kendalltau(t, y)[0] for t in T]; rl = [rmslog(t, y) for t in T]
        t2 = mc(th, lam, 0.2); lr = np.polyfit(t2, y, 1)
        rows.append(dict(version=v, tau_mu02=stats.kendalltau(t2, y)[0], rho=stats.spearmanr(t2, y)[0], r=stats.pearsonr(t2, y)[0],
                         bias=np.mean(t2-y), MAE=np.mean(abs(t2-y)), rmslog=rmslog(t2, y), med_ratio=np.median(t2/y),
                         LR_slope=lr[0], LR_icpt=lr[1], misorder_1K=misorder(t2, y),
                         mu_best_abs=mus[np.argmin(rl)], rmslog_best=min(rl), tau_at_best_abs=taus[np.argmin(rl)],
                         mu_best_tau=mus[np.argmax(taus)], tau_best=max(taus),
                         tau_lambda_only=stats.kendalltau(lam, y)[0], lam_min=lam.min(), lam_max=lam.max()))
    print(pd.DataFrame(rows).set_index('version').T.round(3).to_string())
# which alloys change most
d = df.drop_duplicates('kkr_dir').copy(); d['ratio'] = d.lambda_G/d.lambda_A
d['Tc_A'] = mc(d.theta_mix, d.lambda_mix_A, 0.2); d['Tc_B'] = mc(d.theta_mix, d.lambda_mix_G, 0.2)
print('\nlambda_G/lambda_A over unique alloys:'); print(d.ratio.describe().round(3).to_string())
els = ['Ti','Nb','Zr','Hf','Ta','Mo','W','Y','Re']
print('\nSpearman(ratio, x_el):', {e: round(stats.spearmanr(d[e], d.ratio)[0], 2) for e in els if (d[e] > 0).sum() > 3})
cols = ['kkr_dir','Tc_exp','lambda_mix_A','lambda_mix_G','ratio','Tc_A','Tc_B']
print('\nlargest changes:'); print(pd.concat([d.nsmallest(6,'ratio')[cols], d.nlargest(6,'ratio')[cols]]).round(3).to_string(index=False))
df.to_csv(f'{RES}/fit_cutoff_vs_surface_rows.csv', index=False)
