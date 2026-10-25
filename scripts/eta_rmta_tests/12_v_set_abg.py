"""V-containing literature alloys: fit metrics for eta variants A (last-node cutoff, production), B (surface/MT),
G (GG with MT zero), from results.json / eta_surface.csv / eta_gg.csv. Uses baseline_loader with ELEMENT_SET=None.
Run from notebooks/:  python ../scripts/eta_rmta_tests/12_v_set_abg.py [eta_dir_tag]
eta_dir_tag selects alternative eta csv prefix (e.g. 'nrl' -> eta_surface_nrl.csv ...) — default SRA files."""
import sys, os, numpy as np, pandas as pd
from itertools import combinations
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
RES = '../results/comparison.sra.ew0.6.bz10-10-20'
src = open(os.path.join(HERE, 'baseline_loader.py')).read().replace(
    "ELEMENT_SET = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru']", "ELEMENT_SET = None")
g = {'RES': RES}; exec(src, g); df = g['df']
TAG = sys.argv[1] if len(sys.argv) > 1 else ''
suf = f'_{TAG}' if TAG else ''
es = pd.read_csv(f'{RES}/eta_surface{suf}.csv').set_index('name'); eg = pd.read_csv(f'{RES}/eta_gg{suf}.csv').set_index('name')
df['lambda_A'] = df['lambda']; df['lambda_B'] = df.kkr_dir.map(es.lambda_surface); df['lambda_G'] = df.kkr_dir.map(eg.lambda_gg)
if TAG:  # NRL etas: A-variant lambda from the nrl 'valence' csv if present
    ea = pd.read_csv(f'{RES}/eta_valence{suf}.csv').set_index('name'); df['lambda_A'] = df.kkr_dir.map(ea.lambda_valence)
for v in 'ABG': df[f'lambda_mix_{v}'] = df[f'lambda_{v}'] * (df.thetaDB_K / df.theta_mix)**2
def mc(th, lam, mu):
    den = lam - mu*(1+0.62*lam)
    with np.errstate(all='ignore'): t = th/1.45*np.exp(-1.04*(1+lam)/den)
    return np.where(den > 0, t, 0.0)
def rmslog(t, y): return np.sqrt(np.mean(np.log(np.clip(t, 1e-3, None)/y)**2))
mus = np.round(np.arange(0.05, 0.4001, 0.005), 3)
SETB = ['Ti','Nb','Zr','Hf','Ta','Mo','W','Y','Re','Ru']
out_el = [e for e in g['ELEMENTS'] if e not in SETB]
m_setB = (df[out_el] == 0).all(axis=1); m_V = df.V > 0; m_Sc = (df.Sc > 0) & ~m_V
subsets = [('set B', m_setB), ('V-containing', m_V), ('Sc, no V', m_Sc), ('set B + V', m_setB | m_V), ('all', np.ones(len(df), bool))]
rows = []
for sub, m in subsets:
    d = df[m].dropna(subset=['lambda_A','lambda_B','lambda_G']); y = d.Tc_exp.values
    for v in 'ABG':
        lam = d[f'lambda_mix_{v}'].values; th = d.theta_mix.values
        t2 = mc(th, lam, 0.2); rl = [rmslog(mc(th, lam, mu), y) for mu in mus]
        rows.append(dict(subset=sub, n=len(d), variant=v, tau=stats.kendalltau(t2, y)[0], r=stats.pearsonr(t2, y)[0] if len(d) > 2 else np.nan,
                         MAE=np.mean(abs(t2-y)), bias=np.mean(t2-y), med_ratio=np.median(t2/y), rmslog=rmslog(t2, y),
                         mu_best=mus[np.argmin(rl)], rmslog_best=min(rl), med_ratio_at_best=np.median(mc(th, lam, mus[np.argmin(rl)])/y)))
tab = pd.DataFrame(rows)
pd.set_option('display.width', 200)
print(tab.round(3).to_string(index=False))
d = df[m_V].drop_duplicates('kkr_dir').copy()
for v in 'ABG': d[f'Tc_{v}'] = mc(d.theta_mix, d[f'lambda_mix_{v}'], 0.2)
cols = ['kkr_dir','Tc_exp','V','Ti','theta_mix','lambda_mix_A','lambda_mix_B','lambda_mix_G','Tc_A','Tc_B','Tc_G']
print('\nV-containing alloys (unique compositions):'); print(d[cols].sort_values('V').round(3).to_string(index=False))
d['rG'] = d.lambda_G/d.lambda_A; d['rB'] = d.lambda_B/d.lambda_A
print('\nlambda_B/lambda_A: median %.3f, lambda_G/lambda_A: median %.3f' % (d.rB.median(), d.rG.median()))
print('Spearman(lambda_G/lambda_A, x_V) = %.2f, x_Ti = %.2f' % (stats.spearmanr(d.V, d.rG)[0], stats.spearmanr(d.Ti, d.rG)[0]))
tab.to_csv(f'{RES}/v_set_abg_metrics{suf}.csv', index=False); d.to_csv(f'{RES}/v_set_abg_rows{suf}.csv', index=False)
