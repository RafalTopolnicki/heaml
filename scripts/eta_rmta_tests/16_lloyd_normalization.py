"""Use the Lloyd-formula total DOS at E_F (AkaiKKR 'def=' in the SCF output, per spin, per cell) instead of the
site-projected sum N = sum_l dosef in the 1/N denominator of the GG formula. eta_i^corr = eta_i * N_site_avg / N_Lloyd.
(a) alloy Ta34Nb33Hf8Zr14Ti11 and pure V/Nb per element vs Jaskiewicz / Papaconstantopoulos;
(b) baseline fit (set B, V set) with lambda_A, lambda_G and lambda_G*N_site/N_Lloyd.  Run from notebooks/."""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..'))
RES = '../results/comparison.sra.ew0.6.bz10-10-20'
ll = pd.read_csv(f'{RES}/lloyd_vs_site_dos.csv').set_index('name')
eg = pd.read_csv(f'{RES}/eta_gg.csv').set_index('name'); es = pd.read_csv(f'{RES}/eta_surface.csv').set_index('name')
print('=== (a) per-element eta_G rescaled by N_site_avg/N_Lloyd  [mRy/a0^2]')
JASK = {'Ta': 154.0, 'Nb': 162.0, 'Hf': 160.6, 'Zr': 176.1, 'Ti': 126.2}
for name, els, ref in [('Ta34Nb33Hf8Zr14Ti11', ['Ta', 'Nb', 'Hf', 'Zr', 'Ti'], JASK), ('V', ['V'], {'V': 141.9}), ('Nb', ['Nb'], {'Nb': 157.0}), ('Ta', ['Ta'], {'Ta': 155.0}), ('Ti', ['Ti'], {'Ti': 100.0}), ('Mo', ['Mo'], {'Mo': 119.0}), ('Zr', ['Zr'], {'Zr': 96.0})]:
    f = ll.loc[name, 'site_sum'] / ll.loc[name, 'lloyd_scf']
    line = f'{name:22s} N_site/N_Lloyd = {ll.loc[name,"site_sum"]:.2f}/{ll.loc[name,"lloyd_scf"]:.2f} = {f:.3f} | '
    for el in els:
        g = 1e3 * eg.loc[name, f'{el}_eta_total_gg']; line += f'{el}: G {g:6.1f} -> {g*f:6.1f} (ref {ref[el]:5.1f}, ratio {g*f/ref[el]:.2f})  '
    if len(els) > 1: line += f'| eta_mix G {1e3*eg.loc[name,"eta_mix_gg"]:.1f} -> {1e3*eg.loc[name,"eta_mix_gg"]*f:.1f} (Jask 157.2)'
    print(line)
print('\n=== (b) fit to experiment, theta_mix, mu*=0.2')
src = open(os.path.join(HERE, 'baseline_loader.py')).read().replace("ELEMENT_SET = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru']", "ELEMENT_SET = None")
g = {'RES': RES}; exec(src, g); df = g['df']
df['lambda_A'] = df['lambda']; df['lambda_G'] = df.kkr_dir.map(eg.lambda_gg); df['lambda_B'] = df.kkr_dir.map(es.lambda_surface)
df['fL'] = df.kkr_dir.map(ll.site_sum / ll.lloyd_scf); df['lambda_GL'] = df.lambda_G * df.fL; df['lambda_AL'] = df.lambda_A * df.fL
def mc(th, lam, mu):
    den = lam - mu * (1 + 0.62 * lam)
    with np.errstate(all='ignore'): t = th / 1.45 * np.exp(-1.04 * (1 + lam) / den)
    return np.where(den > 0, t, 0.0)
def rmslog(t, y): return np.sqrt(np.mean(np.log(np.clip(t, 1e-3, None) / y)**2))
mus = np.round(np.arange(0.05, 0.4001, 0.005), 3)
SETB = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru']; out_el = [e for e in g['ELEMENTS'] if e not in SETB]
m_setB = (df[out_el] == 0).all(axis=1); m_V = df.V > 0
rows = []
for sub, m in [('set B', m_setB), ('V-containing', m_V), ('Sc, no V', (df.Sc > 0) & ~m_V), ('set B + V', m_setB | m_V)]:
    d = df[m].dropna(subset=['lambda_A', 'lambda_G', 'fL']); y = d.Tc_exp.values
    for v in ['A', 'B', 'G', 'AL', 'GL']:
        lam = (d[f'lambda_{v}'] * (d.thetaDB_K / d.theta_mix)**2).values; th = d.theta_mix.values
        t2 = mc(th, lam, 0.2); rl = [rmslog(mc(th, lam, mu), y) for mu in mus]; ib = int(np.argmin(rl))
        tb = mc(th, lam, mus[ib])
        rows.append(dict(subset=sub, n=len(d), variant=v, tau=stats.kendalltau(t2, y)[0], r=stats.pearsonr(t2, y)[0], MAE=np.mean(abs(t2 - y)),
                         med_ratio=np.median(t2 / y), rmslog=rmslog(t2, y), mu_best=mus[ib], rmslog_best=rl[ib], tau_best=stats.kendalltau(tb, y)[0], MAE_best=np.mean(abs(tb - y))))
pd.set_option('display.width', 200); print(pd.DataFrame(rows).round(3).to_string(index=False))
print('\nfL = N_site/N_Lloyd over alloys: median %.3f, range %.3f-%.3f' % (df.fL.median(), df.fL.min(), df.fL.max()))
print('Spearman(fL, x_el):', {e: round(stats.spearmanr(df[e], df.fL)[0], 2) for e in ['Ti', 'V', 'Sc', 'Nb', 'Ta', 'Hf', 'Zr', 'Mo', 'W'] if (df[e] > 0).sum() > 5})
