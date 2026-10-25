"""Recompute eta for every process_hea workdir of a results directory in variant GL = Gaspari-Gyorffy surface formula
(cutoff_mode='gg') with the Lloyd total DOS in the 1/N denominator (`def=` of the last SCF iteration of
finalscf_*.out[.gz], macmillan.parse_scf_lloyd_dos) — per element eta_i^GL = X_i / N_Lloyd (Jaskiewicz 2016 Eq. 2),
no KKR run.  Also stores eta_i^G (own site DOS in 1/N), the production A values from results.json and
eta_mix_gl_avg = eta_mix_G * N_site_avg / N_Lloyd (the single-factor rescaling of 16_lloyd_normalization.py).
Then: (a) alloy Ta34Nb33Hf8Zr14Ti11 and pure elements vs literature, (b) Tc fit vs experiment (theta_mix) for
variants A (production) and GL with a mu* scan, subsets set B / V-containing / Sc without V / B+V.
Usage (repo root): python scripts/eta_rmta_tests/18_recompute_baseline_gl.py [results/comparison.sra.ew0.6.bz10-10-20]
Writes <base>/eta_gl.csv and <base>/fit_gl.csv."""
import sys, os, json, numpy as np, pandas as pd
from multiprocessing import Pool
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, 'scripts'))
from common import fort51_path
from macmillan import parse_fort51, parse_scf_lloyd_dos, find_finalscf_out
from macmillan_cutoff import compute_one_combination_cutoff, C_THEORETICAL
BASE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else 'results/comparison.sra.ew0.6.bz10-10-20')

def one(wd):
    try:
        rp = json.load(open(os.path.join(wd, 'run_params.json'))); rj = json.load(open(os.path.join(wd, 'results.json')))
        d = parse_fort51(fort51_path(wd)); ef = float(np.atleast_1d(d['ef'])[0])
        ll = parse_scf_lloyd_dos(find_finalscf_out(os.path.join(wd, 'finalscf'))); NL = ll['def'][0]
        labels = rp['element_labels']; conc = np.asarray(rp['concentrations'], float); conc /= conc.sum()
        row = dict(name=os.path.basename(wd), ef_Ry=ef, N_lloyd=NL, thetaDB_K=rj['thetaDB_K'], mixture_mass=rj['mixture_mass'],
                   mixture_debye_temperature=rj['mixture_debye_temperature'], eta_mix_A=rj['eta_mix'], lambda_A=rj['lambda'])
        eG = eGL = Ns = 0.0
        for b in d['components']:
            if b['spin'] != 1: continue
            b['ef'] = ef; i = b['component'] - 1; el = labels[i]
            g = compute_one_combination_cutoff(b, 'plain', 'none', 'mean', cutoff_mode='gg')
            gl = compute_one_combination_cutoff(b, 'plain', 'none', 'mean', cutoff_mode='gg', total_dos=NL)
            for ch in ('sp', 'pd', 'df'): row[f'{el}_eta_{ch}_gl'] = gl[f'eta_cutoff_{ch}']
            row[f'{el}_eta_total_g'] = g['eta_total_cutoff']; row[f'{el}_eta_total_gl'] = gl['eta_total_cutoff']
            row[f'{el}_Ntot'] = g['Ntot']; row[f'{el}_eta_total_A'] = rj.get(f'{el}_eta_total')
            eG += conc[i] * g['eta_total_cutoff']; eGL += conc[i] * gl['eta_total_cutoff']; Ns += conc[i] * g['Ntot']
        row.update(N_site_avg=Ns, eta_mix_g=eG, eta_mix_gl=eGL, eta_mix_gl_avg=eG * Ns / NL)
        th = rj['thetaDB_K']
        for v in ('g', 'gl', 'gl_avg'):
            row[f'lambda_{v}'] = row[f'eta_mix_{v}'] * C_THEORETICAL / (rj['mixture_mass'] * th**2) if th else np.nan
        return row
    except Exception as e:
        return dict(name=os.path.basename(wd), error=repr(e))

if __name__ == '__main__':
    dirs = sorted(os.path.join(BASE, n) for n in os.listdir(BASE) if os.path.isfile(os.path.join(BASE, n, 'results.json')))
    with Pool(12) as p: rows = p.map(one, dirs)
    df = pd.DataFrame(rows)
    if 'error' in df:
        bad = df[df.error.notna()]; print(f'{len(bad)} failed:'); print(bad[['name', 'error']].to_string()); df = df[df.error.isna()]
    out = os.path.join(BASE, 'eta_gl.csv'); df.to_csv(out, index=False); print(f'{len(df)} ok -> {out}')
    print('N_lloyd/N_site_avg: median %.3f, range %.3f-%.3f' % ((df.N_lloyd / df.N_site_avg).median(), (df.N_lloyd / df.N_site_avg).min(), (df.N_lloyd / df.N_site_avg).max()))
    print('lambda_GL/lambda_A: median %.3f, range %.3f-%.3f' % ((df.lambda_gl / df.lambda_A).median(), (df.lambda_gl / df.lambda_A).min(), (df.lambda_gl / df.lambda_A).max()))
    df = df.set_index('name')
    print('\n=== (a) per element, mRy/a0^2: A (prod) | G | GL = X_i/N_Lloyd | GL_avg (16-style) | literature')
    JASK = {'Ta': 154.0, 'Nb': 162.0, 'Hf': 160.6, 'Zr': 176.1, 'Ti': 126.2}
    for name, els, ref, src in [('Ta34Nb33Hf8Zr14Ti11', ['Ta', 'Nb', 'Hf', 'Zr', 'Ti'], JASK, 'Jaskiewicz 2016'),
                                ('V', ['V'], {'V': 141.9}, 'Papa 1977 (bz20 unconverged)'), ('Nb', ['Nb'], {'Nb': 157.0}, 'Papa 1977 (bz20 unconverged)'),
                                ('Ta', ['Ta'], {'Ta': 155.0}, 'Jaskiewicz Tab. 2'), ('Mo', ['Mo'], {'Mo': 119.0}, 'Papa 1977'),
                                ('Ti', ['Ti'], {'Ti': 100.0}, 'Papa 1977 bcc'), ('Zr', ['Zr'], {'Zr': 96.0}, 'Papa 1977 bcc')]:
        if name not in df.index: print(f'{name}: not in base'); continue
        r = df.loc[name]; f = r.N_site_avg / r.N_lloyd
        print(f'{name:22s} N_site_avg/N_Lloyd {r.N_site_avg:.2f}/{r.N_lloyd:.2f} = {f:.3f}   [{src}]')
        for el in els:
            a, g, gl = 1e3 * r[f'{el}_eta_total_A'], 1e3 * r[f'{el}_eta_total_g'], 1e3 * r[f'{el}_eta_total_gl']
            print(f'    {el:3s} N_i={2*r[f"{el}_Ntot"]:5.2f}  A {a:6.1f} | G {g:6.1f} | GL {gl:6.1f} | GL_avg {g*f:6.1f} | lit {ref[el]:6.1f}  (GL/lit {gl/ref[el]:.2f})')
        if len(els) > 1:
            print(f'    eta_mix: A {1e3*r.eta_mix_A:.1f} | G {1e3*r.eta_mix_g:.1f} | GL {1e3*r.eta_mix_gl:.1f} | GL_avg {1e3*r.eta_mix_gl_avg:.1f} | Jask 157.2')
    print('\n=== (b) Tc vs experiment (theta_mix; lambda_mix = lambda*(theta_KKR/theta_mix)^2)')
    src = open(os.path.join(HERE, 'baseline_loader.py')).read().replace("ELEMENT_SET = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru']", "ELEMENT_SET = None")
    cwd = os.getcwd(); os.chdir(os.path.join(ROOT, 'notebooks'))
    g = {'RES': BASE}; exec(src, g); os.chdir(cwd); b = g['df']
    b['lambda_GL'] = b.kkr_dir.map(df.lambda_gl); b['lambda_G'] = b.kkr_dir.map(df.lambda_g); b['lambda_A'] = b['lambda']
    def mc(th, lam, mu):
        den = lam - mu * (1 + 0.62 * lam)
        with np.errstate(all='ignore'): t = th / 1.45 * np.exp(-1.04 * (1 + lam) / den)
        return np.where(den > 0, t, 0.0)
    def rmslog(t, y): return np.sqrt(np.mean(np.log(np.clip(t, 1e-3, None) / y)**2))
    def metrics(t, y): return dict(tau=stats.kendalltau(t, y)[0], r=stats.pearsonr(t, y)[0], MAE=np.mean(abs(t - y)), med_ratio=np.median(t / y), rmslog=rmslog(t, y))
    mus = np.round(np.arange(0.05, 0.4001, 0.005), 3)
    SETB = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru']; out_el = [e for e in g['ELEMENTS'] if e not in SETB]
    m_B = (b[out_el] == 0).all(axis=1); m_V = b.V > 0
    rows = []
    for sub, m in [('set B', m_B), ('set B, Ti>=30%', m_B & (b.Ti >= 0.3)), ('V-containing', m_V), ('Sc, no V', (b.Sc > 0) & ~m_V), ('set B + V', m_B | m_V)]:
        d = b[m].dropna(subset=['lambda_A', 'lambda_GL']); y = d.Tc_exp.values; th = d.theta_mix.values
        for v in ['A', 'G', 'GL']:
            lam = (d[f'lambda_{v}'] * (d.thetaDB_K / d.theta_mix)**2).values
            rl = [rmslog(mc(th, lam, mu), y) for mu in mus]; ib = int(np.argmin(rl))
            fixed = {'A': 0.2, 'G': 0.2, 'GL': 0.11}[v]
            rows.append(dict(subset=sub, n=len(d), variant=v, mu_fixed=fixed, **{k + '_fixed': val for k, val in metrics(mc(th, lam, fixed), y).items()},
                             mu_best=mus[ib], rmslog_best=rl[ib], med_ratio_best=np.median(mc(th, lam, mus[ib]) / y), MAE_best=np.mean(abs(mc(th, lam, mus[ib]) - y)),
                             lam_min=lam.min(), lam_max=lam.max()))
        if sub == 'set B':
            lam = (d.lambda_GL * (d.thetaDB_K / d.theta_mix)**2).values
            print('  set B, GL median ratio vs mu*: ' + '  '.join(f'{mu:.2f}:{np.median(mc(th, lam, mu)/y):.2f}' for mu in (0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.15)))
    fit = pd.DataFrame(rows); fit.to_csv(os.path.join(BASE, 'fit_gl.csv'), index=False)
    pd.set_option('display.width', 250); print(fit.round(3).to_string(index=False))
