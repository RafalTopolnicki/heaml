"""Data loaders shared by the figure scripts of the paper (PAPER/figures/make_*.py).

Everything is read from the repository (no numbers are typed in here):
  * literature baseline: results/Tc_literature.xlsx + results/comparison.sra.ew0.6.bz10-10-20.gl/
  * optimisation runs:   opt.ratios.gl/ (GL, mu*=0.1; set A Tc_mu0.1, set B Tc_corr / Tc_mu0.1), opt.ratios.mix/ (old A-mode)
  * binaries:            validation.ratios/binaries.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl/
  * thermodynamics:      thermodynamic/ (structure_summary.json of the champions and of the pure elements)
The loaders are copied from the notebook builders (notebooks/build_Tc_Baseline_Sets.py, build_OptGL_mu0.1_Runs.py,
build_OptGL_B_corr2_Runs.py) so that the paper and the notebooks give the same numbers.
"""
import json
import re
import sys
import tarfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import openpyxl
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
FIG = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'scripts'))
import src.process_kkr as pk                       # noqa: E402
from src.elements import ELEMENTS as ELEMENT_DATA  # noqa: E402

assert (dict(pk.TC_CORR_THETA_BCC_K), dict(pk.TC_CORR_SF_ALPHA), pk.TC_CORR_MU) == ({'Sc': 215.0}, {'V': 0.4}, 0.1), \
    'consts.py does not correspond to the corr2 version used in the set-B runs'

SET_A = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'Re', 'Ru']
SET_B = ['Ti', 'Nb', 'Zr', 'Hf', 'Ta', 'Mo', 'W', 'Y', 'V', 'Sc']
RATIO = 0.05           # an element "counts" above this fraction (min_components_ratio of the optimiser)
RY_MEV = 13.605693122994 * 1000

# ----------------------------------------------------------------------------------------------------------------------
# McMillan
# ----------------------------------------------------------------------------------------------------------------------
def mcmillan(theta, lam, mu):
    denom = lam - mu * (1 + 0.62 * lam)
    return theta / 1.45 * np.exp(-1.04 * (1 + lam) / denom) if denom > 0 else 0.0


# ----------------------------------------------------------------------------------------------------------------------
# Literature baseline (identical to notebooks/build_Tc_Baseline_Sets.py)
# ----------------------------------------------------------------------------------------------------------------------
XLSX = ROOT / 'results/Tc_literature.xlsx'
BASELINE_DIR = ROOT / 'results/comparison.sra.ew0.6.bz10-10-20.gl'
EXCLUDE = {'NbTa67MoHfW33'}
XLSX_ELEMENTS = ['Sc', 'Ti', 'Y', 'Zr', 'Nb', 'Mo', 'Ru', 'La', 'Hf', 'Ta', 'W', 'Re', 'V']
COL_NAME, COL_EL0, COL_LINK, COL_TC_EXP, COL_AUTHOR, COL_METHOD, COL_STATUS = 1, 2, 15, 19, 26, 27, 31
COMP_TOL = 2e-3


def _val(cell):
    try:
        return float(cell.value)
    except (TypeError, ValueError):
        return np.nan


def load_baseline():
    """One row per literature entry with a KKR result: experiment, Tc_mu0.1, Tc_mu0.2, Tc_corr2, set membership."""
    kkr = {}
    for d in sorted(BASELINE_DIR.iterdir()):
        p = d / 'results.json'
        if p.is_file():
            j = json.loads(p.read_text())
            if j is None or j.get('thetaDB_K') is None:
                continue
            x = np.array([float(j.get(e) or 0.0) for e in XLSX_ELEMENTS])
            kkr[d.name] = (x / x.sum(), j)
    ws = openpyxl.load_workbook(XLSX, data_only=True)['Tc_literature']
    rows, unmatched = [], []
    for r in range(4, ws.max_row + 1):
        name, tc = ws.cell(r, COL_NAME).value, _val(ws.cell(r, COL_TC_EXP))
        x = np.nan_to_num([_val(ws.cell(r, COL_EL0 + i)) for i in range(len(XLSX_ELEMENTS))])
        if not name or np.isnan(tc) or x.sum() == 0 or name in EXCLUDE:
            continue
        x = x / x.sum()
        k = min(kkr, key=lambda k: np.abs(kkr[k][0] - x).max())
        if np.abs(kkr[k][0] - x).max() > COMP_TOL:
            unmatched.append(name)
            continue
        j = kkr[k][1]
        els = {e for e, xi in zip(XLSX_ELEMENTS, x) if xi > 0}
        lam_mix = j['lambda'] * (j['thetaDB_K'] / j['mixture_debye_temperature']) ** 2
        c = pk.compute_tc_corrected({'thetaDB': j['mixture_debye_temperature'], 'lambda': lam_mix, 'debye_mode': 'mix',
                                     **{f'{e}_Ntot': j.get(f'{e}_Ntot') for e in XLSX_ELEMENTS}},
                                    dict(zip(XLSX_ELEMENTS, kkr[k][0])))
        status = str(ws.cell(r, COL_STATUS).value or '')
        rows.append(dict(name=name, author=str(ws.cell(r, COL_AUTHOR).value or ''), doi=str(ws.cell(r, COL_LINK).value or ''),
                         method=str(ws.cell(r, COL_METHOD).value or ''), kkr_dir=k, Tc_exp=tc,
                         caution=status.startswith('caution'), status=status,
                         in_A=els <= set(SET_A), in_B=els <= set(SET_B), has_VSc=bool(els & {'V', 'Sc'}),
                         n_el=int((x > 0).sum()), **dict(zip(XLSX_ELEMENTS, x)),
                         theta_mix=j['mixture_debye_temperature'], theta_kkr=j['thetaDB_K'],
                         lambda_kkr=j['lambda'], lambda_mix=lam_mix,
                         **{'Tc_mu0.1': mcmillan(j['mixture_debye_temperature'], lam_mix, 0.1),
                            'Tc_mu0.2': mcmillan(j['mixture_debye_temperature'], lam_mix, 0.2),
                            'Tc_kkr_theta_mu0.1': mcmillan(j['thetaDB_K'], j['lambda'], 0.1),
                            'Tc_corr2': c['Tc_corr']},
                         theta_corr=c['theta_corr'], lambda_corr=c['lambda_corr'], lambda_es=c['lambda_es']))
    df = pd.DataFrame(rows)
    noVSc = ~df.has_VSc
    assert np.allclose(df.loc[noVSc, 'Tc_corr2'], df.loc[noVSc, 'Tc_mu0.1'], atol=1e-12, rtol=0)
    df.attrs['unmatched'] = unmatched
    return df


def pair_accuracy(x, y, dT=0.0):
    """Fraction of pairs (i, j) with |y_i - y_j| > dT whose order is reproduced by x (ties in x count as wrong)."""
    good = bad = 0
    for i, j in combinations(range(len(y)), 2):
        if abs(y[i] - y[j]) > dT:
            s = np.sign(y[i] - y[j]) * np.sign(x[i] - x[j])
            good += s > 0
            bad += s <= 0
    return good / (good + bad) if good + bad else np.nan


def metrics(x, y):
    """x = calculated, y = experimental Tc."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    lr = stats.linregress(x, y)
    return {'n': int(len(y)), 'tau': stats.kendalltau(x, y)[0], 'rho': stats.spearmanr(x, y)[0],
            'pair_acc_all': pair_accuracy(x, y, 0.0), 'pair_acc_1K': pair_accuracy(x, y, 1.0),
            'r': lr.rvalue, 'slope': lr.slope, 'intercept': lr.intercept, 'slope0': float(np.sum(x * y) / np.sum(x * x)),
            'MAE': float(np.mean(np.abs(x - y))), 'bias': float(np.mean(x - y)), 'median_ratio': float(np.median(x / y))}


# ----------------------------------------------------------------------------------------------------------------------
# Optimisation runs (identical to notebooks/build_OptGL_mu0.1_Runs.py / build_OptGL_B_corr2_Runs.py)
# ----------------------------------------------------------------------------------------------------------------------
OPT_GL = ROOT / 'opt.ratios.gl'
OPT_OLD = ROOT / 'opt.ratios.mix'
RUN_A = 'sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.mu0.1.acqbeta{beta}.stonerbeta0.mixdebye.mincomp{mc}.r0.05.max0.7.face0.1'
RUN_A_INT = 'sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.gl.interior.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp{mc}.r0.05.max0.7.face0.1'
RUN_B_CORR2 = 'sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.corr2.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp{mc}.r0.05.max0.7.face0.1'
RUN_B_PLAIN = 'sra.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl.mu0.1.acqbeta2.stonerbeta0.mixdebye.mincomp{mc}.r0.05.max0.7.face0.1'
RUN_A_OLD = 'sra.TiNbZrHfTaMoWYReRu.kp10.ew0.6.slim.mu0.2.acqbeta{beta}.stonerbeta0.mixdebye.mincomp{mc}.r0.05.max0.75.face0.1'


def parse_name(name):
    return {el: float(x) for el, x in re.findall(r'([A-Z][a-z]?)(\d+\.\d+)', name)}


def comp_str(row, elements, thr=0.005):
    parts = [(el, row[el]) for el in elements if row.get(el, 0.0) > thr]
    return ' '.join(f'{el}{100 * v:.0f}' for el, v in sorted(parts, key=lambda p: -p[1]))


def comp_tex(row, elements, thr=0.005):
    parts = [(el, row[el]) for el in elements if row.get(el, 0.0) > thr]
    return ''.join(f'{el}$_{{{100 * v:.0f}}}$' for el, v in sorted(parts, key=lambda p: -p[1]))


def iter_results(it_dir):
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
            j = json.loads(rf.read_text())
            if j is not None:
                yield j, (json.loads(sf.read_text()) if sf.exists() else {})


def load_run(run_dir, target, elements, key=None):
    """All evaluated compositions of one run: init set (iteration 0) + every iteration.  Tc = `target`.
    For target 'Tc_corr' the stored value (corr2 in the set-B corr2 runs) is used; the notebook asserts it equals
    process_kkr.compute_tc_corrected on the stored data."""
    run_dir = Path(run_dir)
    params = json.loads((run_dir / 'parameters.json').read_text())
    mc = params['min_components']
    beta = float(params.get('acquisition_beta', np.nan))
    init = pd.read_csv(run_dir / 'init_tc_comparison.csv')
    col = f'{target}_computed'
    comps = pd.DataFrame([parse_name(n) for n in init['name']]).reindex(columns=elements).fillna(0.0)
    init = pd.concat([comps, init[[col]].rename(columns={col: 'Tc'})], axis=1)
    init['iteration'] = 0
    init['source'] = 'init'
    rows = []
    for it_dir in run_dir.glob('iteration_*'):
        it = int(it_dir.name.split('_')[1])
        for r, s in iter_results(it_dir):
            row = {el: float(r.get(el) or 0.0) for el in elements}
            row.update(iteration=it, source=s.get('source', '?'), Tc=r.get(target),
                       lam=r.get('lambda'), theta=r.get('thetaDB_K'), eta=r.get('eta_mix'), VEC=r.get('VEC'),
                       pred=s.get('pred_target'), pred_std=s.get('pred_target_std'), dist=s.get('composition_distance'))
            if target == 'Tc_corr':
                c = pk.compute_tc_corrected(r, {el: row[el] for el in elements})
                assert abs(c['Tc_corr'] - r['Tc_corr']) < 1e-8, (run_dir.name, it)
                row.update(lam_corr=c['lambda_corr'], lam_es=c['lambda_es'], theta_corr=c['theta_corr'])
            rows.append(row)
    opt = pd.DataFrame(rows).sort_values('iteration').reset_index(drop=True)
    df = pd.concat([init, opt], ignore_index=True)
    df['n_comp'] = (df[elements] > RATIO).sum(axis=1).astype(int)
    df['valid'] = df['n_comp'] >= mc
    df['comp'] = df.apply(lambda r: comp_str(r, elements), axis=1)
    df['mincomp'] = mc
    df['acqbeta'] = beta
    df['key'] = key or run_dir.name
    log = (run_dir / 'optimization_log.txt').read_text()
    m = re.search(r'EARLY STOP after iteration (\d+)', log)
    early = int(m.group(1)) if m else None
    return dict(mc=mc, beta=beta, params=params, df=df, early_stop=early,
                n_iter=int(opt['iteration'].max()), n_done=int(opt['iteration'].nunique()), key=key or run_dir.name)


def champion(run):
    """Best valid composition of a run (over init + iterations)."""
    v = run['df'][run['df'].valid]
    return v.loc[v.Tc.idxmax()]


def pareto(runs):
    """Best Tc at exactly k and at >= k elements over all evaluated points of the given runs."""
    U = pd.concat([r['df'] for r in runs], ignore_index=True)
    elements = [c for c in U.columns if c in SET_A + SET_B]
    U = U.drop_duplicates(subset=elements)
    exact = U.groupby('n_comp').Tc.max()
    ge = pd.Series({k: U[U.n_comp >= k].Tc.max() for k in range(1, 11)})
    return U, exact, ge


def load_series(which, mcs=range(2, 11)):
    """which in {'A', 'B_corr2', 'B_plain'}: the mincomp series (beta = 2)."""
    if which == 'A':
        return [load_run(OPT_GL / RUN_A.format(beta=2, mc=mc), 'Tc_mu0.1', SET_A, f'A_mc{mc}') for mc in mcs]
    if which == 'B_corr2':
        return [load_run(OPT_GL / RUN_B_CORR2.format(mc=mc), 'Tc_corr', SET_B, f'Bcorr2_mc{mc}') for mc in mcs]
    if which == 'B_plain':
        return [load_run(OPT_GL / RUN_B_PLAIN.format(mc=mc), 'Tc_mu0.1', SET_B, f'Bplain_mc{mc}') for mc in mcs]
    if which == 'A_old':
        return [load_run(OPT_OLD / RUN_A_OLD.format(beta=2, mc=mc), 'Tc_mu0.2', SET_A, f'Aold_mc{mc}') for mc in mcs]
    raise ValueError(which)


# ----------------------------------------------------------------------------------------------------------------------
# Binaries (validation.ratios/binaries...gl): KKR lines Nb-Zr, Nb-Ti, Nb-Hf, ...
# ----------------------------------------------------------------------------------------------------------------------
BIN_DIR = ROOT / 'validation.ratios/binaries.TiNbZrHfTaMoWYVSc.kp10.ew0.6.gl'


def load_binaries():
    rows = []
    for d in sorted(BIN_DIR.glob('Ti*')):
        p = d / 'results.json'
        comp = {e: float(x) for e, x in re.findall(r'([A-Z][a-z]?)(\d\.\d+)', d.name)}
        nz = {e: v for e, v in comp.items() if v > 0}
        if len(nz) != 2 or not p.exists():
            continue
        j = json.loads(p.read_text())
        if j is None or j.get('thetaDB_K') is None:
            rows.append(dict(system='-'.join(sorted(nz, key=lambda e: -nz[e])), **nz, ok=False))
            continue
        lam_mix = j['lambda'] * (j['thetaDB_K'] / j['mixture_debye_temperature']) ** 2
        c = pk.compute_tc_corrected({'thetaDB': j['mixture_debye_temperature'], 'lambda': lam_mix, 'debye_mode': 'mix',
                                     **{f'{e}_Ntot': j.get(f'{e}_Ntot') for e in SET_B}}, comp)
        rows.append(dict(system='-'.join(sorted(nz, key=lambda e: -nz[e])), **nz, ok=True,
                         theta_mix=j['mixture_debye_temperature'], lambda_mix=lam_mix,
                         **{'Tc_mu0.1': mcmillan(j['mixture_debye_temperature'], lam_mix, 0.1), 'Tc_corr2': c['Tc_corr']}))
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# Thermodynamics of the champions (identical to the TD section of the OptGL notebooks)
# ----------------------------------------------------------------------------------------------------------------------
TD_LIST = ROOT / 'slurm/compositions/td_champions_gl.csv'
EL_TD = ROOT / 'thermodynamic/elements.sra.ew0.6.topola'
STRUCT = ('bcc', 'fcc', 'hcp')
# Pure W and Re are not converged with respect to the energy window at the production ew = 0.6 Ry (the bottom of their
# valence band lies below the contour; W bcc a0 6.26 vs exp 5.98 bohr, Re bcc ground state).  Their references are
# taken from the ew = 1.0 Ry scans (bcc/fcc/hcp, slurm/compositions/td_re_ew1.0.csv).  Converged energies do not depend
# on the window: Ti, Nb, Zr, Ta, Mo, Y differ between ew 0.6 and 1.0 by 1-5 meV, and the champions A mc9/mc10 (bcc at
# a0) by 0.06-0.08 mRy between ew 0.6 and 0.7, so the alloys and the other elements stay at ew 0.6.  (A wider window
# cannot be used for Hf: at ew >= 0.75 its 4f core level enters the contour and the SCF does not converge.)
EL_TD_WIDE = ROOT / 'thermodynamic.ew1.0/elements.sra.topola'
REF_WIDE_EW = {'W', 'Re'}


def element_td(el):
    return read_td((EL_TD_WIDE if el in REF_WIDE_EW else EL_TD) / f'{el}1.0000')


def e_bcc_ref(el):
    """bcc reference energy (Ry) used in the mixing enthalpy: ebulk of elements.py (ew 0.6), or the ew 1.0 scan."""
    return element_td(el)['E_bcc'] if el in REF_WIDE_EW else ELEMENT_DATA[el].ebulk


Z_TO_EL = {22: 'Ti', 41: 'Nb', 40: 'Zr', 72: 'Hf', 73: 'Ta', 42: 'Mo', 74: 'W', 39: 'Y', 75: 'Re', 44: 'Ru', 23: 'V', 21: 'Sc'}


def kkr_input_composition(inp):
    """Concentrations exactly as AkaiKKR used them: the (rounded) values of the input file, normalised to one as in
    specx.f.  Total energies are ~1e4 Ry, so the mixing enthalpy must use these and not the unrounded composition:
    a difference of 5e-7 in x times ~3e4 Ry (Re, W, Hf, Ta vs Ti) is ~0.015 Ry ~ 200 meV."""
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
    comps = {st: kkr_input_composition(sorted((d / st).glob('*.inp'))[0]) for st in STRUCT}
    assert comps['bcc'] == comps['fcc'] == comps['hcp'], d
    assert all(abs(comps['bcc'].get(el, 0) - x) < 1e-5 for el, x in s['composition'].items()), d
    t.update(most_stable=s['most_stable'], a_bcc=s['bcc']['lattice_constant_bohr'], B_bcc=s['bcc']['bulk_modulus_gpa'],
             dE_fcc=(t['E_fcc'] - t['E_bcc']) * RY_MEV, dE_hcp=(t['E_hcp'] - t['E_bcc']) * RY_MEV, comp=comps['bcc'])
    return t


def element_refs(elements):
    rows = []
    for el in elements:
        t = element_td(el)
        rows.append({'el': el, 'ew': 1.0 if el in REF_WIDE_EW else 0.6,
                     'gs_KKR': min(STRUCT, key=lambda st: t[f'E_{st}']), 'dE_fcc': t['dE_fcc'], 'dE_hcp': t['dE_hcp'],
                     'a_bcc': t['a_bcc'], 'E_bcc_minus_ref': (t['E_bcc'] - e_bcc_ref(el)) * RY_MEV,
                     'E_gs_minus_E_bcc': min(0.0, t['dE_fcc'], t['dE_hcp'])})
    return pd.DataFrame(rows).set_index('el')


def champions_td(prefix):
    """prefix in {'A_mu0.1', 'B_corr2', 'B_mu0.1'} (tags of slurm/compositions/td_champions_gl.csv)."""
    REF = element_refs(sorted(set(SET_A + SET_B)))
    lst = pd.read_csv(TD_LIST)
    lst = lst[lst.tag.str.startswith(prefix + '_mc')]
    rows = []
    for _, r in lst.iterrows():
        mc = int(r.tag.rsplit('_mc', 1)[1])
        t = read_td(ROOT / r.outdir / r.dirname)
        c = {el: x for el, x in t['comp'].items() if x > 0}
        e_ref = sum(x * e_bcc_ref(el) for el, x in c.items())
        e_min = min(t[f'E_{st}'] for st in STRUCT)
        s_conf = -sum(x * np.log(x) for x in c.values())
        rows.append({'mincomp': mc, 'Tc': r.Tc, 'comp': c, 'most_stable': t['most_stable'],
                     'dE_fcc': t['dE_fcc'], 'dE_hcp': t['dE_hcp'], 'a_bcc': t['a_bcc'], 'B_bcc': t['B_bcc'],
                     'S_conf_kB': s_conf,
                     'dH_bcc': (t['E_bcc'] - e_ref) * RY_MEV,
                     'dH_gs': (e_min - e_ref) * RY_MEV - sum(x * REF.loc[el, 'E_gs_minus_E_bcc'] for el, x in c.items())})
    return pd.DataFrame(rows).set_index('mincomp').sort_index(), REF


def save_numbers(name, obj):
    """Store numbers used in the text (PAPER/numbers/<name>.json) so that every quoted value can be traced."""
    out = ROOT / 'PAPER/numbers'
    out.mkdir(exist_ok=True)

    def conv(o):
        if isinstance(o, dict):
            return {str(k): conv(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [conv(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if np.isnan(o) else round(float(o), 4)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, pd.Series):
            return conv(o.to_dict())
        if isinstance(o, pd.DataFrame):
            return conv(o.to_dict(orient='index'))
        return o
    (out / f'{name}.json').write_text(json.dumps(conv(obj), indent=1, ensure_ascii=False))
