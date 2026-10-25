"""k-point convergence table: N_site (sum of site DOS at E_F), N_Lloyd (AkaiKKR 'def=' of the LAST SCF iteration, before the
go=dos block whose finite-difference def is meaningless), eta_A (valence cutoff), eta_G (GG surface formula) per element and
eta_mix, and eta_GL = eta_G * N_site / N_Lloyd, as a function of bzqlty.  N both spins in 1/Ry, eta in mRy/a0^2.
Usage (repo root):
  python scripts/eta_rmta_tests/17_kconv_table.py                                   # pure V series (local + results/test.kconv)
  python scripts/eta_rmta_tests/17_kconv_table.py results/test.kconv/TNHZT_a3.36_sra_pbe   # dirs <prefix>_bz<N> from slurm/topola_kconv.sbatch
Composition: <dir>/composition.json ({'anclr': [...], 'conc': [...]}, written by the sbatch) or pure V when absent."""
import sys, os, re, glob, gzip, json, numpy as np
sys.path.insert(0, 'scripts/eta_rmta_tests'); sys.path.insert(0, 'scripts')
from common import fort51_path
from macmillan import parse_fort51
from macmillan_cutoff import compute_one_combination_cutoff
from src.consts import VALENCE_NODES_EXPECTED
from src.elements import ELEMENTS
Z2EL = {e.atomic_number: s for s, e in ELEMENTS.items()}
REFS = {'V': 'ref pure V: Papaconstantopoulos 1977 N=25.4, eta G 4.1/33.8/104.0 = 141.9; Bose 2009 LR N 25.3, eta 147.6',
        'Nb': 'ref pure Nb: Papaconstantopoulos 1977 N=19.4, eta 9.7/56.1/91.2 = 157.0; Bose 2009 LR N 19.9, eta 178.6',
        'Ta34Nb33Hf8Zr14Ti11': 'ref alloy: Jaskiewicz 2016 eta_mix 157.2 (Ta 154.0, Nb 162.0, Hf 160.6, Zr 176.1, Ti 126.2)'}

if len(sys.argv) > 1:
    prefix = sys.argv[1].rstrip('/')
    runs = [(f'bz{int(re.search(r"_bz(\d+)$", d).group(1))}', d) for d in glob.glob(prefix + '_bz*') if re.search(r'_bz(\d+)$', d)]
else:
    runs = [('bz20 prod', 'results/comparison.sra.ew0.6.bz10-10-20/V'), ('bz30 old', 'results/test.nrl.ew0.6/V_sra_pbe_bz30')]
    runs += [(f'bz{bz} bigk', f'results/test.kconv/V_sra_pbe_bz{bz}') for bz in (15, 20, 25, 35, 40, 45, 50, 55, 60)]
runs.sort(key=lambda t: int(re.search(r'\d+', t[0]).group()))

def read_out(wd):
    for pat in ('finalscf_*.out', 'finalscf/*.out', 'finalscf/*.out.gz'):
        for f in sorted(glob.glob(os.path.join(wd, pat))):
            return (gzip.open(f, 'rt') if f.endswith('.gz') else open(f)).read()
    return ''

hdr_done = False
for tag, wd in runs:
    try: f51 = fort51_path(wd)
    except FileNotFoundError: print(f'{tag}: no fort.51 (not finished)'); continue
    cj = os.path.join(wd, 'composition.json')
    if os.path.exists(cj):
        c = json.load(open(cj)); labels = [Z2EL[z] for z in c['anclr']]; conc = np.asarray(c['conc'], float)
    else:
        labels, conc = ['V'], np.array([1.0])
    conc = conc / conc.sum(); name = ''.join(f'{l}{round(100*x)}' if len(labels) > 1 else l for l, x in zip(labels, conc))
    txt = read_out(wd)
    nk = re.findall(r'nk=\s*(\d+)', txt); nk = nk[0] if nk else '?'
    defs = re.findall(r'def=\s*([0-9.]+)', txt.split('go=dos')[0]); NL = 2 * float(defs[-1]) if defs else np.nan
    d = parse_fort51(f51); ef = float(np.atleast_1d(d['ef'])[0])
    if not hdr_done:
        print(REFS.get(name, f'composition {name}'))
        print(f'{"run":10s} {"nk":>6s} {"EF":>7s} {"N_site":>7s} {"N_Lloyd":>8s} {"Ns/NL":>6s} | per element: N  eta_G sp/pd/df/tot  eta_A | eta_mix_G  eta_mix_GL  eta_mix_A')
        hdr_done = True
    Ns = eG = eA = 0.0; cols = []
    for b in d['components']:
        if b['spin'] != 1: continue
        b['ef'] = ef; i = b['component'] - 1; el = labels[i]
        rg = compute_one_combination_cutoff(b, 'plain', 'none', 'mean', cutoff_mode='gg', valence_nodes=VALENCE_NODES_EXPECTED.get(el))
        ra = compute_one_combination_cutoff(b, 'plain', 'none', 'mean', cutoff_mode='valence', valence_nodes=VALENCE_NODES_EXPECTED.get(el))
        e = [1e3 * rg[f'eta_cutoff_{ch}'] for ch in ('sp', 'pd', 'df')]; g = sum(e); a = 1e3 * ra['eta_total_cutoff']
        Ns += conc[i] * 2 * rg['Ntot']; eG += conc[i] * g; eA += conc[i] * a
        cols.append(f'{el} N={2*rg["Ntot"]:5.2f} G {e[0]:4.1f}/{e[1]:5.1f}/{e[2]:5.1f}/{g:5.1f} A {a:5.1f}')
    f = Ns / NL if NL == NL else np.nan
    print(f'{tag:10s} {nk:>6s} {ef:7.4f} {Ns:7.2f} {NL:8.2f} {f:6.3f} | ' + '  '.join(cols) + f' | {eG:6.1f} {eG*f:6.1f} {eA:6.1f}')
