"""Pure bcc elements from results/comparison.sra.ew0.6.bz10-10-20/<El> (SRA, PBE, final SCF bz20):
surface-formula eta (MT; variants D/E with WS renormalisation), phase shifts at E_F from the log-derivative at R_MT,
and partial DOS — to compare with Papaconstantopoulos et al. PRB 15, 4221 (1977) Tables II/III (1 eV/Å² = 20.58 mRy/a0²)."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fort51_path
from macmillan import parse_fort51, reduce_l_block_radials, group_dosef_by_l
from macmillan_cutoff import compute_M_surface
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.special import spherical_jn, spherical_yn
BASE = 'results/comparison.sra.ew0.6.bz10-10-20'
PAPA = {'Ti': (0.130, 1.427, 3.283), 'Zr': (0.246, 2.144, 2.268), 'Nb': (0.470, 2.726, 4.431), 'Mo': (0.069, 1.008, 4.720), 'Y': (0.048, 1.024, 1.266)}
for el in ['Ti', 'Zr', 'Nb', 'Mo', 'Y', 'Hf', 'Ta', 'W']:
    wd = f'{BASE}/{el}'
    if not os.path.isdir(wd): print(el, 'missing'); continue
    a0 = json.load(open(f'{wd}/results.json'))['a0_bohr']
    d = parse_fort51(fort51_path(wd)); ef = float(np.atleast_1d(d['ef'])[0]); b = d['components'][0]
    x = np.array(b['xr'], float)[:-1]; v = np.array(b['v3'], float)[:-1]; R = x[-1]; lnx = np.log(x)
    raw = {int(k): np.array(vv[:-1], float) for k, vv in b['rstr_real_ef'].items()}
    U = reduce_l_block_radials(raw, int(b['mxlcmp']), x, 'mean', 'none'); dos = group_dosef_by_l(b['dosef'], int(b['mxlcmp'])); N = sum(b['dosef'])
    V0 = v[-1]; k = np.sqrt(ef - V0); rws = a0 * (3 / (8 * np.pi))**(1 / 3); rr = np.linspace(R, rws, 2001)
    f = {}; delta = {}
    for l in range(4):
        u = U['spdf'[l]]; up = CubicSpline(lnx, u)(lnx[-1], 1) / R
        D = up / u[-1] - 1.0 / R                       # log-derivative of R_l = u/r at R
        j, n = spherical_jn(l, k * R), spherical_yn(l, k * R)
        jp, np_ = spherical_jn(l, k * R, derivative=True), spherical_yn(l, k * R, derivative=True)
        delta[l] = np.arctan((k * jp - D * j) / (k * np_ - D * n))   # tan δ_l, free solution j cosδ − n sinδ
        A, B = np.linalg.solve([[R * j, R * n], [j + k * R * jp, n + k * R * np_]], [u[-1], up])
        f[l] = simpson(u**2, x=x) / (simpson(u**2, x=x) + simpson((A * rr * spherical_jn(l, k * rr) + B * rr * spherical_yn(l, k * rr))**2, x=rr))
    e = []
    for l in range(3):
        Ms = compute_M_surface(U['spdf'[l]], U['spdf'[l + 1]], x, v, ef, l)
        e.append(1000 * 2 * (l + 1) / ((2 * l + 1) * (2 * l + 3)) * dos['spdf'[l]] * dos['spdf'[l + 1]] / N * Ms**2)
    sp, pd, df = e
    print(f'\n== {el}: a0={a0:.3f} bohr  R_MT={R:.3f}  E_F={ef:.4f} Ry  V(R)={V0:.3f}  N(EF)={2*N:.2f}/Ry (both spins)  n_l/spin: s {dos["s"]:.3f} p {dos["p"]:.3f} d {dos["d"]:.3f} f {dos["f"]:.3f}')
    print(f'   phase shifts δ_s,p,d,f [rad]: {delta[0]:.3f} {delta[1]:.3f} {delta[2]:.3f} {delta[3]:.3f}   sin²(δp−δs) {np.sin(delta[1]-delta[0])**2:.3f} sin²(δd−δp) {np.sin(delta[2]-delta[1])**2:.3f} sin²(δf−δd) {np.sin(delta[3]-delta[2])**2:.3f}')
    print(f'   eta B (surface, MT)  sp/pd/df/tot = {sp:6.1f} / {pd:6.1f} / {df:6.1f} / {sp+pd+df:6.1f}')
    print(f'   eta D (sp,pd WS; df MT)          = {sp*f[0]*f[1]:6.1f} / {pd*f[1]*f[2]:6.1f} / {df:6.1f} / {sp*f[0]*f[1]+pd*f[1]*f[2]+df:6.1f}    f_l = {f[0]:.2f} {f[1]:.2f} {f[2]:.2f} {f[3]:.2f}')
    if el in PAPA:
        p = [20.58 * t for t in PAPA[el]]; print(f'   Papaconstantopoulos 1977          = {p[0]:6.1f} / {p[1]:6.1f} / {p[2]:6.1f} / {sum(p):6.1f}')
