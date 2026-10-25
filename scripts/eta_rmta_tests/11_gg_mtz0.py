"""GG Eq. (7) with the free-wave matching done against the TRUE muffin-tin zero (V = 0 outside R_MT in AkaiKKR; the mesh value
v(R⁻) is the potential just inside the sphere, which jumps to 0 at R). This is the physically correct GG evaluation and
implicitly contains the potential-step contribution ΔV·u_l(R)u_{l+1}(R) that the interior integral / surface form (variant B) omits.
Prints variant B and G (= GG(7), MTZ=0) for pure elements and the alloy, vs Papaconstantopoulos 1977 / Jaśkiewicz 2016."""
import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fort51_path, labels_for
from macmillan import parse_fort51, reduce_l_block_radials, group_dosef_by_l
from macmillan_cutoff import compute_M_surface
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.special import spherical_jn, spherical_yn
def gg(wd, comp=0, V_mtz=0.0):
    d = parse_fort51(fort51_path(wd)); ef = float(np.atleast_1d(d['ef'])[0]); b = d['components'][comp]
    x = np.array(b['xr'], float)[:-1]; v = np.array(b['v3'], float)[:-1]; R = x[-1]; lnx = np.log(x)
    raw = {int(k): np.array(vv[:-1], float) for k, vv in b['rstr_real_ef'].items()}
    U = reduce_l_block_radials(raw, int(b['mxlcmp']), x, 'mean', 'none'); dos = group_dosef_by_l(b['dosef'], int(b['mxlcmp'])); N = sum(b['dosef'])
    k = np.sqrt(ef - V_mtz); dl = {}; r = {}
    for l in range(4):
        u = U['spdf'[l]]; up = CubicSpline(lnx, u)(lnx[-1], 1) / R
        j, y = spherical_jn(l, k * R), spherical_yn(l, k * R); jp, yp = spherical_jn(l, k * R, derivative=True), spherical_yn(l, k * R, derivative=True)
        A, B = np.linalg.solve([[R * j, R * y], [j + k * R * jp, y + k * R * yp]], [u[-1], up]); nrm = A * A + B * B; dl[l] = np.arctan2(-B, A)
        r[l] = dos['spdf'[l]] / ((k / np.pi) * (2 * l + 1) * simpson(u**2, x=x) / nrm)
    G = [1000 * (k * k / (np.pi**2 * N)) * 2 * (l + 1) * np.sin(dl[l + 1] - dl[l])**2 * r[l] * r[l + 1] for l in range(3)]
    Bv = [1000 * 2 * (l + 1) / ((2 * l + 1) * (2 * l + 3)) * dos['spdf'[l]] * dos['spdf'[l + 1]] / N * compute_M_surface(U['spdf'[l]], U['spdf'[l + 1]], x, v, ef, l)**2 for l in range(3)]
    return Bv, G, 2 * N, ef
BASE = 'results/comparison.sra.ew0.6.bz10-10-20'
PAPA = {'Ti': (2.7, 29.4, 67.6), 'Zr': (5.1, 44.1, 46.7), 'Nb': (9.7, 56.1, 91.2), 'Mo': (1.4, 20.7, 97.1), 'Y': (1.0, 21.1, 26.1)}
f = lambda e: f'{e[0]:5.1f}/{e[1]:5.1f}/{e[2]:5.1f}/{sum(e):6.1f}'
print('pure elements (bz20, SRA PBE):      B = surface/MT          G = GG(7), MTZ=0        reference (Papa 1977)')
for el in ['Ti', 'Zr', 'Nb', 'Mo', 'Y', 'Hf', 'Ta', 'W']:
    Bv, G, N, ef = gg(f'{BASE}/{el}')
    print(f'  {el:3s} N={N:5.2f}  {f(Bv)}   {f(G)}   {f(PAPA[el]) if el in PAPA else ""}')
JASK = {'Ta': (0.9, 50.7, 102.4), 'Nb': (4.7, 55.3, 102.0), 'Hf': (1.7, 71.4, 87.5), 'Zr': (7.1, 79.3, 89.7), 'Ti': (4.9, 45.7, 75.6)}
for tag, wd in [('alloy SRA PBE', f'{BASE}/Ta34Nb33Hf8Zr14Ti11'), ('alloy NRL LDA', 'results/test.nrl.ew0.6/TNHZT_a3.36_nrl_vbh')]:
    print(f'\n{tag}:                             B                       G = GG(7), MTZ=0        Jaśkiewicz 2016')
    lab = ['Ta', 'Nb', 'Hf', 'Zr', 'Ti']; conc = [.34, .33, .08, .14, .11]; emB = emG = 0
    for ci, el in enumerate(lab):
        Bv, G, N, ef = gg(wd, ci); emB += conc[ci] * sum(Bv); emG += conc[ci] * sum(G)
        print(f'  {el:3s} N={N:5.2f}  {f(Bv)}   {f(G)}   {f(JASK[el])}')
    print(f'  eta_mix: B {emB:.1f}   G {emG:.1f}   Jaśkiewicz 157.2')
