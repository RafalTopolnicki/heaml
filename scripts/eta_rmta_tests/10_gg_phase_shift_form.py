"""Gaspari–Gyorffy Eq. (7) evaluated directly:  eta = (k²/π² N) Σ_l 2(l+1) sin²(δ_{l+1}−δ_l) (n_l/n_l^(1)) (n_{l+1}/n_{l+1}^(1)),
k² = E_F − V_MT (Ry), N, n_l = AkaiKKR dosef (per spin), n_l^(1) = free-scatterer partial DOS in a sphere of radius a:
n_l^(1)(a) = (k/π)(2l+1) ∫_0^a r² R_l² dr with R_l = j_l cosδ − y_l sinδ outside R_MT (free-wave normalisation).
Evaluated for a = R_MT and a = r_WS, compared with the unit-normalised surface form (variant B) and Papaconstantopoulos 1977.
Also checks GG's statement M_free = sin(δ_{l+1}−δ_l) (ratio printed)."""
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
def analyse(wd, label, a0, comp=0):
    d = parse_fort51(fort51_path(wd)); ef = float(np.atleast_1d(d['ef'])[0]); b = d['components'][comp]
    x = np.array(b['xr'], float)[:-1]; v = np.array(b['v3'], float)[:-1]; R = x[-1]; lnx = np.log(x)
    raw = {int(k): np.array(vv[:-1], float) for k, vv in b['rstr_real_ef'].items()}
    U = reduce_l_block_radials(raw, int(b['mxlcmp']), x, 'mean', 'none'); dos = group_dosef_by_l(b['dosef'], int(b['mxlcmp'])); N = sum(b['dosef'])
    V0 = v[-1]; k = np.sqrt(ef - V0); rws = a0 * (3 / (8 * np.pi))**(1 / 3); rr = np.linspace(R, rws, 4001)
    delta = {}; I_mt = {}; I_ws = {}; scale = {}
    for l in range(4):
        u = U['spdf'[l]]; up = CubicSpline(lnx, u)(lnx[-1], 1) / R
        j, y = spherical_jn(l, k * R), spherical_yn(l, k * R); jp, yp = spherical_jn(l, k * R, derivative=True), spherical_yn(l, k * R, derivative=True)
        A, B = np.linalg.solve([[R * j, R * y], [j + k * R * jp, y + k * R * yp]], [u[-1], up])   # u = A r j + B r y outside
        norm = np.hypot(A, B); delta[l] = np.arctan2(-B, A); scale[l] = 1.0 / norm                # R_free = (A j + B y)/norm = j cosδ − y sinδ
        inside = simpson(u**2, x=x) / norm**2; outside = simpson((A * rr * spherical_jn(l, k * rr) + B * rr * spherical_yn(l, k * rr))**2, x=rr) / norm**2
        I_mt[l] = inside; I_ws[l] = inside + outside
    n1_mt = {l: (k / np.pi) * (2 * l + 1) * I_mt[l] for l in range(4)}; n1_ws = {l: (k / np.pi) * (2 * l + 1) * I_ws[l] for l in range(4)}
    n = {l: dos['spdf'[l]] for l in range(4)}
    print(f'\n== {label}: k²={k*k:.3f} Ry, N/spin={N:.3f};  n_l/n_l^(1): MT ' + ' '.join(f'{n[l]/n1_mt[l]:.2f}' for l in range(4)) + '   WS ' + ' '.join(f'{n[l]/n1_ws[l]:.2f}' for l in range(4)))
    rows = {}
    for l in range(3):
        s2 = np.sin(delta[l + 1] - delta[l])**2
        g_mt = (k * k / (np.pi**2 * N)) * 2 * (l + 1) * s2 * (n[l] / n1_mt[l]) * (n[l + 1] / n1_mt[l + 1])
        g_ws = (k * k / (np.pi**2 * N)) * 2 * (l + 1) * s2 * (n[l] / n1_ws[l]) * (n[l + 1] / n1_ws[l + 1])
        Mu = compute_M_surface(U['spdf'[l]], U['spdf'[l + 1]], x, v, ef, l)                       # unit-normalised u
        Mfree = Mu * scale[l] * scale[l + 1]                                                           # free-wave normalised
        eB = 1000 * 2 * (l + 1) / ((2 * l + 1) * (2 * l + 3)) * n[l] * n[l + 1] / N * Mu**2
        rows[l] = (1000 * g_mt, 1000 * g_ws, eB, Mfree / np.sin(delta[l + 1] - delta[l]))
    print(f'   channel   GG(7) a=MT   GG(7) a=WS   B(surface,unit-MT)   M_free/sin(Δδ)')
    for l in range(3):
        print(f'   {"spdf"[l]+"spdf"[l+1]:5s}  {rows[l][0]:10.1f}  {rows[l][1]:11.1f}  {rows[l][2]:16.1f}   {rows[l][3]:12.4f}')
    print(f'   total  {sum(r[0] for r in rows.values()):10.1f}  {sum(r[1] for r in rows.values()):11.1f}  {sum(r[2] for r in rows.values()):16.1f}')
    return rows
for el in ['Ti', 'Nb', 'Mo', 'Y', 'Zr']:
    a0 = json.load(open(f'{BASE}/{el}/results.json'))['a0_bohr']
    analyse(f'{BASE}/{el}', el, a0)
    if el in PAPA: p = [20.58 * t for t in PAPA[el]]; print(f'   Papaconstantopoulos 1977: {p[0]:.1f} / {p[1]:.1f} / {p[2]:.1f} / {sum(p):.1f}')
print('\n### alloy Ta34Nb33Hf8Zr14Ti11 (SRA PBE), Jaśkiewicz: Ta 0.9/51/102=154, Ti 4.9/46/76=126')
alloy = f'{BASE}/Ta34Nb33Hf8Zr14Ti11'; a0 = json.load(open(f'{alloy}/results.json'))['a0_bohr']
for ci, el in [(0, 'Ta'), (1, 'Nb'), (4, 'Ti')]:
    analyse(alloy, 'alloy ' + el, a0, comp=ci)
