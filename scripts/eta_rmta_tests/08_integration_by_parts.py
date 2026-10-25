"""M_{l,l+1} three ways on the AkaiKKR mesh: direct  ∫ u_l V' u_{l+1} dr  (V' numeric),
by parts  -∫ V (u_l u_{l+1})' dr  (derivative moved onto u; boundary terms vanish: u~r^{l+1} at 0, V(R)=0),
and the GG surface term. Usage: python 08_integration_by_parts.py <workdir|fort.51> [labels]"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fort51_path, labels_for
from macmillan import parse_fort51, reduce_l_block_radials, group_dosef_by_l
from macmillan_cutoff import compute_M_surface
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
P = fort51_path(sys.argv[1]); d = parse_fort51(P); ef = float(np.atleast_1d(d['ef'])[0])
lab = labels_for(sys.argv[1]) or sys.argv[2].split(',')
want = [e for e in ('Ta', 'Nb', 'Ti') if e in lab]
for el in want:
    b = d['components'][lab.index(el)]
    x = np.array(b['xr'], float)[:-1]; v = np.array(b['v3'], float)[:-1]; lnx = np.log(x)
    raw = {int(k): np.array(vv[:-1], float) for k, vv in b['rstr_real_ef'].items()}
    U = reduce_l_block_radials(raw, int(b['mxlcmp']), x, 'mean', 'none'); dos = group_dosef_by_l(b['dosef'], int(b['mxlcmp'])); N = sum(b['dosef'])
    dV_grad = np.gradient(v, x); dV_spl = CubicSpline(lnx, v)(lnx, 1) / x
    print(f'\n=== {el}   V(R)={v[-1]:.2e} Ry   (M in Ry/a0; eta in mRy/a0^2)')
    print(f'{"ch":3s} {"direct(np.gradient)":>20s} {"direct(spline V\')":>18s} {"by parts":>10s} {"surface":>10s} | {"eta direct":>10s} {"eta parts":>10s} {"eta surf":>9s}')
    for l in range(3):
        A, B = U['spdf'[l]], U['spdf'[l + 1]]
        prod = A * B; dprod = CubicSpline(lnx, prod)(lnx, 1) / x
        M_dir = simpson(A * dV_grad * B, x=x); M_spl = simpson(A * dV_spl * B, x=x)
        bt = prod[-1] * v[-1] - prod[0] * v[0]
        M_parts = bt - simpson(v * dprod, x=x)
        M_surf = compute_M_surface(A, B, x, v, ef, l)
        pref = 2 * (l + 1) / ((2 * l + 1) * (2 * l + 3)) * dos['spdf'[l]] * dos['spdf'[l + 1]] / N
        print(f'{"spdf"[l]+"spdf"[l+1]:3s} {M_dir:20.3f} {M_spl:18.3f} {M_parts:10.3f} {M_surf:10.3f} | {1e3*pref*M_dir**2:10.1f} {1e3*pref*M_parts**2:10.1f} {1e3*pref*M_surf**2:9.1f}   (boundary term {bt:.2e})')
    # where does the by-parts integrand live? cumulative from the nucleus
    A, B = U['s'], U['p']; prod = A * B; dprod = CubicSpline(lnx, prod)(lnx, 1) / x; integ = -v * dprod
    cum = [simpson(integ[:i], x=x[:i]) for i in (np.searchsorted(x, r) for r in (1e-3, 1e-2, 0.05, 0.1, 0.3, 1.0))]
    print('  sp by-parts cumulative up to r=1e-3,1e-2,0.05,0.1,0.3,1.0:', ' '.join(f'{c:8.3f}' for c in cum), f' total {-simpson(v*dprod,x=x):.3f}')
