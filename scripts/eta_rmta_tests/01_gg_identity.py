import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fort51_path, labels_for
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
sys.path.insert(0,'/home/rafal/WORK/HEA/ML/scripts')
from macmillan import parse_fort51, reduce_l_block_radials
P=fort51_path(sys.argv[1]); d=parse_fort51(P); ef=float(np.atleast_1d(d['ef'])[0])
Z={'Ta':73,'Ti':22}
lab=labels_for(sys.argv[1]) or sys.argv[2].split(',')
comps={e:lab.index(e) for e in Z if e in lab}
c=274.072  # speed of light in Ry units
for el,ci in comps.items():
    b=d['components'][ci]
    x=np.array(b['xr'],float)[:-1]; v=np.array(b['v3'],float)[:-1]
    raw={int(k):np.array(vv[:-1],float) for k,vv in b['rstr_real_ef'].items()}
    U=reduce_l_block_radials(raw,int(b['mxlcmp']),x,'mean','none')
    print(f'\n################ {el} (Z={Z[el]})  mesh: {len(x)} pts, r_min={x[0]:.2e}, r_max={x[-1]:.3f}')
    dl=np.diff(np.log(x)); print(f'  Δln r: min {dl.min():.4f} max {dl.max():.4f}; points with r<1e-2: {(x<1e-2).sum()}, r<1e-1: {(x<0.1).sum()}')
    # A. potential near the nucleus
    print('  A. r·V(r)/(-2Z) at r =', ' '.join(f'{r:.0e}:{np.interp(r,x,x*v)/(-2*Z[el]):.4f}' for r in (1e-6,1e-5,1e-4,1e-3,1e-2,0.1,0.5)))
    dVn=np.gradient(v,x); dVa=2*Z[el]/x**2
    print('  A. numeric dV/dr ÷ (2Z/r²) at r =', ' '.join(f'{r:.0e}:{np.interp(r,x,dVn/dVa):.3f}' for r in (1e-6,3e-6,1e-5,1e-4,1e-3,1e-2,0.1)))
    # B. small-r power law of u_l: d ln|u| / d ln r
    for l,a in enumerate('spd'):
        u=U[a]; s=np.gradient(np.log(np.abs(u)+1e-300),np.log(x))
        g=np.sqrt((l+1)**2-(2*Z[el]/c)**2) if l==0 else np.sqrt(l*(l+1)+1-(2*Z[el]/c)**2)
        print(f'  B. {a}: d ln u/d ln r at r=1e-6,1e-5,1e-4: {s[np.searchsorted(x,1e-6)]:.3f} {s[np.searchsorted(x,1e-5)]:.3f} {s[np.searchsorted(x,1e-4)]:.3f}'
              f'   (nonrel l+1={l+1}, SRA γ≈{g:.3f});  nodes at r =',
              ' '.join(f'{r:.3g}' for r in x[:-1][np.sign(u[:-1])*np.sign(u[1:])<0]))
    # C. GG identity on [r0, R]: M_num(r0) vs S(R)-S(r0), S(r) built from spline derivatives (nonrelativistic eq.)
    lnx=np.log(x)
    def der(f):
        sp=CubicSpline(lnx,f); return sp(lnx,1)/x
    Vs=CubicSpline(lnx,v); dV=Vs(lnx,1)/x
    for l in range(2):
        A,B=U['spdf'[l]],U['spdf'[l+1]]; k=l+1; Ap,Bp=der(A),der(B)
        S=B*((v-ef+l*(l+1)/x**2)*A+k*A/x**2-k*Ap/x)-Bp*(Ap-k*A/x)
        integ=A*dV*B
        cum=np.array([simpson(integ[i:],x=x[i:]) if i<len(x)-2 else 0 for i in range(len(x))])
        print(f'  C. channel {"spdf"[l]}{"spdf"[l+1]}: r0, M_num(r0→R), S(R)-S(r0), difference')
        for r0 in (1e-6,1e-4,1e-3,3e-3,1e-2,3e-2,0.1,0.3,1.0):
            i=np.searchsorted(x,r0); print(f'       {r0:7.0e} {cum[i]:10.3f} {S[-1]-S[i]:10.3f} {cum[i]-(S[-1]-S[i]):10.3f}')
