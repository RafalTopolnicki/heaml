import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fort51_path, labels_for
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp, simpson
sys.path.insert(0,'/home/rafal/WORK/HEA/ML/scripts')
from macmillan import parse_fort51, reduce_l_block_radials, group_dosef_by_l
P=fort51_path(sys.argv[1]); d=parse_fort51(P); efR=float(np.atleast_1d(d['ef'])[0])
c=137.035999
lab=labels_for(sys.argv[1]) or sys.argv[2].split(',')
comps={e:lab.index(e) for e in ('Ta','Ti') if e in lab}
P2016={'Ta':(0.9,50.7,102.4),'Ti':(4.9,45.7,75.6)}
for el,ci in comps.items():
    b=d['components'][ci]
    x=np.array(b['xr'],float)[:-1]; vR=np.array(b['v3'],float)[:-1]
    raw={int(k):np.array(vv[:-1],float) for k,vv in b['rstr_real_ef'].items()}
    U=reduce_l_block_radials(raw,int(b['mxlcmp']),x,'mean','none')
    dos=group_dosef_by_l(b['dosef'],int(b['mxlcmp'])); N=sum(b['dosef'])
    Vsp=CubicSpline(np.log(x),vR/2); E=efR/2; R=x[-1]
    rf=np.exp(np.linspace(np.log(x[0]),np.log(R),40001)); lnrf=np.log(rf)
    V=lambda r: Vsp(np.log(r)); vf=V(rf); dVf=Vsp(lnrf,1)/rf*2
    def solve(l,sra):
        def f(r,y):
            P,Q=y; Vr=V(r)
            if sra:
                M=1+(E-Vr)/(2*c*c)
                return [2*M*c*Q+P/r, -Q/r+((Vr-E)/c+l*(l+1)/(2*M*c*r*r))*P]
            return [Q, 2*(Vr+l*(l+1)/(2*r*r)-E)*P]
        r0=x[0]; y0=[r0**(l+1), (l*r0**l/(2*c)) if sra else (l+1)*r0**l]
        s=solve_ivp(f,(r0,R),y0,t_eval=rf,method='DOP853',rtol=1e-11,atol=1e-30)
        P=s.y[0]; return P/np.sqrt(simpson(P*P,x=rf))
    def surf(A,B,l):   # GG surface term at R (Ry units: V,E in Ry)
        Ap=CubicSpline(lnrf,A)(lnrf[-1],1)/R; Bp=CubicSpline(lnrf,B)(lnrf[-1],1)/R; k=l+1
        vRy=2*vf[-1]; ERy=efR
        return B[-1]*((vRy-ERy+l*(l+1)/R**2)*A[-1]+k*A[-1]/R**2-k*Ap/R)-Bp*(Ap-k*A[-1]/R)
    print(f'\n######## {el}   (eta in mRy/a0^2: sp / pd / df / total)')
    print(f'  Jaskiewicz 2016              : {P2016[el][0]:7.1f} / {P2016[el][1]:6.1f} / {P2016[el][2]:6.1f} / {sum(P2016[el]):6.1f}')
    for sra in (False,True):
        sol={l:solve(l,sra) for l in range(4)}
        rows={'full integral':[], 'surface formula':[]}
        for l in range(3):
            A,B=sol[l],sol[l+1]
            pref=2*(l+1)/((2*l+1)*(2*l+3))*dos['spdf'[l]]*dos['spdf'[l+1]]/N
            Mn=simpson(A*dVf*B,x=rf); Ms=surf(A,B,l)
            rows['full integral'].append((Mn,1000*pref*Mn**2)); rows['surface formula'].append((Ms,1000*pref*Ms**2))
        tag='SRA   ' if sra else 'nonrel'
        for k,v in rows.items():
            print(f'  {tag} solved, {k:15s}: '+' / '.join(f'{e:6.1f}' for _,e in v)+f' / {sum(e for _,e in v):6.1f}'
                  +'   M = '+' '.join(f'{m:8.3f}' for m,_ in v))
    # AkaiKKR functions, surface formula on AkaiKKR mesh
    lnx=np.log(x); out=[]
    for l in range(3):
        A,B=U['spdf'[l]],U['spdf'[l+1]]; k=l+1
        Ap=CubicSpline(lnx,A)(lnx[-1],1)/R; Bp=CubicSpline(lnx,B)(lnx[-1],1)/R
        Ms=B[-1]*((vR[-1]-efR+l*(l+1)/R**2)*A[-1]+k*A[-1]/R**2-k*Ap/R)-Bp*(Ap-k*A[-1]/R)
        pref=2*(l+1)/((2*l+1)*(2*l+3))*dos['spdf'[l]]*dos['spdf'[l+1]]/N; out.append(1000*pref*Ms**2)
    print(f'  AkaiKKR u, surface formula   : '+' / '.join(f'{e:6.1f}' for e in out)+f' / {sum(out):6.1f}')
    print(f'  partial DOS at EF (per spin, 1/Ry): s {dos["s"]:.3f} p {dos["p"]:.3f} d {dos["d"]:.3f} f {dos["f"]:.3f}  N {N:.3f}')
