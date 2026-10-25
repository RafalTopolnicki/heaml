import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fort51_path, labels_for
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
sys.path.insert(0,'/home/rafal/WORK/HEA/ML/scripts')
from macmillan import parse_fort51, reduce_l_block_radials, group_dosef_by_l, derivative_nonuniform, compute_M
from macmillan_cutoff import find_all_nodes
labels=['Ta','Nb','Hf','Zr','Ti']; conc=[.34,.33,.08,.14,.11]
J={'Ta':(17.2,0.9,50.7,102.4),'Nb':(18.8,4.7,55.3,102.0),'Hf':(16.2,1.7,71.4,87.5),'Zr':(17.2,7.1,79.3,89.7),'Ti':(26.6,4.9,45.7,75.6)}
for path in sys.argv[1:]:
    tag=path; d=parse_fort51(fort51_path(path)); ef=float(np.atleast_1d(d['ef'])[0])
    print(f'\n=== {tag}   E_F={ef:.4f} Ry      eta in mRy/a0^2: sp / pd / df / total;  N(EF) both spins, 1/Ry')
    print(f'{"":4s} {"N_J":>5s} {"N_our":>6s} | {"J: sp/pd/df/tot":>24s} | {"full int.":>24s} | {"surface":>24s} | {"cutoff(prod)":>24s}')
    emix={'full':0,'surf':0,'cut':0}
    for ci,b in enumerate(d['components']):
        x=np.array(b['xr'],float)[:-1]; v=np.array(b['v3'],float)[:-1]; R=x[-1]; lnx=np.log(x)
        raw={int(k):np.array(vv[:-1],float) for k,vv in b['rstr_real_ef'].items()}
        U=reduce_l_block_radials(raw,int(b['mxlcmp']),x,'mean','none'); dos=group_dosef_by_l(b['dosef'],int(b['mxlcmp'])); N=sum(b['dosef'])
        dV=derivative_nonuniform(x,v)
        ef_,es,ec=[],[],[]
        for l in range(3):
            A,B=U['spdf'[l]],U['spdf'[l+1]]; k=l+1
            pref=2*(l+1)/((2*l+1)*(2*l+3))*dos['spdf'[l]]*dos['spdf'[l+1]]/N
            Mf=compute_M(A,B,dV,x,'plain')
            Ap=CubicSpline(lnx,A)(lnx[-1],1)/R; Bp=CubicSpline(lnx,B)(lnx[-1],1)/R
            Ms=B[-1]*((v[-1]-ef+l*(l+1)/R**2)*A[-1]+k*A[-1]/R**2-k*Ap/R)-Bp*(Ap-k*A[-1]/R)
            rc=max(find_all_nodes(x,A)[-1] if find_all_nodes(x,A) else 0, find_all_nodes(x,B)[-1] if find_all_nodes(x,B) else 0)
            i=np.searchsorted(x,rc); Mc=compute_M(A[i:],B[i:],dV[i:],x[i:],'plain')
            ef_.append(1000*pref*Mf**2); es.append(1000*pref*Ms**2); ec.append(1000*pref*Mc**2)
        el=labels[ci]; j=J[el]
        f=lambda a: '/'.join(f'{t:6.1f}' for t in a)+f'/{sum(a):7.1f}'
        print(f'{el:4s} {j[0]:5.1f} {2*N:6.1f} | {j[1]:6.1f}/{j[2]:6.1f}/{j[3]:6.1f}/{sum(j[1:]):7.1f} | {f(ef_)} | {f(es)} | {f(ec)}')
        emix['full']+=conc[ci]*sum(ef_); emix['surf']+=conc[ci]*sum(es); emix['cut']+=conc[ci]*sum(ec)
    print(f'eta_mix: Jaskiewicz 157.2 | full {emix["full"]:.1f} | surface {emix["surf"]:.1f} | cutoff {emix["cut"]:.1f}')
