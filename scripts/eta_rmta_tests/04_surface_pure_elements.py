import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fort51_path
from scipy.interpolate import CubicSpline
sys.path.insert(0,'/home/rafal/WORK/HEA/ML/scripts')
from macmillan import parse_fort51, reduce_l_block_radials, group_dosef_by_l
from macmillan_cutoff import C_THEORETICAL
S=os.path.dirname(os.path.abspath(__file__))
BASE='/home/rafal/WORK/HEA/ML/results/comparison.sra.ew0.6.bz10-10-20'
ALLOY=sys.argv[1] if len(sys.argv)>1 else f'{BASE}/Ta34Nb33Hf8Zr14Ti11'
def surface_eta(path, dE=0.0):
    d=parse_fort51(path); ef=float(np.atleast_1d(d['ef'])[0])+dE
    out=[]
    for b in d['components']:
        x=np.array(b['xr'],float)[:-1]; v=np.array(b['v3'],float)[:-1]; R=x[-1]; lnx=np.log(x)
        raw={int(k):np.array(vv[:-1],float) for k,vv in b['rstr_real_ef'].items()}
        U=reduce_l_block_radials(raw,int(b['mxlcmp']),x,'mean','none'); dos=group_dosef_by_l(b['dosef'],int(b['mxlcmp'])); N=sum(b['dosef'])
        etas=[]
        for l in range(3):
            A,B=U['spdf'[l]],U['spdf'[l+1]]; k=l+1
            Ap=CubicSpline(lnx,A)(lnx[-1],1)/R; Bp=CubicSpline(lnx,B)(lnx[-1],1)/R
            Ms=B[-1]*((v[-1]-ef+l*(l+1)/R**2)*A[-1]+k*A[-1]/R**2-k*Ap/R)-Bp*(Ap-k*A[-1]/R)
            pref=2*(l+1)/((2*l+1)*(2*l+3))*dos['spdf'[l]]*dos['spdf'[l+1]]/N; etas.append(1000*pref*Ms**2)
        out.append(etas)
    return out
base=BASE
ref={'Ta':(155,0.69,180.95,'exp Tc 4.5 K'),'Nb':(152,0.82,92.906,'exp Tc 9.3 K')}
for el,(eta_J,lam_exp,M,note) in ref.items():
    p=fort51_path(f'{base}/{el}')
    import json; j=json.load(open(f'{base}/{el}/results.json')); th=j['thetaDB_K']
    e=surface_eta(p)[0]; tot=sum(e)
    lam=tot/1000*C_THEORETICAL/(M*th**2)
    print(f'{el}: surface eta sp/pd/df = {e[0]:.1f}/{e[1]:.1f}/{e[2]:.1f} total {tot:.1f}  | Jaskiewicz Table2 {eta_J} | lambda(theta_KKR={th:.0f}K) = {lam:.3f} vs exp {lam_exp} | cutoff lambda {j["lambda"]:.3f}, cutoff eta {1000*j["eta_mix"]:.1f}')
# E_F sensitivity, Ta in the alloy (component 0)
p=fort51_path(ALLOY)
for dE in (-0.05,-0.02,0,0.02,0.05):
    e=surface_eta(p,dE)[0]; print(f'TNHZT Ta, E_F{dE:+.2f} Ry: sp/pd/df = {e[0]:.1f}/{e[1]:.1f}/{e[2]:.1f}')
