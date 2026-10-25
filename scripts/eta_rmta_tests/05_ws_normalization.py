import sys, os, json, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import fort51_path, labels_for
from scipy.interpolate import CubicSpline
from scipy.integrate import simpson
from scipy.special import spherical_jn, spherical_yn
sys.path.insert(0,'/home/rafal/WORK/HEA/ML/scripts')
from macmillan import parse_fort51, reduce_l_block_radials, group_dosef_by_l
S=os.path.dirname(os.path.abspath(__file__))
BASE='/home/rafal/WORK/HEA/ML/results/comparison.sra.ew0.6.bz10-10-20'
def analyse(path, a_bohr, labels, ref):
    d=parse_fort51(path); ef=float(np.atleast_1d(d['ef'])[0])
    rws=a_bohr*(3/(8*np.pi))**(1/3)   # bcc, 2 atoms per cell
    print(f'\n=== {path}  a={a_bohr:.4f}  R_MT={np.array(d["components"][0]["xr"])[-2]:.4f}  r_WS={rws:.4f}  E_F={ef:.4f}')
    print(f'{"":4s} {"f_s":>6s} {"f_p":>6s} {"f_d":>6s} {"f_f":>6s} | {"eta_MT sp/pd/df/tot":>28s} | {"eta_WS sp/pd/df/tot":>28s} | {"reference":>28s}')
    for ci,b in enumerate(d['components']):
        x=np.array(b['xr'],float)[:-1]; v=np.array(b['v3'],float)[:-1]; R=x[-1]; lnx=np.log(x)
        raw={int(k):np.array(vv[:-1],float) for k,vv in b['rstr_real_ef'].items()}
        U=reduce_l_block_radials(raw,int(b['mxlcmp']),x,'mean','none'); dos=group_dosef_by_l(b['dosef'],int(b['mxlcmp'])); N=sum(b['dosef'])
        V0=v[-1]; k=np.sqrt(ef-V0)       # constant MT potential outside, E_F > V0
        rr=np.linspace(R,rws,2001)
        f={}; up={}; uR={}
        for l in range(4):
            u=U['spdf'[l]]; uRl=u[-1]; upl=CubicSpline(lnx,u)(lnx[-1],1)/R
            # u = r*(A j_l(kr) + B y_l(kr)); match value & derivative at R
            j=lambda r: r*spherical_jn(l,k*r); y=lambda r: r*spherical_yn(l,k*r)
            jp=lambda r: spherical_jn(l,k*r)+k*r*spherical_jn(l,k*r,derivative=True)
            yp=lambda r: spherical_yn(l,k*r)+k*r*spherical_yn(l,k*r,derivative=True)
            A,B=np.linalg.solve([[j(R),y(R)],[jp(R),yp(R)]],[uRl,upl])
            uo=A*j(rr)+B*y(rr); n_out=simpson(uo**2,x=rr); n_in=simpson(u**2,x=x)
            f[l]=n_in/(n_in+n_out); uR[l]=uRl; up[l]=upl
        eMT=[]; eWS=[]
        for l in range(3):
            A,B=U['spdf'[l]],U['spdf'[l+1]]; kk=l+1; Ap,Bp=up[l],up[l+1]
            Ms=B[-1]*((V0-ef+l*(l+1)/R**2)*A[-1]+kk*A[-1]/R**2-kk*Ap/R)-Bp*(Ap-kk*A[-1]/R)
            pref=2*(l+1)/((2*l+1)*(2*l+3))*dos['spdf'[l]]*dos['spdf'[l+1]]/N
            e=1000*pref*Ms**2; eMT.append(e); eWS.append(e*f[l]*f[l+1])
        g=lambda a: '/'.join(f'{t:6.1f}' for t in a)+f'/{sum(a):6.1f}'
        print(f'{labels[ci]:4s} {f[0]:6.3f} {f[1]:6.3f} {f[2]:6.3f} {f[3]:6.3f} | {g(eMT)} | {g(eWS)} | {ref.get(labels[ci],""):>28s}')
J={'Ta':'0.9/50.7/102.4/154.0','Nb':'4.7/55.3/102.0/162.0','Hf':'1.7/71.4/87.5/160.6','Zr':'7.1/79.3/89.7/176.1','Ti':'4.9/45.7/75.6/126.2'}
L=['Ta','Nb','Hf','Zr','Ti']
alloy=f'{BASE}/Ta34Nb33Hf8Zr14Ti11'
analyse(fort51_path(alloy), json.load(open(f'{alloy}/results.json'))['a0_bohr'], L, J)
nrl='/home/rafal/WORK/HEA/ML/results/test.nrl.ew0.6/TNHZT_a3.36_nrl'
if os.path.exists(nrl): analyse(fort51_path(nrl), 6.34949, L, J)
base=BASE
for el,ref in [('Nb','Rajput 7.5/62.0/98.0/167.5'),('Ta','Jask.T2 155 (tot)')]:
    a0=json.load(open(f'{base}/{el}/results.json'))['a0_bohr']
    analyse(fort51_path(f'{base}/{el}'),a0,[el],{el:ref})
