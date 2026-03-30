'''Example of fitting the Birch-Murnaghan EOS to data'''
import numpy as np
import csv
import sys, getopt
import matplotlib.pyplot as plt
from scipy.optimize import leastsq
import os
import argparse
import pandas as pd
from src.utils import dist_from_si, dist_to_si, energy_from_si, energy_to_si

def volume_to_lattice(vol):
	return (vol*2.0)**(1.0/3.0)

def lattice_to_volume(latt):
	return (latt**3)/2.0

def pressure_volume(v, v0, b, bP):
	P=3.0/2.0*b*((v/v0)**(-7.0/3.0) - (v/v0)**(-5.0/3.0))*(1+3.0/4.0*(bP-4)*((v/v0)**(-2.0/3.0)-1))	
	return P

def find_pressure_lattice(press, v0, b, bP):
	def obj(v, target_p, v0, b, bP):
		err = target_p - pressure_volume(v, v0, b, bP)
#		print(f'{v/v0} err={err}')
		return err
	vtrial = v0*0.9
	vol, ier = leastsq(obj, vtrial, args=(press, v0, b, bP))
	latt = volume_to_lattice(vol)
	return latt

#now we have to create the equation of state function
def Murnaghan(parameters, vol):
    '''
    given a vector of parameters and volumes, return a vector of energies.
    equation From PRB 28,5480 (1983)
    '''
    E0 = parameters[0]
    B0 = parameters[1]
    BP = parameters[2]
    V0 = parameters[3]
    
    volratiosign = np.sign(V0/vol)
    volratio = np.abs(V0/vol)
    # 
    E = E0 + B0*vol/BP*(((volratio)**BP*volratiosign)/(BP-1)+1) - V0*B0/(BP-1.)
    # thrid-order
#    E = E0 + 9*V0*B0/16.0*( ((volratio)**(2.0/3.0)*volratiosign-1)**3*BP + ((volratio)**(2.0/3.0)*volratiosign-1)**2*(6-4*(volratio)**(2.0/3.0)*volratiosign) )

    return E

# and we define an objective function that will be minimized
def objective(pars, y, x):
    #we will minimize this function
    err =  y - Murnaghan(pars,x)
    return err


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--f", type=str, required=True, help="csv file with latt and E in columns")

    args = parser.parse_args()

    dat = pd.read_csv(args.f, quoting=csv.QUOTE_NONE, error_bad_lines=False)
    # keep only converged SCFs
    print(dat.shape)
    dat = dat.loc[dat['2'] == True]
    dat = dat.dropna()
    energy = dat.iloc[:, 1]

    # change to SI
    energy = energy_to_si(energy)
    latt = dist_to_si(dat.iloc[:, 0])

    # compute volume based on lattice constant
    v = lattice_to_volume(latt)


    ### fit a parabola to the data
    # y = ax^2 + bx + c
    a, b, c = np.polyfit(v, energy, 2)  # this is from pylab

    '''
    the parabola does not fit the data very well, but we can use it to get
    some analytical guesses for other parameters.

    V0 = minimum energy volume, or where dE/dV=0
    E = aV^2 + bV + c
    dE/dV = 2aV + b = 0
    V0 = -b/2a

    E0 is the minimum energy, which is:
    E0 = aV0^2 + bV0 + c

    B is equal to V0*d^2E/dV^2, which is just 2a*V0

    and from experience we know Bprime_0 is usually a small number like 4
    '''
    # now here are our initial guesses.
    v0 = -b / (2 * a)
    e0 = a * v0 ** 2 + b * v0 + c
    b0 = 2 * a * v0
    bP = 4

    x0 = [e0, b0, bP, v0] #initial guesses in the same order used in the Murnaghan function

    murnpars, ier = leastsq(objective, x0, args=(energy, v)) #this is from scipy

    #now we make a figure summarizing the results
    vlatt = np.linspace(min(latt), max(latt), 100)
    vvol = lattice_to_volume(vlatt)
    plt.plot(dist_from_si(latt), energy_from_si(energy),'ro')
    #plt.plot(vfit, a*vfit**2 + b*vfit + c,'--',label='parabolic fit')
    y = Murnaghan(murnpars, vvol)
    plt.plot(dist_from_si(vlatt), energy_from_si(y), label='Murnaghan fit')
    plt.xlabel('Lattice (bohr)')
    plt.ylabel('Energy (eV)')
    plt.legend(loc='best')
    plt.savefig('a-eos.png')

    err = np.mean(np.abs(objective(murnpars, energy, v)))
    err = energy_from_si(err)

    v0 = murnpars[3]
    a0 = volume_to_lattice(v0)
    b0 = murnpars[1]/1e9
    bP = murnpars[2]
    E0 = energy_from_si(Murnaghan(murnpars, v0))
    
    print(f'E0= {E0} a0= {dist_from_si(a0)} b0= {b0} bP= {bP} err={err}')
    with open('a-eos.txt', 'w') as f:
        print(f'{E0}, {dist_from_si(a0)}, {b0}, {bP}, {err}', file=f)

if __name__ == "__main__":
    main()
