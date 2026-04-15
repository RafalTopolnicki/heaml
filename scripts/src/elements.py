import numpy as np

ATOMIC_FEATURES = {
    "atomic_radius_pm": {
        "Ti": 140,
        "Nb": 145,
        "Zr": 155,
        "Hf": 155,
        "Ta": 145,
        "Sc": 160,
        "Mo": 145,
        "W": 135,
        "Y": 180,
        "La": 195,
    },
    "electronegativity_pauling": {
        "Ti": 1.54,
        "Nb": 1.60,
        "Zr": 1.33,
        "Hf": 1.30,
        "Ta": 1.50,
        "Sc": 1.36,
        "Mo": 2.16,
        "W": 2.36,
        "Y": 1.22,
        "La": 1.10,
    },
    "valence_electron_count": {
        "Ti": 4,
        "Nb": 5,
        "Zr": 4,
        "Hf": 4,
        "Ta": 5,
        "Sc": 3,
        "Mo": 6,
        "W": 6,
        "Y": 3,
        "La": 3,
    },
    "vec": {
        "Ti": 4,
        "Nb": 5,
        "Zr": 4,
        "Hf": 4,
        "Ta": 5,
        "Sc": 3,
        "Mo": 6,
        "W": 6,
        "Y": 3,
        "La": 3,
    },
    "d_electron_count": {
        "Ti": 2,   # [Ar] 3d2 4s2
        "Nb": 4,   # [Kr] 4d4 5s1
        "Zr": 2,   # [Kr] 4d2 5s2
        "Hf": 2,   # [Xe] 4f14 5d2 6s2
        "Ta": 3,   # [Xe] 4f14 5d3 6s2
        "Sc": 1,   # [Ar] 3d1 4s2
        "Mo": 5,   # [Kr] 4d5 5s1
        "W": 4,    # [Xe] 4f14 5d4 6s2
        "Y": 1,    # [Kr] 4d1 5s2
        "La": 1,   # [Xe] 5d1 6s2
    },
    "vdw_radius_pm": {
        "Ti": 246,
        "Nb": 256,
        "Zr": 252,
        "Hf": 253,
        "Ta": 257,
        "Sc": 258,
        "Mo": 245,
        "W": 249,
        "Y": 275,
        "La": 298,
    },
}

class Element:
    def __init__(self, lattice, bulk_modulus, debye_temperature, atomic_nuber, density, mass):
        self.lattice = lattice
        self.bulk_modulus = bulk_modulus
        self.debye_temperature = debye_temperature
        self.atomic_number = atomic_nuber
        self.density = density
        self.mass = mass

# hcp:bcc lattice constant
# a_bcc = 1.12*a_hcp
# a_bcc = 1.44*a_fcc ?
# lattice constants here are not experimental values but come from KKR claulcations
# NRL EW=0.7 BZ=10 PBE
ELEMENTS = {
    'Ti': Element(atomic_nuber=22, lattice=6.15, bulk_modulus=110, debye_temperature=420, density=4510, mass=47.87), # beta-ti
    'Nb': Element(atomic_nuber=41, lattice=6.25, bulk_modulus=170, debye_temperature=275, density=8570, mass=92.9), #
    'Zr': Element(atomic_nuber=40, lattice=6.73, bulk_modulus=95, debye_temperature=270, density=6150, mass=91.2), # beta-Zr
    'Hf': Element(atomic_nuber=72, lattice=6.83, bulk_modulus=110, debye_temperature=250, density=13310, mass=178.5), # 1.12*3.2A # hcp
    'Ta': Element(atomic_nuber=73, lattice=6.37, bulk_modulus=200, debye_temperature=240, density=15578, mass=180.9), # alpha-Ta
    'Sc': Element(atomic_nuber=21, lattice=6.93, bulk_modulus=57, debye_temperature=355, density=2990, mass=44.95), # 1.12*3.3A # hcp
    'Y': Element(atomic_nuber=39, lattice=6.889, bulk_modulus=41, debye_temperature=210, density=6973, mass=173.05), # ????
    'La': Element(atomic_nuber=57, lattice=7.97, bulk_modulus=28, debye_temperature=135, density=6160, mass=138.9), # 1.12*3.77A
    'Mo': Element(atomic_nuber=42, lattice=5.947, bulk_modulus=250, debye_temperature=423, density=10250, mass=95.95),
    'W': Element(atomic_nuber=74, lattice=5.982, bulk_modulus=310, debye_temperature=400, density=19270, mass=183.8),
}

class HEAClass:
    def __init__(self, labels, concentrations):
        assert len(labels) == len(concentrations)
        self.labels = labels
        cs = np.sum(concentrations, axis=0)
        self.concentrations = [c/cs for c in concentrations]
        self.elements = []
        for lab in self.labels:
            self.elements.append(ELEMENTS[lab])
        self.mixture_lattice = np.sum([c*e.lattice for c, e in zip(self.concentrations, self.elements)])
        self.mixture_bulk_modulus = np.sum([c * e.bulk_modulus for c, e in zip(self.concentrations, self.elements)])
        self.mixture_debye_temperature = np.sum([c * e.debye_temperature for c, e in zip(self.concentrations, self.elements)])
        self.density = np.sum(
            [c * e.density for c, e in zip(self.concentrations, self.elements)])
        self.mass = np.sum(
            [c * e.mass for c, e in zip(self.concentrations, self.elements)])
        self.lattice = None
        self.bulk_modulus = None
        self.debye_temperature = None
    def return_labels(self):
        return ' '.join(self.labels)
    def return_atomic_numbers(self):
        #return ' '.join([str(ELEMENTS[lab].atomic_number) for lab in self.labels])
        return [e.atomic_number for e in self.elements]
    def return_concentrations(self):
        txt = ''
        for c in self.concentrations:
            txt += f'{c}:.4f'
        return txt


