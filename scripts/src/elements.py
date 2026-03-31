import numpy as np

class Element:
    def __init__(self, lattice, bulk_modulus, debye_temperature, atomic_nuber, density):
        self.lattice = lattice
        self.bulk_modulus = bulk_modulus
        self.debye = debye_temperature
        self.atomic_number = atomic_nuber
        self.density = density

ELEMENTS = {
    'Ti': Element(atomic_nuber=22, lattice=5.575, bulk_modulus=110, debye_temperature=420, density=4510),
    'Nb': Element(atomic_nuber=41, lattice=6.229, bulk_modulus=170, debye_temperature=275, density=8570),
    'Zr': Element(atomic_nuber=40, lattice=5.936, bulk_modulus=95, debye_temperature=270, density=6150),
    'Hf': Element(atomic_nuber=72, lattice=6.035, bulk_modulus=110, debye_temperature=250, density=13310),
    'Ta': Element(atomic_nuber=73, lattice=6.020, bulk_modulus=200, debye_temperature=240, density=15578),
    'Sc': Element(atomic_nuber=21, lattice=6.255, bulk_modulus=57, debye_temperature=355, density=2990),
    'Y': Element(atomic_nuber=39, lattice=6.889, bulk_modulus=41, debye_temperature=210, density=6973),
    'La': Element(atomic_nuber=57, lattice=7.508, bulk_modulus=28, debye_temperature=135, density=6160),
    'Mo': Element(atomic_nuber=42, lattice=5.947, bulk_modulus=250, debye_temperature=423, density=10250),
    'W': Element(atomic_nuber=74, lattice=5.982, bulk_modulus=310, debye_temperature=400, density=19270),
}

class HEAClass:
    def __init__(self, labels, concentrations):
        self.labels = labels
        cs = np.sum(concentrations, axis=0)
        self.concentrations = [c/cs for c in concentrations]
        self.elements = []
        for lab, conc in zip(self.labels):
            self.elements.append(ELEMENTS[lab])
        self.mixture_lattice = [c*e.lattice for c, e in zip(self.concentrations, self.elements)]
        self.mixture_bulk_modulus = [c * e.bulk_modulus for c, e in zip(self.concentrations, self.elements)]
        self.mixture_debye = [c * e.bulk_modulus for c, e in zip(self.concentrations, self.elements)]
        self.lattice = None
        self.bulk_modulus = None
        self.debye_temperature = None
    def return_labels(self):
        return ' '.join(self.labels)
    def return_concentrations(self):
        txt = ''
        for c in self.concentrations:
            txt += f' {c}:.4f'
        return txt


