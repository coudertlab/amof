"""
Module containing everythong related to atoms: 
- wrapper for ase atom module
"""

import ase.atoms
import numpy as np

def get_density(atom):
    """
    return density from an ase atom object
    """
    conversion_factor = 1.66053906660 # convert from uma per angstrom to SI (kg/L)
    return conversion_factor * get_total_mass(atom) / atom.get_volume()

def get_number_density(atom):
    """
    return number density in Angstrom^-3 from an ase atom object
    """
    return len(atom) / atom.get_volume()


def get_total_mass(atom):
    return np.sum(atom.get_masses())