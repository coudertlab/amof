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


def select_species_positions(atom, atomic_number):
    """
    return position of atom of species defined by atomic_number
    Args:
        atom: ase atom object
        atomic_number: int, atom selection 

    Return:
        position array of a given species
    """
    if atomic_number is None:
        return atom.get_positions()
    else:
        return atom.get_positions()[atom.get_atomic_numbers()==atomic_number]

def get_atomic_numbers_unique(atom):
    """return list of atomic numbers present in atom"""
    return list(set(atom.get_atomic_numbers()))