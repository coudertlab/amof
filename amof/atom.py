"""
Module containing everything related to ase atom objects: 
- wrapper for ase atom module
"""

import ase
import ase.atoms
import ase.neighborlist
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

def format_cutoff(nb_set_and_cutoff, format='ase', sort_pair = False):
    """
    Return cutoff formatted for different purposes

    Args:
        nb_set_and_cutoff: dict, keys are str indicating pair of neighbors, 
            values are cutoffs float, in Angstrom
        format: str, can be 'ase'
        sort_pair: Bool, if True will sort the tuple of the nb atomic numbers

    Return:
        cutoff_dict: list of cutoff in ase.neighborlist format if format=='ase'
    """
    if format == 'ase':
        cutoff_dict = {} # dict in ase.nl cutoff format
        elements_present_unique = []
        for nn_set, cutoff in nb_set_and_cutoff.items():
            xx = tuple(ase.data.atomic_numbers[i] for i in nn_set.split('-'))
            if sort_pair == True:
                xx = tuple(sorted(xx))
            cutoff_dict[xx] = cutoff
            elements_present_unique += [ase.data.atomic_numbers[i] for i in nn_set.split('-')]
        return cutoff_dict

def get_neighborlist(atom, cutoff_dict):
    """
    Return neighbor list using ase

    Args:
        atom: ase atom object
        cutoff_dict: list of cutoff in ase.neighborlist format
    """
    

    nl_i, nl_j = ase.neighborlist.neighbor_list('ij', atom, cutoff_dict)
    nl = [[] for i in range(atom.get_global_number_of_atoms())]
    for k in range(len(nl_i)):
        i, j = nl_i[k], nl_j[k]
        nl[i].append(j)
    return nl