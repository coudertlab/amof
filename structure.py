"""
Module containing custom methods applied to pymatgen structures 
"""

import pymatgen

def get_center_of_mass(structure, indices):
    """
    Return the center of mass of a selection sites of structure selected by indices

    Args:
        structure: pymatgen structure with periodic sites
        indices: list of int, index of selection in structure.sites
    
    Return:
        center_of_mass: cartesian coordinates of the center of mass
    """
    coords = []
    species = []
    reference_site = indices[0] # coord will be computed in a subcell close to this site
    for i in indices:
        species.append(structure.sites[i].species)
        coords_from_ref_to_i = pymatgen.util.coord.pbc_diff(structure.sites[i].frac_coords, structure.sites[reference_site].frac_coords)
        coords.append(structure.lattice.get_cartesian_coords(coords_from_ref_to_i)) # to cartesian coords
    return pymatgen.core.Molecule(species, coords).center_of_mass + structure.sites[reference_site].coords # add reference_coords