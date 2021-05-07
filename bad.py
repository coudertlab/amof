"""
Module containing bond angle distribution related methods
"""

import ase
import ase.data
# import asap3.analysis.bad

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
import joblib

import logging

import sadi.trajectory
import sadi.files.path
import sadi.atom as satom

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class Bad(object):
    """
    Main class for bad
    """

    def __init__(self):
        """default constructor"""
        self.bad_data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, dr = 0.01, rmax = 'half_cell'):
        """
        constructor of bad class from an ase trajectory object
        dr, rmax in Angstrom
        If rmax is set to 'half_cell', then half of the minimum dimension of the cell is used to ensure no atom is taken into account twice for a given atom (computation is possible beyound this threeshold up to the min cell size)
        """
        bad_class = cls() # initialize class
        bad_class.compute_bad(trajectory, dr, rmax)
        return bad_class # return class as it is a constructor

    @classmethod
    def from_file(cls, filename):
        """
        constructor of bad class from bad file
        """
        bad_class = cls() # initialize class
        bad_class.read_bad_file(filename)
        return bad_class # return class as it is a constructor

    def compute_bad(self, trajectory, dr, rmax):
        """
        compute bad from ase trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        N_species = len(atomic_numbers_unique) # number of different chemical species

        # default option
        if  rmax == 'half_cell':
            rmax = np.min([a for t in trajectory for a in t.get_cell_lengths_and_angles()[0:3]]) / 2

        logger.info("Start computing bad for %s frames with dr = %s and rmax = %s", len(trajectory), dr, rmax)
        bins = int(rmax // dr)
        r = np.arange(bins) * dr        
        self.bad_data = pd.DataFrame({"r": r})

        # Code from the asap3 manual for a trajectory
        badobj = None
        for atoms in trajectory:
            if badobj is None:
                badobj = asap3.analysis.bad.RadialDistributionFunction(atoms, rmax, bins)
            else:
                badobj.atoms = atoms  # Fool badobj to use the new atoms
            badobj.update()           # Collect data
        
        # Total bad        
        bad = badobj.get_bad(groups=0)        
        self.bad_data["X-X"] = bad

        # Partial bads               
        elements = [[(x, y) for y in atomic_numbers_unique] for x in atomic_numbers_unique] # cartesian product of very couple of  species

        # change to np.like 
        partial_bad = [[0 for y in atomic_numbers_unique] for x in atomic_numbers_unique] # need to have same structure as elements but any content is fine as it will be replaced

        for i in range(N_species):
            for j in range(N_species):
                xx = elements[i][j]
                xx_str = ase.data.chemical_symbols[xx[0]] + "-" + ase.data.chemical_symbols[xx[1]]
                partial_bad[i][j] = badobj.get_bad(elements=xx,groups=0)
                self.bad_data[xx_str] = partial_bad[i][j]    
        for i in range(N_species):
            xx = elements[i][i]
            xx_str = ase.data.chemical_symbols[xx[0]] + "-" + ase.data.chemical_symbols[xx[1]]
            self.bad_data[ase.data.chemical_symbols[xx[0]] + "-X"] = sum([partial_bad[i][j] for j in range(N_species)])   

    def write_to_file(self, filename):
        filename = sadi.files.path.append_suffix(filename, 'bad')
        self.bad_data.to_feather(filename)

    def read_bad_file(self, path_to_data):
        path_to_data = sadi.files.path.append_suffix(path_to_data, 'bad')
        self.bad_data = pd.read_feather(path_to_data)
