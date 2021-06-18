"""
Module containing bond angle distribution related methods
"""

import ase
import ase.neighborlist
import ase.data
# import asap3.analysis.bad

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
import joblib
import itertools

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
        self.bad_data = pd.DataFrame({"theta": np.empty([0])})


    @classmethod
    def from_trajectory(cls, trajectory, nb_set_and_cutoff, dtheta = 0.05, parallel = False):
        """
        constructor of rdf class from an ase trajectory object
        Args:
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
            dtheta: float, in degrees, default value of 0.05degree similar to RG
        """
        bad_class = cls() # initialize class
        bad_class.compute_bad(trajectory, nb_set_and_cutoff, dtheta, parallel)
        return bad_class # return class as it is a constructor

    @classmethod
    def from_file(cls, filename):
        """
        constructor of bad class from bad file
        """
        bad_class = cls() # initialize class
        bad_class.read_bad_file(filename)
        return bad_class # return class as it is a constructor

    @staticmethod
    def bad_BAB(atom, A, B, nl):
        """Return BAD distribution of B-A-B angle.

        Args:
            atom: ase atom object
            A, B: either int specifying atom numbers
                    A and B can be "X" (str) and in that case every specie will be taken into account
            nl: neighbor list used
        
        Returns:
            angles: list of floats representing B-A-B angles
                if B = "X", return X-A-Y angle. (all neighbours around A)
                if A and B = "X", return BAD distribution of X-Z-Y angle. (all neighbours around Z, for every Z)
        """    
        atomic_numbers = atom.get_atomic_numbers()  
        angles = [] #one set of 3 atoms will be comprised 1 time
        for a in range(len(atomic_numbers)):
            if A == "X" or atomic_numbers[a] == A:
                indices = nl[a]
                # indices, offsets = nl.get_neighbors(a)
                B_neighbors = [i for i in indices if B == "X" or atomic_numbers[i] == B] #Take couples of B-B neighbours
                comb = itertools.combinations(B_neighbors, 2) # Get all combinations of Bs
                
                angles_indices = []
                for pair in list(comb): 
                    i, j = pair
                    angles_indices.append([i,a,j])
                    # debug
                    # print(i, a, j, atom.get_angle(i,a,j, mic=True))
                    # if atom.get_angle(i,a,j, mic=True) < 90:
                    #     logger.error("angle<90")
                    #     logger.error("%s %s %s %s", i, a, j, atom.get_angle(i,a,j, mic=True))
                    #     with open("low_angle.txt", "w") as text_file:
                    #         text_file.write("%s %s %s %s", i, a, j, atom.get_angle(i,a,j, mic=True))
                    #     atom.write("low_angle.cif")
                    #     import sys
                    #     sys.exit()
    
                if angles_indices != []: # get_angles() doesn't work with an empty list
                    angles += list(atom.get_angles(angles_indices, mic=True))
        return angles
    
    def compute_bad_for_frame(self, atom, cutoff_dict, elements):
        """
        compute bad for ase atom object
        """
        # atom = trajectory[i]
        nl = satom.get_neighborlist(atom, cutoff_dict)
        dic = {}
        for A, B in elements:
            aba_str = "-".join([ase.data.chemical_symbols[C] for C in [B, A, B]])
            dic[aba_str] = self.bad_BAB(atom, A, B, nl)  

        return dic
    
    def compute_bad(self, trajectory, nb_set_and_cutoff, dtheta, parallel):
        """
        compute compute_bad from ase trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))

        cutoff_dict = satom.format_cutoff(nb_set_and_cutoff)
        elements_present_unique =  list(set([ase.data.atomic_numbers[i] for nb_set in nb_set_and_cutoff.keys() for i in nb_set.split('-') ]))

        if len(elements_present_unique) == len(atomic_numbers_unique):
            elements_present_unique.append("X")
        N_species = len(elements_present_unique) # number of different chemical species
            # N_species += 1 # include "X"

        # Partial RDFs               
        elements = [(a, b) for b in elements_present_unique for a in elements_present_unique if (a not in [b, "X"] or ((a, b) == ("X", "X"))) ]

        # nl = ase.neighborlist.NeighborList(cutoff_dict, self_interaction = False, bothways = True)
        # cutoffs = ase.neighborlist.natural_cutoffs(atom, mult = 1.2) # Generate a radial cutoff for every atom based on covalent radii ; mult is a multiplier for all cutoffs
        # nl = ase.neighborlist.NeighborList(cutoffs, self_interaction = False, bothways = True)

        rmax = np.max(list(nb_set_and_cutoff.values()))

        logger.info("Start computing bad for %s frames with dtheta = %s", len(trajectory), dtheta)
        bins = int(180 // dtheta)
        theta_bins = np.arange(bins + 2) * dtheta    
        theta = np.arange(bins + 1) * dtheta + dtheta / 2   
        self.bad_data = pd.DataFrame({"theta": theta})    


        if parallel == False:
            list_of_dict = [self.compute_bad_for_frame(trajectory[i], cutoff_dict, elements) for i in range(len(trajectory))]
        else:
            num_cores = parallel if type(parallel) == int else 18
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_bad_for_frame)(trajectory[i], cutoff_dict, elements) for i in range(len(trajectory)))

        aba_str_keys = list_of_dict[0].keys()
        for aba_str in aba_str_keys:
            angles = []
            for dic in list_of_dict:
                angles += dic[aba_str]
            if angles != []:
                self.bad_data[aba_str] = np.histogram(angles, bins = theta_bins, density=True)[0]


    def write_to_file(self, filename):
        filename = sadi.files.path.append_suffix(filename, 'bad')
        self.bad_data.to_feather(filename)

    def read_bad_file(self, path_to_data):
        path_to_data = sadi.files.path.append_suffix(path_to_data, 'bad')
        self.bad_data = pd.read_feather(path_to_data)
