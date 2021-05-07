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
        self.bad_data = pd.DataFrame({"Step": np.empty([0])})


    @classmethod
    def from_trajectory(cls, trajectory, nb_set_and_cutoff, delta_Step = 1, first_frame = 0, dr = 0.01, parallel = False):
        """
        constructor of rdf class from an ase trajectory object
        Args:
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
            dr: float, in Angstrom
        """
        bad_class = cls() # initialize class
        step = sadi.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        bad_class.compute_bad(trajectory, nb_set_and_cutoff, step, dr, parallel)
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
                indices, offsets = nl.get_neighbors(a)
                B_neighbors = [i for i in indices if B == "X" or atomic_numbers[i] == B] #Take couples of B-B neighbours
                comb = itertools.combinations(B_neighbors, 2) # Get all combinations of Bs
                
                angles_indices = []
                for pair in list(comb): 
                    i, j = pair
                    angles_indices.append([i,a,j])
    
                if angles_indices != []: # get_angles() doesn't work with an empty list
                    angles += list(atom.get_angles(angles_indices, mic=True))
        return angles
    
    
    def compute_bad(self, trajectory, nb_set_and_cutoff, step, dr, parallel):
        """
        compute compute_bad from ase trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        N_species = len(atomic_numbers_unique) # number of different chemical species

        rmax = np.max(list(nb_set_and_cutoff.values()))

        logger.info("Start computing coordination number for %s frames with dr = %s and rmax = %s", len(trajectory), dr, rmax)
        bins = int(rmax // dr)
        r = np.arange(bins) * dr        

        def compute_cn_for_frame(i):
            """
            compute coordination for ase atom object
            """
            atoms = trajectory[i]
            dic = {'Step': step[i]}
            RDFobj = asap3.analysis.rdf.RadialDistributionFunction(atoms, rmax, bins)
            density = satom.get_number_density(atoms)
            for nn_set, cutoff in nb_set_and_cutoff.items():
                xx = tuple(ase.data.atomic_numbers[i] for i in nn_set.split('-'))
                rdf = RDFobj.get_rdf(elements=xx, groups=0)
                dic[nn_set] = get_coordination_number(r, rdf, cutoff, density)
            return dic

        if parallel == False:
            list_of_dict = [compute_cn_for_frame(i) for i in range(len(trajectory))]
        else:
            logger.warning("Parallel mode for coordination number very slow, best to use serial")
            num_cores = parallel if type(parallel) == int else 18
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(compute_cn_for_frame)(i) for i in range(len(trajectory)))

        self.cn_data = pd.DataFrame(list_of_dict)

    def write_to_file(self, filename):
        filename = sadi.files.path.append_suffix(filename, 'bad')
        self.bad_data.to_feather(filename)

    def read_bad_file(self, path_to_data):
        path_to_data = sadi.files.path.append_suffix(path_to_data, 'bad')
        self.bad_data = pd.read_feather(path_to_data)
