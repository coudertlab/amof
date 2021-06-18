"""
Module containing coordination number related methods
"""

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4

import ase
import ase.data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib

import logging

import sadi.trajectory
import sadi.files.path
import sadi.atom as satom

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class CoordinationNumber(object):
    """
    Main class to compute CoordinationNumber
    """

    def __init__(self):
        """default constructor"""
        self.cn_data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, nb_set_and_cutoff, delta_Step = 1, first_frame = 0, parallel = False):
        """
        constructor of rdf class from an ase trajectory object
        Args:
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
            dr: float, in Angstrom
        """
        cn_class = cls() # initialize class
        step = sadi.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        cn_class.compute_cn(trajectory, nb_set_and_cutoff, step, parallel)
        return cn_class # return class as it is a constructor

    def compute_cn(self, trajectory, nb_set_and_cutoff, step, parallel):
        """
        compute coordination from ase trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        N_species = len(atomic_numbers_unique) # number of different chemical species

        logger.info("Start computing coordination number for %s frames", len(trajectory))
        cutoff_dict = satom.format_cutoff(nb_set_and_cutoff)

        def compute_cn_for_frame(atom, step):
            """
            compute coordination for ase atom object
            """
            # atoms = trajectory[i]
            dic = {'Step': step}

            nl = satom.get_neighborlist(atom, cutoff_dict)
            atomic_numbers = atom.get_atomic_numbers()  
            for nb_set, cutoff in nb_set_and_cutoff.items():
                a, b = tuple(ase.data.atomic_numbers[i] for i in nb_set.split('-'))
                cn_list = []
                for i in range(atom.get_global_number_of_atoms()):
                    if atomic_numbers[i] == a:
                        cn_list.append(np.sum(atomic_numbers[nl[i]] == b))
                dic[nb_set] = np.mean(cn_list)
            return dic

        if parallel == False:
            list_of_dict = [compute_cn_for_frame(trajectory[i], step[i]) for i in range(len(trajectory))]
        else:
            logger.warning("Parallel mode for coordination number very slow, best to use serial")
            num_cores = parallel if type(parallel) == int else 18
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(compute_cn_for_frame)(trajectory[i], step[i]) for i in range(len(trajectory)))

        self.cn_data = pd.DataFrame(list_of_dict)

    @classmethod
    def from_file(cls, filename):
        """
        constructor of cn class from msd file
        """
        cn_class = cls() # initialize class
        cn_class.read_cn_file(filename)
        return cn_class # return class as it is a constructor

    def read_cn_file(self, filename):
        """path_to_data: where the cn object is"""
        filename = sadi.files.path.append_suffix(filename, 'cn')
        self.cn_data = pd.read_feather(filename)

    def write_to_file(self, filename):
        filename = sadi.files.path.append_suffix(filename, 'cn')
        self.cn_data.to_feather(filename)
