"""
Module containing msd related methods
"""

import ase
import ase.data

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import logging
import os
import pathlib
import joblib
from numba import jit

import sadi.trajectory
import sadi.atom
import sadi.files.path

logger = logging.getLogger(__name__)

class Msd(object):
    """
    Main class for MSD
    """

    def __init__(self):
        """default constructor"""
        self.msd_data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, delta_Step = 1, first_frame = 0, parallel = False):
        """
        constructor of msd class

        Args:
            trajectory: ase trajectory object
            delta_Step: number of simulation steps between two frames
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        msd_class = cls() # initialize class
        step = sadi.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        msd_class.compute_msd(trajectory, step, parallel)
        return msd_class # return class as it is a constructor

    @staticmethod
    def compute_species_msd(trajectory, atomic_number = None):
        """calculate MSD with real pos (stored in r) compared to PBC pos stored
        in ase (extracted with position)
        if atomic_number is None, compute MSD between all atoms
        """
        r_0 = sadi.atom.select_species_positions(trajectory[0], atomic_number)
        # test reducing mem usage
        # r = np.zeros((len(trajectory), len(r_0), 3))
        # r[0] = r_0 
        r_t = r_0
        MSD = np.zeros(len(trajectory))
        for t in range(1, len(trajectory)):
            r_t_minus_dt = r_t
            dr = np.zeros((len(r_0), 3))
            for j in range(3): #x,y,z
                a = trajectory[t].get_cell()[j,j]
                # dr[:,j] = (sadi.atom.select_species_positions(trajectory[t], atomic_number) - r[t-1]%a)[:,j]
                dr[:,j] = (sadi.atom.select_species_positions(trajectory[t], atomic_number) - r_t_minus_dt%a)[:,j]
                for i in range(len(dr)):
                    if dr[i][j]>a/2:
                        dr[i][j] -= a
                    elif dr[i][j]<-a/2:
                        dr[i][j] += a
            # r[t] = dr + r[t-1]
            # MSD[t] = np.linalg.norm(r[t]-r_0)**2/len(r_0)
            r_t = dr + r_t_minus_dt
            MSD[t] = np.linalg.norm(r_t-r_0)**2/len(r_0)
        return MSD

    def compute_msd(self, trajectory, step, parallel):
        """
        Args:
            trajectory: ase trajectory object
            step: np array, simulation steps
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        logger.info("Start computing msd for %s frames", len(trajectory))
        
        elements = sadi.atom.get_atomic_numbers_unique(trajectory[0])

        Step = step     
        self.msd_data = pd.DataFrame({"Step": Step})
        if not parallel:
            self.msd_data["X"] = self.compute_species_msd(trajectory)

            for x in elements:
                x_str = ase.data.chemical_symbols[x]
                self.msd_data[x_str] = self.compute_species_msd(trajectory, x)
        else:
            x_list = [None] + elements
            num_cores = len(x_list) # default value
            if type(parallel) == int and parallel < num_cores:
                num_cores = parallel
            msd_list = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_species_msd)(trajectory, x) for x in x_list)
            self.msd_data["X"] = msd_list[0]
            for i in range(1, len(x_list)):
                x_str = ase.data.chemical_symbols[x_list[i]]
                self.msd_data[x_str] = msd_list[i]               



    def write_to_file(self, path_to_output):
        """path_to_output: where the MSD object will be written"""
        path_to_output = sadi.files.path.append_suffix(path_to_output, 'msd')
        self.msd_data.to_feather(path_to_output)

    @classmethod
    def from_msd(cls, path_to_msd):
        """
        constructor of msd class from msd file
        """
        msd_class = cls() # initialize class
        msd_class.read_msd_file(path_to_msd)
        return msd_class # return class as it is a constructor

    def read_msd_file(self, path_to_data):
        """path_to_data: where the MSD object is"""
        path_to_data = sadi.files.path.append_suffix(path_to_data, 'msd')
        self.msd_data = pd.read_feather(path_to_data)
