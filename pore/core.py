"""
Module containing pore analysis related methods
"""

import ase
import ase.data

import numpy as np
import pandas as pd
import logging
import os
import pathlib

import sadi.trajectory
import sadi.atom
import sadi.files.path

logger = logging.getLogger(__name__)

class Pore(object):
    """
    Main class for pore analysis
    """

    def __init__(self):
        """default constructor"""
        self.msd_data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, delta_Step = 1):
        """
        constructor of msd class from an ase trajectory object

        """
        msd_class = cls() # initialize class
        msd_class.compute_msd(trajectory, delta_Step)
        return msd_class # return class as it is a constructor

    @staticmethod
    def compute_species_msd(trajectory, atomic_number = None):
        """calculate MSD with real pos (stored in r) compared to PBC pos stored
        in ase (extracted with position)
        if atomic_number is None, compute MSD between all atoms
        """
        r_0 = sadi.atom.select_species_positions(trajectory[0], atomic_number)
        r = np.zeros((len(trajectory), len(r_0), 3))
        r[0] = r_0 
        MSD = np.zeros(len(trajectory))
        for t in range(1, len(trajectory)):
            dr = np.zeros((len(r_0), 3))
            for j in range(3): #x,y,z
                a = trajectory[t].get_cell()[j,j]
                dr[:,j] = (sadi.atom.select_species_positions(trajectory[t], atomic_number) - r[t-1]%a)[:,j]
                for i in range(len(dr)):
                    if dr[i][j]>a/2:
                        dr[i][j] -= a
                    elif dr[i][j]<-a/2:
                        dr[i][j] += a
            r[t] = dr + r[t-1]
            MSD[t] = np.linalg.norm(r[t]-r_0)**2/len(r_0)
        return MSD

    def compute_msd(self, trajectory, delta_Step):
        """
        Args:
            trajectory: ase trajectory object
            delta_Step: number of simulation steps between two frames
        """
        logger.info("Start computing msd for %s frames with delta_Step = %s", len(trajectory), delta_Step)
        
        elements = sadi.atom.get_atomic_numbers_unique(trajectory[0])

        Step = np.arange(len(trajectory)) * delta_Step        
        self.msd_data = pd.DataFrame({"Step": Step})
        self.msd_data["X"] = self.compute_species_msd(trajectory)

        for x in elements:
            x_str = ase.data.chemical_symbols[x]
            self.msd_data[x_str] = self.compute_species_msd(trajectory, x)

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
