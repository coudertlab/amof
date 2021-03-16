"""
Module containing msd related methods
"""

import ase
import ase.data

import matplotlib.pyplot as plt
import numpy as np

import sadi.trajectory
import sadi.atom

class Msd(object):
    """
    Main class for MSD
    """

    def __init__(self):
        """default constructor"""
        self.msd_data = []

    @classmethod
    def from_trajectory(cls, trajectory, dump_freq = 1):
        """
        constructor of msd class from an ase trajectory object

        """
        msd_class = cls() # initialize class
        msd_class.compute_msd(trajectory, dr, rmax)
        return msd_class # return class as it is a constructor

    @staticmethod
    def compute_species_msd(self, trajectory, atomic_number = None):
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

    def compute_msd(self, trajectory, dump_freq):
        """
        Args:
            trajectory: ase trajectory object
            dump_freq: number of simulation steps between two frames
        """
        elements = sadi.atom.get_atomic_numbers_unique(trajectory[0])

        Step = np.arange(len(trajectory)) * dump_freq        
        self.msd_data = pd.DataFrame({"Step": Step})
        self.msd_data["X"] = Msd.compute_species_msd(trajectory)

        for x in elements:
            x_str = ase.data.chemical_symbols[x]
            self.msd_data[x_str] = Msd.compute_species_msd(trajectory, x)

    def write_to_file(self, path_to_output):
        """path_to_output: where the MSD object will be written"""
        self.msd_data.to_feather(path_to_output + ".rdf")

    @classmethod
    def from_msd(cls, path_to_msd):
        """
        constructor of rdf class from rdf file
        """
        msd_class = cls() # initialize class
        msd_class.read_msd_file(path_to_msd)
        return msd_class # return class as it is a constructor

    def read_msd_file(self, path_to_data, add_file_extension = True):
        """add_file_extension: if True will try to add .rdf if not already present"""
        if add_file_extension and path_to_data[-4:] != ".rdf":
            path_to_data += ".rdf"
        self.rdf_data = pd.read_feather(path_to_data)
