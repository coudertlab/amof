"""
Module containing rdf related methods
"""

# TD: clean imports

import ase
# from ase import Atoms, data
from ase import data
# from ase.io import read, write, vasp
# from ase.visualize import view
from ase.io.trajectory import Trajectory

from asap3.analysis.rdf import RadialDistributionFunction

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import glob

import bisect

import logging

import sadi.trajectory

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class Rdf(object):
    """
    Main class for rdf
    """

    def __init__(self):
        """default constructor"""
        self.rdf_data = []
        # self.struct = struct

    # # TBC
    # def cutoff_from_rdf(x, rdf, a, b): 
    #     """ return x such that rdf(x) global minimun between a and b of rdf"""
    #     return x[np.argmin(rdf[(x>a)*(x<b)])+np.size(rdf[x<=a])]

    @classmethod
    def from_trajectory(cls, trajectory, dr = 0.01, rmax = 'half_cell', single_frame = False, last_frames = False):
        """
        constructor of rdf class from sadi trajectory object
        dr, rmax in Angstrom
        If rmax is set to 'half_cell', then half of the minimum dimension of the cell is used to ensure no atom is taken into account twice for a given atom (computation is possible beyound this threeshold up to the min cell size)
        last_frames: number of last frames required
        Seed: specify if seed in traj so that these atoms are removed from the computation"""
        rdf_class = cls() # initialize class
        rdf_class.read_trajectory(path_to_traj, dr, rmax, single_frame, last_frames)
        return rdf_class # return class as it is a constructor

    # # must clean single_frame
    # @classmethod
    # def from_trajectory(cls, path_to_traj, dr = 0.01, rmax = 'half_cell', single_frame = False, last_frames = False):
    #     """
    #     constructor of rdf class from trajectory
    #     dr, rmax in Angstrom
    #     If rmax is set to 'half_cell', then half of the minimum dimension of the cell is used to ensure no atom is taken into account twice for a given atom (computation is possible beyound this threeshold up to the min cell size)
    #     last_frames: number of last frames required
    #     Seed: specify if seed in traj so that these atoms are removed from the computation"""
    #     rdf_class = cls() # initialize class
    #     rdf_class.read_trajectory(path_to_traj, dr, rmax, single_frame, last_frames)
    #     return rdf_class # return class as it is a constructor

    @classmethod
    def from_position(cls, path_to_traj, dr = 0.01, rmax = 'half_cell'):
        """
        constructor of rdf class from position file. Currently accept only lammps-data file format
        last_frames: number of last frames required
        Seed: specify if seed in traj so that these atoms are removed from the computation"""
        rdf_class = cls() # initialize class
        rdf_class.read_trajectory(path_to_traj, dr, rmax, single_frame = True)
        return rdf_class # return class as it is a constructor

    @classmethod
    def from_rdf(cls, path_to_rdf, single_frame = False, last_frames = False):
        """
        constructor of rdf class from rdf file
        """
        rdf_class = cls() # initialize class
        rdf_class.read_rdf_file(path_to_rdf)
        return rdf_class # return class as it is a constructor

    def compute_rdf(self, trajectory, dr, rmax):
        """
        compute rdf from sadi trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        N_species = len(atomic_numbers_unique) # number of different chemical species

        # default option
        if  rmax == 'half_cell':
            rmax = np.min([a for t in trajectory for a in t.get_cell_lengths_and_angles()[0:3]]) / 2
        bins = int(rmax // dr)
        r = np.arange(bins) * dr
        
        self.rdf_data = pd.DataFrame({"r": r})

        # Code from the asap3 manual for a trajectory
        RDFobj = None
        for atoms in traj:
            if RDFobj is None:
                RDFobj = RadialDistributionFunction(atoms, rmax, bins)
            else:
                RDFobj.atoms = atoms  # Fool RDFobj to use the new atoms
            RDFobj.update()           # Collect data
        
        # Total RDF        
        rdf = RDFobj.get_rdf(groups=0)        
        self.rdf_data["X-X"] = rdf

        # Partial RDFs               
        elements = [[(x, y) for y in atomic_numbers_unique] for x in atomic_numbers_unique] # cartesian product of very couple of  species

        # change to np.like 
        partial_rdf = [[0 for y in atomic_numbers_unique] for x in atomic_numbers_unique] # need to have same structure as elements but any content is fine as it will be replaced

        for i in range(N_species):
            for j in range(N_species):
                xx = elements[i][j]
                xx_str = data.chemical_symbols[xx[0]] + "-" + data.chemical_symbols[xx[1]]
                partial_rdf[i][j] = RDFobj.get_rdf(elements=xx,groups=0)
                self.rdf_data[xx_str] = partial_rdf[i][j]    
        for i in range(N_species):
            xx = elements[i][i]
            xx_str = data.chemical_symbols[xx[0]] + "-" + data.chemical_symbols[xx[1]]
            self.rdf_data[data.chemical_symbols[xx[0]] + "-X"] = sum([partial_rdf[i][j] for j in range(N_species)])   

    # def read_trajectory(self, path_to_traj, dr, rmax, single_frame = False, last_frames = False):
    #     """
    #     last_frames: number of last frames required
    #     Seed: specify if seed in traj so that these atoms are removed from the computation"""

    #     logger.info("read trajectory %s", path_to_traj)

    #     if single_frame:
    #         # to be changed with a proper import func or either a class
    #         # format to be changed when first used
    #         atoms = ase.io.read(path_to_traj, format = 'lammps-data', style='charge')

    #         def get_index_closest(myList, myNumber):
    #             """
    #             Assumes myList is sorted. Returns closest value to myNumber.

    #             If two numbers are equally close, return the smallest number.
    #             """
    #             pos = bisect.bisect_left(myList, myNumber)
    #     #         pos = np.searchsorted(myList, myNumber) # twice as slow in tried example
    #             if pos == 0:
    #                 return myList[0]
    #             if pos == len(myList):
    #                 return myList[-1]
    #             before = myList[pos - 1]
    #             after = myList[pos]
    #             if after - myNumber < myNumber - before:
    #                 return pos
    #             else:
    #                 return pos - 1

    #         myList = ase.data.atomic_masses            
    #         atomic_numbers = [get_index_closest(myList, myNumber) for myNumber in atoms.get_masses()]            
    #         atoms.set_atomic_numbers(atomic_numbers)
    #         traj = [atoms]
    #     else:
    #         traj = Trajectory(path_to_traj, mode='r')
    #     if last_frames!=False:
    #     #     output_path += "-lastframes"
    #         traj = [traj[i] for i in range(len(traj) - last_frames, len(traj))] # last <last_frames> frames
             
    #     # H = data.atomic_numbers['H']
    #     # C = data.atomic_numbers['C']
    #     # N = data.atomic_numbers['N']
    #     # Zn = data.atomic_numbers['Zn']

    #     # chemical_symbols = list(set())
    #     atomic_numbers_unique = list(set(traj[0].get_atomic_numbers()))
    #     N_species = len(atomic_numbers_unique) # number of different chemical species

    #     # add this as default option
    #     if  rmax == 'half_cell':
    #         rmax = np.min([a for t in traj for a in t.get_cell_lengths_and_angles()[0:3]]) / 2

    #     bins = int(rmax // dr)
        
    #     # if single_frame:
    #     #     bins = 200
    #     # elif last_frames!=False and last_frames<10:
    #     #     bins = 200
    #     # else:
    #     #     bins = 800
        


    #     r = np.arange(bins) * dr
        
    #     self.rdf_data = pd.DataFrame({"r": r})
    #     # self.rdf_data = pd.DataFrame({"r": r}, index = r) # set index ?

    #     rng = rmax
    #     # Code from the asap3 manual for a trajectory
    #     RDFobj = None
    #     for atoms in traj:
    #         if RDFobj is None:
    #             RDFobj = RadialDistributionFunction(atoms, rng, bins)
    #         else:
    #             RDFobj.atoms = atoms  # Fool RDFobj to use the new atoms
    #         RDFobj.update()           # Collect data
        
    #     # Total RDF
        
    #     rdf = RDFobj.get_rdf(groups=0)
        
    #     self.rdf_data["X-X"] = rdf


            
    #     # elements_list = [H, C, N, Zn]
    #     elements = [[(x, y) for y in atomic_numbers_unique] for x in atomic_numbers_unique] # cartesian product of very couple of  species

    #     partial_rdf = [[0 for y in atomic_numbers_unique] for x in atomic_numbers_unique] # need to have same structure as elements but any content is fine as it will be replaced

    #     for i in range(N_species):
    #         for j in range(N_species):
    #             xx = elements[i][j]
    #             xx_str = data.chemical_symbols[xx[0]] + "-" + data.chemical_symbols[xx[1]]
    #             partial_rdf[i][j] = RDFobj.get_rdf(elements=xx,groups=0)
    #             # plt.plot(x, partial_rdf[i][j], label = xx_str)
    #             # print(xx_str)
    #             # np.save(output_path + ".rdf_" + xx_str, partial_rdf[i][j])
    #             self.rdf_data[xx_str] = partial_rdf[i][j]

    #     # plt.legend()
    #     # plt.savefig(output_path + ".rdf_Y-Y" + ".png")        
    #     # plt.show()        
    #     for i in range(N_species):
    #         xx = elements[i][i]
    #         xx_str = data.chemical_symbols[xx[0]] + "-" + data.chemical_symbols[xx[1]]
    #         # plt.plot(x, partial_rdf[i][i], label = xx_str)
    #         # print(xx_str)
    #         # np.save(output_path + ".rdf_" + data.chemical_symbols[xx[0]] + "-X", sum([partial_rdf[i][j] for j in range(N_species)]))
    #         self.rdf_data[data.chemical_symbols[xx[0]] + "-X"] = sum([partial_rdf[i][j] for j in range(N_species)])
    #     # plt.legend()
    #     # plt.savefig(output_path + ".rdf_X-Y" + ".png")        
    #     # plt.show()       


    def write_to_file(self, path_to_output):
        self.rdf_data.to_feather(path_to_output + ".rdf")

    def read_rdf_file(self, path_to_data):
        self.rdf_data = pd.read_feather(path_to_data + ".rdf")

    def plot(self):
        plt.plot(r, rdf)
        plt.xlabel("$r$ (${\AA}$)")
        plt.ylabel("$g(r)$")
        
        cutoff = cutoff_from_rdf(x, rdf, 3, 3.8)
        plt.axvline(cutoff, linestyle = '--', linewidth = 1)
        plt.text(cutoff + 0.1, 0, "%.2f" % round(cutoff,2))
        plt.savefig(output_path + ".rdf_X-X" + ".png")
        plt.show()

    def average_rdfs(traj_names, output_name):
        """not changed, may be deprecated when data struct turned to df"""
        rdf_prefixes = ["X-X", "Ag-X", "Sb-X", "Te-X", "Ag-Ag", "Sb-Sb", "Te-Te", "Ag-Sb", "Sb-Ag", "Te-Ag", "Ag-Te", "Sb-Te", "Te-Sb"]
        prefixes = [".x"]+[".rdf_" + x for x in rdf_prefixes]
        input_path = "../analysis/data/rdf/"
        output_path = input_path
        
        for p in prefixes:
            rdf_l = [] 
            for traj_name in traj_names:
                rdf_l.append(np.load(input_path + traj_name + p + ".npy"))
            
            rdf_l=np.array(rdf_l)
            rdf = np.average(rdf_l, axis=0)
            np.save(output_path +  output_name + p, rdf)   
    #average_rdfs(["3.c1.pbe","3.c2.pbe","3.c3.pbe"], "3.c.pbe")

class RdfPlotter(object):
    def __init__(self):
        """default constructor"""
        self.multiple_rdf_data = {}
        # self.struct = struct

    def add_rdf(self, path_to_rdf, rdf_name = None):
        if rdf_name is None:
            rdf_name = path_to_rdf
        self.multiple_rdf_data[rdf_name] = rdf.from_rdf(path_to_rdf).rdf_data

    # must clean single_frame
    @classmethod
    def from_multiple_rdf(cls, list_of_path_to_rdf, list_of_rdf_name = None):
        """
        constructor of rdf class from rdf file
        """
        if list_of_rdf_name is None:
            list_of_rdf_name = list_of_path_to_rdf
        rdf_plotter_class = cls() # initialize class
        for i in range(len(list_of_path_to_rdf)):
            rdf_plotter_class.add_rdf(list_of_path_to_rdf[i], list_of_rdf_name[i])
        return rdf_plotter_class # return class as it is a constructor

    def plot(self, nn_set, path_to_plot = None, xlim = None):
        """
        plot the rdf of a given neighbour nn_set (e.g. "X-X" or "Zn-Zn") for every rdf_data in multiple_rdf_data
        """
        # plt.rcParams.update({'font.size': 8})
        # plt.figure(figsize=[6.0, 2.3]) #default on spyder is [6.0, 4.0] 
        for rdf_name, rdf_data in self.multiple_rdf_data.items():
            plt.plot(rdf_data['r'], rdf_data[nn_set], label = rdf_name,  alpha=0.9, linewidth = 1)
    #        cutoff = cutoff_from_rdf(x, rdf, 3, 3.8)
    #        plt.axvline(cutoff, linestyle = '--', linewidth = 1)
    #        plt.text(cutoff + 0.1, 0, "%.2f" % round(cutoff,2))
    #        print(cutoff)
    #        coord = coordination_number(x, rdf, cutoff, phases[t])
    #        print(coord)
        plt.legend()
        plt.xlabel("$r$ (${\AA}$)")
        plt.ylabel("$g(r)$")
        if xlim is not None:
            plt.xlim(xlim[0],xlim[-1])
        plt.title(nn_set)
        if path_to_plot is not None:
            plt.savefig(path_to_plot + ".png", dpi = 300)
        # plt.xlim(2.5,8.5)
        # plt.savefig(path_to_plot+output_traj_name+"_rdf_"+prefix+".png", dpi = 300, bbox_inches='tight')
        plt.show()    
   