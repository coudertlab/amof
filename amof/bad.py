"""
Module containing bond angle distribution related methods
"""
import os

# force numpy to use one thread 
os.environ["OMP_NUM_THREADS"] = "1"  # essential
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # at least one of these 4 is needed
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
os.environ["OPENBLAS_MAIN_FREE"] = "1"

import ase
import ase.neighborlist
import ase.data

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import joblib
import itertools

import logging

import amof.trajectory
import amof.files.path
import amof.atom as amatom

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class CoreBad(object):
    """
    Core material for every Bad function
    """

    @classmethod
    def from_trajectory(cls, trajectory, nb_set_and_cutoff, dtheta = 0.05, normalization = 'total', parallel = False, ):
        """
        constructor of bad class from an ase trajectory object
        Args:
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
            dtheta: float, in degrees, default value of 0.05degree similar to RG
            normalization: str, can be 'total' (entire bad is normalized) or 'partial' (sum of partial is normalized)
        """
        bad_class = cls() # initialize class
        bad_class.compute_bad(trajectory, nb_set_and_cutoff, dtheta, normalization, parallel)
        return bad_class # return class as it is a constructor

    @classmethod
    def from_file(cls, filename):
        """
        constructor of bad class from bad file
        """
        bad_class = cls() # initialize class
        bad_class.read_bad_file(filename)
        return bad_class # return class as it is a constructor

class Bad(CoreBad):
    """
    Main class for bad
    """

    def __init__(self):
        """default constructor"""
        self.data = pd.DataFrame({"theta": np.empty([0])})

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
    
                if angles_indices != []: # get_angles() doesn't work with an empty list
                    angles += list(atom.get_angles(angles_indices, mic=True))
        return angles
    
    def compute_bad_for_frame(self, atom, cutoff_dict, elements):
        """
        compute bad for ase atom object
        """
        # atom = trajectory[i]
        nl = amatom.get_neighborlist(atom, cutoff_dict)
        dic = {}
        for A, B in elements:
            aba_str = "-".join([ase.data.chemical_symbols[C] for C in [B, A, B]])
            dic[aba_str] = self.bad_BAB(atom, A, B, nl)  

        return dic
    
    def compute_bad(self, trajectory, nb_set_and_cutoff, dtheta, normalization, parallel):
        """
        compute compute_bad from ase trajectory object

        normalization not used so far, only one option (ie 'total')
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))

        cutoff_dict = amatom.format_cutoff(nb_set_and_cutoff)
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
        self.data = pd.DataFrame({"theta": theta})    


        if parallel == False:
            list_of_dict = [self.compute_bad_for_frame(trajectory[i], cutoff_dict, elements) for i in range(len(trajectory))]
        else:
            num_cores = parallel if type(parallel) == int else max(joblib.cpu_count() // 2 - 2, 2) # heuristic for 40cores Xeon cpus
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_bad_for_frame)(trajectory[i], cutoff_dict, elements) for i in range(len(trajectory)))

        aba_str_keys = list_of_dict[0].keys()
        for aba_str in aba_str_keys:
            angles = []
            for dic in list_of_dict:
                angles += dic[aba_str]
            if angles != []:
                self.data[aba_str] = np.histogram(angles, bins = theta_bins, density=True)[0]


    def write_to_file(self, filename):
        filename = amof.files.path.append_suffix(filename, 'bad')
        self.data.to_feather(filename)

    def read_bad_file(self, path_to_data):
        path_to_data = amof.files.path.append_suffix(path_to_data, 'bad')
        self.data = pd.read_feather(path_to_data)


class BadByCn(CoreBad):
    """
    Modified Bad class that computes BAD by Coordination Number (BAD for A bonded to 1 B, 2 B, etc.)

    Partial BAD by cn are normalized such that the sum over cn of partial BAD[cn] is of total area 1 (same as Bad class result).
    Each cn has a weight proportional to the number of angles found for A atoms with cn B nn.

    Limited derivation from CoreBad class as the data structure behind is xarray instead of pandas
    Bad could later be adapted with this new datastructures if this class end up routinely used
    """

    def __init__(self):
        """default constructor"""
        self.data = xr.DataArray(np.empty([0,0,0]), 
            coords = [('theta', np.empty([0], dtype='float64')), # one of numpy float
                ('atom_triple', np.empty([0], dtype='str_')), 
                ('cn', np.empty([0], dtype='str_'))], # numpy str is str_
            name = 'bad')

    @staticmethod
    def bad_BAB(atom, A, B, nl):
        """Return BAD distribution of B-A-B angle.

        Args:
            atom: ase atom object
            A, B: either int specifying atom numbers
                    A and B can be "X" (str) and in that case every specie will be taken into account
            nl: neighbor list used
        
        Returns:
            angles: dict of list of floats representing B-A-B angles
                Each dict key is the number of B nb of A
                if B = "X", return X-A-Y angle. (all neighbours around A)
                if A and B = "X", return BAD distribution of X-Z-Y angle. (all neighbours around Z, for every Z)
        """    
        atomic_numbers = atom.get_atomic_numbers()  
        angles = {} #one set of 3 atoms will be comprised 1 time
        for a in range(len(atomic_numbers)):
            if A == "X" or atomic_numbers[a] == A:
                indices = nl[a]
                # indices, offsets = nl.get_neighbors(a)
                B_neighbors = [i for i in indices if B == "X" or atomic_numbers[i] == B] #Take couples of B-B neighbours
                cn = len(B_neighbors)
                if cn > 1 and cn not in angles.keys():
                    angles[cn] = []

                comb = itertools.combinations(B_neighbors, 2) # Get all combinations of Bs
                angles_indices = []
                for pair in list(comb): 
                    i, j = pair
                    angles_indices.append([i,a,j])
    
                if angles_indices != []: # get_angles() doesn't work with an empty list
                    angles[cn] += list(atom.get_angles(angles_indices, mic=True))
        return angles
    
    def compute_bad_for_frame(self, atom, cutoff_dict, elements):
        """
        compute bad for ase atom object
        """
        # atom = trajectory[i]
        nl = amatom.get_neighborlist(atom, cutoff_dict)
        dic = {}
        for A, B in elements:
            aba_str = "-".join([ase.data.chemical_symbols[C] for C in [B, A, B]])
            dic[aba_str] = self.bad_BAB(atom, A, B, nl)  

        return dic
    
    def compute_bad(self, trajectory, nb_set_and_cutoff, dtheta, normalisation, parallel):
        """
        compute compute_bad from ase trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))

        cutoff_dict = amatom.format_cutoff(nb_set_and_cutoff)
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
        # self.data = pd.DataFrame({"theta": theta})    


        if parallel == False:
            list_of_dict = [self.compute_bad_for_frame(trajectory[i], cutoff_dict, elements) for i in range(len(trajectory))]
        else:
            num_cores = parallel if type(parallel) == int else max(joblib.cpu_count() // 2 - 2, 2) # heuristic for 40cores Xeon cpus
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_bad_for_frame)(trajectory[i], cutoff_dict, elements) for i in range(len(trajectory)))

        dic_of_xarray = {}
        aba_str_keys = list_of_dict[0].keys()
        for aba_str in aba_str_keys:
            angles_all = {}
            for dic_aba in list_of_dict:
                for cn, angles in dic_aba[aba_str].items():
                    if cn not in angles_all.keys():
                        angles_all[cn] = []
                    angles_all[cn] += angles
            if angles_all != {}:
                if normalisation == 'partial': 
                    num_angles_all = np.sum(len(ang) for ang in angles_all.values())
                cn_coord, bad_list = [], []
                for cn, ang in angles_all.items():
                    cn_coord.append(cn)
                    normalization_ratio = len(ang)/num_angles_all if normalisation == 'partial' else 1
                    bad_list.append(normalization_ratio * np.histogram(angles_all[cn], bins = theta_bins, density=True)[0])
                ar = xr.DataArray(bad_list, coords={"cn":cn_coord, "theta": theta})
                dic_of_xarray[aba_str] = ar
                1
        xa = xr.Dataset(dic_of_xarray)
        xa = xa.to_array("atom_triple")
        xa = xr.Dataset({'bad': xa}) 
        self.data = xa


    def write_to_file(self, filename):
        filename = amof.files.path.append_suffix(filename, 'bad')
        self.data.to_netcdf(filename)

    def read_bad_file(self, filename):
        filename = amof.files.path.append_suffix(filename, 'bad')
        self.data = xr.open_dataset(filename)
