"""
Module containing rdf related methods
"""

import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4

import ase
import ase.data
import asap3.analysis.rdf

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.interpolate
import joblib

import logging

import amof.trajectory
import amof.files.path
import amof.atom as amatom

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class Rdf(object):
    """
    Main class for rdf
    """

    def __init__(self):
        """default constructor"""
        self.data = pd.DataFrame({"r": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, dr = 0.01, rmax = 'half_cell'):
        """
        Constructor of rdf class 
        
        Args:
            trajectory: an ase trajectory object
            dr, rmax: floats in Angstrom
                If rmax is set to 'half_cell', then half of the minimum dimension of the cell is used 
                to ensure no atom is taken into account twice for a given atom 
                (computation is possible beyond this threeshold up to the min cell size)
        """
        rdf_class = cls() # initialize class
        rdf_class.compute_rdf(trajectory, dr, rmax)
        return rdf_class # return class as it is a constructor

    @classmethod
    def from_rdf(cls, *args):
        logger.exception('from_rdf is deprecated, use from_file instead')

    @classmethod
    def from_file(cls, path_to_rdf):
        """
        constructor of rdf class from rdf file
        """
        rdf_class = cls() # initialize class
        rdf_class.read_rdf_file(path_to_rdf)
        return rdf_class # return class as it is a constructor


    def compute_rdf(self, trajectory, dr, rmax):
        """
        compute rdf from ase trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        N_species = len(atomic_numbers_unique) # number of different chemical species

        rmax_half_cell = np.min([a for t in trajectory for a in t.get_cell_lengths_and_angles()[0:3]]) / 2
        if  rmax == 'half_cell':        # default option
            rmax = rmax_half_cell
        elif rmax > rmax_half_cell:
            logger.info("Specified rmax %s is larger than half cell; will use half_cell rmax", rmax)
            rmax = rmax_half_cell

        logger.info("Start computing rdf for %s frames with dr = %s and rmax = %s", len(trajectory), dr, rmax)
        bins = int(rmax // dr)
        r = np.arange(bins) * dr        
        self.data = pd.DataFrame({"r": r})

        # Code from the asap3 manual for a trajectory
        RDFobj = None
        for atoms in trajectory:
            if RDFobj is None:
                RDFobj = asap3.analysis.rdf.RadialDistributionFunction(atoms, rmax, bins)
            else:
                RDFobj.atoms = atoms  # Fool RDFobj to use the new atoms
            RDFobj.update()           # Collect data
        
        # Total RDF        
        rdf = RDFobj.get_rdf(groups=0)        
        self.data["X-X"] = rdf

        # Partial RDFs               
        elements = [[(x, y) for y in atomic_numbers_unique] for x in atomic_numbers_unique] # cartesian product of very couple of  species

        # change to np.like 
        partial_rdf = [[0 for y in atomic_numbers_unique] for x in atomic_numbers_unique] # need to have same structure as elements but any content is fine as it will be replaced

        for i in range(N_species):
            for j in range(N_species):
                xx = elements[i][j]
                xx_str = ase.data.chemical_symbols[xx[0]] + "-" + ase.data.chemical_symbols[xx[1]]
                partial_rdf[i][j] = RDFobj.get_rdf(elements=xx,groups=0)
                self.data[xx_str] = partial_rdf[i][j]    
        for i in range(N_species):
            xx = elements[i][i]
            xx_str = ase.data.chemical_symbols[xx[0]] + "-" + ase.data.chemical_symbols[xx[1]]
            self.data[ase.data.chemical_symbols[xx[0]] + "-X"] = sum([partial_rdf[i][j] for j in range(N_species)])   

    def write_to_file(self, filename):
        filename = amof.files.path.append_suffix(filename, 'rdf')
        self.data.to_feather(filename)

    def read_rdf_file(self, path_to_data):
        path_to_data = amof.files.path.append_suffix(path_to_data, 'rdf')
        self.data = pd.read_feather(path_to_data)

    def get_coordination_number(self, nn_set, cutoff, density):
        """
        return coordination number
        nn_set: str indicating pair of neighbours
        cutoff: float, in Angstrom
        density: float, no units
        """
        return get_coordination_number(self.data['r'], self.data[nn_set], cutoff, density)



class CoordinationNumber(object):
    """
    Class to compute CoordinationNumber from RDF
    
    Subjected to numerical errors in the integration step
    Best to use amof.cn.CoordinationNumber
    """

    def __init__(self):
        """default constructor"""
        logger.warning('Compute CoordinationNumber from RDF, best to use amof.cn.CoordinationNumber')
        self.data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, nb_set_and_cutoff, delta_Step = 1, first_frame = 0, dr = 0.0001, parallel = False):
        """
        constructor of rdf class from an ase trajectory object
        Args:
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
            dr: float, in Angstrom
        """
        cn_class = cls() # initialize class
        step = amof.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        cn_class.compute_cn(trajectory, nb_set_and_cutoff, step, dr, parallel)
        return cn_class # return class as it is a constructor

    def compute_cn(self, trajectory, nb_set_and_cutoff, step, dr, parallel):
        """
        compute coordination from ase trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        N_species = len(atomic_numbers_unique) # number of different chemical species

        rmax = np.max(list(nb_set_and_cutoff.values()))

        logger.info("Start computing coordination number for %s frames with dr = %s and rmax = %s", len(trajectory), dr, rmax)
        bins = int(rmax // dr)
        r = np.arange(bins) * dr        

        def compute_cn_for_frame(atom, step):
            """
            compute coordination for ase atom object
            """
            # atoms = trajectory[i]
            dic = {'Step': step}
            RDFobj = asap3.analysis.rdf.RadialDistributionFunction(atom, rmax, bins)
            density = amatom.get_number_density(atom)
            for nn_set, cutoff in nb_set_and_cutoff.items():
                xx = tuple(ase.data.atomic_numbers[i] for i in nn_set.split('-'))
                rdf = RDFobj.get_rdf(elements=xx, groups=0)
                dic[nn_set] = get_coordination_number(r, rdf, cutoff, density)            
            return dic

        if parallel == False:
            list_of_dict = [compute_cn_for_frame(trajectory[i], step[i]) for i in range(len(trajectory))]
        else:
            logger.warning("Parallel mode for coordination number very slow, best to use serial")
            num_cores = parallel if type(parallel) == int else max(joblib.cpu_count() // 2 - 2, 2) # heuristic for 40cores Xeon cpus
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(compute_cn_for_frame)(trajectory[i], step[i]) for i in range(len(trajectory)))

        self.data = pd.DataFrame(list_of_dict)

    @classmethod
    def from_file(cls, filename):
        """
        constructor of pore class from msd file
        """
        cn_class = cls() # initialize class
        cn_class.read_cn_file(filename)
        return cn_class # return class as it is a constructor

    def read_cn_file(self, filename):
        """path_to_data: where the cn object is"""
        filename = amof.files.path.append_suffix(filename, 'cn')
        self.data = pd.read_feather(filename)

    def write_to_file(self, filename):
        filename = amof.files.path.append_suffix(filename, 'cn')
        self.data.to_feather(filename)

def get_coordination_number(r, rdf, cutoff, density):
    """
    return coordination number
    x, rdf: arrays of same size
    cutoff: float, in Angstrom
    density: float, number density of the entire system (counting every species) in Angstrom^-3
    """
    mask = (r > 0) & (r < cutoff)
    r = r[mask]
    rdf = rdf[mask]
    integral = scipy.integrate.simps(rdf * (r**2), r)
    return 4 * np.pi * density * integral


class RdfPlotter(object):
    def __init__(self):
        """default constructor"""
        self.multiple_rdf_data = {}
        # self.struct = struct

    def add_rdf(self, path_to_rdf, rdf_name = None):
        if rdf_name is None:
            rdf_name = path_to_rdf
        self.multiple_rdf_data[rdf_name] = Rdf.from_rdf(path_to_rdf).data

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
        for rdf_name, rdf_data in self.multiple_rdf_data.items():
            plt.plot(rdf_data['r'], rdf_data[nn_set], label = rdf_name,  alpha=0.9, linewidth = 1)
        plt.legend()
        plt.xlabel("$r$ (${\AA}$)")
        plt.ylabel("$g(r)$")
        if xlim is not None:
            plt.xlim(xlim[0],xlim[-1])
        plt.title(nn_set)
        if path_to_plot is not None:
            plt.savefig(path_to_plot + ".png", dpi = 300)
        plt.show()    
   

