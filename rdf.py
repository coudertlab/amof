"""
Module containing rdf related methods
"""

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

import sadi.trajectory
import sadi.files.path
import sadi.atom as satom

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class Rdf(object):
    """
    Main class for rdf
    """

    def __init__(self):
        """default constructor"""
        self.rdf_data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, dr = 0.01, rmax = 'half_cell'):
        """
        constructor of rdf class from an ase trajectory object
        dr, rmax in Angstrom
        If rmax is set to 'half_cell', then half of the minimum dimension of the cell is used to ensure no atom is taken into account twice for a given atom (computation is possible beyound this threeshold up to the min cell size)
        """
        rdf_class = cls() # initialize class
        rdf_class.compute_rdf(trajectory, dr, rmax)
        return rdf_class # return class as it is a constructor

    @classmethod
    def from_rdf(cls, path_to_rdf):
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

        # default option
        if  rmax == 'half_cell':
            rmax = np.min([a for t in trajectory for a in t.get_cell_lengths_and_angles()[0:3]]) / 2

        logger.info("Start computing rdf for %s frames with dr = %s and rmax = %s", len(trajectory), dr, rmax)
        bins = int(rmax // dr)
        r = np.arange(bins) * dr        
        self.rdf_data = pd.DataFrame({"r": r})

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
        self.rdf_data["X-X"] = rdf

        # Partial RDFs               
        elements = [[(x, y) for y in atomic_numbers_unique] for x in atomic_numbers_unique] # cartesian product of very couple of  species

        # change to np.like 
        partial_rdf = [[0 for y in atomic_numbers_unique] for x in atomic_numbers_unique] # need to have same structure as elements but any content is fine as it will be replaced

        for i in range(N_species):
            for j in range(N_species):
                xx = elements[i][j]
                xx_str = ase.data.chemical_symbols[xx[0]] + "-" + ase.data.chemical_symbols[xx[1]]
                partial_rdf[i][j] = RDFobj.get_rdf(elements=xx,groups=0)
                self.rdf_data[xx_str] = partial_rdf[i][j]    
        for i in range(N_species):
            xx = elements[i][i]
            xx_str = ase.data.chemical_symbols[xx[0]] + "-" + ase.data.chemical_symbols[xx[1]]
            self.rdf_data[ase.data.chemical_symbols[xx[0]] + "-X"] = sum([partial_rdf[i][j] for j in range(N_species)])   

    def write_to_file(self, filename):
        filename = sadi.files.path.append_suffix(filename, 'rdf')
        self.rdf_data.to_feather(filename)

    def read_rdf_file(self, path_to_data):
        path_to_data = sadi.files.path.append_suffix(path_to_data, 'rdf')
        self.rdf_data = pd.read_feather(path_to_data)

    def get_coordination_number(self, nn_set, cutoff, density):
        """
        return coordination number
        nn_set: str indicating pair of neighbours
        cutoff: float, in Angstrom
        density: float, no units
        """
        return get_coordination_number(self.rdf_data['r'], self.rdf_data[nn_set], cutoff, density)



class CoordinationNumber(object):
    """
    Main class to compute CoordinationNumber
    """

    def __init__(self):
        """default constructor"""
        self.cn_data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, nn_set_and_cutoff, delta_Step = 1, dr = 0.01, parallel = False):
        """
        constructor of rdf class from an ase trajectory object
        Args:
            nn_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
            dr: float, in Angstrom
        """
        cn_class = cls() # initialize class
        cn_class.compute_cn(trajectory, nn_set_and_cutoff, delta_Step, dr, parallel)
        return cn_class # return class as it is a constructor

    def compute_cn(self, trajectory, nn_set_and_cutoff, delta_Step, dr, parallel):
        """
        compute coordination from ase trajectory object
        """
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        N_species = len(atomic_numbers_unique) # number of different chemical species

        rmax = np.max(list(nn_set_and_cutoff.values()))

        logger.info("Start computing rdf for %s frames with dr = %s and rmax = %s", len(trajectory), dr, rmax)
        bins = int(rmax // dr)
        r = np.arange(bins) * dr        

        def compute_cn_for_frame(i):
            """
            compute coordination for ase atom object
            """
            atoms = trajectory[i]
            dic = {'Step': i * delta_Step}
            RDFobj = asap3.analysis.rdf.RadialDistributionFunction(atoms, rmax, bins)
            density = satom.get_number_density(atoms)
            for nn_set, cutoff in nn_set_and_cutoff.items():
                xx = tuple(ase.data.atomic_numbers[i] for i in nn_set.split('-'))
                rdf = RDFobj.get_rdf(elements=xx, groups=0)
                dic[nn_set] = get_coordination_number(r, rdf, cutoff, density)
            return dic

        if parallel == False:
            list_of_dict = [compute_cn_for_frame(i) for i in range(len(trajectory))]
        else:
            num_cores = parallel if type(parallel) == int else 18
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(compute_cn_for_frame)(i) for i in range(len(trajectory)))

        self.cn_data = pd.DataFrame(list_of_dict)

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
        filename = sadi.files.path.append_suffix(filename, 'cn')
        self.cn_data = pd.read_feather(filename)

    def write_to_file(self, filename):
        filename = sadi.files.path.append_suffix(filename, 'cn')
        self.cn_data.to_feather(filename)

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
        self.multiple_rdf_data[rdf_name] = Rdf.from_rdf(path_to_rdf).rdf_data

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
   


#region deprecated code
    # # TBC
    # def cutoff_from_rdf(x, rdf, a, b): 
    #     """ return x such that rdf(x) global minimun between a and b of rdf"""
    #     return x[np.argmin(rdf[(x>a)*(x<b)])+np.size(rdf[x<=a])]

    # def plot(self):
    #     plt.plot(r, rdf)
    #     plt.xlabel("$r$ (${\AA}$)")
    #     plt.ylabel("$g(r)$")
        
    #     cutoff = cutoff_from_rdf(x, rdf, 3, 3.8)
    #     plt.axvline(cutoff, linestyle = '--', linewidth = 1)
    #     plt.text(cutoff + 0.1, 0, "%.2f" % round(cutoff,2))
    #     plt.savefig(output_path + ".rdf_X-X" + ".png")
    #     plt.show()

    # def average_rdfs(traj_names, output_name):
    #     """not changed, may be deprecated when data struct turned to df"""
    #     rdf_prefixes = ["X-X", "Ag-X", "Sb-X", "Te-X", "Ag-Ag", "Sb-Sb", "Te-Te", "Ag-Sb", "Sb-Ag", "Te-Ag", "Ag-Te", "Sb-Te", "Te-Sb"]
    #     prefixes = [".x"]+[".rdf_" + x for x in rdf_prefixes]
    #     input_path = "../analysis/data/rdf/"
    #     output_path = input_path
        
    #     for p in prefixes:
    #         rdf_l = [] 
    #         for traj_name in traj_names:
    #             rdf_l.append(np.load(input_path + traj_name + p + ".npy"))
            
    #         rdf_l=np.array(rdf_l)
    #         rdf = np.average(rdf_l, axis=0)
    #         np.save(output_path +  output_name + p, rdf)   
    # #average_rdfs(["3.c1.pbe","3.c2.pbe","3.c3.pbe"], "3.c.pbe")
#endregion

