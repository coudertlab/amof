"""
Module containing msd related methods
"""

import os

# force numpy to use one thread 
os.environ["OMP_NUM_THREADS"] = "1"  # essential
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # at least one of these 4 is needed
os.environ["MKL_NUM_THREADS"] = "1"  
os.environ["NUMEXPR_NUM_THREADS"] = "1"  
os.environ["OPENBLAS_MAIN_FREE"] = "1"

import ase
import ase.data
from ase.geometry.geometry import wrap_positions

import numpy as np
import pandas as pd
import logging
import joblib

import sadi.trajectory as straj
import sadi.atom
import sadi.files.path

logger = logging.getLogger(__name__)

class Msd(object):
    """
    Main class for MSD
    """

    def write_to_file(self, path_to_output):
        """path_to_output: where the MSD object will be written"""
        path_to_output = sadi.files.path.append_suffix(path_to_output, 'msd')
        self.msd_data.to_feather(path_to_output)

    @classmethod
    def from_msd(cls, *args):
        logger.exception('from_msd is deprecated, use from_file instead')

    @classmethod
    def from_file(cls, path_to_msd):
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


class DirectMsd(Msd):
    """
    Direct MSD

    MSD(t) = \dfrac{1}{N_{particles}} \sum_{i=1}^{N_{particles}} (r_i(t) - r_i(0))^2 
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
        step = straj.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
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


class WindowMsd(Msd):
    """
    Window MSD

    MSD(m) = \frac{1}{N_{particles}} \sum_{i=1}^{N_{particles}} \frac{1}{N-m} \sum_{k=0}^{N-m-1} (\vec{r}_i(k+m) - \vec{r}_i(k))^2
    m is the window length size, usually replaced by time t in publications. 
    To be unambiguous, here time will carry units and m and k will be used as trajectory indexes
    The final data array will be y = MSD, x = m labelled 'Time' 
    """

    def __init__(self):
        """default constructor"""
        self.msd_data = pd.DataFrame({"Time": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, delta_time = 100, max_time = "half", timestep = 1,
             delta_Step = 1, first_frame = 0, parallel = False):
        """
        constructor of msd class

        Args:
            trajectory: ase trajectory object
            delta_time: int, time between two computed values of the MSD, in fs
            max_time: int or str
                if half, then the upper limit is half of the simulation size
            timestep: float, timestep between two frames, in fs
            delta_Step: number of simulation steps between two frames
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        msd_class = cls() # initialize class
        step = straj.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        if max_time == "half":
            max_time = (len(trajectory) // 2) * timestep
        delta_m = delta_time // timestep
        window = np.arange(0, max_time // timestep, delta_m)
        time = timestep * window
        msd_class.compute_msd(trajectory, step, window, time, parallel)
        return msd_class # return class as it is a constructor

    @staticmethod
    def compute_msd_of_m(delta_pos, m):
        """calculate MSD(m) for a given m with real pos (stored in r) compared to PBC pos stored
        in ase (extracted with position)

        Args:
            delta_pos: atomic coordinates of successive displacements
            m: int
            if atomic_number is None, compute MSD between all atoms
        """
        # r_k_minus_m = sadi.atom.select_species_positions(trajectory[0], atomic_number)
        MSD_partial = np.zeros(len(delta_pos) - m)
        r_k_minus_m = delta_pos[0]
        # test reducing mem usage
        # r = np.zeros((len(trajectory), len(r_0), 3))
        # r[0] = r_0 
        r_k = r_k_minus_m[:]
        for k in range(1, m+1): 
            r_k += delta_pos[k]
        for k in range(m, len(delta_pos)): # First looping yields 0
            r_k_minus_1 = r_k
            r_k_minus_m_minus_1 = r_k_minus_m
            # def get_dr(k, r_k_minus_1):
            #     """
            #     !! only works for orthogonal cells
            #     """
            #     dr = np.zeros((len(r_k_minus_1), 3))
            #     for j in range(3): #x,y,z
            #         a = trajectory[k].get_cell()[j,j]
            #         dr[:,j] = (sadi.atom.select_species_positions(trajectory[k], atomic_number) - r_k_minus_1%a)[:,j]
            #         for i in range(len(dr)):
            #             if dr[i][j]>a/2:
            #                 dr[i][j] -= a
            #             elif dr[i][j]<-a/2:
            #                 dr[i][j] += a
            #     return dr
            # dr_k = get_dr(k, r_k_minus_1)
            # dr_k_minus_m = get_dr(k - m, r_k_minus_m_minus_1)

            # def get_dr_new(k, r_k_minus_1):
            #     pos = sadi.atom.select_species_positions(trajectory[k], atomic_number) - r_k_minus_1
            #     cell = trajectory[k].get_cell()
            #     new_dr_k = wrap_positions(pos, cell, center=(0., 0., 0.))
            #     return new_dr_k

            #     # if not np.allclose(pos,dr_k):
            #     #     logger.warning("pos diff from get_dr")
            #     # if not np.allclose(new_dr_k, pos):
            #     #     logger.warning("wrapped pos diff from pos")
            #     #     raise ValueError()

            # dr_k = get_dr_new(k, r_k_minus_1)
            # dr_k_minus_m = get_dr_new(k - m, r_k_minus_m_minus_1)

            dr_k = delta_pos[k]
            dr_k_minus_m = delta_pos[k-m]
            # r[t] = dr + r[t-1]
            # MSD[t] = np.linalg.norm(r[t]-r_0)**2/len(r_0)
            r_k = dr_k + r_k_minus_1
            r_k_minus_m = dr_k_minus_m + r_k_minus_m_minus_1
            MSD_partial[k - m] = np.linalg.norm(r_k - r_k_minus_m)**2/len(r_k_minus_m)
        MSD = np.mean(MSD_partial)
        return MSD

    def compute_msd(self, trajectory, step, window, time, parallel):
        """
        Args:
            trajectory: ase trajectory object
            step: np array, simulation steps
            m: np array, window
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        logger.info("Start computing msd at %s times on a trajectory of %s frames", len(window), len(trajectory))
        
        elements = sadi.atom.get_atomic_numbers_unique(trajectory[0])

        positions_by_elt = [] # each partial trajectory consist of the po
        for x in elements:
            x_str = ase.data.chemical_symbols[x]
            positions_by_elt.append([sadi.atom.select_species_positions(atom, x) for atom in trajectory])
        cell = [atom.get_cell() for atom in trajectory]

        self.msd_data = pd.DataFrame({"Time": time}) 

        def compute_for_every_m(positions, cell):
            delta_pos = straj.get_delta_pos(positions, cell)
            return [self.compute_msd_of_m(delta_pos, m) for m in window]
        if not parallel:
            msd_list = [compute_for_every_m(pos, cell) for pos in positions_by_elt]
        else:
            # x_list = [30] # dev, only Zn
            # x_list = [None, 30] # dev
            # x_list = elements # dev

            # mock X by only keeping Zns
            # x_list = [None] # dev
            # trajectory = [ase.Atoms('Zn16', sadi.atom.select_species_positions(atom, 30),
                    # cell=atom.get_cell(), pbc=True) for atom in trajectory]
            # trajectory = [ase.Atoms(
            #         atom.get_atomic_numbers()[atom.get_atomic_numbers()==30],
            #         sadi.atom.select_species_positions(atom, 30),
            #         cell=atom.get_cell(), pbc=True) for atom in trajectory]
            # trajectory = [sadi.atom.select_species_positions(atom, 30) for atom in trajectory]
            num_cores = len(elements) # default value
            if type(parallel) == int and parallel < num_cores:
                num_cores = parallel
            msd_list = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(compute_for_every_m)(trajectory, x) for x in elements)
        # assign partial msd
        for i in range(len(elements)):
            x_str = ase.data.chemical_symbols[elements[i]]
            self.msd_data[x_str] = msd_list[i]
        # compute total msd
        formula_dict = trajectory[0].symbols.formula._count
        self.msd_data['X'] = self.msd_data.groupby(self.msd_data.index).apply(
            lambda x: 
                np.sum([x[k] * v for k, v in formula_dict.items()])
                / sum(formula_dict.values())
                )         
        self.msd_data            
