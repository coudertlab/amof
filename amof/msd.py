"""
Module containing msd related methods
"""

import os

# force numpy to use one thread 
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # seems to suffice

import ase
import ase.data
from ase.geometry.geometry import wrap_positions

import numpy as np
import pandas as pd
import logging
import joblib

import amof.trajectory as amtraj
import amof.atom
import amof.files.path

logger = logging.getLogger(__name__)

class Msd(object):
    """
    Main class for MSD
    """

    def write_to_file(self, path_to_output):
        """path_to_output: where the MSD object will be written"""
        path_to_output = amof.files.path.append_suffix(path_to_output, 'msd')
        self.data.to_feather(path_to_output)

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
        path_to_data = amof.files.path.append_suffix(path_to_data, 'msd')
        self.data = pd.read_feather(path_to_data)


class DirectMsd(Msd):
    """
    Direct MSD

    MSD(t) = \dfrac{1}{N_{particles}} \sum_{i=1}^{N_{particles}} (r_i(t) - r_i(0))^2 

    This precise implementation only works for orthogonal cell.
    Better to use WindowMSD
    """

    def __init__(self):
        """default constructor"""
        self.data = pd.DataFrame({"Step": np.empty([0])})
        logger.warning('DirectMsd is deprecated and not suitable for non-orthogonal cells, use WindowMsd instead')

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
        step = amtraj.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        msd_class.compute_msd(trajectory, step, parallel)
        return msd_class # return class as it is a constructor

    @staticmethod
    def compute_species_msd(trajectory, atomic_number = None):
        """calculate MSD with real pos (stored in r) compared to PBC pos stored
        in ase (extracted with position)
        if atomic_number is None, compute MSD between all atoms
        """
        r_0 = amof.atom.select_species_positions(trajectory[0], atomic_number)
        r_t = r_0
        MSD = np.zeros(len(trajectory))
        for t in range(1, len(trajectory)):
            r_t_minus_dt = r_t
            dr = np.zeros((len(r_0), 3))
            # ! only works for orthogonal cell
            for j in range(3): #x,y,z
                a = trajectory[t].get_cell()[j,j]
                dr[:,j] = (amof.atom.select_species_positions(trajectory[t], atomic_number) - r_t_minus_dt%a)[:,j]
                for i in range(len(dr)):
                    if dr[i][j]>a/2:
                        dr[i][j] -= a
                    elif dr[i][j]<-a/2:
                        dr[i][j] += a
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
        
        elements = amof.atom.get_atomic_numbers_unique(trajectory[0])

        Step = step     
        self.data = pd.DataFrame({"Step": Step})
        if not parallel:
            self.data["X"] = self.compute_species_msd(trajectory)

            for x in elements:
                x_str = ase.data.chemical_symbols[x]
                self.data[x_str] = self.compute_species_msd(trajectory, x)
        else:
            x_list = [None] + elements
            num_cores = min(len(x_list), joblib.cpu_count()) # default value
            if type(parallel) == int and parallel < num_cores:
                num_cores = parallel
            msd_list = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_species_msd)(trajectory, x) for x in x_list)
            self.data["X"] = msd_list[0]
            for i in range(1, len(x_list)):
                x_str = ase.data.chemical_symbols[x_list[i]]
                self.data[x_str] = msd_list[i]               


class WindowMsd(Msd):
    """
    Window MSD

    MSD(m) = \frac{1}{N_{particles}} \sum_{i=1}^{N_{particles}} \frac{1}{N-m} \sum_{k=0}^{N-m-1} (\vec{r}_i(k+m) - \vec{r}_i(k))^2
    m is the window length size, usually replaced by time t in publications. 
    To be unambiguous, here time will carry units and m and k will be used as trajectory indexes
    The final data array will be y = MSD, x = m labelled 'Time' 

    Time is expressed in fs
    """

    def __init__(self):
        """default constructor"""
        self.data = pd.DataFrame({"Time": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, delta_time = 100, max_time = "half", timestep = 1, parallel = False, unwrap = False):
        """
        constructor of msd class

        Args:
            trajectory: ase trajectory object
            delta_time: int, time between two computed values of the MSD, in fs
            max_time: int or str
                if half, then the upper limit is half of the simulation size
                if input max_time is higher than half of simulation size, will choose the latter
            timestep: int, time between two frames of the trajectory
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
            unwrap: Boolean, if True will unwrap the trajectory before computing the MSD. 
                Use if the input XYZ data was output with a code that automatically wrap the positions 
                without keeping the center of mass constant.
        """
        msd_class = cls() # initialize class
        half_time = (len(trajectory) // 2) * timestep
        if max_time == "half" or max_time > half_time:
            max_time = half_time
        if delta_time < timestep:
            logger.exception("Delta_time should be larger than timestep")
        delta_m = delta_time // timestep
        window = np.arange(0, max_time // timestep, delta_m)
        time = timestep * window
        msd_class.compute_msd(trajectory, window, time, parallel, unwrap)
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
        MSD_partial = np.zeros(len(delta_pos) - m)
        r_k_minus_m = delta_pos[0]
        r_k = r_k_minus_m * 0 # empty array with same shape as r_k_minus_m
        for k in range(0, m+1): 
            r_k += delta_pos[k]
        for k in range(m+1, len(delta_pos)): 
            r_k += delta_pos[k]
            r_k_minus_m += delta_pos[k-m]
            MSD_partial[k - m] = np.linalg.norm(r_k - r_k_minus_m)**2/len(r_k_minus_m)
        MSD = np.mean(MSD_partial)
        return MSD

    def compute_msd(self, trajectory, window, time, parallel, unwrap):
        """
        Args:
            trajectory: ase trajectory object
            window: np array, window
            time: np array, time in fs
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
            unwrap: Boolean, if True will unwrap the trajectory before computing the MSD. 
        """     
        elements = amof.atom.get_atomic_numbers_unique(trajectory[0])

        cell = [atom.get_cell() for atom in trajectory]

        # Unwrap before removing the center of mass by reconstructing the pos by summing the delta_pos
        # Assumes that the timestep isn't too large and that the atoms don't move more than typically half the cell
        if unwrap == True:
            logger.info("Unwrap trajectory before computing msd")
            positions = [atom.get_positions() for atom in trajectory]
            delta_pos = amtraj.get_delta_pos(positions, cell)

            new_pos = positions[0]
            for i in range(1, len(trajectory)):
                new_pos += delta_pos[i]
                trajectory[i].set_positions(new_pos)

        logger.info("Start computing msd at %s times on a trajectory of %s frames", len(window), len(trajectory))

        # Compensate MD drift of center of mass
        for atom in trajectory:
            cg = atom.get_center_of_mass()
            atom.translate(-cg)

        positions_by_elt = [] # each partial trajectory consist of the po
        for x in elements:
            x_str = ase.data.chemical_symbols[x]
            positions_by_elt.append([amof.atom.select_species_positions(atom, x) for atom in trajectory])

        self.data = pd.DataFrame({"Time": time}) 

        def compute_for_every_m(positions, cell):
            delta_pos = amtraj.get_delta_pos(positions, cell)
            return [self.compute_msd_of_m(delta_pos, m) for m in window]

        if not parallel:
            msd_list = [compute_for_every_m(pos, cell) for pos in positions_by_elt]
        else:
            num_cores = len(elements) # default value
            if type(parallel) == int and parallel < num_cores:
                num_cores = parallel
            msd_list = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(compute_for_every_m)(pos, cell) for pos in positions_by_elt)

        # assign partial msd
        for i in range(len(elements)):
            x_str = ase.data.chemical_symbols[elements[i]]
            self.data[x_str] = msd_list[i]
        # compute total msd
        formula_dict = trajectory[0].symbols.formula._count
        self.data['X'] = self.data.groupby(self.data.index).apply(
            lambda x: 
                np.sum([x[k] * v for k, v in formula_dict.items()])
                / sum(formula_dict.values())
                )         
