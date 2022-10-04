"""
Module containing pore analysis related methods
"""

from subprocess import TimeoutExpired
import ase
import ase.data

import numpy as np
import pandas as pd
import logging
import tempfile
import re
import joblib

import amof.trajectory
import amof.atom
import amof.files.path
import amof.pore.pysimmzeopp

logger = logging.getLogger(__name__)

class Pore(object):
    """
    Main class for pore analysis
    """

    def __init__(self):
        """default constructor"""
        self.data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, delta_Step = 1, first_frame = 0, parallel = False):
        """
        constructor of pore class from an ase trajectory object

        """
        pore_class = cls() # initialize class
        step = amof.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        pore_class.compute_surface_volume(trajectory, step, parallel)
        return pore_class # return class as it is a constructor

    def compute_surface_volume(self, trajectory, step, parallel):
        """
        Args:
            trajectory: ase trajectory object
            step: np array, simulation steps
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        logger.info("Start pore analysis for volume and surfaces for %s frames", len(trajectory))
        list_of_dict = []
        if parallel == False:
            for i in range(len(trajectory)):
                logger.debug('compute frame # %s out of %s', i + 1, len(trajectory))
                list_of_dict.append(self.get_surface_volume(trajectory[i], step[i]))
        else:
            if type(parallel) == int:
                num_cores = parallel
            else:
                num_cores = max(joblib.cpu_count() // 2 - 2, 2) # heuristic for 40cores Xeon cpus: less than 20 and nice value for 50 steps
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.get_surface_volume)(trajectory[i], step[i]) for i in range(len(trajectory)))

        list_of_dict = [dic for dic in list_of_dict if dic is not None] # filter pore volume that are not properly computed

        if list_of_dict != []:
            df = pd.DataFrame(list_of_dict)
            self.data = df 
        # else keep empty surface_volume from init

    @staticmethod
    def read_zeopp(filename):
        """
        read surface (sa) and volume (vol) zeopp output
        """
        with open(filename) as f:
            first_line = f.readline().strip('\n')
        split_line = re.split('\ +', first_line)
        split_line = split_line[6:] # remove file name, density and unit cell volume
        list_of_keys = [s.strip(':') for s in split_line[::2]]
        list_of_values = [float(s) for s in split_line[1::2]]
        dic = dict(zip(list_of_keys, list_of_values))
        return dic

    def get_surface_volume(self, atom, step):
        """
        Args:
            atom: ase atom object
            step: int, represent step of frame
        Returns:
            dic: dictionary with output from zeopp vol and sa
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            atom.write(tmpdirname + '/atom.cif')
            try:
                amof.pore.pysimmzeopp.network(tmpdirname + '/atom.cif', sa = True, vol = True)
                sa = self.read_zeopp(tmpdirname + '/atom.sa')
                vol = self.read_zeopp(tmpdirname + '/atom.vol')
                dic = {'Step': step, **sa, **vol}
            except TimeoutExpired:
                logger.warning('Timout expired for ZEOpp. System size: %s; Step: %s', atom.get_global_number_of_atoms(), step)
                dic = None
        return dic

    def write_to_file(self, filename):
        """path_to_output: where the pore object will be written"""
        filename = amof.files.path.append_suffix(filename, 'pore')
        self.data.to_feather(filename)

    @classmethod
    def from_file(cls, filename):
        """
        constructor of pore class from msd file
        """
        pore_class = cls() # initialize class
        pore_class.read_surface_volume_file(filename)
        return pore_class # return class as it is a constructor

    def read_surface_volume_file(self, filename):
        """path_to_data: where the MSD object is"""
        filename = amof.files.path.append_suffix(filename, 'pore')
        self.data = pd.read_feather(filename)
