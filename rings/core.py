"""
Module containing rings analysis related methods
"""

import ase
import ase.data

import numpy as np
import pandas as pd
import logging
import os
import pathlib
import tempfile
import re
import joblib

import sadi.trajectory
import sadi.atom
import sadi.files.path
import sadi.pore.pysimmzeopp

logger = logging.getLogger(__name__)

class Rings(object):
    """
    Main class for rings analysis
    """

    def __init__(self):
        """default constructor"""
        self.rings_data = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, delta_Step = 1, first_frame = 0, parallel = False):
        """
        constructor of rings class from an ase trajectory object

        """
        rings_class = cls() # initialize class
        step = sadi.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        rings_class.compute_rings(trajectory, step, parallel)
        return rings_class # return class as it is a constructor

    def compute_rings(self, trajectory, step, parallel):
        """
        Args:
            trajectory: ase trajectory object
            step: np array, simulation steps
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        logger.info("Start rings analysis for volume and surfaces for %s frames", len(trajectory))
        list_of_dict = []
        if parallel == False:
            for i in range(len(trajectory)):
                logger.debug('compute frame # %s out of %s', i + 1, len(trajectory))
                list_of_dict.append(self.compute_rings_for_frame(trajectory[i], step[i]))
        else:
            if type(parallel) == int:
                num_cores = parallel
            else:
                num_cores = 18 # less than 20 and nice value for 50 steps
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_rings_for_frame)(trajectory[i], step[i]) for i in range(len(trajectory)))

        df = pd.DataFrame(list_of_dict)
        self.surface_volume = df

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

    def compute_rings_for_frame(self, atom, step):
        """
        Args:
            atom: ase atom object
            step: int, represent step of frame
        Returns:
            dic: dictionary with output from zeopp vol and sa
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            atom.write(tmpdirname + 'atom.cif')
            sadi.pore.pysimmzeopp.network(tmpdirname + 'atom.cif', sa = True, vol = True)
            sa = self.read_zeopp(tmpdirname + 'atom.sa')
            vol = self.read_zeopp(tmpdirname + 'atom.vol')
        return {'Step': step, **sa, **vol}

    def write_to_file(self, filename):
        """path_to_output: where the rings object will be written"""
        filename = sadi.files.path.append_suffix(filename, 'rings')
        self.surface_volume.to_feather(filename)

    @classmethod
    def from_file(cls, filename):
        """
        constructor of rings class from msd file
        """
        rings_class = cls() # initialize class
        rings_class.read_surface_volume_file(filename)
        return rings_class # return class as it is a constructor

    def read_rings_file(self, filename):
        """path_to_data: where the rings object is"""
        filename = sadi.files.path.append_suffix(filename, 'rings')
        self.surface_volume = pd.read_feather(filename)
