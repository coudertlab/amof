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
import tempfile
import re

import sadi.trajectory
import sadi.atom
import sadi.files.path
import sadi.pore.pysimmzeopp

logger = logging.getLogger(__name__)

class Pore(object):
    """
    Main class for pore analysis
    """

    def __init__(self):
        """default constructor"""
        self.surface_volume = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, delta_Step = 1):
        """
        constructor of msd class from an ase trajectory object

        """
        pore_class = cls() # initialize class
        pore_class.compute_surface_volume(trajectory, delta_Step)
        return pore_class # return class as it is a constructor

    def compute_surface_volume(self, trajectory, delta_Step):
        """
        Args:
            trajectory: ase trajectory object
            delta_Step: number of simulation steps between two frames
        """
        logger.info("Start pore analysis for volume and surfaces for %s frames with delta_Step = %s", len(trajectory), delta_Step)
        list_of_dict = []
        for i in range(len(trajectory)):
            logger.debug('compute frame # %s out of %s', i + 1, len(trajectory))
            dic = {'Step': i * delta_Step, **self.get_surface_volume(trajectory[i])}
            list_of_dict.append(dic)
        
        # from joblib import Parallel, delayed
        # list_of_dict = Parallel(n_jobs=2)(delayed(self.get_surface_volume)(trajectory[i]) for i in range(len(trajectory)))

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

    def get_surface_volume(self, atom):
        """
        Args:
            atom: ase atom object
        Returns:
            dic: dictionary with output from zeopp vol and sa
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            atom.write(tmpdirname + 'atom.cif')
            sadi.pore.pysimmzeopp.network(tmpdirname + 'atom.cif', res = False, chan = False, psd = False)
            sa = self.read_zeopp(tmpdirname + 'atom.sa')
            vol = self.read_zeopp(tmpdirname + 'atom.vol')
        return {**sa, **vol}

    def write_to_file(self, filename):
        """path_to_output: where the pore object will be written"""
        filename = sadi.files.path.append_suffix(filename, 'pore')
        self.surface_volume.to_feather(filename)

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
        filename = sadi.files.path.append_suffix(filename, 'pore')
        self.surface_volume = pd.read_feather(filename)
