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
import atomman
import itertools

import sadi.trajectory
import sadi.atom as satom
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
    def from_trajectory(cls, trajectory, nb_set_and_cutoff, delta_Step = 1, first_frame = 0, parallel = False):
        """
        constructor of rings class from an ase trajectory object
        Args:
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
        """
        rings_class = cls() # initialize class
        step = sadi.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        rings_class.compute_rings(trajectory, nb_set_and_cutoff, step, parallel)
        return rings_class # return class as it is a constructor

    def compute_rings(self, trajectory, nb_set_and_cutoff, step, parallel):
        """
        Args:
            trajectory: ase trajectory object
            step: np array, simulation steps
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        logger.info("Start rings analysis for volume and surfaces for %s frames", len(trajectory))
        list_of_dict = []

        cutoff_dict = satom.format_cutoff(nb_set_and_cutoff)
        # add 0 where cutoff is not defined
        atomic_numbers_unique = list(set(trajectory[0].get_atomic_numbers()))
        for pair in itertools.combinations_with_replacement(atomic_numbers_unique, 2):
            if pair not in cutoff_dict.keys():
                cutoff_dict[pair] = 0

        # enlarge cutoff dict with 0 for every other values
        if parallel == False:
            for i in range(len(trajectory)):
                logger.debug('compute frame # %s out of %s', i + 1, len(trajectory))
                list_of_dict.append(self.compute_rings_for_frame(trajectory[i], step[i], cutoff_dict))
        else:
            if type(parallel) == int:
                num_cores = parallel
            else:
                num_cores = 18 # less than 20 and nice value for 50 steps
            list_of_dict = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_rings_for_frame)(trajectory[i], step[i], cutoff_dict) for i in range(len(trajectory)))

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

    @staticmethod
    def fill_template(template_name, parameters, path):
        """
        Read, fill and write template

        Args:
            template_name: name of template file used in .ini file
            parameters: dic containing all the necessary input for templating
            path: pathlib path, where to write filled template under the name template_name
        """
        with open(pathlib.Path(__file__).parent.resolve() / 'template' / template_name) as f:
            template = f.read()
        script = atomman.tools.filltemplate(template, parameters, '{', '}')  
        with open(path / template_name, 'w') as f:
            f.write(script)  

    def write_input_files(self, atom, cutoff_dict, path):
        """
        Write Rings input and option files in path

        Args:
            atom: ase atom object
            cutoff_dict: 
            path: pathlib path
        """
        parameters = {}
        parameters['number_of_atoms'] = atom.get_number_of_atoms()
        atomic_numbers_unique = list(set(atom.get_atomic_numbers()))
        elements_present_unique =  [ase.data.chemical_symbols[i] for i in atomic_numbers_unique]
        parameters['number_of_chemical_species'] = len(elements_present_unique)
        parameters['list_of_chemical_species'] = ' '.join(elements_present_unique)
        parameters['rings_maximum_search_depth_divided_by_two'] = 5 
        for i in range(3):
            parameters[f"cell{'abc'[i]}"] = str(atom.cell[i])[1:-1]
        parameters['cutoff_lines'] = ''
        for key, value in cutoff_dict.items():
            line_list = [ase.data.chemical_symbols[i] for i in key] + [str(value), """# \n """]
            parameters['cutoff_lines'] += ' '.join(line_list)
        parameters['Grtot'] = max(cutoff_dict.values())
        self.fill_template('input.inp', parameters, path)
        self.fill_template('options', parameters, path)
        a = 1

    def compute_rings_for_frame(self, atom, step, cutoff_dict):
        """
        Args:
            atom: ase atom object
            step: int, represent step of frame
        Returns:
            dic: dictionary with output from zeopp vol and sa
        """
        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirpath = pathlib.Path(tmpdirname)
            (tmpdirpath / 'data').mkdir(parents=True, exist_ok=True)
            atom.write(tmpdirname + '/data/atom.xyz')
            self.write_input_files(atom, cutoff_dict, tmpdirpath)
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
