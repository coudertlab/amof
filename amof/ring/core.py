"""
Module containing ring statistics analysis related methods

Terminology: 
    singular form 'ring' is employed by default to talk about ring statistics analysis
    plural form 'rings' or 'RINGS' refers to the Rigorous Investigation of Networks Generated using Simulations or ‘RINGS’ code
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
import subprocess
import shlex
import xarray as xr

import amof.trajectory
import amof.atom as amatom
import amof.files.path as ampath
import amof.pore.pysimmzeopp

logger = logging.getLogger(__name__)

class Ring(object):
    """
    Main class for ring statistics analysis analysis

    Search for primitive rings as defined by
        Le Roux, S., & Jund, P. (2010). Ring statistics analysis of topological networks: 
        New approach and application to amorphous GeS2 and SiO2 systems. 
        Computational Materials Science, 49(1), 70–83. 
        https://doi.org/10.1016/j.commatsci.2010.04.023 

    A definition phrased by Franzblau of such rings (named SP ring in this paper) is:
        Given a graph G and a ring R in 6, R is a shortest path ring (SP ring) 
        if R contains a shortest path for each pair of vertices in the ring. 
        That is, distG(y, z ) =dist+ (y, z) for each pair y, z in the ring

        Franzblau, D. S. (1991). Computation of ring statistics for network models of solids. 
        Physical Review B, 44(10), 4925–4930. https://doi.org/10.1103/PhysRevB.44.4925
    """

    def __init__(self, max_search_depth = None, discard_if_potentially_undiscovered_rings = False):
        """default constructor"""
        self.data = xr.DataArray(np.empty([0,0,0]), 
            coords = [('Step', np.empty([0], dtype='int64')), 
                ('ring_size', np.empty([0], dtype='int64')), 
                ('ring_var', np.empty([0], dtype='str_'))], # numpy str is str_
            name = 'ring')
        self.max_search_depth = max_search_depth
        self.discard_if_potentially_undiscovered_rings = discard_if_potentially_undiscovered_rings
        self.report_search = pd.DataFrame({"Step": np.empty([0])})

    @classmethod
    def from_trajectory(cls, trajectory, nb_set_and_cutoff, max_search_depth = 32 , delta_Step = 1, first_frame = 0, parallel = False):
        """
        Constructor of ring class from an ase trajectory object

        Args:
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
            max_search_depth: int, maximum search depth for rings.
        """
        ring_class = cls(max_search_depth = max_search_depth) # initialize class
        nb_set_and_cutoff_list = [nb_set_and_cutoff for i in range(len(trajectory))]
        step = amof.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))
        ring_class.compute_ring(trajectory, nb_set_and_cutoff_list, step, parallel)
        return ring_class # return class as it is a constructor

    @classmethod
    def from_reduced_trajectory(cls, reduced_trajectory, max_search_depth = 32, 
            discard_if_potentially_undiscovered_rings = False, parallel = False):
        """
        Constructor of ring class from a amof ReducedTrajectory object
        
        Args:
            reduced_trajectory: amof ReducedTrajectory object
            nb_set_and_cutoff: dict, keys are str indicating pair of neighbours, 
                values are cutoffs float, in Angstrom
        """
        ring_class = cls(max_search_depth = max_search_depth,
            discard_if_potentially_undiscovered_rings = discard_if_potentially_undiscovered_rings) # initialize class with empty data
        criteria_to_compute_ring = ['connectivity_constructible_with_cutoffs'] # can be expanded in further dev
        criteria_enlarged = ['in_reduced_trajectory'] + criteria_to_compute_ring
        rs = reduced_trajectory.report_search
        rs_traj = rs[rs['in_reduced_trajectory']==True]
        if (len(rs_traj) != 0 and
            np.min([criterium in rs_traj.columns for criterium in criteria_to_compute_ring]) == True):
            compute_ring = rs[criteria_enlarged].all(axis='columns')
            if np.sum(compute_ring) != 0:
                subset_reduced_traj = rs_traj[criteria_to_compute_ring].all(axis='columns')
                nb_set_and_cutoff_list = [eval(i) for i in rs[compute_ring]['nb_set_and_cutoff']]
                step = np.array(rs[compute_ring].index)
                traj = list(itertools.compress(reduced_trajectory.trajectory, subset_reduced_traj))
                ring_class.compute_ring(traj, nb_set_and_cutoff_list, step, parallel)
                return ring_class

        logger.info('No valid frame in reduced trajectory')
        return ring_class # return empty class 

    def compute_ring(self, trajectory, nb_set_and_cutoff_list, step, parallel):
        """
        Args:
            trajectory: ase trajectory object
                nb_set_and_cutoff_list: list of dict, one per frame in trajectory
            step: np array, simulation steps
            parallel: Boolean or int (number of cores to use): whether to parallelize the computation
        """
        logger.info("Start ring analysis for %s frames", len(trajectory))
        result_list = []

        if parallel == False:
            for i in range(len(trajectory)):
                logger.debug('compute frame # %s out of %s', i + 1, len(trajectory))
                result_list.append(self.compute_ring_for_atom(trajectory[i], step[i], nb_set_and_cutoff_list[i]))
        else:
            if type(parallel) == int:
                num_cores = parallel
            else:
                num_cores = max(joblib.cpu_count() // 2 - 2, 2) # heuristic for 40cores Xeon cpus: less than 20 and nice value for 50 steps
            result_list = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(self.compute_ring_for_atom)(
                        trajectory[i], step[i], nb_set_and_cutoff_list[i]) for i in range(len(trajectory)))

        list_report_search = []
        list_of_xarray = []
        for ring_ar, report_search in result_list:
            list_report_search.append(report_search)
            if ring_ar is not None:
                list_of_xarray.append(ring_ar)

        self.report_search = pd.DataFrame(list_report_search).set_index('Step')

        if list_of_xarray != []:
            dic_of_xarray = dict(zip(step, list_of_xarray))
            xa = xr.Dataset(dic_of_xarray)
            xa = xa.to_array("Step", "Step")
            xa = xr.Dataset({'ring': xa}) # one large data_array containing thermo variables
            # xa = xa.to_dataset(dim='thermo') # separate thermo variables
            xa = xa.fillna(0) #  for every ring size found in the entire traj, add 0 at every step if none detected while other ring sizes are
            self.data = xa

    def read_rings_output(self, rstat_path):
        """
        read RINGS output from RINGS-res-5.dat, corresponding to primitive rings 
        read RINGS-res-3.dat, corresponding to King's shortest path rings, to know whether some rings are missing

        Args:
            rstat_path: pathlib path leading to rstat, containing evol-RINGS files

        Returns:
            ar: xarray dataarray containing the rings info
            potentially_undiscovered_rings: float, number potentially_undiscovered_rings 
                    specified in RINGS output files. If 0, means that all are discovered.
        """
        filename = 'RINGS-res-3.dat' # King's shortest path rings
        with open(rstat_path / filename) as f:
            first_line = f.readline().strip('\n')
        searchObj = re.search(r'# Number of rings with n >  (.*) nodes which potentialy exist: (.*)', first_line, re.M|re.I)
        potentially_undiscovered_rings = round(float(searchObj.group(2)))
        
        filename = 'RINGS-res-5.dat' # primitive rings 
        header = 1
        df = pd.read_csv(rstat_path / filename, header = header, escapechar='#', sep='\s+')
        df = df.set_index(' n')
        ar = xr.DataArray(df, dims=("ring_size", "ring_var"))
        return ar, potentially_undiscovered_rings

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

    def write_input_files(self, atom, cutoff_dict, search_depth, path):
        """
        Write RINGS input and option files in path

        Args:
            atom: ase atom object
            cutoff_dict: 
            search_depth: int, rings_maximum_search_depth
            path: pathlib path
        """
        parameters = {}
        parameters['number_of_atoms'] = atom.get_global_number_of_atoms()
        atomic_numbers_unique = list(set(atom.get_atomic_numbers()))
        elements_present_unique =  [ase.data.chemical_symbols[i] for i in atomic_numbers_unique]
        parameters['number_of_chemical_species'] = len(elements_present_unique)
        parameters['list_of_chemical_species'] = ' '.join(elements_present_unique)
        parameters['rings_maximum_search_depth_divided_by_two'] = search_depth // 2 
        for i in range(3):
            parameters[f"cell{'abc'[i]}"] = str(atom.cell[i])[1:-1]
        parameters['cutoff_lines'] = ''
        for key, value in cutoff_dict.items():
            line_list = [ase.data.chemical_symbols[i] for i in key] + [str(value), """# \n """]
            parameters['cutoff_lines'] += ' '.join(line_list)
        parameters['Grtot'] = max(cutoff_dict.values())
        self.fill_template('input.inp', parameters, path)
        self.fill_template('options', parameters, path)

    def compute_ring_for_atom(self, atom, step, nb_set_and_cutoff):
        """
        Args:
            atom: ase atom object
            
        Returns:
            ring_ar: xarray dataarray containing the rings info
            report_search: dic
        """
        report_search = {'Step': step, 
            'Discarded frame': False, 
            'max_search_depth': self.max_search_depth,
            'Discard if potentially undiscovered rings': self.discard_if_potentially_undiscovered_rings,
            'Rings statistics computed with potentially undiscovered rings': False}
        cutoff_dict = amatom.format_cutoff(nb_set_and_cutoff, sort_pair = True)
        # add 0 where cutoff is not defined
        atomic_numbers_unique = list(set(atom.get_atomic_numbers()))
        for pair in itertools.combinations_with_replacement(atomic_numbers_unique, 2):
            if pair not in cutoff_dict.keys():
                cutoff_dict[pair] = 0
        # enlarge cutoff dict with 0 for every other values

        with tempfile.TemporaryDirectory() as tmpdirname:
            tmpdirpath = pathlib.Path(tmpdirname)
            (tmpdirpath / 'data').mkdir(parents=True, exist_ok=True)
            atom.write(tmpdirname + '/data/atom.xyz')
            
            # warning, make sure no user input is possible inside arg. 
            # Here shlex.quote is used as precaution in the only possible variable of the string
            arg = f"cd {shlex.quote(tmpdirname)} && rings input.inp"

            search_depth = min(16, self.max_search_depth)

            ring_ar = None
            potentially_undiscovered_rings = np.inf
            while search_depth <= self.max_search_depth and potentially_undiscovered_rings > 0:
                self.write_input_files(atom, cutoff_dict, search_depth, tmpdirpath)
                
                p = subprocess.Popen(arg, shell=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE) 
                p.wait()

                ring_ar, potentially_undiscovered_rings = self.read_rings_output(tmpdirpath / 'rstat')
                report_search['Final search_depth'] = search_depth
                report_search['Potentially undiscovered rings'] = potentially_undiscovered_rings

                search_depth += 4
            if potentially_undiscovered_rings > 0:
                logger.warning('Rings with n >  %s nodes potentialy exist', self.max_search_depth)
                report_search['Rings statistics computed with potentially undiscovered rings'] = True
                if self.discard_if_potentially_undiscovered_rings == True:
                    report_search['Discarded frame'] = True
                    ring_ar = None # don't add this frame to rings file
        return ring_ar, report_search

    def write_to_file(self, filename):
        """path_to_output: where the ring object will be written"""
        self.data.to_netcdf(ampath.append_suffix(filename, 'ring'))
        self.report_search.to_csv(ampath.append_suffix(filename, 'report_search.csv'))


    @classmethod
    def from_file(cls, filename):
        """
        constructor of ring class from ring file
        """
        ring_class = cls() # initialize class
        ring_class.read_ring_file(filename)
        return ring_class # return class as it is a constructor

    def read_ring_file(self, filename):
        """path_to_data: where the ring object is"""
        filename = ampath.append_suffix(filename, 'ring')
        self.data = xr.open_dataset(filename)
