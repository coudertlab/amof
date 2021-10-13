"""
Module containing functions to reduce a trajectory to its fragments
"""

import pandas as pd
import logging
import joblib

from pymatgen.io.ase import AseAtomsAdaptor
from sadi.coordination.core import SearchError

import sadi.symbols
# import sadi.files.path as spath
import sadi.coordination.zif
import sadi.trajectory

logger = logging.getLogger(__name__)

def reduce_trajectory(trajectory, mof, filename = None, delta_Step = 1, first_frame = 0, parallel = False):
    """
    Conveniant wrapper to reduce trajectory by specifying a specific mof
    For now works for 'ZIF-4', 'ZIF-8', 'ZIF-zni' and 'SALEM-2'

    Args:
        trajectory: ase trajectory object
        mof: str specifying the mof
        filename: str, where to write output files, specify None to avoid writing
        delta_Step: number of simulation steps between two frames
    """
    if mof in ['ZIF-4', 'ZIF-zni', 'SALEM-2']:
        structure_reducer = lambda struct: sadi.coordination.zif.MetalIm(struct, "Zn", dist_margin=1.2, dist_margin_metal=1.6) # 1.6 means 3.088 for Zn-N (one case of 2.92 was observed)
        symbols = sadi.symbols.DummySymbols(['Zn', 'Im']) 
    elif mof in  ['ZIF-8']:
        structure_reducer = lambda struct: sadi.coordination.zif.MetalmIm(struct, "Zn", dist_margin=1.2)
        symbols = sadi.symbols.DummySymbols(['Zn', 'mIm']) 
    else:
        logger.exception('Structure search not available for the mof %s', mof)
    return reduce_trajectory_core(trajectory, structure_reducer, symbols, filename, delta_Step = delta_Step, first_frame = first_frame, parallel = parallel)

def reduce_trajectory_core(trajectory, structure_reducer, symbols, filename = None, delta_Step = 1, first_frame = 0, parallel = False):
        """
        Args:
            trajectory: ase trajectory object
            structure_reducer: a function taking a pymatgen structure as input and returning a CoordinationSearch object
            symbols: a DummySymbol object containing the representation to use for all frames 
                (no option atm to include additionnal symbols on specific frames,
                 parallelisation issues + not adapted to analysis tools to have variable number of atoms)
            delta_Step: number of simulation steps between two frames
            filename: str, where to write output files, specify None to avoid writing
        
        Return:
            reduced_traj: sadi ReducedTrajectory object
        """
        logger.info("Start reducing trajectory for %s frames", len(trajectory))

        step = sadi.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))

        def per_atom(atom, step):
            report_search = {'Step': step}
            try:
                reduced_atom, report_search_atom = reduce_atom(atom, structure_reducer, symbols, filename = None)
                report_search['in_reduced_trajectory'] = reduced_atom is not None
                report_search = {**report_search, **report_search_atom}
                # logger.info(str(report_search))
            except SearchError as e:
                logger.debug('Failed to do reduce frame with error message: ' + e.message)        
                report_search['in_reduced_trajectory'] = False       
                report_search = {**report_search, **e.report_search}
                report_search['Error_message'] = e.message
                reduced_atom = None
            except BaseException as e: # unexpected exception
                logger.debug('Failed to do reduce frame with error message: ' + str(e))        
                report_search['in_reduced_trajectory'] = False       
                report_search['Error_message'] = str(e)
                reduced_atom = None
            return reduced_atom, report_search

        if parallel == False:
            result_list = [per_atom(trajectory[i], step[i]) for i in range(len(trajectory))]
        else:
            num_cores = parallel if type(parallel) == int else 18
            result_list = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(per_atom)(trajectory[i], step[i]) for i in range(len(trajectory)))

        list_report_search = []
        reduced_traj = []
        for reduced_atom, report_search in result_list:
            list_report_search.append(report_search)
            if report_search['in_reduced_trajectory'] == True:
                reduced_traj.append(reduced_atom)
        
        df_report_search = pd.DataFrame(list_report_search).set_index('Step')

        reduced_trajectory = sadi.trajectory.ReducedTrajectory(reduced_traj, df_report_search, symbols)
        if filename is not None:
            reduced_trajectory.write_to_file(filename)
        return reduced_trajectory

def reduce_atom(atom, structure_reducer, symbols, filename = None):
    """
    Args:
        atom: ase atom object
        structure_reducer: a function taking a pymatgen structure as input and returning a CoordinationSearch object
        symbols: a DummySymbol object 
        filename: str, where to write output files. Shouldn't contain any extension
            If None doesn't write

    Return:
        reduced_atom: ase atom with fragments instead of atoms
        report_search: dic
    """
    struct = AseAtomsAdaptor.get_structure(atom)

    searcher = structure_reducer(struct)
    searcher.symbols = symbols # enforce symbols; TBD: add option to get around, 'auto' setup, etc.
    reduced_struct = searcher.reduce_structure()
    report_search = {**{"is_reduced_structure_valid": searcher.is_reduced_structure_valid()}, **searcher.report_search}
    if searcher.is_reduced_structure_valid():
        reduced_atom = AseAtomsAdaptor.get_atoms(reduced_struct)
    else:
        reduced_atom = None
    return reduced_atom, report_search
