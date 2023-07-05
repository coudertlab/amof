"""
Module containing functions to reduce a trajectory to its fragments
"""

import pandas as pd
import logging
import joblib

import time
import multiprocessing
import functools

from pymatgen.io.ase import AseAtomsAdaptor
from amof.coordination.core import SearchError

import amof.symbols
import amof.coordination.zif
import amof.trajectory

logger = logging.getLogger(__name__)

def reduce_trajectory(trajectory, mof, filename = None, dist_margin = 1.2, delta_Step = 1, 
        first_frame = 0, parallel = False, write_mfpx = False, **kwargs):
    """
    Convenient wrapper to reduce trajectory by specifying a specific mof
    For now can handle 'ZIF-4', 'ZIF-8', 'ZIF-zni' and 'SALEM-2' with full functionality
    Or 'ZnCycle' only for detection of the C3N2 cycles (e.g. no mfpx output).
    'ZnCycle' support any ZIF with Zn metal nodes linked to 4 linkers, each linker comprising a unique C3N2 cycle.    
    Thoroughly tested only for 'ZIF-4' and decently tested for 'ZnCycle'

    Args:
        trajectory: ase trajectory object
        mof: str, mof name
        filename: str, where to write output files  
            if None, avoid writing
        dist_margin: float, default tolerance when using the covalent radii criteria to determine if two atoms are neighbours. 
            Default: 1.2
        delta_Step: int, number of simulation steps between two frames
        write_mfpx: bool, if True will write an mfpx file per frame of the reduced trajectory
    """
    dist_margin_metal = kwargs.get('dist_margin_metal', 1.6) # 1.6 means 3.088 for Zn-N (one case of 2.92 was observed)
    dist_margin_H = kwargs.get('dist_margin_H', 1.44)

    if mof in ['ZIF-4', 'ZIF-zni', 'SALEM-2']:
        structure_reducer = lambda struct: amof.coordination.zif.MetalIm(struct, "Zn", 
            dist_margin=dist_margin, 
            dist_margin_metal=dist_margin_metal,
            dist_margin_H = dist_margin_H) 
        symbols = amof.symbols.DummySymbols(['Zn', 'Im']) 
    elif mof in  ['ZIF-8']:
        structure_reducer = lambda struct: amof.coordination.zif.MetalmIm(struct, "Zn", dist_margin=dist_margin)
        symbols = amof.symbols.DummySymbols(['Zn', 'mIm']) 
    elif mof in  ['ZnCycle']:
        structure_reducer = lambda struct: amof.coordination.zif.MetalCycle(struct, "Zn", dist_margin=dist_margin)
        symbols = amof.symbols.DummySymbols(['Zn', 'ImCycle'])      
        if write_mfpx == True:
            logger.error(f'Write mfpx is not implemented for {mof}')   
    else:
        structure_reducer = lambda struct: amof.coordination.NotImplementedSearch(mof)
        symbols = amof.symbols.DummySymbols()
        logger.warning(f'Structure search not implemented for {mof}')
    return reduce_trajectory_core(trajectory, structure_reducer, symbols, filename, 
        delta_Step = delta_Step, first_frame = first_frame, parallel = parallel, write_mfpx = write_mfpx)

def reduce_trajectory_core(trajectory, structure_reducer, symbols, filename = None, delta_Step = 1, 
            first_frame = 0, parallel = False, write_mfpx = False):
        """
        Args:
            trajectory: ase trajectory object
            structure_reducer: a function taking a pymatgen structure as input and returning a CoordinationSearch object
            symbols: a DummySymbol object containing the representation to use for all frames 
                (no option atm to include additional symbols on specific frames,
                 parallelization issues + not adapted to analysis tools to have variable number of atoms)
            delta_Step: int, number of simulation steps between two frames
            filename: str, where to write output files, specify None to avoid writing
            write_mfpx: bool, if True will write an mfpx file per frame of the reduced trajectory
                if no filename is provided, will not write any
        
        Return:
            reduced_traj: amof.trajectory ReducedTrajectory object
        """
        logger.info("Start reducing trajectory for %s frames", len(trajectory))

        step = amof.trajectory.construct_step(delta_Step=delta_Step, first_frame = first_frame, number_of_frames = len(trajectory))

        def per_atom(atom, step, filename):
            report_search = {'Step': step}
            try:
                if filename is not None:
                    filename += f"_{step}" 

                # Non-wraped atoms were found to cause memory leak in CoordinationSearch at line
                # self.all_neighb = self.struct.get_all_neighbors(neighb_max_distance)          
                # It occured for extreme non wrapped atoms located ~50 images away, and didn't cause any issue for non-failing MD simulations
                atom.wrap() 

                reduced_atom, report_search_atom = reduce_atom(atom, structure_reducer, symbols, write_mfpx = write_mfpx, filename = filename)
                report_search['in_reduced_trajectory'] = reduced_atom is not None
                report_search = {**report_search, **report_search_atom}
            except SearchError as e:
                logger.debug('Failed to do reduce frame with error message: ' + e.message)        
                report_search['in_reduced_trajectory'] = False       
                report_search = {**report_search, **e.report_search}
                report_search['Error_message'] = e.message
                reduced_atom = None
            except BaseException as e: # unexpected exception
                logger.debug('Failed to do reduce frame with error message: ' + str(e))        
                report_search['in_reduced_trajectory'] = False       
                report_search['Error_message'] = "Unexpected Base Exception: " + str(e)
                reduced_atom = None
            return reduced_atom, report_search

        if parallel == False:
            result_list = [per_atom(trajectory[i], step[i], filename) for i in range(len(trajectory))]
        else:
            num_cores = parallel if type(parallel) == int else max(joblib.cpu_count() // 2 - 2, 2) # heuristic for 40cores Xeon cpus
            result_list = joblib.Parallel(n_jobs=num_cores)(joblib.delayed(per_atom)(trajectory[i], step[i], filename) for i in range(len(trajectory)))

        list_report_search = []
        reduced_traj = []
        for reduced_atom, report_search in result_list:
            list_report_search.append(report_search)
            if report_search['in_reduced_trajectory'] == True:
                reduced_traj.append(reduced_atom)
        
        df_report_search = pd.DataFrame(list_report_search).set_index('Step')

        reduced_trajectory = amof.trajectory.ReducedTrajectory(reduced_traj, df_report_search, symbols)
        if filename is not None:
            reduced_trajectory.write_to_file(filename)
        return reduced_trajectory


# Forked from https://github.com/joblib/joblib/pull/366#issuecomment-267603530
def with_timeout(timeout):
    def decorator(decorated):
        @functools.wraps(decorated)
        def inner(*args, **kwargs):
            pool = multiprocessing.pool.ThreadPool(1)
            async_result = pool.apply_async(decorated, args, kwargs)
            try:
                return async_result.get(timeout)
            except multiprocessing.TimeoutError:
                raise SearchError('Timeout reached')            
                return 
        return inner
    return decorator

# Timeout of 30min per reduction. If it is still stuck it probably means that there is something wrong with the XYZ (e.g. unphysicaly clustered atoms)
@with_timeout(1800)
def reduce_atom(atom, structure_reducer, symbols, write_mfpx = False, filename = None):
    """
    Args:
        atom: ase atom object
        structure_reducer: a function taking a pymatgen structure as input and returning a CoordinationSearch object
        symbols: a DummySymbol object 
        write_mfpx: bool, if True will write an mfpx file per frame of the reduced trajectory
        filename: str, where to write output files. Shouldn't contain any extension
            If None doesn't write

    Return:
        reduced_atom: ase atom object with fragments instead of atoms
        report_search: dictionary
    """
    struct = AseAtomsAdaptor.get_structure(atom)
    searcher = structure_reducer(struct)
    searcher.symbols = symbols # enforce symbols; TBD: add option to get around, 'auto' setup, etc. 
    reduced_struct = searcher.reduce_structure()
    report_search = {**{"is_reduced_structure_valid": searcher.is_reduced_structure_valid()}, **searcher.report_search}
    if searcher.is_reduced_structure_valid():
        reduced_atom = AseAtomsAdaptor.get_atoms(reduced_struct)
        if write_mfpx == True and filename is not None:
            searcher.write_mfpx(filename)
    else:
        reduced_atom = None
    return reduced_atom, report_search
