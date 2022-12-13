"""
Module containing everything related to trajectories: 
- wrapper for ase trajectory module
- constructor from different files used
- ReducedTrajectory class
"""

# import ase.io.trajectory
import ase.io
from ase.geometry.geometry import wrap_positions

import logging
import bisect
import numpy as np
import gzip
import shutil
import tempfile
import pandas as pd
import pathlib

import amof.atom
import amof.files.path as ampath
import amof.symbols

logger = logging.getLogger(__name__)

class Trajectory(object):
    """
    Wrapper for ase trajectory module
    self.traj: List of frames, each frame is an ase atom object
    """
    def __init__(self):
        """default constructor"""
        self.traj = []
        # self.struct = struct

    @classmethod
    def from_traj(cls, filename, index = None, format = None, unzip = False):
        """
        constructor of trajectory class using ase.io.read

        Args:
            index: index using ase format: 'first_frame:last_frame:step' or slice(first_frame,last_frame,step)
            format: str, Used to specify the file-format. If not given, the file-format will be guessed by the filetype function.
            unzip: if True will unzip file as temporary file before reading it with ase, if False will let ase handle the decompression
        """
        logger.info("Read trajectory %s", filename)
        trajectory_class = cls() # initialize class
        # cls.traj = ase.io.trajectory(filename, mode='r')
        if unzip:
            logger.info("Unzip trajectory file")
            with tempfile.NamedTemporaryFile(buffering=0) as tmp: # buffering messes with ase read in some way, deactivating it
                with gzip.open(filename, 'rb') as f_in:
                    shutil.copyfileobj(f_in, tmp)
                logger.info("Read trajectory with ase")
                trajectory_class.traj = ase.io.read(tmp.name, index, format)
        else:
            logger.info("Read trajectory with ase")
            trajectory_class.traj = ase.io.read(filename, index, format)
        return trajectory_class 

    @classmethod
    def from_lammps_data(cls, filename, atom_style):
        """
        constructor of trajectory class from lammps data file 'filename'
        atom_style: string representing lammps atom_style (e.g. 'charge')
        """
        trajectory_class = cls() # initialize class
        atoms = ase.io.read(filename, format = 'lammps-data', style=atom_style)
        myList = ase.data.atomic_masses            
        atomic_numbers = [cls.get_index_closest(myList, myNumber) for myNumber in atoms.get_masses()]            
        atoms.set_atomic_numbers(atomic_numbers)
        trajectory_class.traj = [atoms]
        return trajectory_class 

    @staticmethod
    def get_index_closest(myList, myNumber):
        """
        Assumes myList is sorted. Returns closest value to myNumber.

        If two numbers are equally close, return the smallest number.
        """
        pos = bisect.bisect_left(myList, myNumber)
        # pos = np.searchsorted(myList, myNumber) # twice as slow in tried example
        if pos == 0:
            return myList[0]
        if pos == len(myList):
            return myList[-1]
        before = myList[pos - 1]
        after = myList[pos]
        if after - myNumber < myNumber - before:
            return pos
        else:
            return pos - 1

    def set_cell(self, cell, set_pbc = True, fit_size = True):
        """
        Args: 
            cell: cell vector of same length than self.traj in valid ase cell format
            set_pbc: wether to set_pbc=True to every frame of traj
            fit_size = if True will limit array size to the smallest if cell and traj to avoid raising an error
        """
        if fit_size and len(self.traj)!=len(cell):
            logger.warning("Mismatch in file sizes; traj: %s vs cell: %s", len(self.traj), len(cell))
            if len(self.traj) > len(cell):
                logger.warning("Reducing traj length")
                self.traj = self.traj[0:len(cell)]
            else:
                logger.warning("reducing cell length")
                cell = cell[0:len(self.traj)]
        for i in range(len(self.traj)):
            self.traj[i].set_cell(cell[i])
            if set_pbc:
                self.traj[i].set_pbc(True)

    def get_traj(self):
        return self.traj


class ReducedTrajectory(object):
    """
    Class representing a reduced trajectory

    Attributes:
        trajectory: ase trajectory object containing the positions of the reduced trajectory
        report_search: pandas dataframe with information per Step of trajectory
        symbols: amof DummySymbols object
    """
    def __init__(self, trajectory = [], 
            report_search = pd.DataFrame({"Step": np.empty([0])}), 
            symbols = amof.symbols.DummySymbols()):
        """Default constructor from attributes
        Mock reducedtrajectory with no frame if no argument supplied"""
        self.trajectory = trajectory
        self.report_search = report_search
        self.symbols = symbols

    @classmethod
    def from_file(cls, filename, sampling = 1, load_trajectory = True):
        """
        Constructor of class from files

        Args: 
            filename: str or path to files without the final suffixes (ie no '.xyz' or '.symbols')
            sampling: int, will sample the trajectory with this rate choosing frames that are in_reduced_trajectory
        """
        if load_trajectory == True:
            logger.info("Read reduced trajectory %s", pathlib.Path(filename).name)
            trajectory = ase.io.read(ampath.append_suffix(filename, 'xyz'), ':', 'xyz')
        else:
            trajectory = []
        report_search = pd.read_csv(ampath.append_suffix(filename, 'report_search.csv'), index_col=0)
        symbols = amof.symbols.DummySymbols.from_file(filename)
        cn_class = cls(trajectory, report_search, symbols) # initialize class
        if sampling != 1:
            cn_class.sample(sampling)
        return cn_class # return class as it is a constructor

    def write_to_file(self, filename):
        """
        Args: 
            filename: str or path to files without the final suffixes (ie no '.xyz' or '.symbols')
        """
        self.report_search.to_csv(ampath.append_suffix(filename, 'report_search.csv'))
        ase.io.write(ampath.append_suffix(filename, 'xyz'), self.trajectory)
        self.symbols.write_to_file(filename)

    def sample(self, sampling):
        """
        Sample trajectory choosing frames that are in_reduced_trajectory

        sampling refers to the frames indexed by integers (0,1,2..) and to Steps
        First time it is used, it is relative to the not reduced trajectory
        Then if the reduced traj has already been sampled, then the rate is relative to the sampled reduced trajectory

        Args: 
            sampling: int, sampling rate relative to not reduced trajectory
        """        
        if len(self.report_search) != 0:
            rs_traj = self.report_search[self.report_search['in_reduced_trajectory']==True]
            in_traj_sampling = round(sampling * len(rs_traj) / len(self.report_search)) # can be zero if few frames in reduced traj
            if in_traj_sampling != 0: 
                self.trajectory = self.trajectory[::in_traj_sampling]  
                self.report_search = rs_traj[::in_traj_sampling]

def read_lammps_data(filename, atom_style):
    """
    constructor of trajectory class from lammps data file 'filename'
    atom_style: string representing lammps atom_style (e.g. 'charge')
    """
    return Trajectory.from_lammps_data(filename, atom_style).get_traj()

def read_lammps_traj(path_to_xyz, index = None, cell = None, unzip_xyz = False):
    """
    Args: 
        index: index using ase format: 'first_frame:last_frame:step' or slice(first_frame,last_frame,step)
        cell: cell vector of same length than self.traj, if None doesn't set cell
        
    Returns:
        traj: ase.trajectory object
    """
    Traj = Trajectory.from_traj(path_to_xyz, index, format = 'xyz', unzip = unzip_xyz)
    if cell is not None:
        Traj.set_cell(cell, set_pbc = True)
    return Traj.get_traj()


def read_cp2k_traj(path_to_xyz, path_to_cell, index = None, unzip_xyz = False):
    """
    Args: 
        index: index using slice ase format: slice(first_frame,last_frame,step). 
        'first_frame:last_frame:step' isn't supported
        
    Returns:
        traj: ase.trajectory object
    """
    Traj = Trajectory.from_traj(path_to_xyz, index, format = 'xyz', unzip = unzip_xyz)
    cell = np.genfromtxt(path_to_cell)
    if len(cell.shape) == 1: # corresponds to single frame in traj
        cell = cell[2:-1] 
        cell = cell[index]
        cell = np.array([cell.reshape(3,3)]) # reshape in 3*3 matrix cell format
    else:
        cell = cell[:,2:-1] 
        cell = cell[index]
        cell = np.array([c.reshape(3,3) for c in cell]) # reshape in 3*3 matrix cell format
    Traj.set_cell(cell, set_pbc = True)
    return Traj.get_traj()


def apply_to_traj(trajectory, function, how):
    """apply function to every atom of trajectory and aggregate using how"""
    if how == 'mean':
        return np.mean([function(atom) for atom in trajectory])

def get_density(trajectory, how = 'mean'):
    """return density of ase trajectory object"""
    return apply_to_traj(trajectory, amof.atom.get_density, how)

def get_number_density(trajectory, how = 'mean'):
    """return number density of ase trajectory object"""
    return apply_to_traj(trajectory, amof.atom.get_number_density, how)

def construct_step(**kwargs):
    """
    Construct step from various constructors

    Args:
        delta_Step: int, number of simulation steps between two frames
        first_frame: int, first step
        last_frame: int, first step
        number_of_frames: int, number of frames
        step: slice object or numpy array
    
    Return:
        step: numpy array containing steps
    """
    delta_Step = kwargs.get('delta_Step', None)
    first_frame = kwargs.get('first_frame', None)
    last_frame = kwargs.get('last_frame', None)
    number_of_frames = kwargs.get('number_of_frames', None)
    step = kwargs.get('step', None)

    try:
        if step is not None:
            if isinstance(step, slice):
                return np.array(list(range(step.start or 0, step.stop, step.step or 1)))
            else:
                return np.array(step)
        elif delta_Step is not None:
            if first_frame is not None and last_frame is not None:
                return np.arange(first_frame, last_frame, delta_Step)
            elif number_of_frames is not None: 
                if first_frame is None and last_frame is not None:
                    first_frame = last_frame - number_of_frames * delta_Step
                if first_frame is not None:
                    return np.arange(first_frame, first_frame + number_of_frames * delta_Step, delta_Step)
        elif number_of_frames is not None:
            if first_frame is not None and last_frame is not None:
                return np.linspace(first_frame, last_frame, number_of_frames)
    except:
        logger.exception("Cannot construct step from provided args")       
        raise ValueError  

def get_delta_pos(pos, cell):
    """
    Converts position coordinates of trajectory into displacements between consecutive frames
    Each displacement will be made to fit in the unit cell around (0,0,0). 
    In other words it will be as small as possible
    The first frame contains the initial positions
    For each frame k the original positions can be retrieved by summing all the displacements from 0 to k

    Args:
        pos: list, a trajectory of atomic positions
        cell: list, a traj of cell values (can be ase cell format)

    Return:
        delta_pos: list, a trajectory of atomic positions
    """
    delta_pos = [pos[0]]
    for k in range(len(pos)-1):
        delta_pos.append(wrap_positions(pos[k+1]-pos[k], cell[k], center=(0., 0., 0.)))
    return delta_pos