"""
Module containing everythong related to trajectories: 
- wrapper for ase trajectory module
- constructor from different files used
"""

# import ase.io.trajectory
import ase.io
import logging
import bisect
import numpy as np

import sadi.atom

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
    def from_traj(cls, filename, index = None, format = None):
        """
        constructor of trajectory class using ase.io.read
        index: index using ase format: 'first_frame:last_frame:step' or slice(first_frame,last_frame,step)
        format: str, Used to specify the file-format. If not given, the file-format will be guessed by the filetype function.
        """
        logger.info("read trajectory %s", filename)
        rdf_class = cls() # initialize class
        # cls.traj = ase.io.trajectory(filename, mode='r')
        rdf_class.traj = ase.io.read(filename, index, format)
        return rdf_class 

    @classmethod
    def from_lammps_data(cls, filename, atom_style):
        """
        constructor of trajectory class from lammps data file 'filename'
        atom_style: string representing lammps atom_style (e.g. 'charge')
        """
        rdf_class = cls() # initialize class
        atoms = ase.io.read(filename, format = 'lammps-data', style=atom_style)
        myList = ase.data.atomic_masses            
        atomic_numbers = [cls.get_index_closest(myList, myNumber) for myNumber in atoms.get_masses()]            
        atoms.set_atomic_numbers(atomic_numbers)
        rdf_class.traj = [atoms]
        return rdf_class 

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
            fit_size = if True will limit array size to the smallest if cell and traj to avoir raising an error
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

def read_lammps_data(filename, atom_style):
    """
    constructor of trajectory class from lammps data file 'filename'
    atom_style: string representing lammps atom_style (e.g. 'charge')
    """
    return Trajectory.from_lammps_data(filename, atom_style).get_traj()

def read_lammps_traj(path_to_xyz, index = None, cell = None):
    """
    Args: 
        index: index using ase format: 'first_frame:last_frame:step' or slice(first_frame,last_frame,step)
        cell: cell vector of same length than self.traj, if None doesn't set cell
        
    Returns:
        traj: ase.trajectory object
    """
    Traj = Trajectory.from_traj(path_to_xyz, index, format = 'xyz')
    if cell is not None:
        Traj.set_cell(cell, set_pbc = True)
    return Traj.get_traj()


def read_cp2k_traj(path_to_xyz, path_to_cell, index = None):
    """
    Args: 
        index: index using slice ase format: slice(first_frame,last_frame,step). 
        'first_frame:last_frame:step' isn't supported
        
    Returns:
        traj: ase.trajectory object
    """
    Traj = Trajectory.from_traj(path_to_xyz, index, format = 'xyz')
    cell = np.genfromtxt(path_to_cell)[:,2:-1] 
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
    return apply_to_traj(trajectory, sadi.atom.get_density, how)

def get_number_density(trajectory, how = 'mean'):
    """return number density of ase trajectory object"""
    return apply_to_traj(trajectory, sadi.atom.get_number_density, how)
