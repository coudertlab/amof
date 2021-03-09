"""
Module containing everythong related to trajectories: 
- wrapper for ase trajectory module
- constructor from different files used
"""

import ase.io.trajectory
import logging
import bisect


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
    def from_traj(cls, filename):
        """
        constructor of trajectory class from ase traj file 'filename'
        """
        logger.info("read trajectory %s", filename)
        rdf_class = cls() # initialize class
        cls.traj = ase.io.trajectory(filename, mode='r')
        return rdf_class 

    # DEV    
    @classmethod
    def from_lammps_data(cls, filename, atom_style = 'charge'):
        """
        constructor of trajectory class from ase traj file 'filename'
        atom_style: string representing lammps atom_style (e.g. 'charge')
        """
        rdf_class = cls() # initialize class
        atoms = ase.io.read(filename, format = 'lammps-data', style=atom_style)
        myList = ase.data.atomic_masses            
        atomic_numbers = [get_index_closest(myList, myNumber) for myNumber in atoms.get_masses()]            
        atoms.set_atomic_numbers(atomic_numbers)
        cls.traj = [atoms]
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