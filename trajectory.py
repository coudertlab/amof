"""
Module containing everythong related to trajectories: 
- wrapper for ase trajectory module
- constructor from different files used
"""

import ase.io.trajectory
import logging

logger = logging.getLogger(__name__)

class Trajectory(object):
    """
    Wrapper for ase trajectory module
    List of frames, each frame is an ase atom object
    """
    def __init__(self):
        """default constructor"""
        self.data = []
        # self.struct = struct

    @classmethod
    def from_traj(cls, path_to_traj, dr = 0.01, rmax = 'half_cell', single_frame = False, last_frames = False):
        """
        constructor of trajectory class from ase traj file
        dr, rmax in Angstrom
        If rmax is set to 'half_cell', then half of the minimum dimension of the cell is used to ensure no atom is taken into account twice for a given atom (computation is possible beyound this threeshold up to the min cell size)
        last_frames: number of last frames required
        Seed: specify if seed in traj so that these atoms are removed from the computation"""
        logger.info("read trajectory %s", path_to_traj)
        rdf_class = cls() # initialize class
        ase.io.trajectory(path_to_traj, mode='r')
        # rdf_class.read_trajectory(path_to_traj, dr, rmax, single_frame, last_frames)
        return rdf_class # return class as it is a constructorf 

    # DEV    
    def from_lammps_data():



        if single_frame:
            # to be changed with a proper import func or either a class
            # format to be changed when first used
            atoms = ase.io.read(path_to_traj, format = 'lammps-data', style='charge')

            def get_index_closest(myList, myNumber):
                """
                Assumes myList is sorted. Returns closest value to myNumber.

                If two numbers are equally close, return the smallest number.
                """
                pos = bisect.bisect_left(myList, myNumber)
        #         pos = np.searchsorted(myList, myNumber) # twice as slow in tried example
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

            myList = ase.data.atomic_masses            
            atomic_numbers = [get_index_closest(myList, myNumber) for myNumber in atoms.get_masses()]            
            atoms.set_atomic_numbers(atomic_numbers)
            traj = [atoms]
        else:
            traj = Trajectory(path_to_traj, mode='r')
        if last_frames!=False:
        #     output_path += "-lastframes"
            traj = [traj[i] for i in range(len(traj) - last_frames, len(traj))] # last <last_frames> frames
             
        # H = data.atomic_numbers['H']
        # C = data.atomic_numbers['C']
        # N = data.atomic_numbers['N']
        # Zn = data.atomic_numbers['Zn']