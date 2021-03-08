from ase.io.trajectory import Trajectory

class Trajectory(object):
    def __init__(self):
        """default constructor"""
        self.multiple_rdf_data = {}
        # self.struct = struct