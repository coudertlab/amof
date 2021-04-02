"""
Module containing methods linked to elastic properties
"""

import math, sys
import numpy as np
import xarray as xr
import os
import pathlib
import logging
import ase.atoms

logger = logging.getLogger(__name__)



h = []
mock_atom = ase.Atoms()

for c in cell:
    mock_atom.set_cell(c)
    h.append(mock_atom.get_cell().array)

h = np.array(h)

class ElasticConstant(object):
    """
    Main class for computing elastic constants
    """

    def __init__(self):
        """default constructor"""
        self.epsilons = []
        self.Cmat = [] 
        # self.msd_data = []

    # cell_traj_short = cell.isel(run_id=0).where(cell.notnull(), drop=True) # xarray where each line is a cell vector

    # Input file name
    # file = sys.argv[1]

    @staticmethod
    def cummean(a):
        """
        Return the cumulative mean of the elements along a given axis.
        Args:
            a : array_like        Input array.
        """
        return np.cumsum(a) / np.arange(1, len(a) + 1)

    def get_epsilons(h):
        # Temperature of the trajectory


        # Read h matrix trajectory
        # Format is: ( a_x    a_y   a_z      b_x     b_y    b_z   c_x      c_y     c_z )
        # # First column in XST file is step number, it is ignored
        # h = np.loadtxt(file, usecols=list(range(1,10)))
        # # Remove the first 20% of the simulation
        # h = h[len(h)/5:]
        # h = np.reshape(h, (-1,3,3))

        # Unit cell averages
        def vector_angle(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            angle = np.arccos(np.dot(v1_u, v2_u))
            if math.isnan(angle):
                if np.dot(v1_u, v2_u) > 0:
                        return 0.0
                else:
                        return np.pi
            return angle

        def h2abc(h):
            return (np.linalg.norm(h[0]), np.linalg.norm(h[1]), np.linalg.norm(h[2]),
                                vector_angle(h[1],h[2]), vector_angle(h[2],h[0]), vector_angle(h[0],h[1]))

        volume = ElasticConstant.cummean(list(map(np.linalg.det, h)))
        abc = list(map(h2abc, h))




        # Calculating the strain matrices

        inv_reference = np.linalg.inv(h[0])
        inv_reference_t = inv_reference.transpose()

        def h2eps(h):
            return (np.dot(inv_reference_t, np.dot(h.transpose(), np.dot(h, inv_reference))) - np.identity(3)) / 2

        epsilons = list(map(h2eps, h))

    def print_cell(abc):
        print('Unit cell averages:')
        print('            a = %.3f' % np.mean([x[0] for x in abc]))
        print('            b = %.3f' % np.mean([x[1] for x in abc]))
        print('            c = %.3f' % np.mean([x[2] for x in abc]))
        print('    alpha = %.3f' % np.rad2deg(np.mean([x[3] for x in abc])))
        print('        beta = %.3f' % np.rad2deg(np.mean([x[4] for x in abc])))
        print('    gamma = %.3f' % np.rad2deg(np.mean([x[5] for x in abc])))
        print('   volume = %.1f' % volume)

    def compute_C(h):
        temperature = 300

        # Elastic constants
        factor = (volume * 1.e-30) / (1.3806488e-23 * temperature)
        CARTESIAN_TO_VOIGT = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))
        VOIGT_FACTORS = (1, 1, 1, 2, 2, 2)

        Smat = np.zeros((6,6))
        for i in range(6):
            a, b = CARTESIAN_TO_VOIGT[i]
            fi = ElasticConstant.cummean([ epsilon[a, b] for epsilon in epsilons ])
            for j in range(i+1):
                u, v = CARTESIAN_TO_VOIGT[j]
                fj = ElasticConstant.cummean([ epsilon[u, v] for epsilon in epsilons ])
                fij = ElasticConstant.cummean([ epsilon[a, b] * epsilon[u, v] for epsilon in epsilons ])
                Smat[i,j] = VOIGT_FACTORS[i] * VOIGT_FACTORS[j] * factor * (fij - fi * fj)

        for i in range(5):
            for j in range(i+1,6):
                Smat[i][j] = Smat[j][i]

        # And now the stiffness matrix (in GPa)
        Cmat = np.linalg.inv(Smat) / 1.e9

        return Cmat

    def print_C(Cmat):
        print('')
        print('Stiffness matrix C (GPa):')
        for i in range(6):
            print('        ', end=' ')
            for j in range(6):
                if j >= i:
                        print(('% 8.2f' % Cmat[i,j]), end=' ')
                else:
                        print('                ', end=' ')
            print('')

        # Eigenvalues
        print('')
        print('Stiffness matrix eigenvalues (GPa):')
        print((6*'% 8.2f') % tuple(np.sort(np.linalg.eigvals(Cmat))))



    def get_final_C(epsilons, volume):
        temperature = 300

        # Elastic constants
        factor = (volume * 1.e-30) / (1.3806488e-23 * temperature)
        CARTESIAN_TO_VOIGT = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))
        VOIGT_FACTORS = (1, 1, 1, 2, 2, 2)

        Smat = np.zeros((6,6))
        for i in range(6):
            a, b = CARTESIAN_TO_VOIGT[i]
            fi = np.mean([ epsilon[a, b] for epsilon in epsilons ])
            for j in range(i+1):
                u, v = CARTESIAN_TO_VOIGT[j]
                fj = np.mean([ epsilon[u, v] for epsilon in epsilons ])
                fij = np.mean([ epsilon[a, b] * epsilon[u, v] for epsilon in epsilons ])
                Smat[i,j] = VOIGT_FACTORS[i] * VOIGT_FACTORS[j] * factor * (fij - fi * fj)

        for i in range(5):
            for j in range(i+1,6):
                Smat[i][j] = Smat[j][i]

        # And now the stiffness matrix (in GPa)
        Cmat = np.linalg.inv(Smat) / 1.e9
        return Cmat


    epsilons = get_epsilons(h)
    volumes = list(map(np.linalg.det, h))
    Cs = []

    for i in range(100, len(h),100):
        volume = np.mean(volumes[:i])
        Cs.append(get_C(epsilons[:i], volume))
    Cs = np.array(Cs)

    # np.save('Cs', Cs)

    a = 1