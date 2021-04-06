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

import sadi.files.path

logger = logging.getLogger(__name__)




class ElasticConstant(object):
    """
    Main class for computing elastic constants
    """

    def __init__(self, h, temperature, final_value = False, step = None):
        """default constructor
        Args:
            h: array_like, unit cell tensor
            final_value: Boolean, If True only one value of C will be computed
            step: Contains step information.
                can be array_like of same length than h containing the step information
        """
        self.temperature = temperature
        self.set_h(h)
        self.set_step(step)
        self.set_volume()
        self.set_epsilons()
        if final_value:
            self.set_final_C() 
        else:
            self.set_every_C() 


    def set_h(self, h):
        """
        Args:
            h: array_like, unit cell tensor, can be of any form that ase set_cell can process
        Results:
            self.h: Format is: ( a_x    a_y   a_z      b_x     b_y    b_z   c_x      c_y     c_z )
        """
        new_h = []
        mock_atom = ase.Atoms()
        for c in h:
            mock_atom.set_cell(c)
            new_h.append(mock_atom.get_cell().array)
        h = np.array(new_h)
        self.h = h
        
    def set_step(self, step):
        if step is None:
            self.step = None
        else:
            self.step = np.array(step)[1:] # remove first step corresponding to Smat = 0

    @staticmethod
    def cummean(a):
        """
        Return the cumulative mean of the elements along a given axis.
        Args:
            a : array_like        Input array.
        """
        return np.cumsum(a) / np.arange(1, len(a) + 1)

    def set_volume(self):
        self.volume = self.cummean(list(map(np.linalg.det, self.h)))

    def set_epsilons(self):
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

        abc = list(map(h2abc, self.h))

        # Calculating the strain matrices

        inv_reference = np.linalg.inv(self.h[0])
        inv_reference_t = inv_reference.transpose()

        def h2eps(h):
            return (np.dot(inv_reference_t, np.dot(h.transpose(), np.dot(h, inv_reference))) - np.identity(3)) / 2

        self.epsilons = np.array(list(map(h2eps, self.h)))

    def set_every_C(self):
        # Elastic constants
        factor = (self.volume * 1.e-30) / (1.3806488e-23 * self.temperature)
        CARTESIAN_TO_VOIGT = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))
        VOIGT_FACTORS = (1, 1, 1, 2, 2, 2)

        Smat = np.zeros((len(factor),6,6))
        for i in range(6):
            a, b = CARTESIAN_TO_VOIGT[i]
            fi = ElasticConstant.cummean([ epsilon[a, b] for epsilon in self.epsilons ])
            for j in range(i+1):
                u, v = CARTESIAN_TO_VOIGT[j]
                fj = ElasticConstant.cummean([ epsilon[u, v] for epsilon in self.epsilons ])
                fij = ElasticConstant.cummean([ epsilon[a, b] * epsilon[u, v] for epsilon in self.epsilons ])
                Smat[:, i,j] = VOIGT_FACTORS[i] * VOIGT_FACTORS[j] * factor * (fij - fi * fj)

        for i in range(5):
            for j in range(i+1,6):
                Smat[:,i,j] = Smat[:,j,i]

        Smat = Smat[1:] # remove first Smat = 0

        # And now the stiffness matrix (in GPa)
        self.Cmat = np.linalg.inv(Smat) / 1.e9

    def set_final_C(self):
        volume = self.volume[-1]

        # Elastic constants
        factor = (volume * 1.e-30) / (1.3806488e-23 * self.temperature)
        CARTESIAN_TO_VOIGT = ((0, 0), (1, 1), (2, 2), (2, 1), (2, 0), (1, 0))
        VOIGT_FACTORS = (1, 1, 1, 2, 2, 2)

        Smat = np.zeros((6,6))
        for i in range(6):
            a, b = CARTESIAN_TO_VOIGT[i]
            fi = np.mean([ epsilon[a, b] for epsilon in self.epsilons ])
            for j in range(i+1):
                u, v = CARTESIAN_TO_VOIGT[j]
                fj = np.mean([ epsilon[u, v] for epsilon in self.epsilons ])
                fij = np.mean([ epsilon[a, b] * epsilon[u, v] for epsilon in self.epsilons ])
                Smat[i,j] = VOIGT_FACTORS[i] * VOIGT_FACTORS[j] * factor * (fij - fi * fj)

        for i in range(5):
            for j in range(i+1,6):
                Smat[i][j] = Smat[j][i]

        # And now the stiffness matrix (in GPa)
        self.Cmat = np.linalg.inv(Smat) / 1.e9

    def write(self, filename):
        da = xr.DataArray(self.Cmat, dims=("Step","col",'row'))
        da["col"] = np.arange(1,7)
        da["row"] = np.arange(1,7)
        if self.step is not None:
            da["Step"] = self.step
        path_to_output = sadi.files.path.append_suffix(filename, 'nc')
        da.to_netcdf(path_to_output)


    # not adapted
    @staticmethod
    def print_cell(abc):
        print('Unit cell averages:')
        print('            a = %.3f' % np.mean([x[0] for x in abc]))
        print('            b = %.3f' % np.mean([x[1] for x in abc]))
        print('            c = %.3f' % np.mean([x[2] for x in abc]))
        print('    alpha = %.3f' % np.rad2deg(np.mean([x[3] for x in abc])))
        print('        beta = %.3f' % np.rad2deg(np.mean([x[4] for x in abc])))
        print('    gamma = %.3f' % np.rad2deg(np.mean([x[5] for x in abc])))
        print('   volume = %.1f' % volume)

    # not adapted
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
