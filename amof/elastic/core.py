"""
Module containing methods linked to elastic properties
"""

import math, sys
import numpy as np
import xarray as xr
import logging
import ase.atoms
import pandas as pd

import amof.files.path as ampath
import amof.elastic.elate as elate

logger = logging.getLogger(__name__)


class ElasticConstant(object):
    """
    Main class for computing elastic constants
    """

    def __init__(self):
        """default constructor, initialize with empty fields"""
        self.temperature = None
        self.h = None
        self.step = None
        self.volume = None
        self.epsilons = None
        # empty data array with proper dims and coords
        self.Cmat = xr.DataArray(np.empty([0,6,6]), 
            coords = [('Step', np.empty([0], dtype='int64')), ('row', np.arange(1,7)), ('col', np.arange(1,7))], 
            name = 'elastic')

    @classmethod
    def from_cell(cls, h, temperature, final_value = False, step = None):
        """default constructor from cell
        Args:
            h: array_like, unit cell tensor, can be of any form that ase set_cell can process
            temperature: float, temperature in K
            final_value: Boolean, If True only one value of C will be computed
            step: Contains step information.
                can be array_like of same length than h containing the step information
        """
        elastic_class = cls()
        elastic_class.temperature = temperature
        elastic_class.set_h(h)
        elastic_class.set_step(step)
        elastic_class.set_volume()
        elastic_class.set_epsilons()
        if final_value:
            elastic_class.set_final_C() 
        else:
            elastic_class.set_every_C()
        return elastic_class 


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
            self.step = np.array(step) 

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

        # remove non inversible elements
        is_inversible = np.linalg.cond(Smat) < 1/sys.float_info.epsilon
        Smat = Smat[is_inversible]
        if self.step is not None:
            self.step = self.step[is_inversible]
        # Smat = Smat[1:] 

        # And now the stiffness matrix (in GPa)
        Cmat = np.linalg.inv(Smat) / 1.e9
        
        # turn to xarray object
        da = xr.DataArray(Cmat, dims=("Step","col",'row'))
        da["col"] = np.arange(1,7)
        da["row"] = np.arange(1,7)
        da.name = 'elastic'
        if self.step is not None:
            da["Step"] = self.step
        self.Cmat = da

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
        Cmat = np.linalg.inv(Smat) / 1.e9
            
        # turn to xarray object
        da = xr.DataArray(Cmat, dims=("col",'row'))
        da["col"] = np.arange(1,7)
        da["row"] = np.arange(1,7)
        da.name = 'elastic'
        if self.step is not None:
            da["Step"] = self.step
        self.Cmat = da

    def write(self, filename):
        path_to_output = ampath.append_suffix(filename, 'elastic')
        self.Cmat.to_netcdf(path_to_output)


    @classmethod
    def from_file(cls, filename):
        """
        constructor of elastic file class from rdf file
        """
        new_class = cls() # initialize class
        new_class.read_elastic_file(filename)
        return new_class # return class as it is a constructor

    def read_elastic_file(self, filename):
        filename = ampath.append_suffix(filename, 'elastic')
        self.Cmat = xr.open_dataset(filename)



    # # not adapted
    # @staticmethod
    # def print_cell(abc):
    #     print('Unit cell averages:')
    #     print('            a = %.3f' % np.mean([x[0] for x in abc]))
    #     print('            b = %.3f' % np.mean([x[1] for x in abc]))
    #     print('            c = %.3f' % np.mean([x[2] for x in abc]))
    #     print('    alpha = %.3f' % np.rad2deg(np.mean([x[3] for x in abc])))
    #     print('        beta = %.3f' % np.rad2deg(np.mean([x[4] for x in abc])))
    #     print('    gamma = %.3f' % np.rad2deg(np.mean([x[5] for x in abc])))
    #     print('   volume = %.1f' % volume)


class MechanicalProperties(object):
    """
    Main class to compute Mechanical Properties from Elastic constants
    Wrapper of ELATE
    """

    def __init__(self):
        """default constructor"""
        self.data = pd.DataFrame()

    @classmethod
    def from_elastic(cls, C):
        """
        constructor of CoordinationNumber class from elastic constant tensor
        Args:
            C: list of list of float, elastic constant tensor
        """
        new_class = cls() # initialize class
        new_class.compute_averages(C)
        return new_class # return class as it is a constructor

    def compute_averages(self, C):
        """
        compute average mechanical properties using ELATE

        C: list of list of float, elastic constant tensor
        """
        el = elate.Elastic(C)
        prop = el.averages()
        df = pd.DataFrame(prop, 
            index = ["voigt", "reuss", "hill"], 
            columns = ["bulk_modulus","youngs_modulus","shear_modulus","poissons_ratio"])
        df.index.name = "averaging_scheme"
        self.data = df

    @classmethod
    def from_file(cls, filename):
        """
        constructor of cn class from msd file
        """
        new_class = cls() # initialize class
        new_class.read_file(filename)
        return new_class # return class as it is a constructor

    def read_file(self, filename):
        """path_to_data: where the cn object is"""
        filename = ampath.append_suffix(filename, 'mech.csv')
        self.data = pd.read_csv(filename, index_col=0)

    def write(self, filename):
        filename = ampath.append_suffix(filename, 'mech.csv')
        self.data.to_csv(filename)



def print_Cmat(Cmat):
    print('')
    print('Stiffness matrix C (GPa):')
    for i in range(6):
        print('    ', end=' ')
        for j in range(6):
            if j >= i:
                    print(('% 8.2f' % Cmat[i,j]), end=' ')
            else:
                    print('        ', end=' ')
        print('')

    # Eigenvalues
    print('')
    print('Stiffness matrix eigenvalues (GPa):')
    print((6*'% 8.2f') % tuple(np.sort(np.linalg.eigvals(Cmat))))
