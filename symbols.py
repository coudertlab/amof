"""
Module for representing symbols not present in the periodic table

Designed for fragment names, but more general terminology:
    name (e.g. fragname such as Im)
    symbol (chemical_symbol, from the periodic table)
"""

import sadi.files.path
import json

from ase.data import chemical_symbols
# Seventh line of the periodic table (from ase.data)
# Arbitrary list for now with elements unlikely to be present in ZIFs, need to be expanded if not long enough
chemical_symbols_seventh_period = ['Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
    'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
    'Lv', 'Ts', 'Og']    

class DummySymbols(object):
    """
    Class allowing handling of any name represented by a string by linking it to a symbol present in the periodic table
    """

    def __init__(self, names = None):
        """
        Args:
            symbol_names: list of str, names used in an atom/structure object
                if None, will create empty class
        """ 
        self.from_name_to_symbol = {}     
        self.from_symbol_to_name = {}
        if names is not None:
            nb_changed_names = 0    
            available_chemical_symbols = [s for s in chemical_symbols_seventh_period if s not in names]
            for name in names:
                if name in chemical_symbols:
                    pt_symbol = name
                else:
                    pt_symbol = available_chemical_symbols[nb_changed_names] # will raise error if insufficient available names
                    nb_changed_names += 1
                self.from_name_to_symbol[name] = pt_symbol
                self.from_symbol_to_name[pt_symbol] = name

    def get_symbol(self, name):
        return self.from_name_to_symbol[name]
    
    def get_name(self, symbol):
        return self.from_symbol_to_name[symbol]
    
    @classmethod
    def from_file(cls, filename):
        """
        constructor of cn class from msd file
        """
        cn_class = cls() # initialize class
        cn_class.read_file(filename)
        return cn_class # return class as it is a constructor

    def read_file(self, filename):
        """path_to_data: where the cn object is"""
        filename = sadi.files.path.append_suffix(filename, 'symbols')
        self.from_name_to_symbol = json.load(open(filename))
        self.from_symbol_to_name = {v: k for k, v in self.from_name_to_symbol.items()}

    def write_to_file(self, filename):
        filename = sadi.files.path.append_suffix(filename, 'symbols')
        with open(filename, 'w') as fp:
            json.dump(self.from_name_to_symbol, fp)

