"""
Module containing functions to reduce a trajectory to its fragments
"""

import pandas as pd
from pymatgen.io.ase import AseAtomsAdaptor

import sadi.symbols

def reduce_trajectory(trajectory, structure_reducer, filename = None):
        """
        Args:
            trajectory: ase trajectory object
            structure_reducer: a function taking a pymatgen structure as input and returning a CoordinationSearch object
            filename: str, where to write output files
                If None doesn't write
        """
        symbols = sadi.symbols.DummySymbols()
        for atom in trajectory:

        df = pd.DataFrame(list_of_dict)
        df.to_csv(results + mof+".csv")
        print(df)

        ase.io.write("ZIF4_15glass01_reduced_traj.xyz", reduced_traj)

def reduce_atom(atom, structure_reducer, symbols, filename = None):
    """
    Args:
        atom: ase atom object
        structure_reducer: a function taking a pymatgen structure as input and returning a CoordinationSearch object
        symbols: a DummySymbol object 
        filename: str, where to write output files. Shouldn't contain any extension
            If None doesn't write

    Return:
        reduced_atom: ase atom with fragments instead of atoms
        report_search: dic

    """
    struct = AseAtomsAdaptor.get_structure(atom)

    searcher = structure_reducer(struct)
    reduced_struct = searcher.reduce_structure()

    if searcher.is_reduced_structure_valid():
        reduced_atom = AseAtomsAdaptor.get_atoms(reduced_struct)
        report_search = searcher.report_search


    # logger.info(conn)
    list_of_dict.append(searcher.report_search)
    logger.info(searcher.report_search)
            # final check, view entire conn
    # for i in range(len(conn)):
    #     print(i, struct[i].species, conn[i], [str(struct[j].species) for j in conn[i]])
