"""
Module containing methods acting on CP2K output files
"""

import logging
import os
import re
import pandas as pd

logger = logging.getLogger(__name__)

def clean_xyz(filename):
    """
    Clean cp2k xyz output:
        - remove duplicate timesteps:
    Works for positional data (positions, velocities, forces, etc.)

    search for lines starting by 'Atoms' and will remove the line before (number of atoms) until the start of the next frame.
    """
    seen_steps = set()
    with open(filename, "r") as fr:
        # lines = f.readlines()
        with open(str(filename)+"_temp_rm_duplicates", "w") as fw:
            previous = None
            write_to_file = True
            for line in fr:
                if line[0:5]==' i = ':
                    step = int(re.search(' i = (.*), time =(.*)', line).group(1))
                    if step not in seen_steps:
                        write_to_file = True
                        seen_steps.add(step)
                    else:
                        logger.info("Removing duplicate %s", step)
                        write_to_file = False
                if write_to_file and previous is not None:
                    fw.write(previous)
                previous = line
            if write_to_file:
                fw.write(previous)
    os.remove(filename)
    os.rename(str(filename)+"_temp_rm_duplicates", filename)


def clean_tabular(filename):
    """
    Clean cp2k tabular output (one line per step)
        - remove duplicate timesteps:
        - remove duplicate headers
    Works for ener, cell, stress
    """
    seen_steps = set()
    with open(filename, "r") as fr:
        # lines = f.readlines()
        with open(str(filename)+"_temp_rm_duplicates", "w") as fw:
            fw.write(fr.readline()) # write header (first line of first file)
            write_to_file = True
            for line in fr:
                if line[0]=='#':
                    write_to_file = False
                else:
                    step = int(re.split('\ +', line)[1])
                    if step not in seen_steps:
                        write_to_file = True
                        seen_steps.add(step)
                    else:
                        logger.info("Removing duplicate %s", line.strip("\n").strip('Atoms.'))
                        write_to_file = False
                if write_to_file:
                    fw.write(line)
    os.remove(filename)
    os.rename(str(filename)+"_temp_rm_duplicates", filename)    


def read_tabular(filename, return_units = False):
    """
    Parses tabular file
    Works for ener, cell, stress

    Args:
        filename: str or pathlib.Path
        return_units: Bool, if true returns unit_dict

    Return:
        df, pandas dataframe representing the data contained in filename
        unit_dict, dict of strings with columns names as keys and units as values
    """
    with open(filename, "r") as fr:
        first_line = fr.readline().strip('\n')
    # first_line = first_line.strip('#').strip('\n')
    columns = re.split('\  +', first_line)[1:] # rm '#' in first position, single whitespaces are tolerated in column name
    names = []
    units = []
    for c in columns:
        if 'Step' in c:
            names.append('Step')
            units.append('')
        else:
            search = re.search('(.*)\[(.*)\]', c)
            names.append(search.group(1).strip('.').strip(' ')) #strip points (abbreviations, not v. practical) and whitespaces (dataframe handling)
            units.append(search.group(2))

    df = pd.read_table(filename, skiprows=1, names=names, sep=r"\s+")
    df = df.set_index('Step')
    if return_units == False:
        return df
    else:
        return df, dict(zip(names, units))