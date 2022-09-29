"""
Module containing methods acting on LAMMPS output files
"""

import logging
import os

logger = logging.getLogger(__name__)

def remove_duplicate_timesteps(filename):
    """
    remove duplicate timesteps in lammps xyz output
    search for lines starting by 'Atoms' and will remove the line before (number of atoms) until the start of the next frame.
    """
    seen_lines = set()
    with open(filename, "r") as fr:
        # lines = f.readlines()
        with open(str(filename)+"_temp_rm_duplicates", "w") as fw:
            previous = None
            write_to_file = True
            for line in fr:
                if line[0:5]=='Atoms':
                    if line not in seen_lines:
                        write_to_file = True
                        seen_lines.add(line)
                    else:
                        logger.info("Removing duplicate %s", line.strip("\n").strip('Atoms.'))
                        write_to_file = False
                if write_to_file and previous is not None:
                    fw.write(previous)
                previous = line
            if write_to_file:
                fw.write(previous)
    os.remove(filename)
    os.rename(str(filename)+"_temp_rm_duplicates", filename)