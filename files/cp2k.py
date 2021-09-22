"""
Module containing methods acting on CP2K output files
"""

import logging
import os
import re

logger = logging.getLogger(__name__)

def clean_xyz(filename):
    """
    Clean cp2k xyz output:
        - remove duplicate timesteps:
    Works for positional data (positions, velocities, forces, etc.)

    search for lines starting by 'Atoms' and will remove the line before (number of atoms) until the start of the next frame.
    """
    seen_lines = set()
    with open(filename, "r") as fr:
        # lines = f.readlines()
        with open(str(filename)+"_temp_rm_duplicates", "w") as fw:
            previous = None
            write_to_file = True
            for line in fr:
                if line[0:5]==' i = ':
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