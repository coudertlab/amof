"""
Module containing basic file operation methods
"""

import gzip
import shutil
import os

def compress(filename):
    """
    Args:
        filename: str, filename without '.gz'
    """
    with open(filename, 'rb') as f_in:
        with gzip.open(filename + '.gz', 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    os.remove(filename)

def decompress(filename, remove = True):
    """
    Args:
        filename: str, filename without '.gz'
        remove: bool, remove compressed file if True
    """
    with gzip.open(filename + '.gz', 'rb') as f_in:
        with open(filename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    if remove == True:
        os.remove(filename+ '.gz')

def concatenate(filenames, output_file):
    """
    Args:
        filename: list of strings
        output files: strings
    """
    with open(output_file,'wb') as wfd:
        for f in filenames:
            with open(f,'rb') as fd:
                shutil.copyfileobj(fd, wfd)