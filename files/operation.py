"""
Module containing basic file operation methods
"""

import gzip
import shutil
import os
import pathlib
import logging

logger = logging.getLogger(__name__)

def compress(filename, remove_if_exists = False):
    """
    Args:
        filename: str, filename without '.gz'
        remove_if_exists: bool, remove uncompressed file if True and a file with .gz already exist
    """
    if not (remove_if_exists == True and pathlib.Path(filename+'.gz').exists()):
        logger.info("compress %s", filename)
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
    logger.info("decompress %s", filename)
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