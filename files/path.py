"""
Module containing path related methods
"""

import pathlib

def append_suffix(path, suffix):
    """
    Append suffix to path if suffix is not already path latest suffix

    Args:
        path: pathlib.Path object or string
        suffix: string object, no initial '.'
    Returns:
        path: pathlib.Path object
    """
    path = pathlib.Path(path)
    if path.suffix != suffix:
        path = path.parent / (path.name + 'suffix')
    return path