"""
Module containing methods to output files readable by molsys:
    - mfpx

forked and adapted from molsys/fileIO/mfpx.py
"""


import numpy
import logging

"""
These functions implement the reading and writing of mfpx files

for the handling of metadata of topologies/embedding the system of keywords is largely extended.
Note: the code still reads old mfpx file of type top correctly but it writes only the new format.

"""

logger = logging.getLogger(__name__)

class DummyMol(object):
    """
    DummyMol object with all the necessary attributes
    Mainly here to provide info on mol arg for write_mfpx
    """
    def __init__(self, elems, xyz, cell, conn, atypes, fragtypes, fragnumbers):
        """default constructor"""
        self.cell = cell
        self.fragtypes = fragtypes
        self.fragnumbers = fragnumbers
        self.elems = elems
        self.xyz = xyz
        self.conn = conn
        self.natoms = len(elems)
        self.atypes = atypes

def write_mfpx(mol, filename):
    """
    Routine, which writes an mfpx file
    :Parameters:
        -mol   (obj) : DummyMol or class with similar attributes
        -filename (str) : filename
    """
    f = open(filename, 'w')
    ### write check ###
    try:
        f.write ### do nothing
    except AttributeError:
        raise IOError("%s is not writable" % f)
    ### write func ###
    ftype = 'xyz'
    f.write('# type %s\n' % ftype)
    f.write('# cellvect %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f %12.6f\n' %\
        tuple(mol.cell.ravel()))
    write_body(f,mol)
    f.close()

def write_body(f, mol):
    """
    Routine, which writes the body of a txyz or a mfpx file
    :Parameters:
        -f      (obj)  : fileobject
        -mol    (obj)  : DummyMol or class with similar attributes
    """
    fragtypes   = mol.fragtypes
    fragnumbers = mol.fragnumbers
    elems       = mol.elems
    xyz         = mol.xyz
    cnct        = mol.conn
    natoms      = mol.natoms
    atypes      = mol.atypes

    xyzl = xyz.tolist()
    for i in range(natoms):
        line = ("%3d %-3s" + 3*"%12.6f" + "   %-24s") % \
            (i+1, elems[i], xyzl[i][0], xyzl[i][1], xyzl[i][2], atypes[i])
        line += ("%-16s %5d ") % (fragtypes[i], fragnumbers[i])
        conn = (numpy.array(cnct[i])+1).tolist()
        if len(conn) != 0:
            line += (len(conn)*"%7d ") % tuple(conn)
        f.write("%s \n" % line)
