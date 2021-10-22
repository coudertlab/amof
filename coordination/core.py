#-*- coding: utf-8 -*-

"""
Main file containing classes for custom neigbour search
"""

# from re import L
import numpy as np
import pymatgen
from pymatgen.core.structure import Structure
from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
from pymatgen.analysis.graphs import StructureGraph
import networkx as nx
from networkx.algorithms.chains import chain_decomposition
from copy import deepcopy
from collections import Counter
import itertools
from scipy import stats

import logging

# from xarray.core import dataset

import sadi.files.path
import sadi.structure
import sadi.symbols

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class SearchError(Exception):
    """Exception raised when coordination search failed.

    Attributes:
        message -- explanation of the error
        report_search -- dictionary describing the search when the error occurred
    """

    def __init__(self, message, report_search = {}):
        self.message = message
        self.report_search = report_search

class CoordinationSearch(object):
    """
    Classes containing general methods to perform a coordination search
    The actual search is to be launched from a subclass mentionning specific species, etc.

    Attributes:
        fragments: dict of dict;  key i of encompassing dict is fragment of fragnumber i
        atypes, fragtypes, fragnumbers, conn, elems: same structure as molsys.mol object
    """
        
    def __init__(self, struct, neighb_max_distance, dist_margin):
        """constructor"""
        self.struct = struct
        self.conn = [[] for i in range(struct.num_sites)]
        self.atypes = ["" for i in range(struct.num_sites)]
        self.fragtypes = ["-1" for i in range(struct.num_sites)]
        self.fragnumbers = [-1 for i in range(struct.num_sites)]
        self.elems = [str(self.struct[i].species)[:-1].lower() for i in range(struct.num_sites)]
        self.fragments = {} 
        self.symbols = sadi.symbols.DummySymbols()
        self.all_neighb = self.struct.get_all_neighbors(neighb_max_distance)
        self.dist_margin = dist_margin
        # initialize report_search with useful descriptors of struct for subsequent report analysis
        self.report_search = {"number_of_atoms":self.struct.num_sites}

    def in_fragment(self, indice):
        """Return True IFF atom at indice 'indice' is in a fragment"""
        return self.fragnumbers[indice] != -1

    def create_fragment(self, fragtype, indices, fragnumber = 'auto'):
        """
        Create new fragment

        Args:
            fragtype: str
            indices: list of int
            fragnumber: int or str. 
                If int, this fragnumber will be used
                If 'auto' will choose the max value (+1) of existing keys
        """
        if fragnumber == 'auto':
            fragnumber = 0 if len(self.fragments.keys()) == 0 else max(self.fragments.keys()) + 1
        indices = list(set(indices)) # remove duplicates
        fragment = {"fragnumber":fragnumber, "fragtype":fragtype, "indices":indices}
        for i in indices:
            self.fragtypes[i] = fragtype
            self.fragnumbers[i] = fragnumber
        self.fragments[fragnumber] = fragment

    def add_to_fragment(self, fragnumber, indices):
        """
        Add indices to fragment

        Args:
            fragnumber: int
            indices: list of int
        """
        fragtype = self.fragments[fragnumber]['fragtype']
        for i in indices:
            self.fragtypes[i] = fragtype
            self.fragnumbers[i] = fragnumber
        self.fragments[fragnumber]['indices'] = list(set(self.fragments[fragnumber]['indices'] + indices))

    def merge_fragments(self, fragnumber, fragnumber_to_absorb):
        """
        Merge fragments by including fragnumber_to_absorb into fragnumber

        Args:
            fragnumber: int
            fragnumber_to_absorb: int
        """
        self.add_to_fragment(fragnumber, self.fragments[fragnumber_to_absorb]['indices'])
        self.fragments.pop(fragnumber_to_absorb)
        
    
    def change_fragnumber(self, fragnumber_old, fragnumber_new):
        """
        Replace fragnumber if fragnumber_new isn't used

        Args:
            fragnumber_old, fragnumber_new: int
        """
        if fragnumber_new in self.fragments.keys():
            raise SearchError('Cannot change fragnumber: not empty in fragment', self.report_search)
        else:
            fragment = self.fragments.pop(fragnumber_old)
            self.create_fragment(fragment['fragtype'], fragment['indices'], fragnumber = fragnumber_new)

    def clean_fragments(self):
        """
        Perform a sequence of clean-up task in fragments
            - Remove empty fragments
            - Renumber fragments to remove gaps between two frag numbers
        """
        # remove empty fragments
        for fragnumber, fragment in self.fragments.items():
            if len(fragment['indices']) == 0:
                self.fragments.pop(fragnumber)

        # Renumber fragments
        fragnumbers_old = list(self.fragments.keys())
        for i in range(len(fragnumbers_old)):
            if i != fragnumbers_old[i]:
                self.change_fragnumber(fragnumbers_old[i], i)

    def reduce_structure(self, preserve_single_fragments = True):
        """
        Reduce the system by turning fragments into atoms 
            - reduce connectivity
            - change structure by creating a pymatgen site per fragment

        Args:
            preserve_single_fragments: Bool, if True, will not rename fragments comprised of a single site
        """
        self.make_frag_conn()
        self.symbols.add_names(list(set(self.fragtypes)))
        species = [''] * len(self.fragments)
        coords = [[0. ,0., 0.]] * len(self.fragments)
        for fragnumber, fragment in self.fragments.items():
            # if preserve_single_fragments and len(fragment['indices']) == 1:
            #     i = fragment['indices'][0]
            #     species[fragnumber] = self.struct.sites[i].species
            #     coords[fragnumber] = self.struct.sites[i].coords
            # else:
            #     # species[fragnumber] = fragment['fragtype']
            #     # species[fragnumber] = {fragment['fragtype']:1}
            #     species[fragnumber] = pymatgen.core.DummySpecies('XX' + fragment['fragtype']) 
            #     # The dummy symbol cannot have any part of first two letters that will constitute an Element symbol. Check DummySpecies description for further info
            #     coords[fragnumber] = sadi.structure.get_center_of_mass(self.struct, fragment['indices'])
            species[fragnumber] = self.symbols.get_symbol(fragment['fragtype'])
            coords[fragnumber] = sadi.structure.get_center_of_mass(self.struct, fragment['indices'])            
        reduced_struct = Structure(self.struct.lattice, species, coords, coords_are_cartesian = True)

        # compute cutoffs defining nb with reduced structure from frag_conn
        list_of_nb = list(set([tuple(sorted((i, j))) for i in range(len(self.frag_conn)) for j in self.frag_conn[i]])) # no duplicates
        bonds = np.array(['-'.join(sorted([species[i], species[j]])) for (i, j) in list_of_nb])
        bonds_unique = list(set(bonds))
        distances = np.array([reduced_struct.get_distance(i, j) for (i, j) in list_of_nb])
        nb_set_and_cutoff = {}
        for nb_set in bonds_unique:
            nb_set_and_cutoff[nb_set] = np.max(distances[bonds==nb_set])
        self.report_search['nb_set_and_cutoff'] = str(nb_set_and_cutoff)
        
        # check that every fragment B within A-B cutoff of A is nb of A in frag_conn
        irregular_nb = []
        irregular_nb_offset = []
        nb_list = reduced_struct.get_neighbor_list(max(nb_set_and_cutoff.values()))
        for k in range(len(nb_list)):
            i, j, distance = nb_list[0][k], nb_list[1][k], nb_list[3][k]
            nb_set = '-'.join(sorted([species[i], species[j]]))
            if (j not in self.frag_conn[i]) and (distance < nb_set_and_cutoff[nb_set]):
                irregular_nb.append(nb_set)
                irregular_nb_offset.append(nb_set_and_cutoff[nb_set] - distance)
        self.report_search['connectivity_constructible_with_cutoffs'] = (len(irregular_nb) == 0)
        if len(irregular_nb) != 0:
            self.report_search['connectivity_wrongly_inferred_from_cutoffs'] = str(dict(Counter(irregular_nb).items()))
            self.report_search['connectivity_wrong_offsets'] = str(stats.describe(irregular_nb_offset))

        # self.struct.to(filename='vesta/not_constructible_connectivity/before_reduction.cif')
        # reduced_struct.to(filename='vesta/not_constructible_connectivity/reduced.cif')

        self.report_search['number_of_nodes'] = reduced_struct.num_sites
        self.report_search['symbols'] = str(self.symbols)
        return reduced_struct

    def make_frag_conn(self):
        """
        generate a fragment connectivity

        Adapted from molsys
        """
        self.frag_conn = []        # fragment connectivity (indices of fragments)
        self.frag_conn_atoms = []  # atoms making the fragemnt connectivity (tuples of atom indices: first in frgmnt, second in other fragmnt)
        # prepare the atom list for the fragments
        for i in range(len(self.fragments)):
            self.frag_conn.append([])
            self.frag_conn_atoms.append([])
        for i, fragment in self.fragments.items():
        # for i,f in enumerate(self.fraglist):
            # determine all external bonds of this fragment
            for ia in fragment['indices']:
                for ja in self.conn[ia]:
                    j = self.fragnumbers[ja]
                    if i != j:
                        # this is an external bond
                        self.frag_conn[i].append(j)
                        self.frag_conn_atoms[i].append((ia,ja))
                        # logger.debug("fragment %d (%s) bonds to fragment %d (%s) %s - %s" %\
                        #               (i, fragment['fragtype'], j, self.fraglist[j], self._mol.atypes[ia], self._mol.atypes[ja]))

    def get_atype(self, i):
        """return atype of atom i formatted as in molsys"""
        atype = self.elems[i] + str(len(self.conn[i])) 
        list_of_nn = [self.elems[i] for j in self.conn[i]]
        counts = Counter(list_of_nn)
        list_of_counts = sorted(counts.items(), key= lambda t: (t[0], t[1])) # sort by alphabetical order to have a unique atype per coordination environment
        atype += "_" + ''.join(str(e) for pair in list_of_counts for e in pair)
        return atype

    def update_atypes(self):
        """update self.atypes"""
        self.atypes = [self.get_atype(i) for i in range(self.struct.num_sites)]

    covalentradius = CovalentRadius().radius # private class variable exclusively used in get_covdist

    def get_covdist(self, i, j): # turn to class at some point to have struct as self. 
        """return sum of covalent radii of the two atoms referenced by their index i and j"""
        return self.covalentradius[self.elems[i].title()] + self.covalentradius[self.elems[j].title()]

    def add_ABbonds(self, graph, A, B, dist_margin = None):
        """
        Add bonds between species A and B to graph graph
        dist_margin: margin to consider neighbourings atoms dist_margin further away than their cov distance
        """
        if dist_margin is None:
            dist_margin = self.dist_margin
        for i in range(self.struct.num_sites):
            if self.struct[i].species == A:
                for site in self.all_neighb[i]:
                    j = site.index
                    if self.struct[j].species == B and site.nn_distance < dist_margin*self.get_covdist(i, j):
                        graph.add_edge(i, j, from_jimage=(0, 0, 0), to_jimage=(0, 0, 0), weight = self.struct.get_distance(i, j), warn_duplicates=False)

    @staticmethod
    def multigraph_to_graph(MG):
        """from networkx tutorial, propagate min of weights"""
        GG = nx.Graph()
        for n, nbrs in MG.adjacency():
            for nbr, edict in nbrs.items():
                minvalue = min([d['weight'] for d in edict.values()])
                GG.add_edge(n, nbr, weight = minvalue)
        return GG

    @classmethod
    def get_chain_decomposition(cls, graph):
        """return chain decomposition as list of list from pymatgen graph"""
        GG = cls.multigraph_to_graph(graph.graph) # pymatgen graph is a multigraph and directed, graph.graph is a nx object
        # test = cls.find_one_cycle_per_node(graph)
        # testb = cls.find_rings(graph)
        return list(chain_decomposition(GG)) # direct use of nx function

    @classmethod
    def find_one_cycle_per_node(cls, graph):
        """for each node, if it exists return a cycle found via depth-first traversal in which node is included.
        Returns:
            cycles_list: list of list of edges"""
        GG = cls.multigraph_to_graph(graph.graph) # pymatgen graph is a multigraph and directed, graph.graph is a nx object
        cycles_list = []
        nodes = list(set(GG.nodes))
        node_in_cycle = {i:False for i in nodes}
        for i in nodes:
            if not node_in_cycle[i]:
                try:
                    cycle = nx.find_cycle(GG, i)
                    cycle_edges = set(itertools.chain.from_iterable(cycle))
                    if i in cycle_edges:
                        for j in cycle_edges:
                            node_in_cycle[j] = True
                        cycles_list.append(cycle)
                except nx.exception.NetworkXNoCycle:
                    pass
        return cycles_list

    @staticmethod
    def are_circularly_identical(arr1, arr2):
        """
        Checks whether two lists of int are circularly identical

        Code from https://stackoverflow.com/a/26924896/8288189
        """
        if len(arr1) != len(arr2):
            return False

        str1 = ' '.join(map(str, arr1))
        str2 = ' '.join(map(str, arr2))
        if len(str1) != len(str2):
            return False

        return str1 in str2 + ' ' + str2

    def find_rings(self, graph, including=None, max_depth = None, exit_if_large_cycle = False, 
            pattern=None, target_number_of_rings = None, exit_if_too_many_rings=False):
        """                
        Find ring structures in the StructureGraph.

        Forked from MoleculeGraph class in pymatgen.analysis.graph
        A ring is defined as a simple cycle in networkx terms
        A simple cycle, or elementary circuit, is a closed path where no node appears twice. Two elementary circuits are distinct if they are not cyclic permutations of each other.

        :param including: list of site indices. If
            including is not None, then find_rings will
            only return those rings including the specified
            sites. By default, this parameter is None, and
            all rings will be returned.
            max_depth: int, maximum cycle size to be found
            exit_if_large_cycle: Bool, if True will raise an Exception if a cycle larger than max_depth is found
            pattern: list of strings representing self.elems to match ('n', 'c', etc.)
            target_number_of_rings: int, number of unique rings
            exit_if_too_many_rings: bool

        :return: dict {index:cycle}. Each
            entry will be a ring (cycle, in graph theory terms) including the index
            found in the Molecule. If there is no cycle including an index, the
            value will be an empty list.
        """

        # Copies graph such that all edges (u, v) matched by edges (v, u)
        undirected = graph.graph.to_undirected()
        directed = undirected.to_directed()

        cycles_nodes = []
        cycles_edges = []

        # Remove all two-edge cycles
        # all_cycles = [c for c in nx.simple_cycles(directed) if len(c) > 2]
        if max_depth is not None:
            all_cycles = []
            for c in nx.simple_cycles(directed):
                if len(c) > 2 and len(c) <= max_depth:
                    all_cycles.append(c)
                    # print(len(all_cycles))
                elif exit_if_large_cycle and len(c) > max_depth:
                    # print(c)
                    raise SearchError('max_depth exceeded in cycle search', self.report_search)
        elif pattern is not None:
            all_cycles = []
            for c in nx.simple_cycles(directed):
                if len(c) == len(pattern):
                    c_pattern = [self.elems[i] for i in c]
                    if self.are_circularly_identical(c_pattern, pattern):
                        all_cycles.append(c)
                        # print(len(all_cycles))
                if exit_if_too_many_rings and len(all_cycles) > target_number_of_rings * 2:
                    # self.plot_conn_as_graph() 
                    raise SearchError('target_number_of_rings exceeded in pattern cycle search', self.report_search)            
        else:
            all_cycles = [c for c in nx.simple_cycles(directed) if len(c) > 2]

        # Using to_directed() will mean that each cycle always appears twice
        # So, we must also remove duplicates
        unique_sorted = []
        unique_cycles = []
        for cycle in all_cycles:
            if sorted(cycle) not in unique_sorted:
                unique_sorted.append(sorted(cycle))
                unique_cycles.append(cycle)

        if including is None:
            cycles_nodes = unique_cycles
        else:
            for i in including:
                for cycle in unique_cycles:
                    if i in cycle and cycle not in cycles_nodes:
                        cycles_nodes.append(cycle)

        for cycle in cycles_nodes:
            edges = []
            for i, e in enumerate(cycle):
                edges.append((cycle[i - 1], e))
            cycles_edges.append(edges)

        return cycles_edges

    def clean_conn(self):
        """
        Remove duplicates indices in conn
        """
        for i in range(len(self.conn)):
            self.conn[i] = list(set(self.conn[i]))

    def plot_conn_as_graph(self, filename = "graph_temp.png"):
        """create graph with every bond present in conn and print it in file"""
        filename = sadi.files.path.append_suffix(filename, 'png')
        graph = StructureGraph.with_empty_graph(self.struct)
        for i in range(self.struct.num_sites):
            for j in self.conn[i]:
                graph.add_edge(i, j, from_jimage=(0, 0, 0), to_jimage=(0, 0, 0), weight = self.struct.get_distance(i, j), warn_duplicates=False)
        graph.draw_graph_to_file(filename, hide_unconnected_nodes=False, hide_image_edges =False, algo='neato') #algo= neato or fdp
        return graph

    def get_A_Bbonds(self, A, B):
        """
        A, B are species
        return a list containing for each atom A the the number of bonds to a B atom, and -1 if not A atom
        """
        A_Bbonds = [-1 for i in range(self.struct.num_sites)]
        for i in range(self.struct.num_sites):
            if self.struct[i].species == A:
                A_Bbonds[i] = 0
                for j in self.conn[i]:
                    if self.struct[j].species == B:
                        A_Bbonds[i] += 1
        return A_Bbonds

    def assign_B_uniquely_to_A_N_coordinated(self, conditionA, conditionB, target_N, use_cov_dist = True, dist_margin=None, report_level = None, report_entry = None, propagate_fragments = False, new_fragments_name = None):
        """
        assign atoms B to atoms A so that B is only assigned once and A end up being at least target_N coordinated. 
        At each step, the closest pair A-B is added to conn
        B can only form new bonds with A.
        conditionA/B are functions of index i in struct
        allowed to exit if not enough nn available
        If an A atom already has more than target_n neighbours, will do nothing
        use_cov_dist can be used to further restrict the search using cov radii, if False, every nn in all_neighb used
        if a string report_entry is supplied, will use it to add as an entry details about the missing nn
        report_level can be 'full' (report every bond formed) or 'undercoordinated' (only those without enough nn)

        Args:
            propagate_fragments: Bool, if True will include every atom of frag(B) in fragment of A if frag(A) exists
                If atom B was in a fragment F, F will be removed
            new_fragments_name: str, if not None will create fragments for atoms A not currently in a fragment with fragname "new_fragments_name". 
                To include B atoms in this new fragment, propagate_fragments must be set to True
        """
        if dist_margin is None:
            dist_margin = self.dist_margin

        def are_A_Ncoordinated(A_conn, A_enough_nn):
            """return False as long as the main loop should keep running"""
            for i in range(len(A_conn)):
                # if len(A_conn[i])>target_N:
                #     logger.warning('len(A_conn[i])>target_N')
                # if (len(A_conn[i])!=target_N) and (A_enough_nn[i]==True):
                if (len(A_conn[i])<target_N) and (A_enough_nn[i]==True):
                    return False
            return True

        # Creation of A_$var variables of length num_of_A_atoms
        A_indices = [] # index of A atom in struct
        A_neighb_indices = [] # indices of nn of A in struct, sorted by increasing distance to A
        A_nn_distances = [] # corresponding distances
        A_conn = [] # contains a temporary copy of conn
        A_enough_nn = [] # True iff A_nn_distances is not empty
        A_new_nb = [] # contains B atoms that will be bounded to A (were not previously in conn)
        
        for i in range(self.struct.num_sites):
            if conditionA(i): 
                A_indices.append(i)
                A_conn.append(deepcopy(self.conn[i])) # starts with preexisting conn
                A_new_nb.append([])
                neighb_set = self.all_neighb[i]
                neighb_set = [n for n in neighb_set if conditionB(n.index)] 
                if use_cov_dist==True:
                    neighb_set = [n for n in neighb_set if n.nn_distance < dist_margin * self.get_covdist(i, n.index)]
                nn_distances = [neighb.nn_distance for neighb in neighb_set]
                sorted_index = np.argsort(nn_distances) # sort nn by distance to nn atom
                A_neighb_indices.append([neighb_set[sorted_index[j]].index for j in range(len(sorted_index))]) # ranked by increasing distance
                A_nn_distances.append([nn_distances[sorted_index[j]] for j in range(len(sorted_index))])
                A_enough_nn.append(not (len(A_conn[-1])<target_N and len(A_nn_distances[-1])==0)) # initialise enough_nn with the current (ie last) element in A_conn and A_nn_distances
        
        while not are_A_Ncoordinated(A_conn, A_enough_nn):
            # take the closest A-B pair and bond them 
            choose_min = [] # candidate of atom A i to be selected
            for i in range(len(A_indices)):
                # if len(A_conn[i])==target_N or A_enough_nn[i]==False:
                if len(A_conn[i])>=target_N or A_enough_nn[i]==False:
                    choose_min.append(np.inf) # so that it is not chosen as min
                else:
                    if A_enough_nn[i] == True and len(A_nn_distances[i]) == 0:
                        a = 1
                    choose_min.append(A_nn_distances[i][0]) # the closest nn is the first as A_nn_distances is sorted
            imin = np.argmin(choose_min)
            B_imin = A_neighb_indices[imin][0]
            A_conn[imin].append(B_imin) # add to A_conn
            A_new_nb[imin].append(B_imin)
            for i in range(len(A_indices)): # remove B_imin as potential candidate for every A atom
                while B_imin in A_neighb_indices[i]:
                    A_nn_distances[i].pop(A_neighb_indices[i].index(B_imin))
                    A_neighb_indices[i].remove(B_imin)
            # flag A atoms missing nn
            for i in range(len(A_indices)):
                # if A_enough_nn[i]==True and len(A_conn[i])<target_N and len(A_nn_distances[i])==0:
                if A_enough_nn[i]==True and len(A_nn_distances[i])==0:
                    A_enough_nn[i]=False
                    # logger.debug("not enough nn for atom id:", A_indices[i], "; N bonds:", C_Nbonds[A_indices[i]], "; missing bonds:", target_N - len(A_conn[i]))
        
        # add A_conn to self.conn
        for i in range(len(A_indices)): 
            A = A_indices[i]
            self.conn[A]=A_conn[i]
            for n in A_conn[i]:
                if A not in self.conn[n]: # if conn wasn't empty at the beginning
                    self.conn[n].append(A)
        

        if report_level=='full':
            list_of_atypes = [self.get_atype(i) for i in range(self.struct.num_sites) if conditionA(i)]
            self.report_search[report_entry] = Counter(list_of_atypes).most_common() # return list sorted by decreasing number of occurances
            if self.report_search[report_entry] != []:
                logger.debug("%s: %s", report_entry, self.report_search[report_entry])

        if report_level=='undercoordinated':
            list_of_atypes = [self.get_atype(i) for i in range(len(A_indices)) if (len(A_conn[i])!=target_N)]
            self.report_search[report_entry] = Counter(list_of_atypes).most_common() # return list sorted by decreasing number of occurances
            if self.report_search[report_entry] != []:
                logger.debug("%s: %s", report_entry, self.report_search[report_entry])

        # # report coordination (formatted as atype) of every atom that endup not having enough nn
        # if report_entry is not None:
        #     list_of_atypes = [self.get_atype(A_indices[i]) for i in range(len(A_indices)) if A_enough_nn[i]==False]
        #     self.report_search[report_entry] = Counter(list_of_atypes).most_common() # return list sorted by decreasing number of occurances
        #     if self.report_search[report_entry] != []:
        #         logger.debug("%s: %s", report_entry, self.report_search[report_entry])

        # Create new fragments for A atoms
        if new_fragments_name is not None:
            for a in A_indices:
                if not self.in_fragment(a):
                    self.create_fragment(new_fragments_name, [a])
        
        # Propagate fragments
        if propagate_fragments:
            for i in range(len(A_indices)): 
                a = A_indices[i]
                if self.in_fragment(a):
                    for b in A_new_nb[i]:
                        if not self.in_fragment(b):
                            self.add_to_fragment(self.fragnumbers[a], [b])
                        else:
                            self.merge_fragments(self.fragnumbers[a], self.fragnumbers[b])

    def get_neighb_cov_dist(self, i, dist_margin=None):
        """return list of nn of atom i using a cutoff based on covalent radii * dist_margin"""
        if dist_margin is None:
            dist_margin = self.dist_margin
        return [n for n in self.all_neighb[i] if n.nn_distance < dist_margin * self.get_covdist(i, n.index)]

    def find_N_closest_cov_dist(self, conditionA, conditionB, target_N, dist_margin = None, verbose = False, report_level = None, report_entry = None, propagate_fragments = False, new_fragments_name = None):
        """
        Find target_N nearest neighbours respecting conditionB to conditionA atoms
        The search for each A is independant, as a result the same B atom can be bonded to several A atoms
        report_level can be 'full' (report every bond formed) or 'undercoordinated' (only those without enough nn)

        Args:
            propagate_fragments: Bool or str if True will include every atom of frag(B) in fragment of A if frag(A) exists
                If atom B was in a fragment F, F will be removed
                If 'reverse' and target_N = 1, will propagate from B to A
            new_fragments_name: str, if not None will create fragments for atoms A not currently in a fragment with fragname "new_fragments_name". 
                To include B atoms in this new fragment, propagate_fragments must be set to True
        """
        if dist_margin is None:
            dist_margin = self.dist_margin

        list_of_undercoordinated = []

        A_indices = [] # index of A atom in struct
        A_new_nb = [] # contains B atoms that will be bounded to A (were not previously in conn)

        for i in range(self.struct.num_sites):
            if conditionA(i):
                A_indices.append(i)
                new_nb = []
                neighb_set = self.get_neighb_cov_dist(i, dist_margin)
                neighb_set = [s for s in neighb_set if conditionB(s.index)]
                if len(neighb_set)<target_N:
                    logger.debug("not enough nn for %s: %s instead of %s",i, len(neighb_set), target_N)
                    for s in neighb_set:
                        logger.debug("%s %s distance: %s conn: %s", str(s.species), s.index, s.nn_distance, str([(str(self.struct[j].species), j) for j in self.conn[s.index]]))
                    list_of_undercoordinated.append(i)
                # take the closest as nn
                nn_distances = [neighb.nn_distance for neighb in neighb_set]
                sorted_index = np.argsort(nn_distances)
                for j in range(min(target_N, len(neighb_set))):
                    nn=neighb_set[sorted_index[j]]
                    new_nb.append(nn.index)
                    self.conn[i].append(nn.index)
                    self.conn[nn.index].append(i)
                A_new_nb.append(new_nb)
        
        if report_level=='full':
            list_of_atypes = [self.get_atype(i) for i in range(self.struct.num_sites) if conditionA(i)]
            self.report_search[report_entry] = Counter(list_of_atypes).most_common() # return list sorted by decreasing number of occurances
            if self.report_search[report_entry] != []:
                logger.debug("%s: %s", report_entry, self.report_search[report_entry])

        if report_level=='undercoordinated':
            list_of_atypes = [self.get_atype(i) for i in list_of_undercoordinated]
            self.report_search[report_entry] = Counter(list_of_atypes).most_common() # return list sorted by decreasing number of occurances
            if self.report_search[report_entry] != []:
                logger.debug("%s: %s", report_entry, self.report_search[report_entry])
        
        # Create new fragments for A atoms
        if new_fragments_name is not None:
            for a in A_indices:
                if not self.in_fragment(a):
                    self.create_fragment(new_fragments_name, [a])

        # Propagate fragments
        if propagate_fragments == True:
            for i in range(len(A_indices)): 
                a = A_indices[i]
                if self.in_fragment(a):
                    for b in A_new_nb[i]:
                        if not self.in_fragment(b):
                            self.add_to_fragment(self.fragnumbers[a], [b])
                        else:
                            self.merge_fragments(self.fragnumbers[a], self.fragnumbers[b])
        elif propagate_fragments == 'reverse':
            if target_N != 1:
                raise SearchError("Propagation ambiguous: Tried to propagate fragment from B to A with target_N not equal to 1", self.report_search) 
            else:
                for i in range(len(A_indices)): 
                    a = A_indices[i]
                    if len(A_new_nb[i]) == 1:
                        b = A_new_nb[i][0]
                        if not self.in_fragment(a):
                            self.add_to_fragment(self.fragnumbers[b], [a])
                        else:
                            self.merge_fragments(self.fragnumbers[b], self.fragnumbers[a])
                        



