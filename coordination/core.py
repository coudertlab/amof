#-*- coding: utf-8 -*-

"""
Main file containing classes for custom neigbour search
"""

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

import logging

import sadi.files.path

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class CoordinationSearch(object):
    """
    Classes containing general methods to perform a coordination search
    The actual search is to be launched from a subclass mentionning specific species, etc.

    Attributes:
        fragments: list of dict;  element i of list is fragment of fragnumber i
        atypes, fragtypes, fragnumbers, conn: same structure as molsys.mol object
    """
        
    def __init__(self, struct, neighb_max_distance, dist_margin):
        """constructor"""
        self.struct = struct
        self.conn = [[] for i in range(struct.num_sites)]
        self.atypes = ["" for i in range(struct.num_sites)]
        self.fragtypes = ["-1" for i in range(struct.num_sites)]
        self.fragnumbers = [-1 for i in range(struct.num_sites)]
        self.fragments = [] 
        self.all_neighb = self.struct.get_all_neighbors(neighb_max_distance)
        self.dist_margin = dist_margin
        # initialize report_search with useful descriptors of struct for subsequent report analysis
        self.report_search = {"number of atoms":self.struct.num_sites}

    def create_fragment(self, fragtype, indices):
        """
        Create new fragment

        Args:
            fragtype: str
            indices: list of int
        """
        fragnumber = len(self.fragments)
        indices = list(set(indices)) # remove duplicates
        fragment = {"fragnumber":fragnumber, "fragtype":fragtype, "indices":indices}
        for i in indices:
            self.fragtypes[i] = fragtype
            self.fragnumbers[i] = fragnumber
        self.fragments.append(fragment)

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

    def get_atype(self, i):
        """return atype of atom i formatted as in molsys"""
        atype = str(self.struct[i].species)[:-1].lower() + str(len(self.conn[i])) 
        list_of_nn = [str(self.struct[j].species)[:-1].lower() for j in self.conn[i]]
        counts = Counter(list_of_nn)
        list_of_counts = sorted(counts.items(), key= lambda t: (t[0], t[1])) # sort by alphabetical order to have a unique atype per coordination environment
        atype += "_" + ''.join(str(e) for pair in list_of_counts for e in pair)
        return atype

    def update_atypes(self):
        """update self.atypes"""
        self.atypes = [self.get_atype(i) for i in range(self.struct.num_sites)]

    covalentradius = CovalentRadius().radius # private class variable exclusively used in get_covdist

    def get_covdist(self, i, j): # turn to class at some poitn to have struct as self. 
        """return sum of covalent radii of the two atoms referenced by their index i and j"""
        return self.covalentradius[str(self.struct[i].specie)] + self.covalentradius[str(self.struct[j].specie)]

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

    @classmethod    
    def find_rings(cls, graph, including=None):
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

    def assign_B_uniquely_to_A_N_coordinated(self, conditionA, conditionB, target_N, use_cov_dist = True, dist_margin=None, report_entry = None):
        """
        assign atoms B to atoms A so that B is only assigned once and A end up being target_N coordinated. 
        At each step, the closest pair A-B is added to conn
        B can only form new bonds with A.
        conditionA/B are functions of index i in struct
        allowed to exit if not enough nn
        use_cov_dist can be used to further restrict the search using cov radii, if False, every nn in all_neighb used
        if a string report_entry is supplied, will use it to add as an entry details about the missing nn
        """
        if dist_margin is None:
            dist_margin = self.dist_margin

        def are_A_Ncoordinated(A_conn, A_enough_nn):
            """return False as long as the main loop should keep running"""
            for i in range(len(A_conn)):
                if (len(A_conn[i])!=target_N) and (A_enough_nn[i]==True):
                    return False
            return True

        # Creation of A_$var variables of length num_of_A_atoms
        A_indices = [] # index of A atom in struct
        A_neighb_indices = [] # indices of nn of A in struct, sorted by increasing distance to A
        A_nn_distances = [] # corresponding distances
        A_conn = [] # contains a temporary copy of conn
        A_enough_nn = [] # True iff A_nn_distances is not empty
        
        for i in range(self.struct.num_sites):
            if conditionA(i): 
                A_indices.append(i)
                A_conn.append(deepcopy(self.conn[i])) # starts with preexisting conn
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
                if len(A_conn[i])==target_N or A_enough_nn[i]==False:
                    choose_min.append(np.inf) # so that it is not chosen as min
                else:
                    choose_min.append(A_nn_distances[i][0]) # the closest nn is the first as A_nn_distances is sorted
            imin = np.argmin(choose_min)
            B_imin = A_neighb_indices[imin][0]
            A_conn[imin].append(B_imin) # add to A_conn
            for i in range(len(A_indices)): # remove B_imin as potential candidate for every A atom
                while B_imin in A_neighb_indices[i]:
                    A_nn_distances[i].pop(A_neighb_indices[i].index(B_imin))
                    A_neighb_indices[i].remove(B_imin)
            # flag A atoms missing nn
            for i in range(len(A_indices)):
                if A_enough_nn[i]==True and len(A_conn[i])<target_N and len(A_nn_distances[i])==0:
                    A_enough_nn[i]=False
                    # logger.info("not enough nn for atom id:", A_indices[i], "; N bonds:", C_Nbonds[A_indices[i]], "; missing bonds:", target_N - len(A_conn[i]))
        
        # add A_conn to self.conn
        for i in range(len(A_indices)): 
            A = A_indices[i]
            self.conn[A]=A_conn[i]
            for n in A_conn[i]:
                if A not in self.conn[n]: # if conn wasn't empty at the beginning
                    self.conn[n].append(A)
        
        # report coordination (formatted as atype) of every atom that endup not having enough nn
        if report_entry is not None:
            list_of_atypes = [self.get_atype(A_indices[i]) for i in range(len(A_indices)) if A_enough_nn[i]==False]
            self.report_search[report_entry] = Counter(list_of_atypes).most_common() # return list sorted by decreasing number of occurances
            logger.info("%s: %s", report_entry, self.report_search[report_entry])

    def get_neighb_cov_dist(self, i, dist_margin=None):
        """return list of nn of atom i using a cutoff based on covalent radii * dist_margin"""
        if dist_margin is None:
            dist_margin = self.dist_margin
        return [n for n in self.all_neighb[i] if n.nn_distance < dist_margin * self.get_covdist(i, n.index)]

    def find_N_closest_cov_dist(self, conditionA, conditionB, target_N, dist_margin = None, verbose = False, report_level = None, report_entry = None):
        """
        Find target_N nearest neighbours respecting conditionB to conditionA atoms
        The search for each A is independant, as a result the same B atom can be bonded to several A atoms
        report_level can be 'full' (report every bond formed) or 'undercoordinated' (only those without enough nn)
        """
        if dist_margin is None:
            dist_margin = self.dist_margin

        list_of_undercoordinated = []

        for i in range(self.struct.num_sites):
            if conditionA(i):
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
                    self.conn[i].append(nn.index)
                    self.conn[nn.index].append(i)
        
        if report_level=='full':
            list_of_atypes = [self.get_atype(i) for i in range(self.struct.num_sites) if conditionA(i)]
            self.report_search[report_entry] = Counter(list_of_atypes).most_common() # return list sorted by decreasing number of occurances
            logger.info("%s: %s", report_entry, self.report_search[report_entry])

        if report_level=='undercoordinated':
            list_of_atypes = [self.get_atype(i) for i in list_of_undercoordinated]
            self.report_search[report_entry] = Counter(list_of_atypes).most_common() # return list sorted by decreasing number of occurances
            logger.info("%s: %s", report_entry, self.report_search[report_entry])

