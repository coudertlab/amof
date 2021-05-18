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

import logging

# create logger without parameters for this module file that will be incorporated by the main file logging parameters
logger = logging.getLogger(__name__)

class CoordinationSearch(object):
    """
    Classes containing general methods to perform a coordination search
    The actual search is to be launched from a subclass mentionning specific species, etc.
    """
        
    def __init__(self, struct, neighb_max_distance, dist_margin):
        """constructor"""
        self.struct = struct
        self.conn = [[] for i in range(struct.num_sites)]
        self.atypes = ["" for i in range(struct.num_sites)]
        self.all_neighb = self.struct.get_all_neighbors(neighb_max_distance)
        self.dist_margin = dist_margin
        # initialize report_search with useful descriptors of struct for subsequent report analysis
        self.report_search = {"number of atoms":self.struct.num_sites}

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
        return list(chain_decomposition(GG)) # direct use of nx function

    def plot_conn_as_graph(self, filename = "graph_temp.png"):
        """create graph with every bond present in conn and print it in file"""
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



class CustomSearchTwo(CoordinationSearch):
    """
    Only for ZIF8
    Principle:
        1. find every cycle of CNCNC
        2. add H based on cov radii to single C and C bonded to one N
        3. bind the remaining H (there should be non for the crystal)
        4. bind the CHn to CN2
        5. bind N and Zn
    """

    def __init__(self, struct, dist_margin = 1.2, dist_margin_zn = 1.5):
        """
        Constructor for custom search 2 for ZIF8 glasses
        dist_margin is the default tolerance when using the covalent radii criteria to determine if two atoms are neighbours
        dist_margin_zn is the specific tolerance for Zn-X bonds as they're not covalent
        """
        self.dist_margin_zn = dist_margin_zn # dist_maring is created in CoordinationSearch.__init__
        neighb_max_distance = self.set_neighb_max_distance(dist_margin, dist_margin_zn)
        CoordinationSearch.__init__(self, struct, neighb_max_distance, dist_margin)
        self.detect_conn()
        self.update_atypes()

    def set_neighb_max_distance(self, dist_margin, dist_margin_zn):
        """set neighb_max_distance to compute all_neighb to the minimal necessary distance"""
        dist_margin_atoms = ['H', 'C', 'N']
        dist_margin_zn_atoms = ['Zn']
        max_cov_organic = np.max([self.covalentradius[A]+self.covalentradius[B] for A in dist_margin_atoms for B in dist_margin_atoms])
        max_cov_zn = np.max([self.covalentradius[A]+self.covalentradius[B] for A in dist_margin_zn_atoms for B in (dist_margin_atoms+dist_margin_zn_atoms)])
        return max(max_cov_organic * dist_margin, max_cov_zn * dist_margin_zn)

    def detect_conn(self):
        """
        main function to detect connectivity
        """
        # shorthand to select different atoms in pymatgen
        Zn = pymatgen.Composition("Zn1")
        H =  pymatgen.Composition("H1")
        C = pymatgen.Composition("C1")
        N = pymatgen.Composition("N1")

        ### Find imid cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.add_ABbonds(graph,N, C)
        self.add_ABbonds(graph,C, C)
        cycles = self.get_chain_decomposition(graph)

        # check sanity of found cycles
        target_number_of_cycles = int(self.struct.num_sites*24/276)
        self.report_search['imid_expected_number_of_cycles'] = (len(cycles) == target_number_of_cycles)
        if not self.report_search['imid_expected_number_of_cycles']:
            logger.warning("number of cycles incorrect")

        cycles_not_of_five = [c for c in cycles if len(c)!=5]
        self.report_search['imid_expected_length_of_cycles'] = (len(cycles_not_of_five) == 0)
        if not self.report_search['imid_expected_length_of_cycles']:
            logger.warning("cycle not of 5 atoms found: %s", cycles_not_of_five)  

        in_cycle = [False for i in range(self.struct.num_sites)]
        self.report_search['imid_atoms_appear_only_once_in_cycles'] = True
        for c in cycles:
            for a, b in c:
                self.conn[a].append(b)
                self.conn[b].append(a)
                if in_cycle[a]==True:
                    logger.warning("atom %s appears in more than one cycle", a)
                    self.report_search['imid_atoms_appear_only_once_in_cycles'] = False
                in_cycle[a]=True

        ### add H based on cov radii to single C and C bonded to one N
        C_Nbonds = self.get_A_Bbonds(C, N)

        logger.debug("number of N nn to C atoms") # debug check
        for i in range(3):
            logger.debug("%s C atoms have %s N nn", C_Nbonds.count(i), i)

        self.assign_B_uniquely_to_A_N_coordinated(lambda i: (C_Nbonds[i] in [0,1]), lambda i: (self.struct[i].species == H),   3, report_entry="C atoms missing H neighbours")

        ### bind the remaining H (there should be non for the crystal)
        H_Cbonds = self.get_A_Bbonds(H, C)    
        self.find_N_closest_cov_dist(lambda i: H_Cbonds[i]==0, lambda i: True, 1, report_level='full', report_entry="H atoms not bonded to C")    
        
        ### link C in cycles (bonded to 2 N) to C bonded to H
        self.find_N_closest_cov_dist(lambda i: C_Nbonds[i]==0, lambda i: C_Nbonds[i]==2, 1, report_level='undercoordinated', report_entry="C in CHn not bonded to any C in imid")        
        
        ### link N to Zn with no constraint on the number of N to Zn
        self.find_N_closest_cov_dist(lambda i: self.struct[i].species==Zn, lambda i: self.struct[i].species==N, 4, dist_margin=self.dist_margin_zn, report_level='undercoordinated', report_entry="undercoordinated Zn")        
