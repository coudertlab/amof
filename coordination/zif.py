"""
Main file containing classes for custom neigbour search in ZIF glasses
Supported: ZIF-8
"""

from numpy import False_
from sadi.coordination import *
import sadi.coordination.buildingunits as bu


class ZifSearch(CoordinationSearch):
    """
    Generic coordination search class for ZIFs comprised of 
        single metal nodes
        imid-based linkers
    Provides generic methods used for all these systems
    Principle:
        1. find every cycle of CNCNC
        2. add H based on cov radii to single C and C bonded to one N
        3. bind the remaining H (there should be non for the crystal)
        4. bind the CHn to CN2
        5. bind N and Zn
    """

    def __init__(self, struct, dist_margin=1.2, dist_margin_metal=1.5, cutoff_metal = None):
        """
        Constructor for custom search 2 for ZIF glasses

        Args:
            dist_margin is the default tolerance when using the covalent radii criteria to determine if two atoms are neighbours
            dist_margin_metal is the specific tolerance for metal-X bonds as they're not covalent
            cutoff_metal: float, overide dist_margin_metal if not None 
                option added but not used so far
        """
        self.dist_margin_metal = dist_margin_metal  # dist_maring is created in CoordinationSearch.__init__
        neighb_max_distance = self.find_neighb_max_distance(
            dist_margin, dist_margin_metal, cutoff_metal)
        CoordinationSearch.__init__(
            self, struct, neighb_max_distance, dist_margin)
        self.detect_conn()
        self.clean_fragments()
        self.update_atypes()

    def find_neighb_max_distance(self, dist_margin, dist_margin_metal, cutoff_metal):
        """return neighb_max_distance to compute all_neighb to the minimal necessary distance"""
        dist_margin_atoms = self.linker.species
        dist_margin_metal_atoms = self.node.species
        max_cov_linker = np.max([self.covalentradius[A]+self.covalentradius[B]
                                 for A in dist_margin_atoms for B in dist_margin_atoms])
        if cutoff_metal is None:
            max_cov_metal = np.max([self.covalentradius[A]+self.covalentradius[B]
                                for A in dist_margin_metal_atoms for B in (dist_margin_atoms+dist_margin_metal_atoms)])
        else: 
            max_cov_metal = cutoff_metal
        return max(max_cov_linker * dist_margin, max_cov_metal * dist_margin_metal)

    def find_ABAcycles(self, A, B, cycle_length, target_number_of_cycles, fragtype = None):
        """
        Find every cycle of the form ABA...BA of length cycle_length for species A and B
        Add the bonds thus found to self.conn
        A can be neighbour to A and B, and B only to A. 
        NOT IMPLEMENTED YET: Only one A-A bond is allowed > check
        For imid, A is C and B is N
        If fragtype is not None, will create new fragments with this fragmame

        Args:
            A, B: string, lowecase string indicating species of A and B (e.g. "h")
            cycle_length: int
            fragtype: str
        """
        # Find cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.add_ABbonds(graph, B, A)
        self.add_ABbonds(graph, A, A)
        # cycles = self.get_chain_decomposition(graph) # give same results for the tested files
        # cycles = self.find_one_cycle_per_node(graph) # give same results for the tested files

        # # time comparison (commented, rm when cleaning)
        # import time
        # start = time.time()
        # try:
        #     cycles = self.find_rings(graph, max_depth=cycle_length, exit_if_large_cycle=False)
        # except SearchError:
        #     pass
        # end = time.time()
        # print("No pattern", end - start)

        # # currently very slow
        # start = time.time()
        # try:
        #     pattern = [str(A)[:-1].lower()] + [str(B)[:-1].lower(), str(A)[:-1].lower()] * int((cycle_length - 1) / 2)
        #     cycles = self.find_rings(graph, pattern=pattern, target_number_of_rings = target_number_of_cycles, exit_if_too_many_rings=False)
        # except SearchError:
        #     pass
        # end = time.time()
        # print("Pattern", end - start)

        pattern = [A] + [B, A] * int((cycle_length - 1) / 2)
        cycles = self.find_rings(graph, pattern=pattern, target_number_of_rings = target_number_of_cycles, 
            exit_if_too_many_rings=False, remove_overlapping_rings=True)

        # check sanity of found cycles
        self.report_search['imid_expected_number_of_cycles'] = (
            len(cycles) == target_number_of_cycles)
        if not self.report_search['imid_expected_number_of_cycles']:
            logger.debug("number of cycles incorrect")
            self.report_search['imid_missing_cycles'] = target_number_of_cycles - len(cycles)

        # TD: remove when sure that pattern search is working fine
        cycles_of_wrong_size = [c for c in cycles if len(c) != cycle_length]
        self.report_search['imid_expected_length_of_cycles'] = (
            len(cycles_of_wrong_size) == 0)
        if not self.report_search['imid_expected_length_of_cycles']:
            logger.debug("cycle not of %s atoms found: %s", cycle_length, 
                           cycles_of_wrong_size)

        in_cycle = [False for i in range(self.struct.num_sites)]
        self.report_search['imid_atoms_appear_only_once_in_cycles'] = True
        for c in cycles:
            for a, b in c:
                self.conn[a].append(b)
                self.conn[b].append(a)
                if in_cycle[a] == True:
                    logger.debug("atom %s appears in more than one cycle", a)
                    self.report_search['imid_atoms_appear_only_once_in_cycles'] = False
                in_cycle[a] = True
        self.clean_conn()
        if fragtype is not None:
            for c in cycles:
                indices = list(set(itertools.chain.from_iterable(c)))
                self.create_fragment(fragtype, indices)
        
        self.report_search['imid_search_successful'] = self.report_search['imid_atoms_appear_only_once_in_cycles'] and self.report_search['imid_expected_number_of_cycles'] and self.report_search['imid_expected_length_of_cycles']

class MetalmIm(ZifSearch):
    """
    Search coordination for ZIFs made of metallic node and mIm (methyldimidazolate, C4N2H5) ligands
    Supports: ZIF-8
    Principle:
        1. find every cycle of CNCNC
        2. add H based on cov radii to single C and C bonded to one N
        3. bind the remaining H (there should be non for the crystal)
        4. bind the CHn to CN2
        5. bind N and Zn
    """

    def __init__(self, struct, metal, dist_margin=1.2, dist_margin_metal=1.5):
        """
        Constructor for coordination search for MetalmIm glasses

        Args:
            struct: pymatgen structure
            metal: str representing metal node
            dist_margin: float, default tolerance when using the covalent radii criteria to determine if two atoms are neighbours
            dist_margin_metal: float, specific tolerance for metal-X bonds as they're not covalent
        """
        self.node = bu.SingleMetal(metal, 4)
        self.linker = bu.ImidazoleBased("mIm", "C4N2H5")
        ZifSearch.__init__(self, struct, dist_margin, dist_margin_metal)

    
    def detect_conn(self):
        """
        main function to detect connectivity
        """
        # shorthand to select different atoms in pymatgen
        metal_atom = pymatgen.core.Composition(f"{self.node.name}1")
        H = "h"
        C = "c"
        N = "n"

        # Find imid cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.find_ABAcycles(C, N, cycle_length = 5, 
            target_number_of_cycles = self.elems.count("n") / 2, 
            fragtype = self.linker.name)

        # add H based on cov radii to single C and C bonded to one N
        C_Nbonds = self.get_A_Bbonds(C, N)

        logger.debug("number of N nn to C atoms")  # debug check
        for i in range(3):
            logger.debug("%s C atoms have %s N nn", C_Nbonds.count(i), i)

        self.assign_B_uniquely_to_A_N_coordinated(lambda i: (C_Nbonds[i] in [0, 1]), lambda i: (
            self.elems[i] == "h"),   3, report_entry="C atoms missing H neighbours", 
            propagate_fragments = True, new_fragments_name = 'methyl')

        # bind the remaining H (there should be non for the crystal)
        # create a new fragment called irregular_H that can propagate to the entire imid or to Zn
        H_Cbonds = self.get_A_Bbonds(H, C)
        self.find_N_closest_cov_dist(
            lambda i: H_Cbonds[i] == 0, lambda i: True, 1, report_level='full', 
            report_entry="H atoms not bonded to C", 
            propagate_fragments = True, new_fragments_name = 'irregular_H')

        # link C in cycles (bonded to 2 N) to C bonded to H
        self.find_N_closest_cov_dist(lambda i: C_Nbonds[i] == 0, lambda i: C_Nbonds[i] == 2, 1,
            report_level='undercoordinated', report_entry="C in CHn not bonded to any C in imid", 
            propagate_fragments = 'reverse')

        # link N to metal_atom with no constraint on the number of N to metal_atom
        self.find_N_closest_cov_dist(lambda i: self.elems[i] == metal_atom, lambda i: self.elems[i] == N,
            self.node.target_coordination, dist_margin=self.dist_margin_metal, report_level='undercoordinated', 
            report_entry=f"undercoordinated {self.node.name}", new_fragments_name = self.node.name)



class MetalIm(ZifSearch):
    """
    Search coordination for ZIFs made of metallic node and Im (imidazolate, C3N2H3) ligands
    Supports: ZIF-4, ZIF-zni, SALEM-2
    Principle:
        1. find every cycle of CNCNC
        2. add H based on cov radii to every C
        3. bind the remaining H (there should be non for the crystal)
        4. bind N and Zn
    """

    def __init__(self, struct, metal, dist_margin=1.2, dist_margin_metal=1.5):
        """
        Constructor for coordination search for MetalmIm glasses

        Args:
            struct: pymatgen structure
            metal: str representing metal node
            dist_margin: float, default tolerance when using the covalent radii criteria to determine if two atoms are neighbours
            dist_margin_metal: float, specific tolerance for metal-X bonds as they're not covalent
        """
        self.node = bu.SingleMetal(metal, 4)
        self.linker = bu.ImidazoleBased("Im", "C3N2H3")
        ZifSearch.__init__(self, struct, dist_margin, dist_margin_metal)

    
    def detect_conn(self):
        """
        main function to detect connectivity
        """
        # shorthand to select different atoms in pymatgen
        metal_atom = self.node.name.lower()
        H = "h"
        C = "c"
        N = "n"

        # Find imid cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.find_ABAcycles(C, N, cycle_length = 5, target_number_of_cycles = self.elems.count("n") / 2,
            fragtype = self.linker.name)

        # hard way to force the reduction to work by ignoring the failed imid search: to be investigated
        if not self.report_search['imid_search_successful']:
            raise SearchError('Imid search failed', self.report_search)

        # add H based on cov radii to every C
        self.assign_B_uniquely_to_A_N_coordinated(lambda i: (self.elems[i] == C), lambda i: (
            self.elems[i] == H),   3, report_level = 'undercoordinated', report_entry="C atoms missing H neighbours",

            propagate_fragments = True, new_fragments_name = 'irregular_C',
            dist_margin=self.dist_margin * 1.2) # quick fix for ab intio zif4_15glass

        # bind the remaining H (there should be non for the crystal)
        H_Cbonds = self.get_A_Bbonds(H, C)
        self.find_N_closest_cov_dist(
            lambda i: H_Cbonds[i] == 0, lambda i: True, 1, report_level='full', report_entry="H atoms not bonded to C",
            propagate_fragments = True, new_fragments_name = 'irregular_H',
            dist_margin=self.dist_margin * 1.2) # quick fix for ab intio zif4_15glass

        # link N to metal_atom with no constraint on the number of N to metal_atom
        self.find_N_closest_cov_dist(lambda i: self.elems[i] == metal_atom, lambda i: self.elems[i] == N,
            self.node.target_coordination, dist_margin=self.dist_margin_metal, report_level='undercoordinated',
            report_entry=f"undercoordinated {self.node.name}", new_fragments_name = self.node.name)

    def is_reduced_structure_valid(self):
        """
        For now, only accept the search if nothing else then Im and Zn are found
        """
        return len(self.symbols.from_name_to_symbol) == 2