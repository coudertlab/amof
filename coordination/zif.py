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

    def __init__(self, struct, dist_margin=1.2, dist_margin_metal=1.5, cutoff_metal = None, ignore_H_in_reduction = True):
        """
        Constructor for custom search 2 for ZIF glasses

        Args:
            dist_margin is the default tolerance when using the covalent radii criteria to determine if two atoms are neighbours
            dist_margin_metal is the specific tolerance for metal-X bonds as they're not covalent
            cutoff_metal: float, overide dist_margin_metal if not None 
                option added but not used so far
            ignore_H_in_reduction: bool, if True imids identification is only done by the cycle
                The errors in bonding Hs are however reported, and the resulting reduced structure 
                    is computed with the H taken into account (barycenter).
        """
        self.dist_margin_metal = dist_margin_metal  # dist_margin is created in CoordinationSearch.__init__
        self.ignore_H_in_reduction = ignore_H_in_reduction
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
        For imid, A is C and B is N
        If fragtype is not None, will create new fragments with this fragmame

        Args:
            A, B: string, lowecase string indicating species of A and B (e.g. "h")
            cycle_length: int
            fragtype: str
        """
        graph = StructureGraph.with_empty_graph(self.struct)
        self.add_ABbonds(graph, B, A)
        self.add_ABbonds(graph, A, A)

        pattern = [A] + [B, A] * int((cycle_length - 1) / 2)
        cycles = self.find_rings(graph, pattern=pattern, target_number_of_rings = target_number_of_cycles, 
            exit_if_too_many_rings=False, remove_overlapping_rings=True)

        # check sanity of found cycles
        report_entry_1 = 'Expected number of cycles'
        self.report_search[report_entry_1] = (
            len(cycles) == target_number_of_cycles)
        if not self.report_search[report_entry_1]:
            logger.debug("number of cycles incorrect")
            self.report_search['Number of missing cycles'] = target_number_of_cycles - len(cycles)

        report_entry_2 = 'Atoms appear only once in cycles'
        in_cycle = [False for i in range(self.struct.num_sites)]
        self.report_search[report_entry_2] = True
        for c in cycles:
            for a, b in c:
                self.conn[a].append(b)
                self.conn[b].append(a)
                if in_cycle[a] == True:
                    logger.debug("atom %s appears in more than one cycle", a)
                    self.report_search[report_entry_2] = False
                in_cycle[a] = True
        self.clean_conn()

        if fragtype is not None:
            for c in cycles:
                indices = list(set(itertools.chain.from_iterable(c)))
                self.create_fragment(fragtype, indices)
        
        self.report_search['Cycle search successful'] = self.report_search[report_entry_1] and self.report_search[report_entry_2]

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

    def __init__(self, struct, metal, dist_margin=1.2, dist_margin_metal=1.5, ignore_H_in_reduction = True):
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
        ZifSearch.__init__(self, struct, dist_margin = dist_margin, 
            dist_margin_metal = dist_margin_metal, 
            ignore_H_in_reduction = ignore_H_in_reduction)
    
    def detect_conn(self):
        """
        main function to detect connectivity
        """
        # Find imid cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.find_ABAcycles("c", "n", cycle_length = 5, 
            target_number_of_cycles = self.elems.count("n") / 2, 
            fragtype = self.linker.name)

        # hard way to force the reduction to work by ignoring the failed imid search
        if not self.report_search['Cycle search successful']:
            raise SearchError('Cycle search failed', self.report_search)

        # Connect H
        H_perfectly_connected = True

        # add H based on cov radii to single C and C bonded to one N
        new_fragments_name = 'methyl'
        report_entry = "C atoms missing H neighbours"
        C_Nbonds = self.get_A_Bbonds("c", "n")

        logger.debug("number of N nn to C atoms")  # debug check
        for i in range(3):
            logger.debug("%s C atoms have %s N nn", C_Nbonds.count(i), i)

        self.assign_B_uniquely_to_A_N_coordinated(
            lambda i: (C_Nbonds[i] in [0, 1]), 
            lambda i: (self.elems[i] == "h"),
            3, 
            report_entry=report_entry, 
            propagate_fragments = True, new_fragments_name = new_fragments_name)
        H_perfectly_connected = H_perfectly_connected and self.report_search[report_entry] == []

        # bind the remaining H (there should be non for the crystal)
        # create a new fragment called irregular_H that can propagate to the entire imid or to Zn
        H_Cbonds = self.get_A_Bbonds("h", "c")
        new_fragments_name = self.linker.name if self.ignore_H_in_reduction else 'irregular_H'
        report_entry = "H atoms not bonded to C"
        self.find_N_closest_cov_dist(
            lambda i: H_Cbonds[i] == 0, 
            lambda i: True, 
            1, 
            report_level='full', report_entry=report_entry, 
            propagate_fragments = True, new_fragments_name = new_fragments_name)
        H_perfectly_connected = H_perfectly_connected and self.report_search[report_entry] == []

        self.report_search['H perfectly connected'] = H_perfectly_connected

        # link C in cycles (bonded to 2 N) to C bonded to H
        self.find_N_closest_cov_dist(
            lambda i: C_Nbonds[i] == 0, 
            lambda i: C_Nbonds[i] == 2, 
            1, 
            report_level='undercoordinated', report_entry="C in CHn not bonded to any C in imid", 
            propagate_fragments = 'reverse')

        # link N to metal_atom with no constraint on the number of N to metal_atom
        metal_atom = self.node.name.lower()
        self.find_N_closest_cov_dist(
            lambda i: self.elems[i] == metal_atom, 
            lambda i: self.elems[i] == "n",
            self.node.target_coordination, 
            dist_margin=self.dist_margin_metal, report_level='undercoordinated', 
            report_entry=f"undercoordinated {self.node.name}", new_fragments_name = self.node.name)

    def is_reduced_structure_valid(self):
        """
        For now, only accept the search if nothing else then mIm and Zn are found
        """
        return len(self.symbols.from_name_to_symbol) == 2

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

    def __init__(self, struct, metal, dist_margin=1.2, dist_margin_metal=1.5, ignore_H_in_reduction = True):
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
        ZifSearch.__init__(self, struct, dist_margin = dist_margin, 
            dist_margin_metal = dist_margin_metal, 
            ignore_H_in_reduction = ignore_H_in_reduction)

    
    def detect_conn(self):
        """
        main function to detect connectivity
        """
        # Find imid cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.find_ABAcycles("c", "n", cycle_length = 5, target_number_of_cycles = self.elems.count("n") / 2,
            fragtype = self.linker.name)

        # hard way to force the reduction to work by ignoring the failed imid search: to be investigated
        if not self.report_search['Cycle search successful']:
            raise SearchError('Cycle search failed', self.report_search)

        # Connect H
        H_perfectly_connected = True
        
        # add H based on cov radii to every C
        new_fragments_name = self.linker.name if self.ignore_H_in_reduction else 'irregular_C'
        report_entry = "C atoms missing H neighbours"
        self.assign_B_uniquely_to_A_N_coordinated(
            lambda i: (self.elems[i] == "c"), 
            lambda i: (self.elems[i] == "h"),  
            3, 
            report_level = 'undercoordinated', report_entry = report_entry,
            propagate_fragments = True, new_fragments_name = new_fragments_name,
            dist_margin=self.dist_margin * 1.2) # quick fix for ab intio zif4_15glass
        H_perfectly_connected = H_perfectly_connected and self.report_search[report_entry] == []

        # bind the remaining H (there should be non for the crystal)
        H_Cbonds = self.get_A_Bbonds("h", "c")
        new_fragments_name = self.linker.name if self.ignore_H_in_reduction else 'irregular_H'
        report_entry = "H atoms not bonded to C"
        self.find_N_closest_cov_dist(
            lambda i: H_Cbonds[i] == 0, 
            lambda i: True, 
            1, 
            report_level='full', report_entry= report_entry,
            propagate_fragments = True, new_fragments_name = new_fragments_name,
            dist_margin=self.dist_margin * 1.2) # quick fix for ab intio zif4_15glass
        H_perfectly_connected = H_perfectly_connected and self.report_search[report_entry] == []

        self.report_search['H perfectly connected'] = H_perfectly_connected

        # link N to metal_atom with no constraint on the number of N to metal_atom
        metal_atom = self.node.name.lower()
        self.find_N_closest_cov_dist(
            lambda i: self.elems[i] == metal_atom, 
            lambda i: self.elems[i] == "n",
            self.node.target_coordination, 
            dist_margin=self.dist_margin_metal, report_level='undercoordinated',
            report_entry=f"undercoordinated {self.node.name}", new_fragments_name = self.node.name)

    def is_reduced_structure_valid(self):
        """
        For now, only accept the search if nothing else then Im and Zn are found
        """
        return len(self.symbols.from_name_to_symbol) == 2