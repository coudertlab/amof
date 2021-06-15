"""
Main file containing classes for custom neigbour search in ZIF glasses
Supported: ZIF-8
"""

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

    def __init__(self, struct, dist_margin=1.2, dist_margin_metal=1.5):
        """
        Constructor for custom search 2 for ZIF8 glasses
        dist_margin is the default tolerance when using the covalent radii criteria to determine if two atoms are neighbours
        dist_margin_metal is the specific tolerance for metal-X bonds as they're not covalent
        """
        self.dist_margin_metal = dist_margin_metal  # dist_maring is created in CoordinationSearch.__init__
        neighb_max_distance = self.find_neighb_max_distance(
            dist_margin, dist_margin_metal)
        CoordinationSearch.__init__(
            self, struct, neighb_max_distance, dist_margin)
        self.detect_conn()
        self.update_atypes()

    def find_neighb_max_distance(self, dist_margin, dist_margin_metal):
        """return neighb_max_distance to compute all_neighb to the minimal necessary distance"""
        dist_margin_atoms = self.linker.species
        dist_margin_metal_atoms = self.node.species
        max_cov_linker = np.max([self.covalentradius[A]+self.covalentradius[B]
                                 for A in dist_margin_atoms for B in dist_margin_atoms])
        max_cov_metal = np.max([self.covalentradius[A]+self.covalentradius[B]
                                for A in dist_margin_metal_atoms for B in (dist_margin_atoms+dist_margin_metal_atoms)])
        return max(max_cov_linker * dist_margin, max_cov_metal * dist_margin_metal)

    def find_ABAcycles(self, A, B, cycle_length, target_number_of_cycles):
        """
        Find every cycle of the form ABA...BA of length cycle_length for species A and B
        Add the bonds thus found to self.conn
        A can be neighbour to A and B, and B only to A. 
        NOT IMPLEMENTED YET: Only one A-A bond is allowed > check
        For imid, A is C and B is N

        Args:
            A, B: pymatgen.Composition objects, species of A and B
            cycle_length: int
        """
        # Find cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.add_ABbonds(graph, B, A)
        self.add_ABbonds(graph, A, A)
        cycles = self.get_chain_decomposition(graph)
        cycles = self.find_one_cycle_per_node(graph)
        cycles = self.find_rings(graph)
        # check sanity of found cycles
        self.report_search['imid_expected_number_of_cycles'] = (
            len(cycles) == target_number_of_cycles)
        if not self.report_search['imid_expected_number_of_cycles']:
            logger.warning("number of cycles incorrect")

        cycles_of_wrong_size = [c for c in cycles if len(c) != cycle_length]
        self.report_search['imid_expected_length_of_cycles'] = (
            len(cycles_of_wrong_size) == 0)
        if not self.report_search['imid_expected_length_of_cycles']:
            logger.warning("cycle not of %s atoms found: %s", cycle_length, 
                           cycles_of_wrong_size)

        in_cycle = [False for i in range(self.struct.num_sites)]
        self.report_search['imid_atoms_appear_only_once_in_cycles'] = True
        for c in cycles:
            for a, b in c:
                self.conn[a].append(b)
                self.conn[b].append(a)
                if in_cycle[a] == True:
                    logger.warning("atom %s appears in more than one cycle", a)
                    self.report_search['imid_atoms_appear_only_once_in_cycles'] = False
                in_cycle[a] = True

        self.plot_conn_as_graph('find_rings.png')
        print(1)

    # def rest(self):
    #     # add H based on cov radii to single C and C bonded to one N
    #     C_Nbonds = self.get_A_Bbonds(C, N)

    #     logger.debug("number of N nn to C atoms")  # debug check
    #     for i in range(3):
    #         logger.debug("%s C atoms have %s N nn", C_Nbonds.count(i), i)

    #     self.assign_B_uniquely_to_A_N_coordinated(lambda i: (C_Nbonds[i] in [0, 1]), lambda i: (
    #         self.struct[i].species == H),   3, report_entry="C atoms missing H neighbours")

    #     # bind the remaining H (there should be non for the crystal)
    #     H_Cbonds = self.get_A_Bbonds(H, C)
    #     self.find_N_closest_cov_dist(
    #         lambda i: H_Cbonds[i] == 0, lambda i: True, 1, report_level='full', report_entry="H atoms not bonded to C")

    #     # link C in cycles (bonded to 2 N) to C bonded to H
    #     self.find_N_closest_cov_dist(lambda i: C_Nbonds[i] == 0, lambda i: C_Nbonds[i] == 2, 1,
    #                                  report_level='undercoordinated', report_entry="C in CHn not bonded to any C in imid")

    #     # link N to Zn with no constraint on the number of N to Zn
    #     self.find_N_closest_cov_dist(lambda i: self.struct[i].species == Zn, lambda i: self.struct[i].species == N,
    #                                  4, dist_margin=self.dist_margin_metal, report_level='undercoordinated', report_entry="undercoordinated Zn")


class MetalmIm(ZifSearch):
    """
    Search coordination for ZIFs made of metallic node and mIm (methyldimidazolate, C4N2H5) ligands
    Supports: ZIF8
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
        H = pymatgen.core.Composition("H1")
        C = pymatgen.core.Composition("C1")
        N = pymatgen.core.Composition("N1")

        # Find imid cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.find_ABAcycles(C, N, cycle_length = 5, target_number_of_cycles = self.struct.species.count(pymatgen.core.Element("N")) / 2)

        # add H based on cov radii to single C and C bonded to one N
        C_Nbonds = self.get_A_Bbonds(C, N)

        logger.debug("number of N nn to C atoms")  # debug check
        for i in range(3):
            logger.debug("%s C atoms have %s N nn", C_Nbonds.count(i), i)

        self.assign_B_uniquely_to_A_N_coordinated(lambda i: (C_Nbonds[i] in [0, 1]), lambda i: (
            self.struct[i].species == H),   3, report_entry="C atoms missing H neighbours")

        # bind the remaining H (there should be non for the crystal)
        H_Cbonds = self.get_A_Bbonds(H, C)
        self.find_N_closest_cov_dist(
            lambda i: H_Cbonds[i] == 0, lambda i: True, 1, report_level='full', report_entry="H atoms not bonded to C")

        # link C in cycles (bonded to 2 N) to C bonded to H
        self.find_N_closest_cov_dist(lambda i: C_Nbonds[i] == 0, lambda i: C_Nbonds[i] == 2, 1,
                                     report_level='undercoordinated', report_entry="C in CHn not bonded to any C in imid")

        # link N to metal_atom with no constraint on the number of N to metal_atom
        self.find_N_closest_cov_dist(lambda i: self.struct[i].species == metal_atom, lambda i: self.struct[i].species == N,
                                     self.node.target_coordination, dist_margin=self.dist_margin_metal, report_level='undercoordinated', report_entry=f"undercoordinated {self.node.name}")



class MetalIm(ZifSearch):
    """
    Search coordination for ZIFs made of metallic node and Im (imidazolate, C3N2H3) ligands
    Supports: ZIF4, ZIF-zni, SALEM-2
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
        metal_atom = pymatgen.core.Composition(f"{self.node.name}1")
        H = pymatgen.core.Composition("H1")
        C = pymatgen.core.Composition("C1")
        N = pymatgen.core.Composition("N1")

        # Find imid cycles (C-N-C-N-C)
        graph = StructureGraph.with_empty_graph(self.struct)
        self.find_ABAcycles(C, N, cycle_length = 5, target_number_of_cycles = self.struct.species.count(pymatgen.core.Element("N")) / 2)

        # add H based on cov radii to every C
        self.assign_B_uniquely_to_A_N_coordinated(lambda i: (self.struct[i].species == C), lambda i: (
            self.struct[i].species == H),   3, report_entry="C atoms missing H neighbours")

        # bind the remaining H (there should be non for the crystal)
        H_Cbonds = self.get_A_Bbonds(H, C)
        self.find_N_closest_cov_dist(
            lambda i: H_Cbonds[i] == 0, lambda i: True, 1, report_level='full', report_entry="H atoms not bonded to C")

        # link N to metal_atom with no constraint on the number of N to metal_atom
        self.find_N_closest_cov_dist(lambda i: self.struct[i].species == metal_atom, lambda i: self.struct[i].species == N,
                                     self.node.target_coordination, dist_margin=self.dist_margin_metal, report_level='undercoordinated', report_entry=f"undercoordinated {self.node.name}")
