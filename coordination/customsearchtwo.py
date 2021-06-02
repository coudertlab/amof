p"""
Legacy version of custom search - need to be removed when sadi implementation working
"""

from core import CoordinationSearch

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
