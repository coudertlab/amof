"""
Classes for nodes and linkers representation
"""

class BuildingUnit(object):
    """
    Abstract class for Building Units objects: nodes and linkers
    """
    def __init__(self) -> None:
        super().__init__()

class Node(BuildingUnit):
    """
    Generic class for representing nodes
    """
    def __init__(self) -> None:
        super().__init__()

class SingleMetal(Node):
    """
    Class for representing single metal nodes
    """
    def __init__(self, metal, target_coordination) -> None:
        """
        Args:
            metal: str, metal species (e.g. "Zn")
            target_coordination: int
        """
        super().__init__()
        self.name = metal
        self.species = [metal]
        self.target_coordination = target_coordination

class Linker(BuildingUnit):
    """
    Generic class for representing linkers
    """
    def __init__(self) -> None:
        super().__init__()

class ImidazoleBased(Linker):
    """
    Class for representing imidazole-based linkers
    """
    def __init__(self, name, formula) -> None:
        """
        Args:
            name: str, name of linker
            formula: str, chemical formula (e.g. "C3H3N2")
        """        
        super().__init__()
        self.name = name
        self.formula = formula
        self.species = ["C", "H", "N"] #temporary fix, to be changed (smiles & co)

class ImidazoleCycle(ImidazoleBased):
    """
    Class for representing the CNCNC cycle as generic imidazole-based linker
    """
    def __init__(self) -> None:
        """
        Args:
            name: str, name of linker
            formula: str, chemical formula (e.g. "C3H3N2")
        """        
        super().__init__("ImCycle", "C3N2")
        self.species = ["C", "N"] # override 
