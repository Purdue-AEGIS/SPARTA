from oodesign import Node
from models.model import Model
from scipy import sparse as sps


class NodeModel(Model):
    def __init__(self, node_obj: Node):
        self.obj = node_obj

    def get_id(self):
        return self.obj.id

    def get_phases_without_n(self) -> list[str]:
        phases = self.get_phases()
        return [ph for ph in phases if ph != "N"]
    
    def get_basetype(self):
        return "node"
    
    def get_M_powerflow(self) -> tuple[sps.coo_array, sps.coo_array]:
        raise ValueError("Not Applicable to Nodes")
