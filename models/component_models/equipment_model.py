from oodesign import Equipment
from models.model import Model


class EquipmentModel(Model):
    def __init__(self, equip_obj: Equipment) -> None:
        self.obj = equip_obj

        self.name = self.obj.name
        self.n_ph = self.obj.n_ph
        # {phase -> int}, e.g. {'A': 1, 'B': 1, 'C': 1}
        self.phases_dict = self.obj.phases

        # to be updated by subclasses
        self.num_eqns = None
        self.num_vars = None
        self.var_offset = None

    def get_id(self):
        return self.obj.id