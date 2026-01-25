from oodesign import *

from models.component_models.source_model import SourceModel
# from models.component_models.source_model__gfl_study import SourceModel


class InverterModel(SourceModel):
    def __init__(self, inverter_obj: Source):
        super().__init__(inverter_obj)


class GFLInverterModel(InverterModel):
    def __init__(self, inverter_obj: GFLInverter):
        super().__init__(inverter_obj)

    def get_basetype(self):
        return "source"

    def get_id(self):
        return self.obj.id


class GFMInverterModel(InverterModel):
    def __init__(self, inverter_obj: GFMInverter):
        super().__init__(inverter_obj)

        vnom = list(self.nominal_voltage.values())
        if len(vnom) > 1:
            # we assume all phases to have the same magnitude
            assert vnom[0] == vnom[1]
        self.nominal_voltage_rms = vnom[0]

    def get_basetype(self):
        return "source"

    def get_id(self):
        return self.obj.id
