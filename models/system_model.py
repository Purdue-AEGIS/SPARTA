from oodesign import *
from models.component_models.line_model import *

# from models.component_models.line_model_neg import *
from models.component_models.equipment_model import *
from models.component_models.transformer_model import *
from models.component_models.capacitor_model import *
from models.component_models.volt_regulator_model import *
from models.component_models.switch_model import *

from models.component_models.source_model import *
# from models.component_models.source_model__gfl_study import *

# from models.component_models.load_model import *  # use for sync study as well
from models.component_models.load_model import *
# from models.component_models.load_model__multi_gfm_study import *

from models.component_models.node_model import *

# from models.component_models.inverters.GFMinverter_model  import *
# from models.component_models.inverters.GFMinverter_model_emid  import *
# from models.component_models.inverters.GFMinverter_model_emid_brf  import *
# from models.component_models.inverters.GFMinverter_model_emid_brf_refactor  import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor  import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV_wip import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV_refstudy13 import *
from models.component_models.inverters.GFMinverter_model_r0 import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV_vczero import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV_vir import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_wip  import *

# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_unifi  import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_unifi_init  import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_vir_imp  import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_vir_imp_sw  import *
# from models.component_models.inverters.GFLinverter_model import *
# from models.component_models.inverters.GFLinverter_model_refactor import *
# from models.component_models.inverters.GFLinverter_model_refactor_dc import *
from models.component_models.inverters.GFLinverter_model_refactor_brf import *

from models.model import Model


def create_equipment_model(equip: Equipment) -> EquipmentModel:
    if isinstance(equip, Transformer) and equip.config.startswith("ThreePhaseDeltaWye"):
        return ThreePhaseDYStepDownTransformerModel(equip)

    elif isinstance(equip, Transformer) and equip.config.startswith("ThreePhaseWyeWye"):
        return ThreePhaseYYStepDownTransformerModel(equip)

    elif isinstance(equip, Transformer) and equip.config.startswith(
        "ThreePhaseDeltaDelta"
    ):
        return ThreePhaseDDStepDownTransformerModel(equip)

    elif isinstance(equip, ShuntCapacitor) and equip.config == "StarShuntCapacitor":
        return StarShuntCapacitorModel(equip)

    elif isinstance(equip, ShuntCapacitor) and equip.config == "DeltaShuntCapacitor":
        return DeltaShuntCapacitorModel(equip)

    elif isinstance(equip, Regulator):
        return VoltRegulatorModel(equip)

    elif isinstance(equip, Switch):
        return SwitchModel(equip)

    print(f">> equip: {equip}")
    print(f">> equip.type: '{equip.config}'")
    raise ValueError(f"unknown equipment type: {equip}")


def create_load_model(load: Load) -> LoadModel:
    if isinstance(load, StarConstantPowerLoad):
        return StarConstantPowerLModel(load)

    elif isinstance(load, StarConstantImpedanceLoad):
        return StarConstantImpedanceLModel(load)

    elif isinstance(load, StarConstantCurrentLoad):
        return StarConstantCurrentLModel(load)

    elif isinstance(load, DeltaConstantImpedanceLoad):
        return DeltaConstantImpedanceLModel(load)

    elif isinstance(load, DeltaConstantCurrentLoad):
        return DeltaConstantCurrentLModel(load)

    elif isinstance(load, DeltaConstantPowerLoad):
        return DeltaConstantPowerLModel(load)

    raise ValueError(f"unknown load type: {load}")


def create_source_model(source: Source) -> SourceModel:
    if isinstance(source, ConstantVoltageSource):
        return ConstantVoltageModel(source)

    elif isinstance(source, GFMInverter3Ph):
        # input("continue making a GFM?")
        return GFMInverter3PhModel(source)

    elif isinstance(source, GFLInverter3Ph):
        # input("continue making a GFL?")
        return GFLInverter3PhModel(source)

    elif isinstance(source, ConstantVoltageSeqSource):
        return ConstantVoltageSeqModel(source)

    elif isinstance(source, ConstantVoltageAdjSource):
        return ConstantVoltageAdjModel(source)

    raise ValueError(f"unknown source type: {source}")


class SystemModel:
    def __init__(self, system: System):
        self.system = system

        self.nodes = [NodeModel(node) for node in system.nodes]
        self.lines = [LineModel(line) for line in system.lines]
        self.equipments = [create_equipment_model(equip) for equip in system.equipments]
        self.loads = [create_load_model(load) for load in system.loads]
        self.sources = [create_source_model(source) for source in system.sources]

        self.components = self.lines + self.equipments + self.loads + self.sources

        # TODO:
        # inverters
        # generators

        # check that the ids are unique
        id_list = [comp.get_id() for comp in self.components]
        id_set = set(id_list)
        assert (
            len(id_list) == len(id_set),
            "non-unique ids found [PLEASE INVESTIGATE]",
        )

    def get_source_by_id(self, source_id) -> SourceModel:
        for source in self.sources:
            if source.get_id() == source_id:
                return source
        raise ValueError("no source found with id: {source_id}")

    def get_component_by_id(self, comp_id) -> Model:
        for component in self.components:
            if component.get_id() == comp_id:
                return component
        raise ValueError("no component found with id: {comp_id}")

    def get_node_by_id(self, node_id) -> NodeModel:
        for node in self.nodes:
            if node.get_id() == node_id:
                return node
        raise ValueError("no node found with id: {node_id}")
