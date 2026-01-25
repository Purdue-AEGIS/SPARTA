# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 12:13:46 2023
"""

"""
Overall Class Hierarchy:
    
    DSItem:
        Line:
            OverheadLine: (could b a subclass if needed)
            UndergroundLine:(could be a subclass if needed)
        Node:
            Source Node (this information does not come from xml)            
            LoadNode: 
                
        Equipment:
            Transformer:
            Generator:(to be added with other equipments)
"""


import numpy as np
from dataclasses import dataclass
from pprint import pformat
from enum import Enum, StrEnum
from const import StudyType, NodeSide
import copy


@dataclass
class DSItem:
    """the base class for all items in our system"""

    id: str

    def update_pu_base(self, base_power: float, base_voltage: float):
        raise NotImplementedError

    def __eq__(self, other):
        if not isinstance(other, DSItem):
            return False
        else:
            return self.id == other.id


@dataclass
class Node(DSItem):
    """a connection point"""

    phases: dict[str, int]
    voltage: float

    base_power: float = None
    base_voltage: float = None

    def update_pu_base(self, base_power: float, base_voltage: float):
        self.base_power = base_power
        self.base_voltage = base_voltage

    def __eq__(self, other):
        if not isinstance(other, Node):
            return False
        else:
            return self.id == other.id

    def __hash__(self):
        # Convert phases dict to a sorted tuple of items for hashing
        phases_tuple = tuple(sorted(self.phases.items()))
        return hash((phases_tuple, self.voltage))


def get_node_on_side(dsitem: DSItem, side: NodeSide) -> Node:
    if side == NodeSide.FROM:
        return dsitem.terminal.from_node
    elif side == NodeSide.TO:
        return dsitem.terminal.to_node
    elif side == NodeSide.AT:
        return dsitem.terminal.at_node
    else:
        raise ValueError("uknown side: {side}")


@dataclass
class NTerminal:
    @classmethod
    def get_num_term(cls) -> int:
        raise NotImplementedError


@dataclass
class SingleTerminal(NTerminal):
    at_node: Node

    @classmethod
    def get_num_term(cls) -> int:
        return 1


@dataclass
class TwoTerminal(NTerminal):
    from_node: Node
    to_node: Node

    @classmethod
    def get_num_term(cls) -> int:
        return 2


@dataclass
class TertiaryTerminal(NTerminal):
    from_node: Node
    to_node1: Node
    to_node2: Node

    @classmethod
    def get_num_term(cls) -> int:
        return 3


@dataclass
class Line(DSItem):
    """Line is a superset of all overhead and underground lines
    A line could be single phase or multiphase"""

    name: str
    # from_node: Node
    # to_node: Node
    terminal: TwoTerminal
    phases: dict[str, int]
    # phasing: list[str]
    # numer of phases in this line
    n_ph: int
    has_capacitance: bool
    resistance_mat: np.ndarray
    reactance_mat: np.ndarray
    admittance_mat: np.ndarray

    base_power: float = None
    base_voltage: float = None
    resistance_mat_act: np.ndarray = None
    reactance_mat_act: np.ndarray = None
    admittance_mat_act: np.ndarray = None
    Z_base: float = None

    def __hash__(self):
        return hash(self.id)

    def update_pu_base(self, base_power: float, base_voltage: float):
        self.base_power = base_power
        self.base_voltage = base_voltage

        self.Z_base = self.base_voltage**2 / self.base_power

        self.resistance_mat_act = copy.deepcopy(self.resistance_mat)
        self.reactance_mat_act = copy.deepcopy(self.reactance_mat)
        self.admittance_mat_act = copy.deepcopy(self.admittance_mat)

        self.resistance_mat = self.resistance_mat / self.Z_base
        self.reactance_mat = self.reactance_mat / self.Z_base
        self.admittance_mat = self.admittance_mat / self.Z_base




@dataclass
class Equipment(DSItem):
    """base class for all Equipment: transformer, generator etc."""

    name: str
    # from_node: Node
    # to_node: Node | None
    terminal: NTerminal
    n_ph: int
    phases: dict[str, int]


class SwitchState(StrEnum):
    Open = "open"
    Closed = "closed"


@dataclass
class Switch(Equipment):
    name: str
    terminal: TwoTerminal
    phases: dict[str, int]
    state: SwitchState
    nominal_voltage: dict[str, float]


class TransformerConfig(Enum):
    D_D = "ThreePhaseDeltaDelta"
    D_GrY = "ThreePhaseDeltaWye"
    D_Y = "ThreePhaseDeltaWye"
    D_GrW = "ThreePhaseDeltaWye"
    Y_Y = "ThreePhaseWyeWye"
    GrY_D = "ThreePhaseWyeDelta"
    GrY_Y = "ThreePhaseWyeWye"
    GrY_GrY = "ThreePhaseWyeWye"
    Y_D = "ThreePhaseWyeDelta"


@dataclass
class Transformer(Equipment):
    """this class inherits from Equipment class and creates object for transformers
    in the network"""

    pri_volt: float  # in V
    sec_volt: float  # in V
    turns_ratio: float
    VA: float  # in VA
    resistance_mat: np.ndarray
    inductance_mat: np.ndarray
    config: str

    base_power: float = None
    base_voltage: float = None
    resistance_mat_act: np.ndarray = None
    inductance_mat_act: np.ndarray = None
    base_voltage_secondary: float = None

    def update_pu_base(self, base_power, base_voltage):
        self.base_power = base_power
        self.base_voltage = base_voltage
        self.base_voltage_secondary = self.base_voltage / self.turns_ratio

        self.Z_base = self.base_voltage**2 / self.base_power

        # backup
        self.pri_volt_act = copy.deepcopy(self.pri_volt)
        self.sec_volt_act = copy.deepcopy(self.sec_volt)
        self.resistance_mat_act = copy.deepcopy(self.resistance_mat)
        self.inductance_mat_act = copy.deepcopy(self.inductance_mat)

        # pu updates
        self.pri_volt = self.pri_volt / self.base_voltage
        self.sec_volt = self.sec_volt / self.base_voltage
        self.resistance_mat = self.resistance_mat / self.Z_base
        self.inductance_mat = self.inductance_mat / self.Z_base


@dataclass
class Generator(Equipment):
    # Base values
    VA_rating: float
    V_rating_LL: float
    power_factor: float
    poles_count: float
    speed: float  # r/min
    gen_turbine_inertia: float  # Joules.s^2
    # parameters in ohms
    rs: float
    rkd: float
    rfd: float
    rkq1: float
    rkq2: float
    Xls: float
    Xq: float
    Xd: float
    Xlkq1: float
    Xlfd: float
    Xlkq2: float
    Xlkd: float


@dataclass
class LoadType(Enum):
    Y_PQ = "StarConstantPowerLoad"
    Y_Z = "StarConstantImpedanceLoad"
    Y_I = "StarConstantCurrentLoad"
    D_PQ = "DeltaConstantPowerLoad"
    D_Z = "DeltaConstantImpedanceLoad"
    D_I = "DeltaConstantCurrentLoad"


@dataclass
class Load(DSItem):
    # This class describes the load model and the active reactive power consumption
    name: str
    # type: LoadType  # Y/D, const.Power(PQ)/const.imp(Z)/const. curr(I)
    # at_node: Node
    terminal: SingleTerminal
    phases: list[str]
    active_power: dict[str, float]
    reactive_power: dict[str, float]
    n_ph: int
    nominal_voltage: dict[str, float]
    stepchange: bool

    def update_pu_base(self, base_power, base_voltage):
        self.base_power = base_power
        self.base_voltage = base_voltage

        # backup
        self.active_power_act = copy.deepcopy(self.active_power)
        self.reactive_power_act = copy.deepcopy(self.reactive_power)
        self.nominal_voltage_act = copy.deepcopy(self.nominal_voltage)

        # pu updates
        for ph in self.phases:
            self.active_power[ph] = self.active_power[ph] / self.base_power
            self.reactive_power[ph] = self.reactive_power[ph] / self.base_power
            self.nominal_voltage[ph] = self.nominal_voltage[ph] / self.base_voltage


@dataclass
class StarConstantPowerLoad(Load):
    pass


@dataclass
class StarConstantImpedanceLoad(Load):
    pass


@dataclass
class StarConstantCurrentLoad(Load):
    iconst: dict[str, float]
    power_factor: dict[str, float]

    base_power: float = None
    base_voltage: float = None
    active_power_act: dict[str, float] = None
    reactive_power_act: dict[str, float] = None
    nominal_voltage_act: dict[str, float] = None
    iconst_act: dict[str, float] = None

    def update_pu_base(self, base_power, base_voltage):
        super().update_pu_base(base_power, base_voltage)

        # backup
        self.iconst_act = copy.deepcopy(self.iconst)

        # pu updates
        self.I_base = self.base_power / self.base_voltage
        for ph in self.phases:
            self.iconst[ph] = self.iconst[ph] / self.I_base


@dataclass
class DeltaConstantImpedanceLoad(Load):
    pass


@dataclass
class DeltaConstantCurrentLoad(Load):
    iconst: dict[str, float]
    power_factor: dict[str, float]

    base_power: float = None
    base_voltage: float = None
    active_power_act: dict[str, float] = None
    reactive_power_act: dict[str, float] = None
    nominal_voltage_act: dict[str, float] = None
    iconst_act: dict[str, float] = None

    def update_pu_base(self, base_power, base_voltage):
        super().update_pu_base(base_power, base_voltage)

        # backup
        self.iconst_act = copy.deepcopy(self.iconst)

        # pu updates
        self.I_base = self.base_power / self.base_voltage
        for ph in self.phases:
            self.iconst[ph] = self.iconst[ph] / self.I_base


@dataclass
class DeltaConstantPowerLoad(Load):
    pass


class ShuntCapType(Enum):
    Y = "StarShuntCapacitor"
    D = "DeltaShuntCapacitor"


@dataclass
class ShuntCapacitor(Equipment):
    """Shunt Capacitor is a device that is used to improve the power
    factor of the system"""

    # at_node: Node
    terminal: SingleTerminal
    power: dict[str, float]
    n_ph: int
    nominal_voltage: dict[str, float]
    config: str


@dataclass
class Source(DSItem):
    name: str
    # at_node: Node
    terminal: SingleTerminal
    phases: list[str]
    n_ph: int
    nominal_voltage: dict[str, float]

    # base_power: float = None
    # base_voltage: float = None
    # nominal_voltage_act: dict[str, float] = None

    def update_pu_base(self, base_power, base_voltage):
        self.base_power = base_power
        self.base_voltage = base_voltage

        # backup
        self.nominal_voltage_act = copy.deepcopy(self.nominal_voltage)

        # pu updates
        for ph in self.phases:
            self.nominal_voltage[ph] = self.nominal_voltage[ph] / self.base_voltage


@dataclass
class ConstantVoltageSource(Source):
    pass


@dataclass
class ConstantVoltageSeqSource(Source):
    pass


@dataclass
class ConstantVoltageAdjSource(Source):
    pass


class InverterSequenceControl(StrEnum):
    NORMAL = "normal"
    SEQUENCE_CONTROL = "sequence_control"


@dataclass
# parent class for inverters
class Inverter(Source):
    Vdc: float
    Pb: float  # W
    V_base: float
    # delta: float

    # # Inverter switch and diode parameters (From 61016 AVM case study)
    # Vsw: float  # V
    # Vd: float  # V
    # rsw: float  # ohm
    # rd: float  # ohm

    # control: InverterSequenceControl

    def update_pu_base(self, base_power, base_voltage):
        raise NotImplementedError


class InverterPrimaryControl(StrEnum):
    DROOP = "droop"
    SYNCHRONOUS = "synchronous"


class InverterSingleDualControl(StrEnum):
    SINGLE = "single"
    DUAL = "dual"


@dataclass
class InverterControl:
    primary_control: InverterPrimaryControl
    single_dual_control: InverterSingleDualControl


@dataclass
class GFMInverter(Inverter):
    pass


@dataclass
class GFLInverter(Inverter):
    pass


@dataclass
class GFMInverter3Ph(GFMInverter):
    # Vdc: float
    # Pb: float
    # V_base: float
    delta: float
    La1: float
    Lb1: float
    Lc1: float
    raL1: float
    rbL1: float
    rcL1: float
    La2: float
    Lb2: float
    Lc2: float
    raL2: float
    rbL2: float
    rcL2: float
    Ca: float
    Cb: float
    Cc: float
    raC: float
    rbC: float
    rcC: float
    Vsw: float
    Vd: float
    rsw: float
    rd: float
    kw: float
    kq: float
    taus: float
    Kpv: float      # pu
    tauiv: float
    Imx: float      # pu
    Kpc: float      # pu
    tauic: float
    Vmx: float      # pu
    # vqcap_star: float
    # vdcap_star: float
    controls: InverterControl
    switch_state: SwitchState


@dataclass
class GFMInverter1Ph(GFMInverter):
    pass


@dataclass
class GFLInverter(Inverter):
    pass

@dataclass
class GFLInverter3Ph(GFLInverter):
    delta: float
    # Filter parameters
    La1: float
    Lb1: float
    Lc1: float
    raL1: float
    rbL1: float
    rcL1: float
    La2: float
    Lb2: float
    Lc2: float
    raL2: float
    rbL2: float
    rcL2: float
    Ca: float
    Cb: float
    Cc: float
    raC: float
    rbC: float
    rcC: float
    Vsw: float
    Vd: float
    rsw: float
    rd: float
    # controller constants
    Kpc: float      # pu
    Kic: float
    # k: float
    Kpc: float
    Kic: float
    # Pref: float  # W
    # Qref: float  # W
    # PLL
    Kppll: float
    Kipll: float

# theirs
@dataclass
class Inverter_3ph(Inverter):
    # Filter parameters
    La1: float
    Lb1: float
    Lc1: float
    raL1: float
    rbL1: float
    rcL1: float
    La2: float
    Lb2: float
    Lc2: float
    raL2: float
    rbL2: float
    rcL2: float
    Ca: float
    Cb: float
    Cc: float
    raC: float
    rbC: float
    rcC: float


# theirs
@dataclass
class GFL_inverter_3ph(Inverter_3ph):
    # controller constants
    k: float  # pu
    Kpc: float  # pu
    Kic: float  # pu
    Pref: float  # W
    Qref: float  # W
    # PLL
    Kppll: float  # pu
    Kipll: float  # pu


# theirs
@dataclass
class GFM_inverter_3ph(Inverter_3ph):
    # constants
    # Values from Hugo paper
    # P-w droop
    # constants (from Hugo paper)
    kw: float  # pu
    taus: float  # ms

    # VR
    Kpv: float  # Proportional constant for PI voltage controller (pu)
    tauiv: float  # Time constant for PI voltage controlelr (s)
    Imx: float  # pu

    # CCH
    Kpc: float  # 0.6631 #Proportional constant for PI current controller (pu)
    tauic: float  # TIme constant for PI current controlelr (s)

    # Inputs
    vqcap_star: float
    vdcap_star: float


# theirs
@dataclass
class Inverter_1ph(Inverter):
    # Filter parameters
    L1: float
    rL1: float
    L2: float
    rL2: float
    C: float
    rC: float


# theirs
@dataclass
class GFL_inverter_1ph(Inverter_1ph):
    # PLL
    Kppll: float  # pu
    Kipll: float  # pu
    # Current Controller
    k: float  # pu
    L: float  # H
    Kpc: float  # pu
    Kic: float  # pu
    Pref: float  # W
    Qref: float  # W
    igdi_ref: float  # pu


# theirs
@dataclass
class GFM_inverter_1ph(Inverter_1ph):
    # constants (from Hugo paper)
    kw: float  # pu
    Pemx: float  # pu
    taus: float  # ms
    # VR
    Kpv: float  # Proportional constant for PI voltage controller (pu)
    tauiv: float  # Time constant for PI voltage controlelr (s)
    Imx: float


class RegulatorControl(StrEnum):
    MANUAL = "manual"
    AUTOMATIC = "automatic"


@dataclass
class Regulator(Equipment):
    terminal: TwoTerminal
    bandwidth: float
    pt_ratio: float
    ct_primary: float
    ct_secondary: float
    voltage_level: dict[str, float]
    r_setting: dict[str, float]
    x_setting: dict[str, float]
    control: RegulatorControl
    tap_setting: dict[str, int]
    reg_type: str  # 'A' or 'B'
    effective_reg_ratio: dict[str, float]
    nominal_voltage: dict[str, float]

    # aux info
    # xfmr_sec_v_mag: float
    # i_line_mag: float

class System:
    def __init__(
        self,
        study_type: int,
        simulation_time: float,
        lines: list[Line],
        nodes: list[Node],
        equipments: list[Equipment],
        # src_nodes: list[Node],
        loads: list[Load],
        sources: list[Source],
        inverters: list[Inverter],
        generators: list[Generator],
        slack_bus: Node,
        w_nominal: float,
        base_power: float | None = None,
        base_voltage: float | None = None,
    ):
        self.lines: list[Line] = lines
        self.nodes: list[Node] = nodes
        self.equipments: list[Equipment] = equipments
        # self.src_nodes = src_nodes
        self.loads = loads
        self.sources = sources
        self.inverters = inverters
        self.generators = generators  
        self.components = lines + equipments + loads + sources + inverters + generators
        # construct additional lists of each equipment type
        self.equip_transformers = self._transformers()
        self.slack_bus = slack_bus
        self.w_nominal = w_nominal
        self.base_power = base_power
        self.base_voltage = base_voltage

        if study_type == 0:
            self.study_type = StudyType.DYNAMIC
        elif study_type == 1:
            self.study_type = StudyType.POWERFLOW

        self.simulation_time = simulation_time

    def _transformers(self) -> list[Transformer]:
        result = []
        for equip in self.equipments:
            if isinstance(equip, Transformer):
                result.append(equip)
        return result

    def validate():
        """validate the items in the System"""
        pass

    def generate_system_matrices(self):
        pass

    def run(self, plot_results=True):
        # construct the system matrices
        # do the simulation
        # print/plot results
        pass

    def get_node_idx_in_nodes(self, node: Node) -> int:
        for idx, curr_node in enumerate(self.nodes):
            if node == curr_node:
                return idx
        print("[!] node not found in nodes. Please investigate.")
        print(f"node: {node}")
        print(f"nodes: {self.nodes}")
        exit(1)

    def get_node_idx_for_node_id(self, node_id: str) -> int:
        for idx, curr_node in self.nodes:
            if curr_node.node_id == node_id:
                return idx
        print("[!] node_id not found in nodes. Please investigate.")
        print(f"node_id: {node_id}")
        print(f"nodes: {self.nodes}")
        exit(1)

    def get_node_by_id(self, node_id: str) -> Node:
        for node in self.nodes:
            if node.id == node_id:
                return node
        raise ValueError("No node found with node_id:{node_id}")

    def __str__(self) -> str:
        res = "System(\n"
        # res += f"lines: {pformat(self.lines)},\n"
        res += f"nodes: {pformat(self.nodes)},\n"
        res += f"components: {pformat(self.components)}\n"
        res += ")"
        return res


def convert_to_pu(system: System):
    assert system.base_power is not None, "base power not set"
    assert system.base_voltage is not None, "base voltage not set"

    # start from the slack bus node
    r = PUFlowRecord(
        node=system.slack_bus,
        base_power=system.base_power,
        base_voltage=system.base_voltage,
    )

    # update the slack bus node
    system.slack_bus.update_pu_base(system.base_power, system.base_voltage)
    src = [src for src in system.sources if src.terminal.at_node == r.node]
    assert len(src) == 1, f"src: {src} not found"
    src[0].update_pu_base(system.base_power, system.base_voltage)

    worklist = [r]
    visited = set()
    print(f">> worklist: {worklist}")
    while worklist:
        r = worklist.pop(0)
        print(f">> r: {r}")
        # assert r not in visited, f"visited: {visited}, r: {r}"
        if r in visited:
            print(f"[!] already visited: {r}")
            continue

        r.node.update_pu_base(r.base_power, r.base_voltage)

        visited.add(r)

        # adjacent components (from)
        adj_components = adjacent_components(
            r.node, [NodeSide.FROM, NodeSide.AT], system.components
        )
        # skip the source
        if r.node == system.slack_bus:
            adj_components = [
                comp for comp in adj_components if not isinstance(comp, Source)
            ]

        print(f">> adj_components count: {len(adj_components)}")

        # 1. update the component
        for comp in adj_components:
            comp.update_pu_base(r.base_power, r.base_voltage)

            if isinstance(comp, Transformer):
                new_base_voltage = comp.base_voltage_secondary
            else:
                new_base_voltage = comp.base_voltage

            # 2. append other side node to worklist
            assert comp.terminal.get_num_term() in [
                1,
                2,
            ], f"num_term: {comp.get_num_term()} not supported"
            if comp.terminal.get_num_term() == 2:
                next_r = PUFlowRecord(
                    node=get_node_on_side(comp, NodeSide.TO),
                    base_voltage=new_base_voltage,
                    base_power=r.base_power,
                )
                worklist.append(next_r)


def adjacent_components(
    node: Node, sides: list[NodeSide], components: list[DSItem]
) -> list[DSItem]:
    print(f">> looking for adjacent components of node: {node}")
    result = list()

    for comp in components:
        # print(f">> comp: {comp}")
        for side in sides:
            if side == NodeSide.AT and comp.terminal.get_num_term() == 2:
                continue
            if (
                side in [NodeSide.FROM, NodeSide.TO]
                and comp.terminal.get_num_term() == 1
            ):
                continue
            comp_node = get_node_on_side(comp, side)
            # print(f">> comp_node: {comp_node}")
            if comp_node == node:  # comp_node.__eq__(node):
                # print(f">> found adjacent")
                result.append(comp)

    result = list(result)
    return result


@dataclass
class PUFlowRecord:
    node: Node
    base_power: float
    base_voltage: float

    def __eq__(self, other):
        if not isinstance(other, PUFlowRecord):
            return False
        return (
            self.node == other.node
            and self.base_power == other.base_power
            and self.base_voltage == other.base_voltage
        )

    def __hash__(self):
        return hash((self.node, self.base_power, self.base_voltage))
