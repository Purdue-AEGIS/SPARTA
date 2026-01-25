from models.system_model import SystemModel
import numpy as np
import scipy.sparse as sps
from collections import defaultdict
from models.model import Model
from models.model import ValType
from const import NodeSide
from models.component_models.node_model import *
from models.component_models.source_model import *

# from models.component_models.load_model import *   # use for sync study as well
from models.component_models.load_model import *
# from models.component_models.load_model__multi_gfm_study import *

from models.component_models.transformer_model import *
from models.component_models.line_model import *
from models.component_models.switch_model import SwitchModel

# from models.component_models.line_model_neg import *
from models.component_models.capacitor_model import *

# GFMs:
# from models.component_models.equipment_model.volt_regulator_model import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV_refstudy13 import *
from models.component_models.inverters.GFMinverter_model_r0 import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_QV_vir import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_wip import *

# GFLs:
# from models.component_models.inverters.GFLinverter_model_refactor import *
# from models.component_models.inverters.GFLinverter_model_refactor_dc import *
from models.component_models.inverters.GFLinverter_model_refactor_brf import *

# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_unifi import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_unifi_init import *
# from models.component_models.inverters.GFMinverter_model_SVPWM_brf_refactor_vir_imp import *
from simulation.powerflow import Powerflow
import itertools
import matplotlib.pyplot as plt
import json
import utils
from pprint import pformat
import sys
from copy import deepcopy
import pandas as pd
import sksundae as sun

from assimulo.solvers import IDA
from assimulo.problem import Implicit_Problem

# import pickle
import dill as pickle


class Dynamic:
    def __init__(self, model: SystemModel, tsim=0.5):
        self.resfn_count = 0
        self.tsim = tsim

        # run the powerflow first
        self.powerflow = Powerflow(model)
        self.powerflow.run()
        self.powerflow.print_y(self.powerflow.y_final)
        self.powerflow.save_results_xlsx("Powerflow_results.xlsx")

        self.eqn_offset, self.var_offset = {}, {}
        # set w for this simulation as obtained from powerflow
        self.wg = model.system.w_nominal

        # some maps to speed up loopups
        # { node_id -> [comp_id, ...]}
        self.node_components_map = defaultdict(list)
        # { comp_id -> [(node_id, side), ...]}
        self.component_nodes_map = defaultdict(list)
        self.compute_node_component_maps(model)

        # self.stages = [(0, "stage1"), (0.3, "stage2")]
        self.stages = [(0, "stage1"), (0.1, "stage2")]
        # self.stages = [(0, "stage1")]

        self.M_by_stage = {}
        self.K_by_stage = {}

        for stage_start_time, stage in self.stages:
            # 1. M
            self.M = None
            self.stack_M_comp(model, stage)

            self.interface(model)
            print(f"M.shape (after v,i interfacing): {self.M.shape}")

            self.interface_w(model)
            print(f"M.shape (after w interfacing - final): {self.M.shape}")
            assert self.M.shape[0] == self.M.shape[1], f"shape: {self.M.shape}"

            self.M = self.M.tocsc()
            self.M_by_stage[stage] = self.M

            print(f":: dynamic eqn_offset:\n{pformat(self.eqn_offset)}")
            print(f":: dynamic var_offset:\n{pformat(self.var_offset)}")
            # input("continue?")

            # self.num_eqns = sum(comp.num_eqns_dynamic for comp in model.components)
            # print(f"tmp:: self.num_eqns: {self.num_eqns}")
            # assert self.M.shape[0] == self.num_eqns
            # self.num_vars = sum(comp.num_vars_dynamic for comp in model.components)
            # print(f"tmp:: self.num_vars: {self.num_vars}")
            # assert self.M.shape[1] == self.num_vars

            # 2. K
            self.K: sps.coo_array = None
            self.stack_K_comp(model, stage)
            # print(f"tmp:: self.K.shape: {self.K.shape}")
            num_rows_to_add = self.M.shape[0] - self.K.shape[0]
            num_cols_to_add = self.M.shape[1] - self.K.shape[1]
            bottom_right = sps.lil_matrix(
                (num_rows_to_add, num_cols_to_add), dtype=float
            )
            self.K = sps.bmat(
                [[self.K, None], [None, bottom_right]], format="coo", dtype=float
            )

            self.K = self.K.tocsc()
            self.K_by_stage[stage] = self.K

            # assert that K and M are the same size
            print(f"K.shape : {self.K.shape}")
            print(f"M.shape : {self.M.shape}")
            assert self.K.shape == self.M.shape

        # # tmp:
        # assert not np.allclose(
        #     self.M_by_stage["stage1"].toarray(), self.M_by_stage["stage2"].toarray()
        # )

        assert len(self.M_by_stage) > 0
        assert len(self.K_by_stage) > 0

        first_stage = self.stages[0][1]
        self.M = self.M_by_stage[first_stage]
        self.K = self.K_by_stage[first_stage]

        # 3. fy
        # self.fy = sps.lil_matrix((self.M.shape[0], 1), dtype=float)
        self.fy = np.zeros(self.M.shape[0], dtype=float)

        # 4. u
        self.u = np.zeros(self.M.shape[0], dtype=float)

        # 5. algvars (indices)
        # this list should correspond to the empty columns of K
        self.algebraic_idx = self.compute_algebraic_idx(model)
        # print(f"tmp:: type(self.algebraic_idx) : {type(self.algebraic_idx)}")
        # print(f"tmp:: self.algebraic_idx :\n{self.algebraic_idx}")

        # init y0 from powerflow solution
        # init yp0 (zero for now)

    def interface(self, model: SystemModel):
        self.interface_voltages(model)
        print(f"M.shape (after v interfacing): {self.M.shape}")
        self.interface_currents(model)
        print(f"M.shape (after i interfacing): {self.M.shape}")

    def interface_voltages(self, model: SystemModel):
        for node in model.nodes:
            node_v_key = f"{node.get_id()}_v"
            self.eqn_offset[node_v_key] = self.M.shape[0]
            self.var_offset[node_v_key] = self.M.shape[1]

            adjacent: list[Model] = []
            for comp_id in self.node_components_map[node.get_id()]:
                comp_model = model.get_component_by_id(comp_id)
                adjacent.append(comp_model)

            # NOTE:
            # Based on num_term of component, check if this node is the from_node/at_node
            # of this component.
            # If so, we need to apply phasing while interfacing.

            # add the columns for node_v vars to M_iface
            node_phases = node.get_phases()
            node_phases_without_n = [ph for ph in node_phases if ph != "N"]
            Z_node_v = sps.lil_matrix(
                (self.M.shape[0], len(node_phases_without_n)), dtype=float
            )
            self.M = sps.bmat([[self.M, Z_node_v]], format="coo")

            for comp in adjacent:
                if hasattr(comp, "phasing"):
                    raise NotImplementedError("phasing is not implemented yet")

                # if there is no phasing involve, then the node *must* have the phases
                # required at the component
                self._validate_node_to_comp_phases(node, comp)

                # at this point:
                # 1. no phasing is involved
                # 2. component phases are a subset of node phases

                # print(f">> interfacing node ({node}) with comp ({comp})...")
                new_M_rows = self.interface_node_comp_voltages(node, comp)
                # print(f">> interfacing node ({node}) with comp ({comp}) done.")

                # stack the component matrix in the "M_iface"
                self.M = sps.bmat(
                    [[self.M], [new_M_rows]],
                    format="coo",
                )

    def interface_currents(self, model: SystemModel):
        """
        1) This function creates the interface equation by summing up the currents leaving each node and directed
           towards the connected components.
        2) It takes self object and object of SystemModel containing models of all the components.
        3) It returns the all_comp_iface_imat containing the KCL equations for each phase of each node.
        4) Unlike voltage interface there will be one equation for each phase of a node, summing up all the currents leaving that node-phase.
        6) To implement above following steps are implemented:
            a) Iterate over each node.
            b) Get all the components connected to a node (using node_components_map) in a list named adjacent
            c) Iterate over each component in the adjacent. Iterate over each phase of the node
            d) Check if the node is a 'from_node' or 'to_node' for two terminal components or 'at_node' for single terminal
            e) Based on above and the ph get the local index of the "I" in that component and add the global offset to get the index of the "I" in the
                y vector of M_comp.
            f) Set all the "I" to 1 since they are leaving the present node and sum up to zero.
        7) Above steps are repeated for each node and the all_comp_iface_imat matrix is compiled.
        8) The matrix corresponding to the node injections is all zeros of with same no. or rows as all_comp_iface_imat matrix.

        """
        # kcl at nodes
        for node in model.nodes:
            node_i_key = f"{node.get_id()}_i"
            self.eqn_offset[node_i_key] = self.M.shape[0]
            self.var_offset[node_i_key] = self.M.shape[1]

            comp_iface_imat = None
            node_iface_imat = None
            M_num_vars = self.M.shape[1]

            adjacent: list[Model] = []
            for comp_id in self.node_components_map[node.get_id()]:
                comp_model = model.get_component_by_id(comp_id)
                adjacent.append(comp_model)

            # NOTE:
            # Based on num_term of component, check if this node is the from_node/at_node
            # of this component.
            # If so, we need to apply phasing while interfacing.

            # one equation will have all components,
            # so the iteration would be on the phases of the node.

            node_phases = node.get_phases()
            node_phases_without_n = [ph for ph in node_phases if ph != "N"]

            for ph in node_phases_without_n:
                # one equation will have all components
                new_M_row = sps.lil_matrix((1, M_num_vars))
                new_node_i_row = sps.lil_matrix((1, len(node_phases_without_n)))
                for comp in adjacent:
                    if ph not in comp.get_phases():
                        continue
                    side = comp.get_node_side(node.get_id())
                    comp_local_i_idx = comp.get_local_idx_dynamic("I", ph, side)
                    comp_i_idx = self.var_offset[comp.get_id()] + comp_local_i_idx
                    new_M_row[0, comp_i_idx] = 1

                comp_iface_imat = sps.bmat(
                    [[comp_iface_imat], [new_M_row]], format="coo"
                )

                node_iface_imat = sps.bmat(
                    [[node_iface_imat], [new_node_i_row]], format="coo"
                )

            self.M = sps.bmat(
                [[self.M, None], [comp_iface_imat, node_iface_imat]],
                format="coo",
            )

        print(f">> M.shape (after i node interfacing - mid): {self.M.shape}")

        # current injections at nodes
        # - eqn will be of the form: -Inode -Isources -Iloads = 0
        M_num_vars = self.M.shape[1]
        for node in model.nodes:
            node_i_key = f"{node.get_id()}_i"
            node_i_offset = self.var_offset[node_i_key]

            adjacent_stc: list[Model] = []
            for comp_id in self.node_components_map[node.get_id()]:
                comp_model = model.get_component_by_id(comp_id)
                if comp_model.get_num_term() == 1:
                    adjacent_stc.append(comp_model)

            node_phases = node.get_phases()
            node_phases_without_n = [ph for ph in node_phases if ph != "N"]

            for ph in node_phases_without_n:
                new_M_row = sps.lil_matrix((1, M_num_vars))

                local_node_iph_offset = node_phases_without_n.index(ph)
                node_iph_idx = node_i_offset + local_node_iph_offset
                new_M_row[0, node_iph_idx] = -1

                # phase currents being injected at this node due to stc
                for comp in adjacent_stc:
                    if ph not in comp.get_phases():
                        continue
                    side = comp.get_node_side(node.get_id())
                    comp_local_i_idx = comp.get_local_idx_dynamic("I", ph, side)
                    comp_i_idx = self.var_offset[comp.get_id()] + comp_local_i_idx
                    if isinstance(comp, SourceModel):
                        # new_M_row[0, comp_i_idx] = -1
                        new_M_row[0, comp_i_idx] = 1
                    elif isinstance(comp, LoadModel):
                        new_M_row[0, comp_i_idx] = -1
                    elif isinstance(comp, ShuntCapacitorModel):
                        new_M_row[0, comp_i_idx] = -1
                    else:
                        raise ValueError(f"unhandled component type: {comp}")

                self.M = sps.bmat([[self.M], [new_M_row]], format="coo")

    def _interface_w_const_v_src(self, model: SystemModel):
        all_comp_iface_wmat = None
        M_num_vars = self.M.shape[1]

        # 1. interface all components (except GFM and GFL)
        for comp in model.components:
            print(f"tmp1:: comp: {comp}")
            if (
                isinstance(comp, GFLInverterModel)
                or isinstance(comp, GFMInverterModel)
                or isinstance(comp, SwitchModel)
            ):
                continue

            print(f">> interfacing w for non-constsrc comp: {comp}")
            comp_local_w_idx = comp.get_local_idx_dynamic("w", None, None)
            comp_w_idx = self.var_offset[comp.get_id()] + comp_local_w_idx

            new_M_row = sps.lil_matrix((1, M_num_vars))
            new_M_row[0, comp_w_idx] = -1

            print(f"tmp:: adding row to all_comp_iface_wmat (1)")
            all_comp_iface_wmat = sps.bmat([[all_comp_iface_wmat], [new_M_row]])

        # additional eqn for wg = u (below M_comp)
        Z_row = sps.lil_matrix((1, M_num_vars))
        print(f"tmp:: adding row to all_comp_iface_wmat (0)")
        all_comp_iface_wmat = sps.bmat([[all_comp_iface_wmat], [Z_row]], format="coo")

        # wg col
        num_eqns = all_comp_iface_wmat.shape[0]
        global_w_mat = sps.lil_matrix((num_eqns, 1))
        global_w_mat[:, 0] = 1
        global_w_mat[-1, 0] = -1

        # combine
        self.M = sps.bmat(
            [[self.M, None], [all_comp_iface_wmat, global_w_mat]],
            format="coo",
        )

    def _interface_w_gfm(self, model: SystemModel):
        all_comp_iface_wmat = None
        M_num_vars = self.M.shape[1]

        num_gfm_inverter_src = 0

        # 1. interface all non-gfm components (except GFLInverters)
        for comp in model.components:
            print(f"tmp:: comp: {comp}")
            if isinstance(comp, GFLInverterModel) or isinstance(comp, SwitchModel):
                continue

            if isinstance(comp, GFMInverterModel):
                num_gfm_inverter_src += 1
                # all gfms are to be interfaced together (ref. hugo paper)
                continue

            print(f">> interfacing w for non-inverter comp: {comp}")
            comp_local_w_idx = comp.get_local_idx_dynamic("w", None, None)
            comp_w_idx = self.var_offset[comp.get_id()] + comp_local_w_idx

            new_M_row = sps.lil_matrix((1, M_num_vars))
            new_M_row[0, comp_w_idx] = -1

            print(f"tmp:: adding row to all_comp_iface_wmat (1)")
            all_comp_iface_wmat = sps.bmat([[all_comp_iface_wmat], [new_M_row]])

        # 2. interface gfm-components:
        # one row connecting wg with wighted average of all gfm's w
        new_M_row = sps.lil_matrix((1, M_num_vars))
        for comp in model.components:
            if isinstance(comp, GFMInverterModel):
                comp_local_w_idx = comp.get_local_idx_dynamic("w", None, None)
                comp_w_idx = self.var_offset[comp.get_id()] + comp_local_w_idx
                # TODO: this should ideally follow weighted average (see Hugo Paper eqn 43)
                # currently we are just assuming equal weights.
                new_M_row[0, comp_w_idx] = -1 / num_gfm_inverter_src
        print(f"tmp:: adding row to all_comp_iface_wmat (2)")
        all_comp_iface_wmat = sps.bmat(
            [[all_comp_iface_wmat], [new_M_row]], format="coo"
        )

        # wg col
        num_eqns = all_comp_iface_wmat.shape[0]
        global_w_mat = sps.lil_matrix((num_eqns, 1))
        global_w_mat[:, 0] = 1

        # combine
        self.M = sps.bmat(
            [[self.M, None], [all_comp_iface_wmat, global_w_mat]],
            format="coo",
        )

    def interface_w(self, model: SystemModel):
        # [x] constant v source cannot work with GFM (for now)
        # [x] if constant v source then wg <- u

        # find if we have a constant voltage source amongst our midst
        has_constant_v_src = False
        has_gfm_inverter_src = False
        for comp in model.components:
            if isinstance(comp, ConstantVoltageModel):
                has_constant_v_src = True
            elif isinstance(comp, GFMInverterModel):
                has_gfm_inverter_src = True

        if has_constant_v_src:
            # ConstSrc + GFLs + GFMs
            self._interface_w_const_v_src(model)
        elif has_gfm_inverter_src:
            # GFMs + GFLs
            self._interface_w_gfm(model)
        else:
            raise ValueError("atleast one constantvsrc or GFM is required")

        # print(f"tmp:: all_comp_iface_wmat.shape : {all_comp_iface_wmat.shape}")
        # print(f"tmp:: global_w_mat.shape : {global_w_mat.shape}")

    def _validate_node_to_comp_phases(self, node: NodeModel, comp: Model):
        node_phases = node.get_phases()
        node_phases_set = set(node_phases)

        comp_phases = comp.get_phases()
        comp_phases_set = set(comp_phases)

        if not comp_phases_set.issubset(node_phases_set):
            print(f"[!] comp: {pformat(comp.__dict__)}")
            raise ValueError(
                f"phases of component ({comp.get_id()}) are not a subset of phases of node ({node.get_id()}) [PLEASE INVESTIGATE]"
            )

    def interface_node_comp_voltages(
        self, node: NodeModel, comp: Model
    ) -> tuple[sps.coo_array]:
        """
        1) This function implements the interfacing of the node voltages with the component voltages for a node-component pair.
        2) It takes self object, node and component model as input.
        3) It returns the M matrix rows and the node voltage matrix rows for the present node-component pair.
        4) To implement above following steps are implemented:
            a) Check whether the node is 'from-node' or 'to-node' for two terminal components or 'at-node' for single terminal components.
            b) Get the phases of the node and the component.
            c) Get the phases of the node and the component without the neutral phase.
            d) Check whether the component is single phase or multi-phase.
            e) If single phase, equate the respective phases of the node and the component.
            f) If multi-phase, equate the respective phase differences of the node and the component.
            g) Stack the M matrix rows and the node voltage matrix rows for each phase of the node and the component.
        5) Above steps are repeated for all the phases of the node and the component.
        6) The M matrix rows and the node voltage matrix rows are stacked for all the phases of the node and the component.
        7) The final M matrix rows and the node voltage matrix rows are returned.
        """

        # print(
        #     f">> interface_node_comp_voltages(): \nnode={node} \ncomp={comp} \nval_type={val_type}"
        # )

        # check whether the node is 'from-node' or 'to-node' for two terminal components or 'at-node' for single terminal components.
        side = comp.get_node_side(node.get_id())

        M_num_cols = self.M.shape[1]
        node_phases = node.get_phases()
        node_phases_without_n = [
            ph for ph in node_phases if ph != "N"
        ]  # iterate over each node phase except the neutral phase.
        comp_phases = comp.get_phases()
        comp_phases_without_n = [
            ph for ph in comp_phases if ph != "N"
        ]  # iterate over each component phase except the neutral phase.

        new_M_rows = None
        key = f"{node.get_id()}_v"

        if len(comp_phases_without_n) == 1:
            # single phase:
            # equate respective phases
            ph = comp_phases_without_n[0]
            new_M_row = sps.lil_matrix((1, M_num_cols))

            comp_local_v_idx = comp.get_local_idx_dynamic("V", ph, side)
            comp_v_idx = self.var_offset[comp.get_id()] + comp_local_v_idx
            new_M_row[0, comp_v_idx] = -1

            # print(f">> interface_node_comp_voltages(): comp_v_idx = {comp_v_idx}")

            node_local_v_idx = node_phases_without_n.index(ph)
            node_v_idx = self.var_offset[key] + node_local_v_idx
            new_M_row[0, node_v_idx] = 1

            # print(f">> interface_node_comp_voltages(): node_v_idx = {node_v_idx}")

            new_M_rows = new_M_row

        elif len(comp_phases_without_n) == 2:
            # 2 phase
            # TODO: am: change phase names from a/b to ph1/ph2
            # ph1
            ph = comp_phases_without_n[0]

            comp_local_v_idx_a = comp.get_local_idx_dynamic("V", ph, side)
            comp_v_idx_a = self.var_offset[comp.get_id()] + comp_local_v_idx_a

            node_local_v_idx_a = node_phases_without_n.index(ph)
            node_v_idx_a = self.var_offset[key] + node_local_v_idx_a

            # ph2
            ph = comp_phases_without_n[1]

            comp_local_v_idx_b = comp.get_local_idx_dynamic("V", ph, side)
            comp_v_idx_b = self.var_offset[comp.get_id()] + comp_local_v_idx_b

            node_local_v_idx_b = node_phases_without_n.index(ph)
            node_v_idx_b = self.var_offset[key] + node_local_v_idx_b

            # vph1, vph2
            new_M_row1 = sps.lil_matrix((1, M_num_cols))
            new_M_row1[0, comp_v_idx_a] = 1
            new_M_row1[0, node_v_idx_a] = -1

            new_M_row2 = sps.lil_matrix((1, M_num_cols))
            new_M_row2[0, comp_v_idx_b] = 1
            new_M_row2[0, node_v_idx_b] = -1

            new_M_rows = sps.bmat([[new_M_row1], [new_M_row2]], format="coo")

        else:
            # 3 phase
            # equate respective phase diffs
            # ph0 = comp_phases_without_n[0]
            # comp_phases_without_n.append(ph0)

            # a
            ph = comp_phases_without_n[0]

            comp_local_v_idx_a = comp.get_local_idx_dynamic("V", ph, side)
            comp_v_idx_a = self.var_offset[comp.get_id()] + comp_local_v_idx_a

            node_local_v_idx_a = node_phases_without_n.index(ph)
            node_v_idx_a = self.var_offset[key] + node_local_v_idx_a

            # b
            ph = comp_phases_without_n[1]

            comp_local_v_idx_b = comp.get_local_idx_dynamic("V", ph, side)
            comp_v_idx_b = self.var_offset[comp.get_id()] + comp_local_v_idx_b

            node_local_v_idx_b = node_phases_without_n.index(ph)
            node_v_idx_b = self.var_offset[key] + node_local_v_idx_b

            # c
            ph = comp_phases_without_n[2]

            comp_local_v_idx_c = comp.get_local_idx_dynamic("V", ph, side)
            comp_v_idx_c = self.var_offset[comp.get_id()] + comp_local_v_idx_c

            node_local_v_idx_c = node_phases_without_n.index(ph)
            node_v_idx_c = self.var_offset[key] + node_local_v_idx_c

            # va, vb, vc
            new_M_row1 = sps.lil_matrix((1, M_num_cols))
            new_M_row1[0, comp_v_idx_a] = 1
            new_M_row1[0, node_v_idx_a] = -1

            new_M_row2 = sps.lil_matrix((1, M_num_cols))
            new_M_row2[0, comp_v_idx_b] = 1
            new_M_row2[0, node_v_idx_b] = -1

            new_M_row3 = sps.lil_matrix((1, M_num_cols))
            new_M_row3[0, comp_v_idx_c] = 1
            new_M_row3[0, node_v_idx_c] = -1

            # print(
            #     f">> self.var_offset_imag[comp.get_id()]: {self.var_offset_imag[comp.get_id()]}"
            # )
            # print(f">> interface_node_comp_voltages(): comp_v_idx_a = {comp_v_idx_a}")
            # print(f">> interface_node_comp_voltages(): node_v_idx_a = {node_v_idx_a}")
            # print(f">> interface_node_comp_voltages(): comp_v_idx_b = {comp_v_idx_b}")
            # print(f">> interface_node_comp_voltages(): node_v_idx_b = {node_v_idx_b}")
            # print(f">> interface_node_comp_voltages(): comp_v_idx_c = {comp_v_idx_c}")
            # print(f">> interface_node_comp_voltages(): node_v_idx_c = {node_v_idx_c}")

            new_M_rows = sps.bmat(
                [[new_M_row1], [new_M_row2], [new_M_row3]], format="coo"
            )

        return new_M_rows

    def y0(self, zero_start=False) -> np.ndarray:
        # return sps.lil_matrix((self.M.shape[1], 1), dtype=float)
        # return self.initial_y()
        return self.initial_y(zero_start)

        # return np.zeros(self.M.shape[1], dtype=float)

    def yp0(self, y0_dyn: np.ndarray, zero_start=False) -> np.ndarray:
        return self.initial_yp(y0_dyn, zero_start)
        # return np.zeros(self.M.shape[1], dtype=float)
        # return sps.lil_matrix((self.M.shape[1], 1), dtype=float)

    def stack_K_comp(self, model: SystemModel, stage=None):
        for comp in model.components:
            K_comp = comp.get_K_dynamic(stage)
            # if isinstance(comp, SwitchModel):
            #     K_comp = comp.get_K_dynamic(stage)
            # else:
            #     K_comp = comp.get_K_dynamic()
            comp_id = comp.get_id()

            if self.K is None:
                self.eqn_offset[comp_id] = 0
                self.var_offset[comp_id] = 0
                self.K = K_comp

            else:
                self.eqn_offset[comp_id] = self.K.shape[0]
                self.var_offset[comp_id] = self.K.shape[1]
                self.K = sps.bmat([[self.K, None], [None, K_comp]], format="coo")

    def stack_M_comp(self, model: SystemModel, stage=None):
        """
        1. This function stacks the M matrices of all the components in the network.
        2. It takes self, model as input, where models is the object of SystemModel class containing all the components.
        3. It iterates over all the components and calls the get_M_powerflow function of each component.
        4. The M matrices of all the components are stacked to form the M_comp matrix on the self object.
        5. The eqn_offset and var_offset are updated for each component.
        6. ordering -> line + equip + load + source
        """
        for comp in model.components:
            M_comp = comp.get_M_dynamic(stage)
            print(f">> stacking M for {comp}")
            # if isinstance(comp, SwitchModel):
            #     M_comp = comp.get_M_dynamic(stage)
            # else:
            #     M_comp = comp.get_M_dynamic()
            # print(f"M_comp : {M_comp.shape}")
            comp_id = comp.get_id()
            self._stack_diagonally_M(M_comp, comp_id)

    def _stack_diagonally_M(self, comp_mat: sps.coo_array, comp_id):
        if self.M is None:
            self.eqn_offset[comp_id] = 0
            self.var_offset[comp_id] = 0
            self.M = comp_mat
        else:
            self.eqn_offset[comp_id] = self.M.shape[0]
            self.var_offset[comp_id] = self.M.shape[1]
            self.M = sps.bmat([[self.M, None], [None, comp_mat]], format="coo")

    def compute_algebraic_idx(self, model: SystemModel) -> list[int]:
        assert self.K is not None
        # just get the column indices of K which are zero.
        csc = self.K.tocsc()
        nnz_per_col = csc.getnnz(axis=0)
        empty_col_indices = np.where(nnz_per_col == 0)[0]
        # print(f"tmp:: empty_col_indices: {empty_col_indices}")
        return empty_col_indices.tolist()

    def run(self):
        input("powerflow done, continue?")

        tspan = np.linspace(0, self.tsim, 1000)
        zero_start = True
        y0 = self.y0(zero_start=zero_start)
        yp0 = self.yp0(y0, zero_start=zero_start)

        # inverter = self.powerflow.model.components[2]
        # idx_inverter_start = self.var_offset["inverter0"]
        # idx_theta = idx_inverter_start + inverter.var_offset_dynamic["theta"]
        # yp0[idx_theta] = 376.991118

        model = Implicit_Problem(self.resfn_assimulo, y0, yp0, 0)
        sim = IDA(model)
        assimulo_alg_var = np.ones(y0.shape[0])
        assimulo_alg_var[self.algebraic_idx] = 0
        sim.algvar = assimulo_alg_var
        # sim.make_consistent("IDA_Y_INIT")
        sim.make_consistent("IDA_YA_YDP_INIT")
        sim.suppress_alg = True
        sim.maxh = 0.0001
        sim.atol = 1e-4
        sim.rtol = 1e-4
        sim.report_continuously = True
        sim.verbosity = 20
        t, y, yprime = sim.simulate(self.tsim)

        print(f"[***] simulation ran")

        self.process_result_assimulo(t, y)

        exit(1)

        self.solver = sun.ida.IDA(
            self.resfn,
            atol=1e-3,
            rtol=1e-4,
            algebraic_idx=self.algebraic_idx,
            # max_num_steps=100000,
            # calc_initcond='yp0',
            # calc_init_dt=-0.01,
            # min_step = 50e-6,
            # max_step = .001
        )
        # y0 = self.y0()
        # yp0 = self.yp0()
        # print(f"tmp:: y0.shape : {y0.shape}")
        # print(f"tmp:: yp0.shape : {yp0.shape}")
        assert y0.shape == yp0.shape
        assert len(y0.shape) == 1
        assert len(yp0.shape) == 1
        self.soln = self.solver.solve(tspan, y0, yp0)
        print(f"tmp:: soln:\n{self.soln}")

        if self.soln.success:
            self.process_result_sundae()
        else:
            print(f"[!] failed :(")

    def process_result_assimulo(self, t, y):
        # save the results in a pickle file
        with open("gfm_synchronizing_study.pkl", "wb") as f:
            pickle.dump(self.powerflow.model, f)
            pickle.dump(self.var_offset, f)
            pickle.dump(t, f)
            pickle.dump(y, f)

        # Basic result processing: plot voltages and currents for the first load (if present)
        loads = self.powerflow.model.loads
        if not loads:
            print("No loads found for result processing.")
            return

        # load
        load = loads[0]
        idx_load_start = self.var_offset[load.get_id()]
        idx_load_v_start = idx_load_start + load.get_local_idx_dynamic(
            "V", ph=None, side=None
        )
        load_num_phases = len([ph for ph in load.get_phases() if ph != "N"])
        load_v = y[:, idx_load_v_start : idx_load_v_start + load_num_phases]

        # i
        idx_load_i_start = idx_load_start + load.get_local_idx_dynamic(
            "i", ph=None, side=None
        )
        load_i = y[:, idx_load_i_start : idx_load_i_start + load_num_phases]

        plt.figure()
        plt.title("Load Voltages (Assimulo)")
        plt.plot(t, load_v)
        plt.xlabel("Time (s)")
        plt.ylabel("Voltage")
        plt.legend([f"V_{ph}" for ph in load.get_phases() if ph != "N"])
        plt.grid()
        plt.show()

        plt.figure()
        plt.title("Load Currents (Assimulo)")
        plt.plot(t, load_i)
        plt.xlabel("Time (s)")
        plt.ylabel("Current")
        plt.legend([f"I_{ph}" for ph in load.get_phases() if ph != "N"])
        plt.grid()
        plt.show()

        wg = y[:, -1]
        plt.figure()
        plt.title("wg (Assimulo)")
        plt.plot(t, wg)
        plt.xlabel("Time (s)")
        plt.ylabel("freq")
        plt.legend("wg")
        plt.grid()
        # plt.show()

        # # capactior - V
        sources = self.powerflow.model.sources

        inverter = None
        for source in sources:
            if isinstance(source, InverterModel):
                inverter = source
                break

        if inverter:
            idx_inv = self.var_offset[inverter.get_id()]

            idx_filter_V_start = idx_inv + inverter.get_local_idx_dynamic(
                "V", ph=None, side=None
            )
            idx_filter_V_end = idx_filter_V_start + 3
            filter_v = y[:, idx_filter_V_start:idx_filter_V_end]
            filter_vc = y[:, idx_filter_V_start + 3 : idx_filter_V_end + 3]

            idx_filter_I_start = idx_inv + inverter.get_local_idx_dynamic(
                "I", ph=None, side=None
            )
            idx_filter_I_end = idx_filter_I_start + 3
            filter_i = y[:, idx_filter_I_start:idx_filter_I_end]

            idx_inverter_w = idx_inv + inverter.get_local_idx_dynamic(
                "w", ph=None, side=None
            )
            inverter_w = y[:, idx_inverter_w : idx_inverter_w + 1]

            plt.figure()
            plt.title("Inverter Filter terminal Voltages (Assimulo)")
            plt.plot(t, filter_v)
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage")
            plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
            plt.grid()
            # plt.show()

            plt.figure()
            plt.title("Inverter Filter Vc Voltages (Assimulo)")
            plt.plot(t, filter_vc)
            plt.xlabel("Time (s)")
            plt.ylabel("Voltage")
            plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
            plt.grid()

            plt.figure()
            plt.title("Inverter Filter terminal Current (Assimulo)")
            plt.plot(t, filter_i)
            plt.xlabel("Time (s)")
            plt.ylabel("Current")
            plt.legend([f"I_{ph}" for ph in inverter.get_phases() if ph != "N"])
            plt.grid()
            # plt.show()

            idx_filter_I_start = idx_inv + inverter.get_local_idx_dynamic(
                "I", ph=None, side=None
            )
            idx_filter_I_end = idx_filter_I_start + 3
            filter_I = y[:, idx_filter_I_start:idx_filter_I_end]

            plt.figure()
            plt.title("Inverter Filter terminal Current (Assimulo)")
            plt.plot(t, filter_I)
            plt.xlabel("Time (s)")
            plt.ylabel("Current")
            plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
            plt.grid()
            plt.show()

        #     idx_inv_cap_V_start = (
        #         idx_inv + inverter.get_local_idx_dynamic("V", ph=None, side=None) + 3
        #     )
        #     idx_inv_cap_V_end = idx_inv_cap_V_start + 3
        #     cap_v = y[:, idx_inv_cap_V_start:idx_inv_cap_V_end]

        #     idx_w = (idx_inv + inverter.get_local_idx_dynamic("w", ph = None, side = None))

        #     idx_duty_cycle_start = (
        #         idx_inv + inverter.get_local_idx_dynamic("m_abc", ph=None, side=None)
        #     )
        #     idx_duty_cycle_end = idx_duty_cycle_start + 3

        #     idx_Vin_qdref_start = (
        #         idx_inv + inverter.get_local_idx_dynamic("v_qd_ref", ph=None, side=None)
        #     )
        #     idx_Vin_qdref_end = idx_Vin_qdref_start + 2

        #     idx_i_vir_start = (
        #         idx_inv + inverter.get_local_idx_dynamic("i_vir_qd", ph=None, side=None)
        #     )
        #     idx_i_vir_end = idx_i_vir_start + 2

        #     i_vir_qd = y[:, idx_i_vir_start : idx_i_vir_end ]

        #     idx_vgrid_qd_start = (
        #         idx_inv + inverter.get_local_idx_dynamic("Vgrid_qd", ph=None, side=None)
        #     )
        #     idx_vgrid_qd_end = idx_i_vir_start + 2

        #     v_grid_qd = y[:, idx_vgrid_qd_start : idx_vgrid_qd_end ]

        #     idx_vinverter_qd_start = (
        #         idx_inv + inverter.get_local_idx_dynamic("Vgrid_qd", ph=None, side=None)
        #     )
        #     idx_vinverter_qd_end = idx_i_vir_start + 2

        #     v_inverter_qd = y[:, idx_vinverter_qd_start : idx_vinverter_qd_end ]

        #     w_inverter = y[:, idx_w]

        #     duty_cycle = y[:,  idx_duty_cycle_start : idx_duty_cycle_end]

        #     Vin_qdref = y[:, idx_Vin_qdref_start : idx_Vin_qdref_end]

        #     plt.figure()
        #     plt.title("Inverter virtual current (Assimulo)")
        #     plt.plot(t, i_vir_qd)
        #     plt.xlabel("Time (s)")
        #     plt.ylabel("Voltage")
        #     plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
        #     plt.grid()

        #     plt.figure()
        #     plt.title("Inverter grid voltage qd (Assimulo)")
        #     plt.plot(t, v_grid_qd)
        #     plt.xlabel("Time (s)")
        #     plt.ylabel("Voltage")
        #     plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
        #     plt.grid()

        #     plt.figure()
        #     plt.title("Inverter voltage qd (Assimulo)")
        #     plt.plot(t, v_inverter_qd)
        #     plt.xlabel("Time (s)")
        #     plt.ylabel("Voltage")
        #     plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
        #     plt.grid()

        #     plt.show()

        # plt.figure()
        # plt.title("Inverter Capacitor Voltages (Assimulo)")
        # plt.plot(t, cap_v)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Voltage")
        # plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
        # plt.grid()

        # plt.figure()
        # plt.title("Inverter Voltage qdref (Assimulo)")
        # plt.plot(t, Vin_qdref)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Voltage qdref")
        # plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
        # plt.grid()

        # plt.figure()
        # plt.title("Inverter w (Assimulo)")
        # plt.plot(t, w_inverter)
        # plt.xlabel("Time (s)")
        # plt.ylabel("w")
        # plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
        # plt.grid()

        # plt.figure()
        # plt.title("Inverter w (Assimulo)")
        # plt.plot(t, duty_cycle)
        # plt.xlabel("Time (s)")
        # plt.ylabel("duty_cycle")
        # plt.legend([f"V_{ph}" for ph in inverter.get_phases() if ph != "N"])
        # plt.grid()

        # plt.figure()
        # plt.title("Inverter Capacitor Currents (Assimulo)")
        # plt.plot(t, cap_i)
        # plt.xlabel("Time (s)")
        # plt.ylabel("Current")
        # plt.legend([f"I_{ph}" for ph in inverter.get_phases() if ph != "N"])
        # plt.grid()
        # plt.show()

    def process_result_sundae(self):
        # load - v (all phases)

        loads = self.powerflow.model.loads
        assert len(loads) >= 1

        load = loads[0]
        idx_load_start = self.var_offset[load.get_id()]
        idx_load_v_start = idx_load_start + load.get_local_idx_dynamic(
            "V", ph=None, side=None
        )
        load_num_phases = len([ph for ph in load.get_phases() if ph != "N"])
        load_v = self.soln.y[:, idx_load_v_start : idx_load_v_start + load_num_phases]

        # i
        idx_load_i_start = idx_load_start + load.get_local_idx_dynamic(
            "i", ph=None, side=None
        )
        load_i = self.soln.y[:, idx_load_i_start : idx_load_i_start + load_num_phases]

        plt.plot(self.soln.t, load_v)
        plt.show()

        plt.plot(self.soln.t, load_i)
        plt.show()

        print(f"")

    def get_stage(self, t) -> str:
        if len(self.stages) == 1:
            return self.stages[0][1]

        for i, (start_time, stage) in enumerate(self.stages):
            if i == len(self.stages) - 1:
                return stage
            if t >= start_time and t < self.stages[i + 1][0]:
                return stage

        raise ValueError("should not be reachable")

    def resfn_assimulo(self, t, y: np.ndarray, yp: np.ndarray) -> np.ndarray:
        self.resfn_count += 1
        print("-----------------------------------")
        print(f"t: {t}")
        print(f"resfn_count: {self.resfn_count}")
        # print(f"tmp:: before type(res) : {type(res)}")
        # print(f"tmp:: res.shape : {res.shape}")

        stage = self.get_stage(t)

        self.M = self.M_by_stage[stage]
        self.K = self.K_by_stage[stage]

        self.update_fy(t, y, yp, stage)
        self.update_u(t, y, stage)

        # # tmp:
        # idx_inv = self.var_offset["inverter0"]
        # inv = self.powerflow.model.components[2]
        # idx_inv_w = idx_inv + inv.var_offset_dynamic["w"]
        # print(f"tmp:: inv w: {y[idx_inv_w]}")

        term1 = self.K @ yp
        term2 = self.M @ y
        term3 = self.fy
        term4 = self.u

        res = self.K @ yp + self.M @ y + self.fy + self.u
        # print(f"res:\n{pformat(res)}")
        res_norm = np.linalg.norm(res, ord=np.inf)
        print(f"res norm: {np.linalg.norm(res, ord=np.inf)}")
        max_res_idx = np.argmax(np.abs(res))
        print(f"res norm idx: {max_res_idx}")

        # if abs(y[idx_inv_w]) > 0:
        #     # input("continue?")
        #     # exit(1)

        # if res_norm < 1e-6:
        #     print("check res_norm")
        #     exit(1)

        # print("dummy")
        # input("continue?")

        return res

    def resfn(self, t, y: np.ndarray, yp: np.ndarray, res):
        self.resfn_count += 1
        print(f"t: {t}")
        print(f"resfn_count: {self.resfn_count}")
        # print(f"tmp:: before type(res) : {type(res)}")
        # print(f"tmp:: res.shape : {res.shape}")

        # if t > abs(1e-9):
        #     print(f"move ahead of t!")
        #     exit(1)

        self.update_fy(t, y)
        self.update_u(t, y)

        term1 = self.K @ yp
        term2 = self.M @ y
        term3 = self.fy
        term4 = self.u

        # tmp::
        # inverter_var_offset = self.var_offset["inverter0"]
        # inverter = self.powerflow.model.sources[0]
        # idx_vg = inverter_var_offset + inverter.var_offset_dynamic["Es_abc"]
        # print(f"Es_a : {y[idx_vg]}")
        # print(f"Es_b : {y[idx_vg + 1]}")
        # print(f"Es_c : {y[idx_vg + 2]}")

        # print(f"tmp:: self.M[132]: {self.M.toarray()[132]}")
        # print(f"tmp:: self.M[132][187]: {self.M[132].toarray()[187]}")
        # print(f"tmp:: self.M[132][188]: {self.M[132].toarray()[188]}")
        # print(f"tmp:: self.M[132][189]: {self.M[132][189]}")

        res[:] = self.K @ yp + self.M @ y + self.fy + self.u
        print(f"res:\n{pformat(res)}")
        res_norm = np.linalg.norm(res, ord=np.inf)
        # if abs(res_norm) < 1e-6:
        #     print("something changed!!")
        #     exit(1)
        print(f"res norm: {np.linalg.norm(res, ord=np.inf)}")
        max_res_idx = np.argmax(np.abs(res))
        print(f"res norm idx: {max_res_idx}")
        # print("dummy")
        # input("continue?")

    def update_fy(self, t: float, y, yp, stage=None):
        # print(f"tmp:: type(y) : {type(y)}")
        self.fy[:] = 0

        for comp in self.powerflow.model.components:
            # print(f"tmp:: update_fy for comp: {comp}")
            y_comp = self._get_comp_y(comp, y)
            yp_comp = self._get_comp_yp(comp, yp)
            comp_fy = comp.get_fy_dynamic(t, y_comp, yp_comp, stage)
            # if isinstance(comp, SwitchModel):
            #     comp_fy = comp.get_fy_dynamic(t, y_comp, stage)
            # else:
            #     comp_fy = comp.get_fy_dynamic(t, y_comp)

            comp_row_start = self.eqn_offset[comp.get_id()]
            comp_row_end = comp_row_start + comp.num_eqns_dynamic
            # print(f"type of comp_fy :{type(comp_fy)}")
            # print(f"type of comp_row_start :{type(comp_row_start)}")
            # print(f"type of comp_row_end :{type(comp_row_end)}")
            self.fy[comp_row_start:comp_row_end] = comp_fy

    def update_u(self, t: float, y, stage=None):
        self.u[:] = 0

        has_constant_v_src = False

        for comp in self.powerflow.model.components:
            # print(f"tmp:: update_u for comp: {comp}")
            if isinstance(comp, ConstantVoltageModel):
                has_constant_v_src = True

            y_comp = self._get_comp_y(comp, y)
            comp_u = comp.get_u_dynamic(t, y_comp)

            comp_row_start = self.eqn_offset[comp.get_id()]
            comp_row_end = comp_row_start + comp.num_eqns_dynamic
            self.u[comp_row_start:comp_row_end] = comp_u

        if has_constant_v_src:
            self.u[-1] = self.wg

    def _get_comp_y(self, comp: Model, y):
        comp_idx_start = self.var_offset[comp.get_id()]
        comp_idx_end = comp_idx_start + comp.num_vars_dynamic
        y_comp = y[comp_idx_start:comp_idx_end]
        return y_comp
    
    def _get_comp_yp(self, comp: Model, yp):
        comp_idx_start = self.var_offset[comp.get_id()]
        comp_idx_end = comp_idx_start + comp.num_vars_dynamic
        yp_comp = yp[comp_idx_start:comp_idx_end]
        return yp_comp

    def compute_node_component_maps(self, model: SystemModel):
        for comp in model.components:
            if comp.num_term == 1:
                node = comp.obj.terminal.at_node
                self.node_components_map[node.id].append(comp.get_id())
                self.component_nodes_map[comp.get_id()].append((node.id, NodeSide.AT))

            elif comp.num_term == 2:
                from_node = comp.obj.terminal.from_node
                self.node_components_map[from_node.id].append(comp.get_id())
                self.component_nodes_map[comp.get_id()].append(
                    (from_node.id, NodeSide.FROM)
                )
                to_node = comp.obj.terminal.to_node
                self.node_components_map[to_node.id].append(comp.get_id())
                self.component_nodes_map[comp.get_id()].append(
                    (to_node.id, NodeSide.TO)
                )

            else:
                raise NotImplementedError

    def initial_yp(self, y0_dyn: np.ndarray, zero_start: bool) -> np.ndarray:
        y_pf_final = self.powerflow.y_dict(self.powerflow.y_final)
        y_complex = np.array([cmplx for (_var, (cmplx, _mag)) in y_pf_final])

        yp0_dyn = np.zeros(self.M.shape[1], dtype=float)

        w_nom = self.powerflow.model.system.w_nominal

        # components
        for comp in self.powerflow.model.components:
            comp_var_idx_start = self.powerflow.var_offset_real[comp.get_id()]
            comp_var_idx_end = comp_var_idx_start + comp.num_vars
            y_pf_comp = y_complex[comp_var_idx_start:comp_var_idx_end]

            comp_var_idx_dyn_start = self.var_offset[comp.get_id()]
            comp_var_idx_dyn_end = comp_var_idx_dyn_start + comp.num_vars_dynamic

            y0_dyn_comp = y0_dyn[comp_var_idx_dyn_start:comp_var_idx_dyn_end]

            if zero_start:
                yp0_dyn_comp = comp.initial_yp_dynamic_zero(
                    y0_dyn_comp, y_pf_comp, w_nom
                )
            else:
                yp0_dyn_comp = comp.initial_yp_dynamic(y0_dyn_comp, y_pf_comp, w_nom)

            yp0_dyn[comp_var_idx_dyn_start:comp_var_idx_dyn_end] = yp0_dyn_comp

        return yp0_dyn

    def initial_y(self, zero_start: bool) -> np.ndarray:
        """
        Needs y from powerflow
        pure real variables to be initialised as it is
        complex variables have magnitude and angle so they will be initialized as mag*cos(w*t)
        w = w_global obtained in powerflow
        t = 0
        """
        # [ (var, (complex, mag)) ]
        y_pf_final = self.powerflow.y_dict(self.powerflow.y_final)
        y_complex = np.array([cmplx for (_var, (cmplx, _mag)) in y_pf_final])
        # print(f"tmp:: len(y_complex) : {len(y_complex)}")

        # # TODO: modify this when we need to "init" inverters
        # # remove the variables related to powerflow equations
        # num_vars_dynamic = self.M.shape[1]
        # y_filtered = y_complex[: (num_vars_dynamic - 1)]
        # print(f"tmp:: len(y_filtered) : {len(y_filtered)}")
        # assert len(y_filtered) == self.M.shape[1] - 1

        # # y = ymag * cos(angle)
        # # need:
        # # - ymag
        # # - angle
        # # given:
        # # y_re, y_im

        # # ymag = np.abs(y_complex)
        # # angle = np.angle(val, deg=True)

        # y0_lst = []
        # for yvar in y_filtered:
        #     mag = np.abs(yvar)
        #     angle = np.angle(yvar)
        #     val = mag * np.cos(angle)
        #     y0_lst.append(val)

        # y0_lst.append(self.powerflow.y_final[-1, 0])

        # y0 = np.array(y0_lst)
        # return y0

        # new:
        y0_dyn = np.zeros(self.M.shape[1], dtype=float)

        w_nom = self.powerflow.model.system.w_nominal

        # components
        for comp in self.powerflow.model.components:
            print(f"initial_y for comp: {comp}")
            comp_var_idx_start = self.powerflow.var_offset_real[comp.get_id()]
            comp_var_idx_end = comp_var_idx_start + comp.num_vars
            y_comp = y_complex[comp_var_idx_start:comp_var_idx_end]

            if zero_start:
                y0_dyn_comp = comp.initial_guess_dynamic_zero(y_comp, w_nom)
            else:
                y0_dyn_comp = comp.initial_guess_dynamic(y_comp, w_nom)

            comp_var_idx_dyn_start = self.var_offset[comp.get_id()]
            comp_var_idx_dyn_end = comp_var_idx_dyn_start + comp.num_vars_dynamic
            y0_dyn[comp_var_idx_dyn_start:comp_var_idx_dyn_end] = y0_dyn_comp

        # nodes:
        if not zero_start:
            for node in self.powerflow.model.nodes:
                # v:
                v_key = f"{node.get_id()}_v_re"
                vre_pf_idx_start = self.powerflow.var_offset_real[v_key]
                vre_pf_idx_end = vre_pf_idx_start + len(node.get_phases_without_n())
                vre = self.powerflow.y_final[vre_pf_idx_start:vre_pf_idx_end]
                v_key = f"{node.get_id()}_v_im"
                vim_pf_idx_start = self.powerflow.var_offset_imag[v_key]
                vim_pf_idx_end = vim_pf_idx_start + len(node.get_phases_without_n())
                vim = self.powerflow.y_final[vim_pf_idx_start:vim_pf_idx_end]
                v = vre + 1j * vim
                v = v.toarray().flatten()
                v_abs = np.abs(v)
                v_angle = np.angle(v)
                v = np.sqrt(2) * v_abs * np.cos(v_angle)

                node_v_key = f"{node.get_id()}_v"
                node_v_dyn_idx_start = self.var_offset[node_v_key]
                node_v_dyn_idx_end = node_v_dyn_idx_start + len(
                    node.get_phases_without_n()
                )
                y0_dyn[node_v_dyn_idx_start:node_v_dyn_idx_end] = v

                # i:
                i_key = f"{node.get_id()}_i_re"
                ire_pf_idx_start = self.powerflow.var_offset_real[i_key]
                ire_pf_idx_end = ire_pf_idx_start + len(node.get_phases_without_n())
                ire = self.powerflow.y_final[ire_pf_idx_start:ire_pf_idx_end]
                i_key = f"{node.get_id()}_i_im"
                iim_pf_idx_start = self.powerflow.var_offset_imag[i_key]
                iim_pf_idx_end = iim_pf_idx_start + len(node.get_phases_without_n())
                iim = self.powerflow.y_final[iim_pf_idx_start:iim_pf_idx_end]
                i = ire + 1j * iim
                i = i.toarray().flatten()
                i_mag = np.abs(i)
                i_angle = np.angle(i)
                i = np.sqrt(2) * i_mag * np.cos(i_angle)

                node_i_key = f"{node.get_id()}_i"
                node_i_dyn_idx_start = self.var_offset[node_i_key]
                node_i_dyn_idx_end = node_i_dyn_idx_start + len(
                    node.get_phases_without_n()
                )
                y0_dyn[node_i_dyn_idx_start:node_i_dyn_idx_end] = i

        # wg:
        if not zero_start:
            y0_dyn[-1] = self.powerflow.y_final[-1, 0]

        return y0_dyn
