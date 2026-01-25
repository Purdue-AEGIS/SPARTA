from models.system_model import SystemModel
import numpy as np
import scipy.sparse as sps
from collections import defaultdict
from models.model import Model
from models.model import ValType
import const
from const import NodeSide
from models.component_models.node_model import *

from models.component_models.source_model import *
# from models.component_models.source_model__gfl_study import *

from models.component_models.load_model import *
# from models.component_models.load_model__multi_gfm_study import *

from models.component_models.transformer_model import *
from models.component_models.switch_model import *
from models.component_models.line_model import *

# from models.component_models.line_model_neg import *
from models.component_models.capacitor_model import *
from models.component_models.volt_regulator_model import *

import itertools
import matplotlib.pyplot as plt
import json
import utils
from pprint import pformat
import sys
from copy import deepcopy
import pandas as pd


class Powerflow:
    def __init__(self, model: SystemModel):
        """
        1. This class solves the powerflow equations for the given network.
        2. First it loads the M and u matrices for all the components in the network.
        3. Then it interfaces the components with the nodes.The interfacing is done in three steps:
            a) Interface the voltages
            b) Interface the currents
            c) Interface the w variables of components with the global w.
        4. Final matrix is formed by stacking the M and u matrices of all the components and the interface matrices.
        5. TODO: Implementation of Newton-Raphson method to solve the system of equations for powerflow.

        """
        self.model = model

        # Create the empty matrices, for stacking component models
        self.M_comp_split = None
        self.u_comp_split = None

        self.M_comp_real = None
        self.M_comp_imag = None

        self.u_comp_real = None
        self.u_comp_imag = None

        # self.u_comp = None

        # Create the empty matrices, for stacking the interface equations below the component models
        # self.M_iface = None
        # self.u_iface = None

        # offset for equations and variables. eqn_offset pertains to the rows and var_offset pertains to the columns

        self.eqn_offset_real, self.eqn_offset_imag = {}, {}
        self.var_offset_real, self.var_offset_imag = {}, {}

        # some maps to speed up loopups
        # { node_id -> [comp_id, ...]}
        self.node_components_map = defaultdict(list)
        # { comp_id -> [(node_id, side), ...]}
        self.component_nodes_map = defaultdict(list)
        self.compute_node_component_maps(model)

        # These function calls will stack the M and u matrices for all the components in the network and make them an attribute of the class.
        self.stack_M_comp(model)
        self.stack_u_comp(model)

        print(f">> self.var_offset_real: \n{pformat(self.var_offset_real)}")
        print(f">> self.var_offset_imag: \n{pformat(self.var_offset_imag)}")

        # plt.spy(self.M_comp_real)
        # plt.title("M_comp_real")
        # plt.show()

        # plt.spy(self.M_comp_imag)
        # plt.title("M_comp_imag")
        # plt.show()

        # print(f"M_comp.shape: {self.M_comp.shape}")
        # print(f"u_comp.shape: {self.u_comp.shape}")

        # M and u should have the same row count
        assert self.M_comp_real.shape[0] == self.u_comp_real.shape[0]
        assert self.M_comp_imag.shape[0] == self.u_comp_imag.shape[0]

        self.M_comp_split = sps.bmat(
            [[self.M_comp_real, None], [None, self.M_comp_imag]],
            dtype=float,
            format="csc",
        )
        self.u_comp_split = sps.bmat(
            [[self.u_comp_real], [self.u_comp_imag]], format="coo"
        )

        assert self.M_comp_split.shape[0] == self.u_comp_split.shape[0]

        # plt.spy(self.M_comp_split)
        # plt.title("M_comp_split")
        # plt.show()

        # These function calls will interface the components and then update the w of each component with that of global w.
        self.interface(model)

        print(
            f"M_iface_split.shape (after v,i interfacing): {self.M_iface_split.shape}"
        )
        print(
            f"u_iface_split.shape (after v,i interfacing): {self.u_iface_split.shape}"
        )

        # M and u should have the same row count
        assert self.M_iface_split.shape[0] == self.u_iface_split.shape[0]

        # plt.spy(self.M_iface_split)
        # plt.title("M_iface_split")
        # plt.show()

        # print(f"M_iface dtype (after interface): {self.M_iface.dtype}")

        self.add_powerflow_eqns(model)

        print(f"M_iface_split.shape (after powerflow eqns): {self.M_iface_split.shape}")
        print(f"u_iface_split.shape (after powerflow eqns): {self.u_iface_split.shape}")

        # M and u should have the same row count
        assert self.M_iface_split.shape[0] == self.u_iface_split.shape[0]

        # print(f"M_iface dtype (after powerflow): {self.M_iface.dtype}")

        # plt.spy(self.M_iface)
        # plt.show()

        self.interface_w(model)

        print(f"M_iface_split.shape (after w): {self.M_iface_split.shape}")
        print(f"u_iface_split.shape (after w): {self.u_iface_split.shape}")

        # M and u should have the same row count
        assert self.M_iface_split.shape[0] == self.u_iface_split.shape[0]

        # print(f">> self.node_offset_real: \n{pformat(self.var_offset_real)}")
        # print(f">> self.node_offset_imag: \n{pformat(self.var_offset_imag)}")

        # print(">> spy plot of M_iface_split")
        # plt.spy(self.M_iface_split)
        # plt.grid(visible=True)
        # # Set custom tick locations at every 10 units
        # plt.xticks(np.arange(0, self.M_iface_split.shape[1], 10))
        # plt.yticks(np.arange(0, self.M_iface_split.shape[0], 10))
        # plt.title("M_iface_split", fontsize=20)
        # plt.show()

        print(f"M_iface_split.shape (final): {self.M_iface_split.shape}")
        print(f"u_iface_split.shape (final): {self.u_iface_split.shape}")

        # M_iface_split_rank = np.linalg.matrix_rank(self.M_iface_split.toarray())
        # print(f">> M_iface_split rank: {M_iface_split_rank}")
        # # input("continue?")

        print(f"eqn_offset_real: {pformat(self.eqn_offset_real)}")
        print(f"eqn_offset_imag: {pformat(self.eqn_offset_imag)}")
        print(f"var_offset_real: {pformat(self.var_offset_real)}")
        print(f"var_offset_imag: {pformat(self.var_offset_imag)}")

        # tmp:
        # print(f"line[0] var_offsets: {pformat(self.model.lines[0].var_offset)}")
        # print(f"line[1] var_offsets: {pformat(self.model.lines[1].var_offset)}")
        if self.model.equipments:
            print(
                f"equipment var_offsets: {pformat(self.model.equipments[0].var_offset)}"
            )
        # print(f"source var_offsets: {pformat(self.model.sources[0].var_offset)}")
        # print(f"load var_offsets: {pformat(self.model.loads[0].var_offset)}")

        # tmp:
        # print(f"line num_vars: {pformat(self.model.lines[0].num_vars)}")
        if self.model.equipments:
            print(f"euquipment num_vars: {pformat(self.model.equipments[0].num_vars)}")
        # print(f"source num_vars: {pformat(self.model.sources[0].num_vars)}")
        # print(f"load num_vars: {pformat(self.model.loads[0].num_vars)}")

        # tmp:
        # print(f"line num_eqns: {pformat(self.model.lines[0].num_eqns)}")
        if self.model.equipments:
            print(f"euquipment num_eqns: {pformat(self.model.equipments[0].num_eqns)}")
        # print(f"source num_eqns: {pformat(self.model.sources[0].num_eqns)}")
        # print(f"load num_eqns: {pformat(self.model.loads[0].num_eqns)}")

        self.composite_var_list = self._generate_composite_var_list()
        self.composite_eqn_list = self._generate_composite_eqn_list()

        # reserving memory
        # self.fy = sps.lil_matrix((self.M_iface_split.shape[0], 1), dtype=complex)
        self.fy_split = sps.lil_matrix((self.M_iface_split.shape[0], 1), dtype=float)
        # self.pd_fy = sps.lil_matrix(self.M_iface.shape, dtype=float)
        self.pd_fy_split = sps.lil_matrix(self.M_iface_split.shape, dtype=float)
        self.jacobian = sps.lil_matrix(self.M_iface_split.shape, dtype=float)

        # tmp:
        arr = self.M_iface_split.toarray()
        header = ",".join(self.composite_var_list)
        np.savetxt(
            "M_iface_split.csv",
            arr,
            delimiter=",",
            fmt="%.5f",
            header=header,
            comments="",
        )

    def _generate_composite_eqn_list(self) -> list[str]:
        # components
        # nodes
        # w_g

        result = []

        # components
        # re:
        for comp in self.model.components:
            num_eqn = comp.num_eqns
            comp_eqns = []
            for i in range(1, num_eqn + 1):
                eqn = f"{comp.get_id()}__{i}__re"
                comp_eqns.append(eqn)
            result.extend(comp_eqns)
        # im:
        for comp in self.model.components:
            num_eqn = comp.num_eqns_complex
            comp_eqns = []
            for i in range(1, num_eqn + 1):
                eqn = f"{comp.get_id()}__{i}__im"
                comp_eqns.append(eqn)
            result.extend(comp_eqns)

        print(f">>> len(result) (after stacking) = {len(result)}")

        # nodes - V
        # - this would really depend on the number of equations for each component-node pair.
        # re:
        for node in self.model.nodes:
            node_eqns = []
            for comp_id in self.node_components_map[node.get_id()]:
                comp = self.model.get_component_by_id(comp_id)
                phases_without_n = [ph for ph in comp.get_phases() if ph != "N"]
                for ph in phases_without_n:
                    eqn = f"{node.get_id()}__{comp.get_id()}_v_{ph}__re"
                    node_eqns.append(eqn)
            result.extend(node_eqns)
        for node in self.model.nodes:
            node_eqns = []
            for comp_id in self.node_components_map[node.get_id()]:
                comp = self.model.get_component_by_id(comp_id)
                phases_without_n = [ph for ph in comp.get_phases() if ph != "N"]
                for ph in phases_without_n:
                    eqn = f"{node.get_id()}__{comp.get_id()}_v_{ph}__im"
                    node_eqns.append(eqn)
            result.extend(node_eqns)

        print(f">>> len(result) (after v interface) = {len(result)}")

        # nodes - I
        # - this would be the same number as the number of phases at this node.
        # re:
        for node in self.model.nodes:
            node_eqns = []
            for ph in node.get_phases_without_n():
                eqn = f"{node.get_id()}__i_{ph}__re"
                node_eqns.append(eqn)

            for ph in node.get_phases_without_n():
                eqn = f"{node.get_id()}__i_{ph}_stc__re"
                node_eqns.append(eqn)

            result.extend(node_eqns)
        # im:
        for node in self.model.nodes:
            node_eqns = []
            for ph in node.get_phases_without_n():
                eqn = f"{node.get_id()}__i_{ph}__im"
                node_eqns.append(eqn)

            for ph in node.get_phases_without_n():
                eqn = f"{node.get_id()}__i_{ph}_stc__im"
                node_eqns.append(eqn)

            result.extend(node_eqns)

        print(f">>> len(result) after i interface = {len(result)}")

        # nodes - S
        # - this would be the same number as the nubmer of phases at this node.
        # re:
        for node in self.model.nodes:
            node_eqns = []
            for ph in node.get_phases_without_n():
                eqn = f"{node.get_id()}__s_{ph}__re"
                node_eqns.append(eqn)
            result.extend(node_eqns)
        # im:
        for node in self.model.nodes:
            node_eqns = []
            for ph in node.get_phases_without_n():
                eqn = f"{node.get_id()}__s_{ph}__im"
                node_eqns.append(eqn)
            result.extend(node_eqns)

        print(f">>> len(result) after s interface = {len(result)}")

        # wg
        for comp in self.model.components:
            if isinstance(comp, SwitchModel):
                continue
            eqn = f"{comp.get_id()}__w"
            result.append(eqn)
        result.append("w_g")

        print(f">>> len(result) final = {len(result)}")

        assert len(result) == self.M_iface_split.shape[0], (
            f"len(result) = {len(result)}, but self.M_iface_split.shape[0]={self.M_iface_split.shape[0]}"
        )

        return result

    def _generate_composite_var_list(self) -> list[str]:
        # components
        # nodes
        # w_g

        result = []

        # components
        # re : vars_real + vars_complex
        for comp in self.model.components:
            var_offsets = [
                (offset, var) for (var, offset) in comp.var_offset_real.items()
            ]
            var_offsets.sort()

            for i, (offset, var) in enumerate(var_offsets):
                if i == len(var_offsets) - 1:
                    count = comp.num_vars_real - offset
                else:
                    count = var_offsets[i + 1][0] - offset
                comp_type = comp.get_basetype()
                result.extend([f"{comp_type}__{comp.get_id()}__{var}__re"] * count)

            var_offsets = [
                (offset, var) for (var, offset) in comp.var_offset_complex.items()
            ]
            var_offsets.sort()

            for i, (offset, var) in enumerate(var_offsets):
                if i == len(var_offsets) - 1:
                    count = comp.num_vars_complex - offset
                else:
                    count = var_offsets[i + 1][0] - offset
                comp_type = comp.get_basetype()
                result.extend([f"{comp_type}__{comp.get_id()}__{var}__re"] * count)

            # print(f">> after comp (real) ({comp.get_id()})...")
            # print(f">> len(result) = {len(result)}")

        # im : vars_complex
        for comp in self.model.components:
            var_offsets = [
                (offset, var) for (var, offset) in comp.var_offset_complex.items()
            ]
            var_offsets.sort()

            for i, (offset, var) in enumerate(var_offsets):
                if i == len(var_offsets) - 1:
                    count = comp.num_vars_complex - offset
                else:
                    count = var_offsets[i + 1][0] - offset
                comp_type = comp.get_basetype()
                result.extend([f"{comp_type}__{comp.get_id()}__{var}__im"] * count)

            # print(f">> after comp (complex) ({comp.get_id()})...")
            # print(f">> len(result) = {len(result)}")

        # nodes - V
        # re
        for node in self.model.nodes:
            phases = node.get_phases_without_n()
            for phase in phases:
                result.append(f"node__{node.get_id()}__V_{phase}__re")
        # im
        for node in self.model.nodes:
            phases = node.get_phases_without_n()
            for phase in phases:
                result.append(f"node__{node.get_id()}__V_{phase}__im")

        # nodes - I
        # re
        for node in self.model.nodes:
            phases = node.get_phases_without_n()
            for phase in phases:
                result.append(f"node__{node.get_id()}__I_{phase}__re")
        # im
        for node in self.model.nodes:
            phases = node.get_phases_without_n()
            for phase in phases:
                result.append(f"node__{node.get_id()}__I_{phase}__im")

        # nodes - S
        # re
        for node in self.model.nodes:
            phases = node.get_phases_without_n()
            for phase in phases:
                result.append(f"node__{node.get_id()}__S_{phase}__re")
        # im
        for node in self.model.nodes:
            phases = node.get_phases_without_n()
            for phase in phases:
                result.append(f"node__{node.get_id()}__S_{phase}__im")

        result.append(f"wg")

        assert len(result) == self.M_iface_split.shape[1], (
            f"len(result) = {len(result)}, but self.M_iface_split.shape[1]={self.M_iface_split.shape[1]}"
        )

        return result

    # returns [var_ph1, var_ph2, ...]
    def _get_reg_line_data(self, reg_id: str, var: str) -> list[complex]:
        assert var in ["V", "I"]

        reg = self.model.get_component_by_id(reg_id)
        ph = reg.get_phases()[0]

        reg_local_idx = reg.get_local_idx(var, ValType.REAL, ph, NodeSide.TO)
        reg_I_ph_idx_real = self.var_offset_real[reg_id] + reg_local_idx
        var_real = self.y_final[reg_I_ph_idx_real : reg_I_ph_idx_real + reg.n_ph]

        reg_local_idx = reg.get_local_idx(var, ValType.IMAG, ph, NodeSide.TO)
        reg_I_ph_idx_imag = self.var_offset_imag[reg_id] + reg_local_idx
        var_imag = self.y_final[reg_I_ph_idx_imag : reg_I_ph_idx_imag + reg.n_ph]

        return var_real + 1j * var_imag

    def run(self):
        # regulator #
        have_reg, have_reg_auto_control, auto_reg_ids = False, False, []
        for comp in self.model.components:
            if isinstance(comp, VoltRegulatorModel):
                have_reg = True
                if comp.obj.control == "automatic":
                    have_reg_auto_control = True
                    auto_reg_ids.append(comp.get_id())

        # if we have a reg && if reg control is "automatic"
        if have_reg and have_reg_auto_control:
            # run with 0 tap
            self.y_final = self.newton_raphson_powerflow()

            self.print_y(self.y_final)

            # calc tap from result
            # we need the Iline for the regulator
            for reg_id in auto_reg_ids:
                vline = self._get_reg_line_data(reg_id, "V")
                iline = self._get_reg_line_data(reg_id, "I")
                reg: VoltRegulatorModel = self.model.get_component_by_id(reg_id)
                reg.calc_tap_and_update(vline, iline)
                # update global M for the updated regulator params
                M_reg_new_re, M_reg_new_im = reg.get_M_powerflow()

                # re
                start_row_idx = self.eqn_offset_real[reg_id]
                end_row_idx = start_row_idx + reg.num_eqns
                start_col_idx = self.var_offset_real[reg_id]
                end_col_idx = start_col_idx + reg.num_vars
                self.M_iface_split = self.M_iface_split.tocsc()
                self.M_iface_split[
                    start_row_idx:end_row_idx, start_col_idx:end_col_idx
                ] = M_reg_new_re

                # im
                start_row_idx = self.eqn_offset_imag[reg_id]
                end_row_idx = start_row_idx + reg.num_eqns_complex
                start_col_idx = self.var_offset_imag[reg_id]
                end_col_idx = start_col_idx + reg.num_vars_complex
                self.M_iface_split[
                    start_row_idx:end_row_idx, start_col_idx:end_col_idx
                ] = M_reg_new_im

            # TODO: clear state

            input("continue ph2?")

        # run
        # implement the Newton Raphson method here
        self.y_final = self.newton_raphson_powerflow()

    ############################################################################################################################################################################
    """
    Below are the function definitions required for loading component models and interfacing them with the nodes.
    """

    def add_powerflow_eqns(self, model: SystemModel):
        def _powerflow_eqn(val_type: ValType):
            for node in model.nodes:
                # key = f"{node.get_id()}_s"
                # self.var_offset[key] = self.M_iface.shape[1]
                if val_type == ValType.REAL:
                    key = f"{node.get_id()}_s_re"
                    self.eqn_offset_real[key] = self.M_iface_split.shape[0]
                    self.var_offset_real[key] = self.M_iface_split.shape[1]
                elif val_type == ValType.IMAG:
                    key = f"{node.get_id()}_s_im"
                    self.eqn_offset_imag[key] = self.M_iface_split.shape[0]
                    self.var_offset_imag[key] = self.M_iface_split.shape[1]

                node_phases = node.get_phases()
                node_phases_without_n = [ph for ph in node_phases if ph != "N"]

                new_node_powerflow_mat = sps.identity(
                    len(node_phases_without_n), format="coo"
                )

                self.M_iface_split = sps.bmat(
                    [[self.M_iface_split, None], [None, -new_node_powerflow_mat]],
                    format="coo",
                )

                new_uiface_mat = sps.lil_matrix((len(node_phases_without_n), 1))
                self.u_iface_split = sps.bmat(
                    [[self.u_iface_split], [new_uiface_mat]], format="coo"
                )

        _powerflow_eqn(ValType.REAL)
        _powerflow_eqn(ValType.IMAG)

    def interface_w(self, model: SystemModel):
        all_comp_iface_wmat = None
        M_iface_num_vars = self.M_iface_split.shape[1]

        for comp in model.components:
            if isinstance(comp, SwitchModel):
                continue
            # print(f">> comp: {comp}")
            comp_local_w_idx = comp.get_local_idx("w", ValType.REAL, None, None)
            comp_w_idx = self.var_offset_real[comp.get_id()] + comp_local_w_idx

            new_M_row = sps.lil_matrix((1, M_iface_num_vars))
            new_M_row[0, comp_w_idx] = -1

            all_comp_iface_wmat = sps.bmat([[all_comp_iface_wmat], [new_M_row]])

        # eqn for global w
        Z_row = sps.lil_matrix((1, M_iface_num_vars))
        all_comp_iface_wmat = sps.bmat([[all_comp_iface_wmat], [Z_row]], format="coo")

        num_eqns = all_comp_iface_wmat.shape[0]
        global_w_mat = sps.lil_matrix((num_eqns, 1))
        global_w_mat[:, 0] = 1
        global_w_mat[-1, 0] = -1

        self.M_iface_split = sps.bmat(
            [[self.M_iface_split, None], [all_comp_iface_wmat, global_w_mat]],
            format="coo",
        )

        num_w_iface_eqns = all_comp_iface_wmat.shape[0]
        all_comp_w_iface_u = sps.lil_matrix((num_w_iface_eqns, 1))
        self.u_iface_split = sps.bmat(
            [[self.u_iface_split], [all_comp_w_iface_u]], format="csc"
        )

        self.u_iface_split[-1] = const.w_nominal

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

    def interface_voltages(self, model: SystemModel):
        """
        1. This function adds equations that equates the component terminal voltage to the node voltage.
        2. It takes self object and model which is the object of SystemModel containing all the component models.
        3. To get the the voltage equations following steps are implemented:
            a) Outerloop: Iterate over every node.
            b) Innerloop: Iterate over every component connected to that node (using 'node_component_map')
            c) set the offsets for adding the voltage interface equations.
            c) Call the interface_node_comp_voltages function, that gives the equation for the present node-component pair.
            d) Above function returns the rows to be stacked below the Mcomp matrix and the corresponding node_voltages in form a matrix
            e) Once all components are iterated over the node we get a node voltage matrix for that node along with the matrix of eqns for all the components
                connected to the node.
            g) Above process is repeated for all nodes and all_comp_iface_vmat and all_node_iface_vmat matrices are formed.
        """
        self.M_iface_split = self.M_comp_split

        # real
        for node in model.nodes:
            node_v_key = f"{node.get_id()}_v_re"
            self.eqn_offset_real[node_v_key] = self.M_iface_split.shape[0]
            self.var_offset_real[node_v_key] = self.M_iface_split.shape[1]

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
                (self.M_iface_split.shape[0], len(node_phases_without_n)), dtype=float
            )
            self.M_iface_split = sps.bmat(
                [[self.M_iface_split, Z_node_v]], format="coo"
            )

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
                new_M_iface_rows = self.interface_node_comp_voltages(
                    node, comp, ValType.REAL
                )
                # print(f">> interfacing node ({node}) with comp ({comp}) done.")

                # stack the component matrix in the "M_iface"
                self.M_iface_split = sps.bmat(
                    [[self.M_iface_split], [new_M_iface_rows]],
                    format="coo",
                )

        # imag
        for node in model.nodes:
            node_v_key = f"{node.get_id()}_v_im"
            self.eqn_offset_imag[node_v_key] = self.M_iface_split.shape[0]
            self.var_offset_imag[node_v_key] = self.M_iface_split.shape[1]

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
                (self.M_iface_split.shape[0], len(node_phases_without_n)), dtype=float
            )
            self.M_iface_split = sps.bmat(
                [[self.M_iface_split, Z_node_v]], format="coo"
            )

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
                new_M_iface_rows = self.interface_node_comp_voltages(
                    node, comp, ValType.IMAG
                )
                # print(f">> interfacing node ({node}) with comp ({comp}) done.")

                # stack the component matrix in the "M_iface"
                self.M_iface_split = sps.bmat(
                    [[self.M_iface_split], [new_M_iface_rows]],
                    format="coo",
                    dtype=float,
                )

    def interface_currents(
        self, model: SystemModel
    ) -> tuple[sps.coo_array, sps.coo_array]:
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

        def _interface(val_type: ValType):
            for node in model.nodes:
                # node_i_key = f"{node.get_id()}_i_re"
                if val_type == ValType.REAL:
                    node_i_key = f"{node.get_id()}_i_re"
                    self.eqn_offset_real[node_i_key] = self.M_iface_split.shape[0]
                    self.var_offset_real[node_i_key] = self.M_iface_split.shape[1]
                else:
                    node_i_key = f"{node.get_id()}_i_im"
                    self.eqn_offset_imag[node_i_key] = self.M_iface_split.shape[0]
                    self.var_offset_imag[node_i_key] = self.M_iface_split.shape[1]

                comp_iface_imat = None
                node_iface_imat = None
                M_iface_num_vars = self.M_iface_split.shape[1]

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
                    new_M_row = sps.lil_matrix((1, M_iface_num_vars))
                    new_node_i_row = sps.lil_matrix((1, len(node_phases_without_n)))
                    for comp in adjacent:
                        if ph not in comp.get_phases():
                            continue
                        side = comp.get_node_side(node.get_id())
                        comp_local_i_idx = comp.get_local_idx("I", val_type, ph, side)
                        # comp_i_idx = self.var_offset_real[comp.get_id()] + comp_local_i_idx
                        if val_type == ValType.REAL:
                            comp_i_idx = (
                                self.var_offset_real[comp.get_id()] + comp_local_i_idx
                            )
                        elif val_type == ValType.IMAG:
                            comp_i_idx = (
                                self.var_offset_imag[comp.get_id()] + comp_local_i_idx
                            )
                        new_M_row[0, comp_i_idx] = 1

                    comp_iface_imat = sps.bmat(
                        [[comp_iface_imat], [new_M_row]], format="coo"
                    )

                    node_iface_imat = sps.bmat(
                        [[node_iface_imat], [new_node_i_row]], format="coo"
                    )

                self.M_iface_split = sps.bmat(
                    [[self.M_iface_split, None], [comp_iface_imat, node_iface_imat]],
                    format="coo",
                )

            print(
                f">> M_iface_split.shape (after i node interfacing - mid): {self.M_iface_split.shape}"
            )

            # current injections at nodes
            # - eqn will be of the form: -Inode -Isources -Iloads = 0
            M_iface_num_vars = self.M_iface_split.shape[1]
            for node in model.nodes:
                # node_i_key = f"{node.get_id()}_i_re"
                if val_type == ValType.REAL:
                    node_i_key = f"{node.get_id()}_i_re"
                    node_i_offset = self.var_offset_real[node_i_key]
                elif val_type == ValType.IMAG:
                    node_i_key = f"{node.get_id()}_i_im"
                    node_i_offset = self.var_offset_imag[node_i_key]

                adjacent_stc: list[Model] = []
                for comp_id in self.node_components_map[node.get_id()]:
                    comp_model = model.get_component_by_id(comp_id)
                    if comp_model.get_num_term() == 1:
                        adjacent_stc.append(comp_model)

                node_phases = node.get_phases()
                node_phases_without_n = [ph for ph in node_phases if ph != "N"]

                for ph in node_phases_without_n:
                    new_M_row = sps.lil_matrix((1, M_iface_num_vars))

                    local_node_iph_offset = node_phases_without_n.index(ph)
                    node_iph_idx = node_i_offset + local_node_iph_offset
                    new_M_row[0, node_iph_idx] = -1

                    # phase currents being injected at this node due to stc
                    for comp in adjacent_stc:
                        if ph not in comp.get_phases():
                            continue
                        side = comp.get_node_side(node.get_id())
                        comp_local_i_idx = comp.get_local_idx("I", val_type, ph, side)
                        # comp_i_idx = self.var_offset_real[comp.get_id()] + comp_local_i_idx
                        if val_type == ValType.REAL:
                            comp_i_idx = (
                                self.var_offset_real[comp.get_id()] + comp_local_i_idx
                            )
                        elif val_type == ValType.IMAG:
                            comp_i_idx = (
                                self.var_offset_imag[comp.get_id()] + comp_local_i_idx
                            )
                        if isinstance(comp, SourceModel):
                            new_M_row[0, comp_i_idx] = -1
                        elif isinstance(comp, LoadModel):
                            new_M_row[0, comp_i_idx] = -1
                        elif isinstance(comp, ShuntCapacitorModel):
                            new_M_row[0, comp_i_idx] = -1
                        else:
                            raise ValueError(f"unhandled component type: {comp}")

                    self.M_iface_split = sps.bmat(
                        [[self.M_iface_split], [new_M_row]], format="coo"
                    )

        _interface(ValType.REAL)
        _interface(ValType.IMAG)

    # returns: M, u
    def interface(self, model: SystemModel) -> tuple[sps.coo_array, sps.coo_array]:
        """
        1) This function creates the final interface matrix consisting of both the voltage and current equations
        2) It calls the interface_voltages function to get the voltage interface equation matrix along with the corresponding node voltage matrix
        3) It calls the interface_currents function to get the current equation matrix for each node and its associated components.
        4) It stacks the M_comp i.e. all component model matrix, voltage interface matrix and current interface matrix to give the final matrix that contains
         all component level and interfacing equations.
        5) Also creates the u_vector corresponding the this final matrix.
        """
        self.interface_voltages(model)
        print(f"M_iface_split.shape (after v interfacing): {self.M_iface_split.shape}")
        # print(f"u_iface.shape (after v interfacing): {self.u_iface.shape}")

        # print(f">> self.var_offset_real: {pformat(self.var_offset_real)}")
        # print(f">> self.var_offset_imag: {pformat(self.var_offset_imag)}")

        self.interface_currents(model)
        print(f"M_iface_split.shape (after i interfacing): {self.M_iface_split.shape}")
        # print(f"u_iface.shape (after i interfacing): {self.u_iface.shape}")

        # print(f">> self.var_offset_real: {pformat(self.var_offset_real)}")
        # print(f">> self.var_offset_imag: {pformat(self.var_offset_imag)}")

        # create the complete u_vector corresponding to the above matrix.
        num_eqns = self.M_iface_split.shape[0]
        curr_u_rows = self.u_comp_split.shape[0]
        num_rows_to_add = num_eqns - curr_u_rows
        Z_rows = sps.lil_matrix((num_rows_to_add, 1), dtype=float)
        self.u_iface_split = sps.bmat(
            [[self.u_comp_split], [Z_rows]], format="coo", dtype=float
        )
        print(
            f">> u_iface_split.shape (after i interfacing): {self.u_iface_split.shape}"
        )

    def interface_node_comp_voltages(
        self, node: NodeModel, comp: Model, val_type: ValType
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

        M_iface_num_cols = self.M_iface_split.shape[1]
        node_phases = node.get_phases()
        node_phases_without_n = [
            ph for ph in node_phases if ph != "N"
        ]  # iterate over each node phase except the neutral phase.
        comp_phases = comp.get_phases()
        comp_phases_without_n = [
            ph for ph in comp_phases if ph != "N"
        ]  # iterate over each component phase except the neutral phase.

        new_M_rows = None
        if val_type == ValType.REAL:
            key = f"{node.get_id()}_v_re"
        elif val_type == ValType.IMAG:
            key = f"{node.get_id()}_v_im"
        else:
            raise ValueError(f"unhandled val_type: {val_type}")

        if len(comp_phases_without_n) == 1:
            # single phase:
            # equate respective phases
            ph = comp_phases_without_n[0]
            new_M_row = sps.lil_matrix((1, M_iface_num_cols))

            comp_local_v_idx = comp.get_local_idx("V", val_type, ph, side)
            # comp_v_idx = self.var_offset_real[comp.get_id()] + comp_local_v_idx
            if val_type == ValType.REAL:
                comp_v_idx = self.var_offset_real[comp.get_id()] + comp_local_v_idx
            elif val_type == ValType.IMAG:
                comp_v_idx = self.var_offset_imag[comp.get_id()] + comp_local_v_idx
            new_M_row[0, comp_v_idx] = -1

            # print(f">> interface_node_comp_voltages(): comp_v_idx = {comp_v_idx}")

            node_local_v_idx = node_phases_without_n.index(ph)
            # node_v_idx = self.var_offset_real[key] + node_local_v_idx
            if val_type == ValType.REAL:
                node_v_idx = self.var_offset_real[key] + node_local_v_idx
            elif val_type == ValType.IMAG:
                node_v_idx = self.var_offset_imag[key] + node_local_v_idx
            new_M_row[0, node_v_idx] = 1

            # print(f">> interface_node_comp_voltages(): node_v_idx = {node_v_idx}")

            new_M_rows = new_M_row

        elif len(comp_phases_without_n) == 2:
            # 2 phase
            # TODO: am: change phase names from a/b to ph1/ph2
            # ph1
            ph = comp_phases_without_n[0]

            comp_local_v_idx_a = comp.get_local_idx("V", val_type, ph, side)
            # comp_v_idx_a = self.var_offset_real[comp.get_id()] + comp_local_v_idx_a
            if val_type == ValType.REAL:
                comp_v_idx_a = self.var_offset_real[comp.get_id()] + comp_local_v_idx_a
            elif val_type == ValType.IMAG:
                comp_v_idx_a = self.var_offset_imag[comp.get_id()] + comp_local_v_idx_a

            node_local_v_idx_a = node_phases_without_n.index(ph)
            # node_v_idx_a = self.var_offset_real[key] + node_local_v_idx_a
            if val_type == ValType.REAL:
                node_v_idx_a = self.var_offset_real[key] + node_local_v_idx_a
            elif val_type == ValType.IMAG:
                node_v_idx_a = self.var_offset_imag[key] + node_local_v_idx_a

            # ph2
            ph = comp_phases_without_n[1]

            comp_local_v_idx_b = comp.get_local_idx("V", val_type, ph, side)
            # comp_v_idx_b = self.var_offset_real[comp.get_id()] + comp_local_v_idx_b
            if val_type == ValType.REAL:
                comp_v_idx_b = self.var_offset_real[comp.get_id()] + comp_local_v_idx_b
            elif val_type == ValType.IMAG:
                comp_v_idx_b = self.var_offset_imag[comp.get_id()] + comp_local_v_idx_b

            node_local_v_idx_b = node_phases_without_n.index(ph)
            # node_v_idx_b = self.var_offset_real[key] + node_local_v_idx_b
            if val_type == ValType.REAL:
                node_v_idx_b = self.var_offset_real[key] + node_local_v_idx_b
            elif val_type == ValType.IMAG:
                node_v_idx_b = self.var_offset_imag[key] + node_local_v_idx_b

            # vph1, vph2
            new_M_row1 = sps.lil_matrix((1, M_iface_num_cols))
            new_M_row1[0, comp_v_idx_a] = 1
            new_M_row1[0, node_v_idx_a] = -1

            new_M_row2 = sps.lil_matrix((1, M_iface_num_cols))
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

            comp_local_v_idx_a = comp.get_local_idx("V", val_type, ph, side)
            # comp_v_idx_a = self.var_offset_real[comp.get_id()] + comp_local_v_idx_a
            if val_type == ValType.REAL:
                comp_v_idx_a = self.var_offset_real[comp.get_id()] + comp_local_v_idx_a
            elif val_type == ValType.IMAG:
                comp_v_idx_a = self.var_offset_imag[comp.get_id()] + comp_local_v_idx_a

            node_local_v_idx_a = node_phases_without_n.index(ph)
            # node_v_idx_a = self.var_offset_real[key] + node_local_v_idx_a
            if val_type == ValType.REAL:
                node_v_idx_a = self.var_offset_real[key] + node_local_v_idx_a
            elif val_type == ValType.IMAG:
                node_v_idx_a = self.var_offset_imag[key] + node_local_v_idx_a

            # b
            ph = comp_phases_without_n[1]

            comp_local_v_idx_b = comp.get_local_idx("V", val_type, ph, side)
            # comp_v_idx_b = self.var_offset_real[comp.get_id()] + comp_local_v_idx_b
            if val_type == ValType.REAL:
                comp_v_idx_b = self.var_offset_real[comp.get_id()] + comp_local_v_idx_b
            elif val_type == ValType.IMAG:
                comp_v_idx_b = self.var_offset_imag[comp.get_id()] + comp_local_v_idx_b

            node_local_v_idx_b = node_phases_without_n.index(ph)
            # node_v_idx_b = self.var_offset_real[key] + node_local_v_idx_b
            if val_type == ValType.REAL:
                node_v_idx_b = self.var_offset_real[key] + node_local_v_idx_b
            elif val_type == ValType.IMAG:
                node_v_idx_b = self.var_offset_imag[key] + node_local_v_idx_b

            # c
            ph = comp_phases_without_n[2]

            comp_local_v_idx_c = comp.get_local_idx("V", val_type, ph, side)
            # comp_v_idx_c = self.var_offset_real[comp.get_id()] + comp_local_v_idx_c
            if val_type == ValType.REAL:
                comp_v_idx_c = self.var_offset_real[comp.get_id()] + comp_local_v_idx_c
            elif val_type == ValType.IMAG:
                comp_v_idx_c = self.var_offset_imag[comp.get_id()] + comp_local_v_idx_c

            node_local_v_idx_c = node_phases_without_n.index(ph)
            # node_v_idx_c = self.var_offset_real[key] + node_local_v_idx_c
            if val_type == ValType.REAL:
                node_v_idx_c = self.var_offset_real[key] + node_local_v_idx_c
            elif val_type == ValType.IMAG:
                node_v_idx_c = self.var_offset_imag[key] + node_local_v_idx_c

            # va, vb, vc
            new_M_row1 = sps.lil_matrix((1, M_iface_num_cols))
            new_M_row1[0, comp_v_idx_a] = 1
            new_M_row1[0, node_v_idx_a] = -1

            new_M_row2 = sps.lil_matrix((1, M_iface_num_cols))
            new_M_row2[0, comp_v_idx_b] = 1
            new_M_row2[0, node_v_idx_b] = -1

            new_M_row3 = sps.lil_matrix((1, M_iface_num_cols))
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

    def stack_u_comp(self, model: SystemModel):
        """
        1. This function stacks the u vector of all the components in the network.
        2. It takes self, model as input, where models is the object of SystemModel class containing all the components.
        3. It iterates over all the components and calls the get_u_powerflow function of each component.
        4. The u vector of all the components are stacked to form the u_comp on the self object.
        5. The eqn_offset created while stacking the M matrices acts as the offset for the u vector.
        6. order -> line + equip + load + src
        """
        # lines
        for line in model.lines:
            (u_line_real, u_line_imag) = line.get_u_powerflow()
            self.u_comp_real = sps.bmat(
                [[self.u_comp_real], [u_line_real]], format="coo"
            )
            self.u_comp_imag = sps.bmat(
                [[self.u_comp_imag], [u_line_imag]], format="coo"
            )

        # equipments
        for equip in model.equipments:
            (u_equip_real, u_equip_imag) = equip.get_u_powerflow()
            self.u_comp_real = sps.bmat(
                [[self.u_comp_real], [u_equip_real]], format="coo"
            )
            self.u_comp_imag = sps.bmat(
                [[self.u_comp_imag], [u_equip_imag]], format="coo"
            )

        # loads
        for load in model.loads:
            u_load_real, u_load_imag = load.get_u_powerflow()
            self.u_comp_real = sps.bmat(
                [[self.u_comp_real], [u_load_real]], format="coo"
            )
            self.u_comp_imag = sps.bmat(
                [[self.u_comp_imag], [u_load_imag]], format="coo"
            )

        # sources
        for source in model.sources:
            u_source_real, u_source_imag = source.get_u_powerflow()
            self.u_comp_real = sps.bmat(
                [[self.u_comp_real], [u_source_real]], format="coo"
            )
            self.u_comp_imag = sps.bmat(
                [[self.u_comp_imag], [u_source_imag]], format="coo"
            )

    def _stack_diagonally_M(
        self, comp_mat_real: sps.coo_array, comp_mat_imag: sps.coo_array, comp_id
    ):
        # real
        if self.M_comp_real is None:
            self.eqn_offset_real[comp_id] = 0
            self.var_offset_real[comp_id] = 0
            self.M_comp_real = comp_mat_real
        else:
            self.eqn_offset_real[comp_id] = self.M_comp_real.shape[0]
            self.var_offset_real[comp_id] = self.M_comp_real.shape[1]
            self.M_comp_real = sps.bmat(
                [[self.M_comp_real, None], [None, comp_mat_real]], format="coo"
            )

        # imag
        if self.M_comp_imag is None:
            self.eqn_offset_imag[comp_id] = 0
            self.var_offset_imag[comp_id] = 0
            self.M_comp_imag = comp_mat_imag
        else:
            self.eqn_offset_imag[comp_id] = self.M_comp_imag.shape[0]
            self.var_offset_imag[comp_id] = self.M_comp_imag.shape[1]
            self.M_comp_imag = sps.bmat(
                [[self.M_comp_imag, None], [None, comp_mat_imag]], format="coo"
            )

    def stack_M_comp(self, model: SystemModel):
        """
        1. This function stacks the M matrices of all the components in the network.
        2. It takes self, model as input, where models is the object of SystemModel class containing all the components.
        3. It iterates over all the components and calls the get_M_powerflow function of each component.
        4. The M matrices of all the components are stacked to form the M_comp matrix on the self object.
        5. The eqn_offset and var_offset are updated for each component.
        6. ordering -> line + equip + load + source
        """
        for comp in self.model.components:
            print(f">> stacking M for {comp}")
            M_comp_real, M_comp_imag = comp.get_M_powerflow()
            # print(f"M_comp : {M_comp.shape}")
            comp_id = comp.get_id()
            self._stack_diagonally_M(M_comp_real, M_comp_imag, comp_id)

            # plt.spy(M_comp_real)
            # plt.title(f"stacked {comp_id} (real)")
            # plt.show()

        # now we need to shift all imag offset by the number of real vars
        for k in self.eqn_offset_imag.keys():
            self.eqn_offset_imag[k] += self.M_comp_real.shape[0]
        for k in self.var_offset_imag.keys():
            self.var_offset_imag[k] += self.M_comp_real.shape[1]

    # for performance reason, we are re-using memory
    def update_fy(self, y: sps.coo_array):
        # reset self.fy
        self.fy_split[:] = 0

        # components
        for comp in self.model.components:
            # print(f">> updating fy for {comp}")
            y_comp_real, y_comp_imag = self._get_comp_y(comp, y)
            comp_fy_real, comp_fy_imag = comp.get_fy_powerflow(y_comp_real, y_comp_imag)
            # print(f">> comp_fy_real: {comp_fy_real.toarray()}")
            # print(f">> comp_fy_imag: {comp_fy_imag.toarray()}")
            # input("continue?")

            comp_re_row_start = self.eqn_offset_real[comp.get_id()]
            comp_re_row_end = comp_re_row_start + comp.num_eqns
            comp_im_row_start = self.eqn_offset_imag[comp.get_id()]
            comp_im_row_end = comp_im_row_start + comp.num_eqns_complex
            self.fy_split[comp_re_row_start:comp_re_row_end] = comp_fy_real
            self.fy_split[comp_im_row_start:comp_im_row_end] = comp_fy_imag

        # nodes
        for node in self.model.nodes:
            node_num_phases = len(node.get_phases_without_n())

            # v
            node_vre, node_vim = self._get_node_y(node, y, var="v")
            node_v = node_vre + 1j * node_vim

            # i
            node_ire, node_iim = self._get_node_y(node, y, var="i")
            node_i = node_ire + 1j * node_iim

            # powerflow
            node_s = node_v.multiply(node_i.conjugate())
            key = f"{node.get_id()}_s_re"
            node_s_re_eqn_start = self.eqn_offset_real[key]
            node_s_re_eqn_end = node_s_re_eqn_start + node_num_phases
            key = f"{node.get_id()}_s_im"
            node_s_im_eqn_start = self.eqn_offset_imag[key]
            node_s_im_eqn_end = node_s_im_eqn_start + node_num_phases
            self.fy_split[node_s_re_eqn_start:node_s_re_eqn_end] = node_s.real
            self.fy_split[node_s_im_eqn_start:node_s_im_eqn_end] = node_s.imag

        # self.fy_split = sps.bmat([[self.fy.real], [self.fy.imag]], format="coo")

    # returns ([real parts of *all* vars, imaginary parts of complex vars])
    def _get_comp_y(
        self, comp: Model, y: sps.coo_array
    ) -> tuple[sps.coo_array, sps.coo_array]:
        comp_re_idx_start = self.var_offset_real[comp.get_id()]
        comp_re_idx_end = comp_re_idx_start + comp.num_vars
        y_real = y[comp_re_idx_start:comp_re_idx_end]

        comp_im_idx_start = self.var_offset_imag[comp.get_id()]
        comp_im_idx_end = comp_im_idx_start + comp.num_vars_complex
        y_imag = y[comp_im_idx_start:comp_im_idx_end]

        return (y_real, y_imag)

    def _get_node_y(
        self, node: NodeModel, y: sps.coo_array, var: str
    ) -> tuple[sps.coo_array, sps.coo_array]:
        assert var in ["i", "v"]

        node_num_phases = len(node.get_phases_without_n())

        key = f"{node.get_id()}_{var}_re"
        node_re_idx_start = self.var_offset_real[key]
        node_re_idx_end = node_re_idx_start + node_num_phases
        y_real = y[node_re_idx_start:node_re_idx_end]

        key = f"{node.get_id()}_{var}_im"
        node_im_idx_start = self.var_offset_imag[key]
        node_im_idx_end = node_im_idx_start + node_num_phases
        y_imag = y[node_im_idx_start:node_im_idx_end]

        return (y_real, y_imag)

    # # updates y in place
    # # values are the complex values to be updated in y
    # def set_node_fy(self, node: NodeModel, var: str, values: sps.coo_array):
    #     assert var in ["s"]

    #     node_num_phases = len(node.get_phases_without_n())
    #     assert node_num_phases == values.shape[0]

    #     key = f"{node.get_id()}_{var}_re"
    #     node_re_idx_start = self.var_offset_real[key]
    #     node_re_idx_end = node_re_idx_start + node_num_phases
    #     y[node_re_idx_start:node_re_idx_end] = values.real

    #     key = f"{node.get_id()}_{var}_im"
    #     node_im_idx_start = self.var_offset_imag[key]
    #     node_im_idx_end = node_im_idx_start + node_num_phases
    #     y[node_im_idx_start:node_im_idx_end] = values.imag

    # for performance reason, we are re-using memory
    def update_jacobian(self, y: sps.coo_array):
        # reset
        # self.pd_fy[:] = 0
        self.pd_fy_split[:] = 0

        for comp in self.model.components:
            # print(f">> updating jacobian for {comp}")
            y_real, y_imag = self._get_comp_y(comp, y)
            rr, ri, ir, ii = comp.get_pd_fy_split(y_real, y_imag)

            # print(f">> rr.shape: {rr.shape}")
            # print(f">> ri.shape: {ri.shape}")
            # print(f">> ir.shape: {ir.shape}")
            # print(f">> ii.shape: {ii.shape}")
            # print(f">> comp.num_eqns: {comp.num_eqns}")
            # print(f">> comp.num_eqns_complex: {comp.num_eqns_complex}")
            # print(f">> comp.num_vars: {comp.num_vars}")
            # print(f">> comp.num_vars_complex: {comp.num_vars_complex}")

            assert rr.shape[0] == comp.num_eqns
            assert rr.shape[1] == comp.num_vars
            assert ri.shape[0] == comp.num_eqns
            assert ri.shape[1] == comp.num_vars_complex
            assert ir.shape[0] == comp.num_eqns_complex
            assert ir.shape[1] == comp.num_vars
            assert ii.shape[0] == comp.num_eqns_complex
            assert ii.shape[1] == comp.num_vars_complex

            comp_re_row_start = self.eqn_offset_real[comp.get_id()]
            comp_re_row_end = comp_re_row_start + comp.num_eqns
            comp_re_col_start = self.var_offset_real[comp.get_id()]
            comp_re_col_end = comp_re_col_start + comp.num_vars

            comp_im_row_start = self.eqn_offset_imag[comp.get_id()]
            comp_im_row_end = comp_im_row_start + comp.num_eqns_complex
            comp_im_col_start = self.var_offset_imag[comp.get_id()]
            comp_im_col_end = comp_im_col_start + comp.num_vars_complex

            self.pd_fy_split[
                comp_re_row_start:comp_re_row_end, comp_re_col_start:comp_re_col_end
            ] = rr

            self.pd_fy_split[
                comp_re_row_start:comp_re_row_end, comp_im_col_start:comp_im_col_end
            ] = ri

            self.pd_fy_split[
                comp_im_row_start:comp_im_row_end, comp_re_col_start:comp_re_col_end
            ] = ir

            self.pd_fy_split[
                comp_im_row_start:comp_im_row_end, comp_im_col_start:comp_im_col_end
            ] = ii

            # plt.spy(comp_pd_fy)
            # plt.show()

        for node in self.model.nodes:
            node_num_phases = len(node.get_phases_without_n())

            # v
            key = f"{node.get_id()}_v_re"
            node_v_re_start = self.var_offset_real[key]
            key = f"{node.get_id()}_v_im"
            node_v_im_start = self.var_offset_imag[key]

            key = f"{node.get_id()}_i_re"
            node_i_re_start = self.var_offset_real[key]
            key = f"{node.get_id()}_i_im"
            node_i_im_start = self.var_offset_imag[key]

            # powerflow
            key = f"{node.get_id()}_s_re"
            node_s_re_eqn_start = self.eqn_offset_real[key]
            key = f"{node.get_id()}_s_im"
            node_s_im_eqn_start = self.eqn_offset_imag[key]

            # s_re
            for offset in range(node_num_phases):
                s_re_row = node_s_re_eqn_start + offset

                v_re_col_offset = node_v_re_start + offset
                v_re = y[v_re_col_offset, 0]
                v_im_col_offset = node_v_im_start + offset
                v_im = y[v_im_col_offset, 0]

                i_re_col_offset = node_i_re_start + offset
                i_re = y[i_re_col_offset, 0]
                i_im_col_offset = node_i_im_start + offset
                i_im = y[i_im_col_offset, 0]

                # v_re
                self.pd_fy_split[s_re_row, v_re_col_offset] = i_re
                # v_im
                self.pd_fy_split[s_re_row, v_im_col_offset] = i_im
                # i_re
                self.pd_fy_split[s_re_row, i_re_col_offset] = v_re
                # i_im
                self.pd_fy_split[s_re_row, i_im_col_offset] = v_im

            # s_im
            for offset in range(node_num_phases):
                s_im_row = node_s_im_eqn_start + offset

                v_re_col_offset = node_v_re_start + offset
                v_re = y[v_re_col_offset, 0]
                v_im_col_offset = node_v_im_start + offset
                v_im = y[v_im_col_offset, 0]

                i_re_col_offset = node_i_re_start + offset
                i_re = y[i_re_col_offset, 0]
                i_im_col_offset = node_i_im_start + offset
                i_im = y[i_im_col_offset, 0]

                # v_re
                self.pd_fy_split[s_im_row, v_re_col_offset] = -i_im
                # v_im
                self.pd_fy_split[s_im_row, v_im_col_offset] = i_re
                # i_re
                self.pd_fy_split[s_im_row, i_re_col_offset] = v_im
                # i_im
                self.pd_fy_split[s_im_row, i_im_col_offset] = -v_re

        # plt.spy(self.pd_fy_split)
        # plt.title("pd_fy_split", fontsize=20)
        # plt.show()

        self.jacobian = self.M_iface_split + self.pd_fy_split

    # returns: [ (var, (complex, mag)) ]
    # var format: "source__source1__i"
    def y_dict(self, y: sps.coo_array) -> list:
        # print(f">> composite_var_list: {pformat(self.composite_var_list)}")

        # OLD ====
        # y_lst = [float(v[0]) for v in y.toarray()]
        # y_re = np.array(y_lst[: self.M_iface.shape[1]])
        # y_im = np.array(y_lst[self.M_iface.shape[1] :])
        # y_complex = y_re + 1j * y_im
        # y_mag = np.abs(y_complex)
        # y_result = zip(y_complex, y_mag)
        # ========

        y_lst = [float(v[0]) for v in y.toarray()]
        vars = []
        y_real, y_imag = [], []

        # COMPONENTS:
        # - vars:
        # for vars, just take the real vars of the components and remove the __re suffix.
        comp_num_vars_real = self.M_comp_real.shape[1]
        vars.extend(
            [
                var.removesuffix("__re")
                for var in self.composite_var_list[:comp_num_vars_real]
            ]
        )

        # - y_real
        y_real.extend(y_lst[:comp_num_vars_real])

        # - y_imag
        for comp in self.model.components:
            comp_id = comp.get_id()
            # for each component, the imaginary parts start at self.var_offset_imag[comp_id]
            y_imag.extend([0.0] * comp.num_vars_real)
            idx_start = self.var_offset_imag[comp_id]
            idx_end = idx_start + comp.num_vars_complex
            y_imag.extend(y_lst[idx_start:idx_end])

        check_count = 0
        for comp in self.model.components:
            check_count += comp.num_vars

        assert (len(y_real) == len(y_imag)) and (
            len(y_real) == len(vars) and len(y_real) == check_count
        )

        # nodes:
        # - vars
        # find the min and max index for any of node vars that relate to voltage (real) i.e.
        # 1. var startswith "node__"
        # 2. var contains "_V_"
        # 3. var endswith "__re"
        idx_vars = list(enumerate(self.composite_var_list))
        # print(f">> idx_vars: {idx_vars}")
        node_v_re_idx_vars = [
            (i, v)
            for (i, v) in idx_vars
            if v.startswith("node__") and "_V_" in v and v.endswith("__re")
        ]
        assert len(node_v_re_idx_vars) > 0
        node_v_re_idx_min = min([i for (i, v) in node_v_re_idx_vars])
        node_v_re_idx_max = max([i for (i, v) in node_v_re_idx_vars])
        node_v_vars_count = node_v_re_idx_max - node_v_re_idx_min + 1

        check_count = 0
        for node in self.model.nodes:
            check_count += len(node.get_phases_without_n())

        assert node_v_vars_count == check_count

        node_v_vars = [
            var.removesuffix("__re")
            for var in self.composite_var_list[
                node_v_re_idx_min : node_v_re_idx_max + 1
            ]
        ]
        assert len(node_v_vars) == check_count
        vars.extend(node_v_vars)

        # - y_real
        node_v_y_real = y_lst[node_v_re_idx_min : node_v_re_idx_min + node_v_vars_count]
        y_real.extend(node_v_y_real)

        # - y_imag
        node_v_y_imag = y_lst[
            node_v_re_idx_max + 1 : node_v_re_idx_max + 1 + node_v_vars_count
        ]
        y_imag.extend(node_v_y_imag)

        assert len(node_v_y_real) == len(node_v_y_imag)

        # ============

        # 1. var startswith "node__"
        # 2. var contains "_I_"
        # 3. var endswith "__re"
        idx_vars = list(enumerate(self.composite_var_list))
        # print(f">> idx_vars: {idx_vars}")
        node_i_re_idx_vars = [
            (i, v)
            for (i, v) in idx_vars
            if v.startswith("node__") and "_I_" in v and v.endswith("__re")
        ]
        assert len(node_i_re_idx_vars) > 0
        node_i_re_idx_min = min([i for (i, v) in node_i_re_idx_vars])
        node_i_re_idx_max = max([i for (i, v) in node_i_re_idx_vars])
        node_i_vars_count = node_i_re_idx_max - node_i_re_idx_min + 1

        check_count = 0
        for node in self.model.nodes:
            check_count += len(node.get_phases_without_n())

        assert node_i_vars_count == check_count

        node_i_vars = [
            var.removesuffix("__re")
            for var in self.composite_var_list[
                node_i_re_idx_min : node_i_re_idx_max + 1
            ]
        ]
        assert len(node_i_vars) == check_count
        vars.extend(node_i_vars)

        # - y_real
        node_i_y_real = y_lst[node_i_re_idx_min : node_i_re_idx_min + node_i_vars_count]
        y_real.extend(node_i_y_real)

        # - y_imag
        node_i_y_imag = y_lst[
            node_i_re_idx_max + 1 : node_i_re_idx_max + 1 + node_i_vars_count
        ]
        y_imag.extend(node_i_y_imag)

        # ============

        # 1. var startswith "node__"
        # 2. var contains "_S_"
        # 3. var endswith "__re"
        # idx_vars = list(enumerate(self.composite_var_list))
        # print(f">> idx_vars: {idx_vars}")
        node_s_re_idx_vars = [
            (i, v)
            for (i, v) in idx_vars
            if v.startswith("node__") and "_S_" in v and v.endswith("__re")
        ]
        assert len(node_s_re_idx_vars) > 0
        node_s_re_idx_min = min([i for (i, v) in node_s_re_idx_vars])
        node_s_re_idx_max = max([i for (i, v) in node_s_re_idx_vars])
        node_s_vars_count = node_s_re_idx_max - node_s_re_idx_min + 1

        check_count = 0
        for node in self.model.nodes:
            check_count += len(node.get_phases_without_n())

        assert node_s_vars_count == check_count

        node_s_vars = [
            var.removesuffix("__re")
            for var in self.composite_var_list[
                node_s_re_idx_min : node_s_re_idx_max + 1
            ]
        ]
        assert len(node_s_vars) == check_count
        vars.extend(node_s_vars)

        # - y_real
        node_s_y_real = y_lst[node_s_re_idx_min : node_s_re_idx_min + node_s_vars_count]
        y_real.extend(node_s_y_real)

        # - y_imag
        node_s_y_imag = y_lst[
            node_s_re_idx_max + 1 : node_s_re_idx_max + 1 + node_s_vars_count
        ]
        y_imag.extend(node_s_y_imag)

        # ================

        assert len(y_real) == len(y_imag)
        assert len(y_real) == len(vars)

        # print(f">> vars: {vars}")
        # print(f">> y_real: {y_real}")
        # print(f">> y_imag: {y_imag}")

        y_real = np.array(y_real)
        y_imag = np.array(y_imag)
        y_complex = y_real + 1j * y_imag
        y_mag = np.abs(y_complex)
        y_result = zip(y_complex, y_mag)
        result = list(zip(vars, y_result))
        return result

    def print_y(self, y: sps.coo_array) -> str:
        result = self.y_dict(y)
        res = f"y:\n{pformat(result)}"
        print(res)
        return res

    def print_residuals(self, residuals: sps.coo_array):
        eqns = self.composite_eqn_list
        residuals_arr = residuals[:, 0].toarray()
        result = list(zip(eqns, residuals_arr))
        print(f"residuals:\n{pformat(result)}")

        print(">> non-zero residuals:")
        for eqn, residual in result:
            if abs(residual) > 1e-9:
                print(f"eqn: {eqn}, residual: {residual}")

    def newton_raphson_powerflow(self, tol=1e-6, num_iter=500) -> sps.coo_array | None:
        # resume-here
        y = self.initial_guess()
        assert y.shape[0] == self.M_iface_split.shape[1]
        # print shape of y
        print(f"y.shape: {y.shape}")
        # print(f"y_0: {y}")
        # print("y_0:")
        # self.print_y(y)
        # input("continue?")

        # plt.spy(y)
        # plt.title("y_initial", fontsize=20)
        # plt.show()

        prev_residual = None

        iter = 0
        while True:
            print("======================")
            print(f"> iter = {iter}")
            print("======================")
            self.update_fy(y)

            print(f"fy_split.shape: {self.fy_split.shape}")

            # print(f">> newton_raphson_powerflow(): y.shape: {y.shape}")
            # input("continue?")

            # plt.spy(self.fy_split)
            # plt.title("fy_split", fontsize=20)
            # plt.show()

            residual = self.M_iface_split * y + self.fy_split + self.u_iface_split
            # print(f"M_iface_split dtype: {self.M_iface_split.dtype}")
            # print(f"y dtype: {y.dtype}")
            # print(f"fy_split dtype: {self.fy_split.dtype}")
            # print(f"u_iface_split dtype: {self.u_iface_split.dtype}")

            # print(f"y: {pformat([(i, v[0]) for (i, v) in enumerate(y.toarray())])}")
            # print(f"residual: {residual}")
            # print(f"residual.dtype: {residual.dtype}")
            # self.print_residuals(residual)

            if iter >= num_iter:
                self.print_residuals(residual)
                self.print_y(y)
                print("[!] DID NOT CONVERGE")
                sys.exit(1)

            residual_norm = max(abs(residual))

            if residual_norm < tol:
                print(f"y:\n{y}")
                print("[*] converged!")
                return y
            else:
                self.update_jacobian(y)
                # print(f"self.jacobian: {self.jacobian}")
                print(f"jacobian.dtype: {self.jacobian.dtype}")

                self.print_residuals(residual)

                jacobian_rank = np.linalg.matrix_rank(self.jacobian.toarray())
                print(f">> jacobian_rank: {jacobian_rank}")

                # plt.spy(self.jacobian)
                # plt.grid(visible=True)
                # # Set custom tick locations at every 10 units
                # plt.xticks(np.arange(0, self.M_iface_split.shape[1], 10))
                # plt.yticks(np.arange(0, self.M_iface_split.shape[0], 10))
                # plt.title("jacobian", fontsize=20)
                # plt.show()

                # eigenvals = sps.linalg.eigs(self.jacobian, return_eigenvectors=False)
                # print("\neigenvalues (jacobian):", np.abs(eigenvals))
                # print(f"\nresidual.shape: {residual.shape}")
                # print(f"\njacobian.shape: {self.jacobian.shape}")
                delta_y = sps.linalg.spsolve(self.jacobian, -residual)
                # print(
                #     f"delta_y: {pformat(list(zip(self.composite_var_list, delta_y)))}"
                # )
                # print(f"y.shape: {y.shape}")
                # print(f"delta_y.shape: {delta_y.T.shape}")
                # print(f"type(delta_y): {type(delta_y)}")
                y += sps.coo_array(delta_y.reshape(-1, 1), dtype=float).tocsc()
                iter += 1

                # input("continue?")

    def _node_voltages(self) -> dict[str, float]:
        def all_initialized(result: dict) -> bool:
            for node_id, ph_dict in result.items():
                for ph, val in ph_dict.items():
                    if val is None:
                        print(
                            f">> all_initialized(): node not initialized fully: node={node_id}, ph_dict={ph_dict}"
                        )
                        return False
            return True

        # { node_id -> { ph -> voltage | None } }
        result: dict[str, dict[str, float | None]] = {}
        for node in self.model.system.nodes:
            print(f">> node_id:{node.id}, phases:{node.phases}")
            ph_dict = {}
            for ph in node.phases.keys():
                if ph != "N" and node.phases[ph] != 0:
                    ph_dict[ph] = None
            result[node.id] = ph_dict

        iter = 0
        while True:
            print(f">> iter: {iter}")
            if all_initialized(result):
                break

            for node in self.model.nodes:
                # get the adjacent models for this node
                adjacent: list[Model] = []
                for comp_id in self.node_components_map[node.get_id()]:
                    comp_model = self.model.get_component_by_id(comp_id)
                    adjacent.append(comp_model)

                for ph in node.get_phases_without_n():
                    if result[node.get_id()][ph] != None:
                        continue
                    # see if we can get this phase's voltage from any adjacent component
                    for comp in adjacent:
                        if ph not in comp.get_phases():
                            continue

                        # cases:
                        # - comp is a line
                        # - comp is a xfmr
                        # - comp is not a line or xfmr

                        # - comp is a line
                        if isinstance(comp, LineModel) or isinstance(
                            comp, VoltRegulatorModel
                        ):
                            side = comp.get_node_side(node.get_id())

                            other_side = None
                            if side == NodeSide.FROM:
                                other_side = NodeSide.TO
                            elif side == NodeSide.TO:
                                other_side = NodeSide.FROM
                            else:
                                raise ValueError("this should never be reached")

                            # print(f">> comp: {comp}")
                            # print(f">> other_side: {other_side}")
                            other_side_node = comp.get_node_on_side(other_side)

                            if (
                                ph in other_side_node.phases
                                and result[other_side_node.id][ph]
                            ):
                                result[node.get_id()][ph] = result[other_side_node.id][
                                    ph
                                ]
                                break

                        elif isinstance(comp, TransformerModel):
                            # - comp is a xfmr
                            side = comp.get_node_side(node.get_id())

                            nominal_ph_voltage = None
                            if side == NodeSide.FROM:
                                nominal_ph_voltage = comp.obj.pri_volt
                            elif side == NodeSide.TO:
                                nominal_ph_voltage = comp.obj.sec_volt
                            else:
                                raise ValueError("this should never be reached")

                            result[node.get_id()][ph] = nominal_ph_voltage
                            break

                        else:
                            result[node.get_id()][ph] = comp.nominal_voltage[ph]
                            break
            # print(f"result: {pformat(result)}")
            # input("continue?")
            iter += 1

        print(f">> result (nominal): {pformat(result)}")

        # now convert all "nominal-voltages" to "phasors" based on their phase
        for node_id, ph_dict in result.items():
            print(f">> node_id:{node_id}, ph_dict:{ph_dict}")
            v_phasor = {k: v for (k, v) in utils.get_vector_phasors(ph_dict).items()}
            result[node_id] = v_phasor

        return result

    def initial_guess(self) -> sps.coo_array:
        y_0_split = sps.lil_matrix((self.M_iface_split.shape[1], 1), dtype=float)

        # global w
        w_val = const.w_nominal
        y_0_split[-1, 0] = w_val

        # nodes
        node_voltages = self._node_voltages()
        assert len(node_voltages) == len(self.model.nodes)

        print(f"node_voltages: \n{pformat(node_voltages)}")
        for node_id, v_phasors in node_voltages.items():
            node = self.model.get_node_by_id(node_id)
            v_phasors = np.array(list(v_phasors.values())).reshape(-1, 1)

            # key = f"{node_id}_v"
            # idx_node_vre_start = self.var_offset[key]
            # idx_node_vre_end = idx_node_vre_start + len(node.get_phases_without_n())
            # idx_node_vim_start = idx_node_vre_start + self.M_iface.shape[1]
            # idx_node_vim_end = idx_node_vre_end + self.M_iface.shape[1]

            idx_node_vre_start = self.var_offset_real[f"{node_id}_v_re"]
            idx_node_vre_end = idx_node_vre_start + len(node.get_phases_without_n())
            idx_node_vim_start = self.var_offset_imag[f"{node_id}_v_im"]
            idx_node_vim_end = idx_node_vim_start + len(node.get_phases_without_n())

            print(f">> v_phasors: {v_phasors}")
            print(f">> idx_node_vre_start: {idx_node_vre_start}")
            print(f">> idx_node_vre_end: {idx_node_vre_end}")
            print(f">> idx_node_vim_start: {idx_node_vim_start}")
            print(f">> idx_node_vim_end: {idx_node_vim_end}")

            y_0_split[idx_node_vre_start:idx_node_vre_end, 0] = v_phasors.real
            y_0_split[idx_node_vim_start:idx_node_vim_end, 0] = v_phasors.imag

        # components
        for comp in self.model.components:
            # print(f">> comp: {comp}")

            # if comp is a line, then we need to pass in the "voltage" phasor as well,
            # which we can obtain from any adjacent node
            vals = {"w": w_val}
            if isinstance(comp, LineModel):
                from_node = comp.get_node_on_side(NodeSide.FROM)
                v_phasors: dict[str, complex] = node_voltages[from_node.id]
                vals["v_phasors"] = deepcopy(v_phasors)

            comp_y_0 = comp.initial_guess(vals=vals)

            assert comp_y_0.shape[0] == comp.num_vars
            assert comp_y_0.shape[1] == 1

            idx_comp_re_start = self.var_offset_real[comp.get_id()]
            idx_comp_re_end = idx_comp_re_start + comp.num_vars
            idx_comp_im_start = self.var_offset_imag[comp.get_id()]
            idx_comp_im_end = idx_comp_im_start + comp.num_vars_complex

            y_0_split[idx_comp_re_start:idx_comp_re_end, 0] = comp_y_0.real
            y_0_split[idx_comp_im_start:idx_comp_im_end, 0] = comp_y_0[
                comp.num_vars_real :
            ].imag

        # # print(">> spy for y_0")
        # # plt.spy(y_0)
        # # plt.show()

        return y_0_split.tocsc()

    def save_results_xlsx(self, fname: str):
        # sheets:
        # - nodes
        # - loads
        # - sources
        # - transformers

        y_lst = self.y_dict(self.y_final)

        node_entries = [(key, val) for (key, val) in y_lst if key.startswith("node__")]
        line_entries = [(key, val) for (key, val) in y_lst if key.startswith("line__")]
        load_entries = [(key, val) for (key, val) in y_lst if key.startswith("load__")]
        source_entries = [
            (key, val) for (key, val) in y_lst if key.startswith("source__")
        ]
        xfmr_entries = [(key, val) for (key, val) in y_lst if key.startswith("xfmr__")]

        # - nodes
        data_nodes = []
        for key, (val, mag) in node_entries:
            node_profile = key.removeprefix("node__")
            angle = np.angle(val, deg=True)
            record = {
                "node_profile": node_profile,
                "magnitude": mag,
                "angle": angle,
                "real_part": val.real,
                "imag_part": val.imag,
            }
            data_nodes.append(record)
        df_node = pd.DataFrame.from_records(data_nodes)

        # - loads
        # print(f"load_entries: {pformat(load_entries)}")
        # data_loads = []
        # for (key, (val, mag)) in load_entries:
        #     load_profile = key.removeprefix("load__")
        #     angle = np.angle(val, deg=True)
        #     record = {
        #         "load_profile": load_profile,
        #         "magnitude": mag,
        #         "angle": angle,
        #         "real_part": val.real,
        #         "imag_part": val.imag
        #     }
        #     data_loads.append(record)

        # - xfmrs
        data_xfmrs = []
        for key, (val, mag) in xfmr_entries:
            xfmr_profile = key.removeprefix("xfmr__")
            angle = np.angle(val, deg=True)
            record = {
                "xfmr_profile": xfmr_profile,
                "magnitude": mag,
                "angle": angle,
                "real_part": val.real,
                "imag_part": val.imag,
            }
            data_xfmrs.append(record)
        df_xfmrs = pd.DataFrame.from_records(data_xfmrs)

        # finally write to the resutls sheet in one go
        with pd.ExcelWriter(fname) as writer:
            df_node.to_excel(writer, sheet_name="nodes", index=False)
            # df_load.to_excel(writer, sheet_name="loads", index=False)
            # df_source.to_excel(writer, sheet_name="sources", index=False)
            df_xfmrs.to_excel(writer, sheet_name="xfmrs", index=False)
        print(f">> saved results to {fname}.")
