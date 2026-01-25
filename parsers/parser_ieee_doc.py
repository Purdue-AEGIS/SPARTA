# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 15:48:27 2023

@author: anonymous
"""

"""
This script initiates the object oriented model based on the
information provided by the user entered in manual_mode.

"""

########################################################################################
"""
Definitions of helper functions

"""
# import networkx as nx-
import importlib
from pprint import pformat


# Define this constant to convert ft to miles.
from parsers.parser_utils import *
from collections import namedtuple
import numpy as np
from oodesign import *
from oodesign import (
    GFMInverter3Ph,
    InverterControl,
    InverterPrimaryControl,
    InverterSingleDualControl,
)
import parsers.parser_utils as parser_utils
import utils
from catalog.line_config_catalog import line_config
import sys

# from equipment_catalog import generator_config
from catalog.inverter_catalog import inverter_config
from catalog.generator_catalog import generator_config

import const

MILE_PER_FT = 0.0001893939394


def _extract_RXY_matrix(upp_tri_matrix: np.ndarray, type_: str) -> np.ndarray:
    """
    Extracts the resistance, reactance, or admittance matrix based on the type provided.

    Parameters:
        upp_tri_matrix (np.ndarray): Upper triangular matrix containing either resistance, reactance, or admittance values.
        type_ (str): Type of matrix to extract. Should be either 'R', 'X', or 'Y'.

    Returns:
        np.ndarray: Symmetric matrix containing either resistance, reactance, or admittance values.
    """
    if type_ == "R":
        upp_tri = upp_tri_matrix[:, 0::2]
    elif type_ == "X":
        upp_tri = upp_tri_matrix[:, 1::2]
    elif type_ == "Y":
        upp_tri = upp_tri_matrix[:, :]
    else:
        raise ValueError("Invalid type. Please provide either 'R', 'X', or 'Y'.")

    matrix = upp_tri + upp_tri.T - np.diag(upp_tri.diagonal())

    return matrix


def _get_transformer_zmatrix(
    kVA_rating: float, sec_volt_kV: float, percent_r, percent_X, w: float
) -> tuple[np.ndarray, np.ndarray]:
    """
    This function generates the impedane matrix from the data given in the
    document.
    """

    # Calculate Z_base:
    Z_base = ((sec_volt_kV**2) * 1000) / kVA_rating

    # convert percentage reactance and resistance in PU to ohm
    resistance_mat = np.identity(3) * (percent_r / 100) * Z_base
    reactance_mat = np.identity(3) * (percent_X / 100) * Z_base
    inductance_mat = reactance_mat / w
    # print(f"impedance_matrix : {impedance_matrix}")
    return resistance_mat, inductance_mat


"""
This function is to model the distributed load as follows:
 -split the line and create a node at 1/4th distance from source node
 (node_A)
 -place 2/3 of the total load at this new node
 -add the remaining 1/3 of the total load at the end node of the line
 (node_B)
 -the two lines so created replace the original line in the network
"""


def distributed_to_spotload(
    load: dict,
    nodes: list[Node],
    lines: list[Line],
    loads: list[Load],
    cnt: int,
    input_data,
):
    # The line between node_A and node_b is modelled as distributed load
    # So we split this line to create two new lines and replace the original line
    # with the two new line

    # the splitting of line creates a new spot load at 1/4th distance from node_A
    # this spot load is 2/3rd of total load on distributed line and rest one third
    # is added to the spot load of node_B

    """
    Kersting's Tb - Section 3.4.3-The Exact Lumped Load Model
    """
    # First create a spot load
    line_to_remove_id = None
    for line in lines:
        if (line.terminal.from_node.id == load["NodeA"]) and (
            line.terminal.to_node.id == load["NodeB"]
        ):
            new_node = Node(
                id=line.terminal.to_node.id + "1",
                phases=line.terminal.to_node.phases,
                voltage=line.terminal.to_node.voltage,
            )
            if new_node not in nodes:
                nodes.append(new_node)

            # create first part of split lines
            newline_1 = Line(
                name=line.name + "-1",
                id=line.id + "-1",
                terminal=TwoTerminal(line.terminal.from_node, new_node),
                phases=line.phases,
                # resistance matrix for 1st part of split line is 1/4th of original line matrix
                resistance_mat=(1 / 4) * line.resistance_mat,
                reactance_mat=(1 / 4) * line.reactance_mat,
                admittance_mat=(1 / 4) * line.admittance_mat,
                has_capacitance=input_data.include_capacitance_of_lines,
                n_ph=line.n_ph,
            )
            lines.append(newline_1)
            # create second part of split lines
            newline_2 = Line(
                name=line.name + "-2",
                id=line.id + "-2",
                terminal=TwoTerminal(new_node, line.terminal.to_node),
                phases=line.phases,
                resistance_mat=(3 / 4) * line.resistance_mat,
                reactance_mat=(3 / 4) * line.reactance_mat,
                admittance_mat=(3 / 4) * line.admittance_mat,
                has_capacitance=input_data.include_capacitance_of_lines,
                n_ph=line.n_ph,
            )
            lines.append(newline_2)

            line_to_remove_id = line.id

            load_2_3_dict = {
                "Node": new_node.id,
                "Model": load["Model"],
                "active_power": utils.calculate_terms(
                    load["active_power"], 2 / 3, operation_type="multiply"
                ),
                "reactive_power": utils.calculate_terms(
                    load["reactive_power"], 2 / 3, operation_type="multiply"
                ),
                "nominal_voltage": load["nominal_voltage"],
            }
            new_spot_load = IEEEDocParser._parse_load(load_2_3_dict, len(loads), nodes)
            loads.append(new_spot_load)

            # create a spot load with 2/3 of total distributed load lumped at new load
            # load_model = LoadType[load["Model"]]
            # phases = load["nominal_voltage"].keys()
            # new_spot_load = Load(
            #     name=f"dist_spot_load{cnt}" + "-1",
            #     type=load_model.value,
            #     node=new_node,
            #     phases=phases,
            #     nominal_voltage=load["nominal_voltage"],
            #     active_power=utils.calculate_terms(
            #         load["active_power"], 2 / 3, operation_type="multiply"
            #     ),
            #     reactive_power=utils.calculate_terms(
            #         load["reactive_power"], 2 / 3, operation_type="multiply"
            #     ),
            #     n_ph=len(load["active_power"]),
            # )
            # loads.append(new_spot_load)

            # new_spot_load = Load(
            #     name=f"dist_spot_load{cnt}" + "-2",
            #     type=load_model.value,
            #     node=get_node_by_id(load["NodeB"], nodes),
            #     phases=phases,
            #     nominal_voltage=load["nominal_voltage"],
            #     active_power=utils.calculate_terms(
            #         load["active_power"], 1 / 3, operation_type="multiply"
            #     ),
            #     reactive_power=utils.calculate_terms(
            #         load["reactive_power"], 1 / 3, operation_type="multiply"
            #     ),
            #     n_ph=len(load["active_power"]),
            # )

            # loads.append(new_spot_load)

            load_1_3_dict = {
                "Node": line.terminal.to_node.id,
                "Model": load["Model"],
                "active_power": utils.calculate_terms(
                    load["active_power"], 1 / 3, operation_type="multiply"
                ),
                "reactive_power": utils.calculate_terms(
                    load["reactive_power"], 1 / 3, operation_type="multiply"
                ),
                "nominal_voltage": load["nominal_voltage"],
            }
            new_spot_load = IEEEDocParser._parse_load(load_1_3_dict, len(loads), nodes)
            loads.append(new_spot_load)

    # above two lines should replace the original line so it is removed
    line_to_remove = None
    for line in lines:
        if line_to_remove_id == line.id:
            line_to_remove = line
            break
    lines.remove(line_to_remove)


def distributed_to_spotload2(
    load: dict,
    nodes: list[Node],
    lines: list[Line],
    loads: list[Load],
    cnt: int,
    input_data,
):
    # The line between node_A and node_b is modelled as distributed load
    # So we split this line to create two new lines and replace the original line
    # with the two new line

    # the splitting of line creates a new spot load at 1/4th distance from node_A
    # this spot load is 2/3rd of total load on distributed line and rest one third
    # is added to the spot load of node_B

    """
    Kersting's Tb - Section 3.4.3-The Exact Lumped Load Model
    """
    # First create a spot load
    line_to_remove_id = None
    for line in lines:
        if (line.terminal.from_node.id == load["NodeA"]) and (
            line.terminal.to_node.id == load["NodeB"]
        ):
            new_node = Node(
                id=line.terminal.to_node.id + "1",
                phases=line.terminal.to_node.phases,
                voltage=line.terminal.to_node.voltage,
            )
            if new_node not in nodes:
                nodes.append(new_node)

            # create first part of split lines
            newline_1 = Line(
                name=line.name + "-1",
                id=line.id + "-1",
                terminal=TwoTerminal(line.terminal.from_node, new_node),
                phases=line.phases,
                # resistance matrix for 1st part of split line is 1/4th of original line matrix
                resistance_mat=(1 / 2) * line.resistance_mat,
                reactance_mat=(1 / 2) * line.reactance_mat,
                admittance_mat=(1 / 2) * line.admittance_mat,
                has_capacitance=input_data.include_capacitance_of_lines,
                n_ph=line.n_ph,
            )
            lines.append(newline_1)
            # create second part of split lines
            newline_2 = Line(
                name=line.name + "-2",
                id=line.id + "-2",
                terminal=TwoTerminal(new_node, line.terminal.to_node),
                phases=line.phases,
                resistance_mat=(1 / 2) * line.resistance_mat,
                reactance_mat=(1 / 2) * line.reactance_mat,
                admittance_mat=(1 / 2) * line.admittance_mat,
                has_capacitance=input_data.include_capacitance_of_lines,
                n_ph=line.n_ph,
            )
            lines.append(newline_2)

            line_to_remove_id = line.id

            load_2_3_dict = {
                "Node": new_node.id,
                "Model": load["Model"],
                "active_power": utils.calculate_terms(
                    load["active_power"], 1, operation_type="multiply"
                ),
                "reactive_power": utils.calculate_terms(
                    load["reactive_power"], 1, operation_type="multiply"
                ),
                "nominal_voltage": load["nominal_voltage"],
            }
            new_spot_load = IEEEDocParser._parse_load(load_2_3_dict, len(loads), nodes)
            loads.append(new_spot_load)

            # load_1_3_dict = {
            #     "Node": line.terminal.to_node.id,
            #     "Model": load["Model"],
            #     "active_power": utils.calculate_terms(
            #         load["active_power"], 1 / 3, operation_type="multiply"
            #     ),
            #     "reactive_power": utils.calculate_terms(
            #         load["reactive_power"], 1 / 3, operation_type="multiply"
            #     ),
            #     "nominal_voltage": load["nominal_voltage"],
            # }
            # new_spot_load = IEEEDocAdapter._parse_load(load_1_3_dict, len(loads), nodes)
            # loads.append(new_spot_load)

    # above two lines should replace the original line so it is removed
    line_to_remove = None
    for line in lines:
        if line_to_remove_id == line.id:
            line_to_remove = line
            break
    lines.remove(line_to_remove)


ConfigData = namedtuple(
    # "ConfigData", ["R_matrix", "X_matrix", "Y_matrix", "Phases", "Phasing"]
    "ConfigData",
    ["R_matrix", "X_matrix", "Y_matrix", "Phases"],
)

def node_ids_to_str(data: list[dict]) -> list[dict]:
    for item in data:
        if "Node_A" in item:
            item["Node_A"] = str(item["Node_A"])
        if "Node_B" in item:
            item["Node_B"] = str(item["Node_B"])
        if "NodeA" in item:
            item["NodeA"] = str(item["NodeA"])
        if "NodeB" in item:
            item["NodeB"] = str(item["NodeB"])
        if "Node" in item:
            item["Node"] = str(item["Node"])
    return data



def clean_line_seg_data(line_seg_data: list[dict]) -> list[dict]:
    for line in line_seg_data:
        line["Node_A"] = str(line["Node_A"])
        line["Node_B"] = str(line["Node_B"])
    return line_seg_data


def clean_spot_load_data(spot_load_data: list[dict]) -> list[dict]:
    for load in spot_load_data:
        load["Node"] = str(load["Node"])
    return spot_load_data


def clean_dist_load_data(dist_load_data: list[dict]) -> list[dict]:
    for load in dist_load_data:
        load["NodeA"] = str(load["NodeA"])
        load["NodeB"] = str(load["NodeB"])
    return dist_load_data


def _count_ph_line_list(phases) -> list:
    """
    Counts the number of phases for each line in the system.

    Returns:
            list: A list containing the count of phases for each line.
    """

    ph = {"A": 0, "B": 0, "C": 0}
    for key in phases:
        if key == "N":
            continue
        ph[key] = 1

    count = sum(np.array(list(ph.values())))

    # valid_phases = "ABC"
    # count = 0
    # for ph in valid_phases:
    #     if phases[ph] == 1:
    #         count += 1

    return count


class IEEEDocParser:
    def from_ieee_input_data(filename) -> System:
        # import the data from the ieee_13_input_data.py file
        # input_data = __import__(filename)
        input_data = importlib.import_module(filename)

        line_seg_data = clean_line_seg_data(input_data.line_seg_data)
        spot_load_data = clean_spot_load_data(input_data.spot_load_data)
        dist_load_data = clean_dist_load_data(input_data.dist_load_data)

        transformer_data = node_ids_to_str(input_data.transformer_data)
        shunt_cap_data = node_ids_to_str(input_data.capacitor_data)
        inverter_data = node_ids_to_str(input_data.inverter_data)
        generator_data = node_ids_to_str(input_data.generator_data)
        volt_reg_data = node_ids_to_str(input_data.volt_reg_data)
        switch_data = node_ids_to_str(input_data.switch_data)
        source_data = node_ids_to_str(input_data.source_data)

        # get the nominal frequency from user
        nominal_freq = input_data.nominal_frequency
        # calculate angular velocity
        # w_nominal = 2 * np.pi * nominal_freq
        const.w_nominal = 2 * np.pi * nominal_freq

        # init config
        config_data = IEEEDocParser.init_config(line_config)
        # parse nodes
        nodes = []
        for line in line_seg_data:
            from_node_id = line["Node_A"]
            from_node = Node(
                id=from_node_id,
                phases={"A": 0, "B": 0, "C": 0, "N": 0},
                voltage=None,
            )
            to_node_id = line["Node_B"]
            to_node = Node(
                id=to_node_id,
                phases={"A": 0, "B": 0, "C": 0, "N": 0},
                voltage=None,
            )
            if from_node not in nodes:
                nodes.append(from_node)
            if to_node not in nodes:
                nodes.append(to_node)
        # parse lines
        lines = []
        for i, line in enumerate(line_seg_data):
            # skip if the line is a xfmr / substation / vreg
            continue_outer = False
            for prefix in ["XFM", "Substation", "VReg", "sw"]:
                if line["config"].startswith(prefix):
                    continue_outer = True
                    break
            if continue_outer:
                continue

            if line["config"] not in line_config:
                raise ValueError(f"unknown config: {line['config']}")
            line_id = f"line{i}"
            line_name = line_id

            from_node_id = line["Node_A"]
            from_node = get_node_by_id(from_node_id, nodes)

            to_node_id = line["Node_B"]
            to_node = get_node_by_id(to_node_id, nodes)

            config_id = line["config"]
            phases = config_data[config_id].Phases
            phases = remove_zero_val_dict(phases)
            R_matrix = line["Length_ft"] * MILE_PER_FT * config_data[config_id].R_matrix
            R_matrix = remove_zero_rows_cols(R_matrix)
            X_matrix = line["Length_ft"] * MILE_PER_FT * config_data[config_id].X_matrix
            X_matrix = remove_zero_rows_cols(X_matrix)

            # print(f">> line: {from_node_id} -> {to_node_id}")
            # print(f">> R_matrix: {R_matrix}")
            # print(f">> X_matrix: {X_matrix}")
            # input("Press Enter to continue...")

            n_ph = _count_ph_line_list(phases)
            
            Y_matrix = line["Length_ft"] * MILE_PER_FT * config_data[config_id].Y_matrix

            # NOTE: this causes problem if C is a zero matrix (3x3), hecne commenting this out.
            Y_matrix = remove_zero_rows_cols(Y_matrix)
            if Y_matrix.size == 0:
                Y_matrix = np.zeros((n_ph, n_ph), dtype=float)

            # print(f"Y_matrix: {Y_matrix}")
            # print(f"type(R_matrix): {type(R_matrix)}")
            # input("continue?")

            # phasing = config_data[config_id].Phasing

            line = Line(
                name=line_name,
                id=line_id,
                # from_node=from_node,
                # to_node=to_node,
                terminal=TwoTerminal(from_node, to_node),
                phases=phases,
                # phasing=phasing,
                n_ph=n_ph,
                resistance_mat=R_matrix,
                reactance_mat=X_matrix,
                admittance_mat=Y_matrix,
                has_capacitance=input_data.include_capacitance_of_lines,
            )
            lines.append(line)
            # update node phases
            for ph in "ABCN":
                if ph in phases:
                    from_node.phases[ph] = max(from_node.phases[ph], phases[ph])
                    to_node.phases[ph] = max(to_node.phases[ph], phases[ph])

        equipments = []
        # transformers
        for _, transformer in enumerate(transformer_data):
            for _, line in enumerate(line_seg_data):
                if line["config"] == transformer["Name"]:
                    name = transformer["Name"]
                    equipment_id = transformer["Name"]
                    from_node = get_node_by_id(line["Node_A"], nodes)
                    to_node = get_node_by_id(line["Node_B"], nodes)
                    high_wdg = transformer["high_winding_connect"]
                    low_wdg = transformer["low_winding_connect"]
                    pri_volt = 1e3 * transformer["kV_high"]
                    sec_volt = 1e3 * transformer["kV_low"]
                    VA = 1e3 * transformer["kVA"]
                    phases = {"A": 1, "B": 1, "C": 1}
                    n_ph = 3
                    conn = high_wdg + "_" + low_wdg
                    config = TransformerConfig[conn].value
                    if high_wdg == "D" and (low_wdg == "GrY" or low_wdg == "GrW"):
                        if pri_volt > sec_volt:
                            config += "StepDown"
                        else:
                            config += "StepUp"

                    elif (high_wdg == "GrY" or high_wdg == "GrW") and low_wdg == "D":
                        if pri_volt > sec_volt:
                            config += "StepDown"
                        else:
                            config += "StepUp"

                    elif high_wdg == "D" and low_wdg == "D":
                        if pri_volt > sec_volt:
                            config += "StepDown"
                        else:
                            config += "StepUp"

                    print(f">> config: {config}")
                    turns_ratio = pri_volt / sec_volt
                    R_mat, L_mat = _get_transformer_zmatrix(
                        transformer["kVA"],
                        transformer["kV_low"],
                        transformer["R_percent_pu"],
                        transformer["X_percent_pu"],
                        const.w_nominal,
                    )
                    transformer_obj = Transformer(
                        name=name,
                        id=equipment_id,
                        terminal=TwoTerminal(from_node, to_node),
                        phases=phases,
                        pri_volt=pri_volt,
                        sec_volt=sec_volt,
                        turns_ratio=turns_ratio,
                        VA=VA,
                        resistance_mat=R_mat,
                        inductance_mat=L_mat,
                        config=config,
                        n_ph=n_ph,
                    )
                    equipments.append(transformer_obj)
                    # update node phases
                    for ph in "ABC":
                        from_node.phases[ph] = max(from_node.phases[ph], phases[ph])
                        to_node.phases[ph] = max(to_node.phases[ph], phases[ph])
        # # Create a directed graph
        # g = nx.DiGraph()
        # for line_seg in line_seg_data:
        #     g.add_edge(
        #         line_seg["Node_A"],
        #         line_seg["Node_B"],
        #         length=line_seg["Length_ft"],
        #         config=line_seg["config"],
        #     )
        # # check if there is a Substation transformer
        # substation_transformer = False
        # for transformer in transformer_data:
        #     if transformer["Name"] == "Substation":
        #         substation_transformer = True
        #         break
        # for edge in g.edges(data=True):
        #     if edge[2]["length"] == 0.0:
        #         print(f"Edge {edge} has zero length.")
        #         for transformer in transformer_data:
        #             if edge[2]["config"] == transformer["Name"]:
        #                 if transformer["Name"] == "Substation":
        #                     g.nodes[edge[0]]["nominal_voltage"] = transformer["kV_high"]
        #                     g.nodes[edge[1]]["nominal_voltage"] = transformer["kV_low"]
        #                     # find all the ancestors and descendants
        #                     # to the edge before and after transformer
        #                     p = nx.ancestors(g, edge[0])
        #                     s = nx.descendants(g, edge[1])

        #                     # add nominal voltage to the p and s and on
        #                     # either side of the transformer based on
        #                     #  transformer data
        #                     for node in p:
        #                         g.nodes[node]["nominal_voltage"] = transformer[
        #                             "kV_high"
        #                         ]
        #                     for node in s:
        #                         g.nodes[node]["nominal_voltage"] = transformer["kV_low"]
        #                 else:
        #                     if not substation_transformer:
        #                         p = list(nx.ancestors(g, edge[1]))
        #                         for node in p:
        #                             g.nodes[node]["nominal_voltage"] = transformer[
        #                                 "kV_high"
        #                             ]
        #                     s = list(nx.descendants(g, edge[1]))
        #                     # add edge[1] also to the successor list
        #                     s.append(edge[1])
        #                     print(f"Successors of {edge[1]}: {s}")
        #                     for node in s:
        #                         g.nodes[node]["nominal_voltage"] = transformer["kV_low"]
        # # set the nominal voltage of each node in nodes based on g.nodes
        # for node in nodes:
        #     node.voltage = g.nodes[node.node_id]["nominal_voltage"] * 1e3 / (np.sqrt(3))

        # capacitors
        for i, cap in enumerate(shunt_cap_data):
            cap_config = ShuntCapType[cap["connection"]].value
            # phases = adapter_utils.find_phases(cap["nominal_voltage"])
            phases = {}
            for ph, v in cap["nominal_voltage"].items():
                if v == 0:
                    phases[ph] = 0
                else:
                    phases[ph] = 1
            terminal = SingleTerminal(at_node=get_node_by_id(str(cap["Node"]), nodes))
            nominal_voltage = {ph: 1e3 * v for ph, v in cap["nominal_voltage"].items()}
            power = {ph: 1e3 * p for ph, p in cap["power"].items()}
            equipments.append(
                ShuntCapacitor(
                    name=f"cap{i + 1}",
                    id=f"cap{i + 1}",
                    terminal=terminal,
                    phases=phases,
                    power=power,
                    n_ph=len(power),
                    nominal_voltage=nominal_voltage,
                    config=cap_config,
                )
            )
            # update node phases
            for ph in "ABC":
                if ph in phases:
                    terminal.at_node.phases[ph] = max(
                        terminal.at_node.phases[ph], phases[ph]
                    )

        # volt_regulators
        for i, volt_reg in enumerate(volt_reg_data):
            for _, line in enumerate(line_seg_data):
                if line["config"] == volt_reg["Name"]:
                    from_node = get_node_by_id(line["Node_A"], nodes)
                    to_node = get_node_by_id(line["Node_B"], nodes)
                    terminal = TwoTerminal(from_node, to_node)
                    volt_reg_obj = IEEEDocParser._parse_volt_reg(
                        volt_reg, i, terminal, equipments
                    )
                    equipments.append(volt_reg_obj)
                    print(f">> tap_setting: {volt_reg_obj.tap_setting}")
                    # update node phases
                    for ph in "ABC":
                        if ph in volt_reg_obj.phases:
                            from_node.phases[ph] = max(
                                from_node.phases[ph], volt_reg_obj.phases[ph]
                            )
                            to_node.phases[ph] = max(
                                to_node.phases[ph], volt_reg_obj.phases[ph]
                            )

        # switches
        for i, switch in enumerate(switch_data):
            for line in line_seg_data:
                if line["config"] == switch["Name"]:
                    from_node = get_node_by_id(line["Node_A"], nodes)
                    to_node = get_node_by_id(line["Node_B"], nodes)
                    terminal = TwoTerminal(from_node, to_node)
                    switch_obj = IEEEDocParser._parse_switch(
                        switch, i, terminal
                    )
                    equipments.append(switch_obj)
                    # update node phases
                    for ph in "ABC":
                        if ph in switch_obj.phases:
                            from_node.phases[ph] = max(
                                from_node.phases[ph], switch_obj.phases[ph]
                            )
                            to_node.phases[ph] = max(
                                to_node.phases[ph], switch_obj.phases[ph]
                            )

        loads = []
        for i, load in enumerate(spot_load_data):
            load_obj = IEEEDocParser._parse_load(load, i, nodes)
            loads.append(load_obj)

        # create spot load for each distributed load
        for i, load in enumerate(dist_load_data):
            # distributed_to_spotload(load, nodes, lines, loads, i, input_data)
            distributed_to_spotload2(load, nodes, lines, loads, i, input_data)

        if input_data.study_of_interest == "Dynamic":
            input_data.study_of_interest = 0
        elif input_data.study_of_interest == "PowerFlow":
            input_data.study_of_interest = 1
        elif input_data.study_of_interest == "PowerFlow++":
            input_data.study_of_interest = 2
        else:
            raise ValueError("Invalid study of interest", input_data.study_of_interest)

        # pu
        if hasattr(input_data, "base_power"):
            base_power = input_data.base_power
        else:
            base_power = None

        if hasattr(input_data, "base_voltage"):
            base_voltage = input_data.base_voltage
        else:
            base_voltage = None

        assert (base_power is None and base_voltage is None) or (
            base_power is not None and base_voltage is not None
        ), "Either both base_power and base_voltage should be provided or none of them"

        sources = []
        # parsing of balanced source
        for i, source in enumerate(source_data):
            source_obj = IEEEDocParser._parse_source(source, i, nodes)
            sources.append(source_obj)

        inverters = IEEEDocParser._parse_inverters(
            inverter_data,
            nodes,
            const.w_nominal,
            base_power=base_power,
            base_voltage=base_voltage,
        )
        for i, inverter_obj in enumerate(inverters):
            sources.append(inverter_obj)
        # inverters = []

        generators = IEEEDocParser._parse_generators(generator_data, nodes)

        # # parsing of source defined in sequence components
        # for i, source in enumerate(input_data.source_data_with_seq):
        #     source_obj = IEEEDocAdapter._parse_source_with_seq(source, i, nodes)
        #     sources.append(source_obj)

        # slack bus
        slack_bus = get_node_by_id(str(input_data.slack_bus), nodes)

        # finally the system
        system = System(
            study_type=input_data.study_of_interest,
            lines=lines,
            nodes=nodes,
            equipments=equipments,
            sources=sources,
            loads=loads,
            inverters=inverters,
            generators=generators,
            simulation_time=input_data.simulation_time,
            slack_bus=slack_bus,
            w_nominal=const.w_nominal,
            base_power=base_power,
            base_voltage=base_voltage,
        )

        return system
    

    @staticmethod
    def _parse_switch(
        switch_dict: dict, i: int, terminal: TwoTerminal
    ) -> Switch:
        phases = {}
        for ph, v in switch_dict["nominal_voltage"].items():
            if v == 0:
                phases[ph] = 0
            else:
                phases[ph] = 1
        n_ph = sum(phases.values())

        nominal_voltage = {
            ph: 1e3 * v for ph, v in switch_dict["nominal_voltage"].items()
        }

        assert switch_dict["state"] in (SwitchState.Open.value, SwitchState.Closed.value)

        switch_obj = Switch(
            name=f"switch{i + 1}",
            id=f"switch{i + 1}",
            terminal=terminal,
            nominal_voltage=nominal_voltage,
            n_ph=n_ph,
            phases=phases,
            state=switch_dict["state"]
        )

        return switch_obj

    @staticmethod
    def _parse_volt_reg(
        volt_reg_dict: dict, i: int, terminal: TwoTerminal, equipments: list[Equipment]
    ) -> Regulator:
        phases = {}
        for ph, v in volt_reg_dict["nominal_voltage"].items():
            if v == 0:
                phases[ph] = 0
            else:
                phases[ph] = 1
        n_ph = sum(phases.values())

        # # - find the transformer for which this voltage regulator is connected
        # xfmrs = []
        # for equipment in equipments:
        #     if isinstance(equipment, Transformer):
        #         if equipment.terminal.to_node == terminal.from_node:
        #             xfmrs.append(equipment)
        # assert (
        #     len(xfmrs) == 1
        # ), "Voltage regulator should be connected to exactly one transformer"
        # xfmr = xfmrs[0]

        # TODO: set the regulator type
        reg_type = "A"
        # reg_type = "B"

        if volt_reg_dict["control"] == "manual":
            control = RegulatorControl.MANUAL
        elif volt_reg_dict["control"] == "automatic":
            control = RegulatorControl.AUTOMATIC
        else:
            raise ValueError("Invalid control type")

        tap_setting, effective_reg_ratio = {}, {}
        for ph in phases:
            if phases[ph] == 1:
                # TODO: calculate the PF based on the actual load connected in the network
                # TODO: this would be different for different phases, based on the kind of load connected
                # pf = 0.9
                tap_setting[ph] = volt_reg_dict["tap_setting"][ph]
                effective_reg_ratio[ph] = utils.calc_effective_reg_ratio(
                    tap_setting[ph], reg_type
                )
            # else:
            #     tap_setting[ph], effective_reg_ratio[ph] = None, None

        nominal_voltage = {
            ph: 1e3 * v for ph, v in volt_reg_dict["nominal_voltage"].items()
        }

        volt_reg_obj = Regulator(
            name=f"volt_reg{i + 1}",
            id=f"volt_reg{i + 1}",
            terminal=terminal,
            bandwidth=volt_reg_dict["bandwidth"],
            pt_ratio=volt_reg_dict["PT_ratio"],
            ct_primary=volt_reg_dict["CT_primary"],
            ct_secondary=volt_reg_dict["CT_secondary"],
            voltage_level=volt_reg_dict["voltage_level"],
            r_setting=volt_reg_dict["R_setting"],
            x_setting=volt_reg_dict["X_setting"],
            control=control,
            tap_setting=tap_setting,
            reg_type=reg_type,
            effective_reg_ratio=effective_reg_ratio,
            nominal_voltage=nominal_voltage,
            # xfmr_sec_v_mag=xfmr.sec_volt / np.sqrt(3),
            # xfmr_sec_v_mag=4.16 / np.sqrt(3),
            # i_line_mag=xfmr.kVA / (np.sqrt(3) * xfmr.sec_volt),
            # i_line_mag=5000 / (np.sqrt(3) * 4.16),
            n_ph=n_ph,
            phases=phases,
        )
        return volt_reg_obj

    @staticmethod
    def _parse_load(load_dict: dict, i: int, nodes: list[Node]) -> Load:
        load_type = LoadType[load_dict["Model"]]
        print(f">> load_type: {load_type}")

        # phases = adapter_utils.find_phases(load_dict["nominal_voltage"])
        phases = {}
        for ph, v in load_dict["nominal_voltage"].items():
            if v == 0:
                phases[ph] = 0
            else:
                phases[ph] = 1

        if "step_change" in load_dict.keys():
            step_change = load_dict["step_change"]
        else:
            step_change = False

        nominal_voltage = {
            ph: 1e3 * v for (ph, v) in load_dict["nominal_voltage"].items()
        }
        active_power = {ph: 1e3 * p for (ph, p) in load_dict["active_power"].items()}
        reactive_power = {
            ph: 1e3 * q for (ph, q) in load_dict["reactive_power"].items()
        }

        # footgun on enum comparison!!
        if load_type.name == LoadType.Y_I.name:
            # calculate iconst
            nominal_v_phasors = np.array(
                list(utils.get_vector_phasors(nominal_voltage).values())
            )
            P = np.array(list(active_power.values()))
            Q = np.array(list(reactive_power.values()))
            S = P + 1j * Q
            iconst = (S / nominal_v_phasors).conj()
            iconst = iconst.reshape(-1, 1)
            iconst = np.abs(iconst)

            # calculate power_factor
            power_factor = np.cos(np.angle(S))
            power_factor = power_factor.reshape(1, -1).tolist()[0]
            power_factor: dict[str, float] = {
                k: v for k, v in zip(phases.keys(), power_factor)
            }
            print(f"power_factor: {power_factor}")
            # sys.exit(1)

            # create obj
            terminal = SingleTerminal(at_node=get_node_by_id(load_dict["Node"], nodes))
            load_obj = StarConstantCurrentLoad(
                name=f"load{i + 1}",
                id=f"load{i + 1}",
                # at_node=get_node_by_id(load_dict["Node"], nodes),
                terminal=terminal,
                phases=phases,
                active_power=active_power,
                reactive_power=reactive_power,
                n_ph=len(active_power),
                nominal_voltage=nominal_voltage,
                stepchange=step_change,
                iconst=iconst,
                power_factor=power_factor,
            )

        elif load_type.name == LoadType.D_I.name:
            # calculate iconst
            V_phasors = utils.get_vector_phasors(nominal_voltage)
            V_line_line = utils.get_phase_phase_values(V_phasors)
            V_line_line_abs = {k: np.abs(v) for k, v in V_line_line.items()}
            Vll = {k: v for k, v in V_line_line_abs.items() if k in active_power}
            Vll = 1e3 * np.array(list(Vll.values()))

            P = 1e3 * np.array(list(active_power.values()))
            Q = 1e3 * np.array(list(reactive_power.values()))
            S = P + 1j * Q

            iconst = (S / Vll).conj()
            iconst = iconst.reshape(-1, 1)
            iconst = np.abs(iconst)

            # calculate power_factor
            power_factor = np.cos(np.angle(S))
            power_factor = power_factor.reshape(1, -1).tolist()[0]
            power_factor: dict[str, float] = {
                k: v for k, v in zip(phases.keys(), power_factor)
            }
            print(f">> power_factor: {power_factor}")
            # sys.exit(1)

            # create obj
            terminal = SingleTerminal(at_node=get_node_by_id(load_dict["Node"], nodes))
            load_obj = DeltaConstantCurrentLoad(
                name=f"load{i + 1}",
                id=f"load{i + 1}",
                # at_node=get_node_by_id(load_dict["Node"], nodes),
                terminal=terminal,
                phases=phases,
                active_power=active_power,
                reactive_power=reactive_power,
                n_ph=len(active_power),
                nominal_voltage=nominal_voltage,
                stepchange=step_change,
                iconst=iconst,
                power_factor=power_factor,
            )

        else:
            load_obj_cls = globals()[load_type.value]
            terminal = SingleTerminal(at_node=get_node_by_id(load_dict["Node"], nodes))
            load_obj = load_obj_cls(
                name=f"load{i + 1}",
                id=f"load{i + 1}",
                # at_node=get_node_by_id(load_dict["Node"], nodes),
                terminal=terminal,
                phases=phases,
                active_power=active_power,
                reactive_power=reactive_power,
                n_ph=len(active_power),
                nominal_voltage=nominal_voltage,
                stepchange=step_change,
            )

        print(f">> load_obj: {load_obj}")

        return load_obj

    @staticmethod
    def _parse_source(source_dict: dict, i: int, nodes: list[Node]) -> Source:
        nominal_voltage = {
            ph: 1e3 * v for (ph, v) in source_dict["nominal_voltage"].items()
        }

        # phases = adapter_utils.find_phases(source["nominal_voltage"])
        phases = {}
        for ph, v in nominal_voltage.items():
            if v == 0:
                phases[ph] = 0
            else:
                phases[ph] = 1

        if "source_type" not in source_dict:
            raise ValueError("'source_type' needs to be specified")

        terminal = SingleTerminal(
            at_node=get_node_by_id(str(source_dict["Node"]), nodes)
        )

        if source_dict["source_type"] == "ConstantVoltage":
            source_obj = ConstantVoltageSource(
                name=f"source{i + 1}",
                id=f"source{i + 1}",
                # at_node=get_node_by_id(str(source_dict["Node"]), nodes),
                terminal=terminal,
                phases=phases,
                nominal_voltage=nominal_voltage,
                n_ph=len(phases),
            )

        # elif source_dict["source_type"] == "GFMInverter3Ph":
        #     controls = InverterControl(
        #         primary_control=InverterPrimaryControl(source_dict["primary_control"]),
        #         single_dual_control=InverterSingleDualControl(source_dict["single_dual_control"]),
        #     )
        #     source_obj = GFMInverter3Ph(
        #         name=f"source{i + 1}",
        #         id=f"source{i + 1}",
        #         # at_node=get_node_by_id(str(source_dict["Node"]), nodes),
        #         terminal=terminal,
        #         phases=phases,
        #         nominal_voltage=nominal_voltage,
        #         n_ph=len(phases),
        #         La1=source_dict["La1"],
        #         Lb1=source_dict["Lb1"],
        #         Lc1=source_dict["Lc1"],
        #         raL1=source_dict["raL1"],
        #         rbL1=source_dict["rbL1"],
        #         rcL1=source_dict["rcL1"],
        #         La2=source_dict["La2"],
        #         Lb2=source_dict["Lb2"],
        #         Lc2=source_dict["Lc2"],
        #         raL2=source_dict["raL2"],
        #         rbL2=source_dict["rbL2"],
        #         rcL2=source_dict["rcL2"],
        #         Ca=source_dict["Ca"],
        #         Cb=source_dict["Cb"],
        #         Cc=source_dict["Cc"],
        #         raC=source_dict["raC"],
        #         rbC=source_dict["rbC"],
        #         rcC=source_dict["rcC"],
        #         controls=controls,
        #     )

        elif source_dict["source_type"] == "ConstantSeqVoltage":
            source_obj = ConstantVoltageSeqSource(
                name=f"source{i + 1}",
                id=f"source{i + 1}",
                # at_node=get_node_by_id(str(source_dict["Node"]), nodes),
                terminal=terminal,
                phases=phases,
                nominal_voltage=nominal_voltage,
                n_ph=len(phases),
            )

        else:
            raise NotImplementedError

        return source_obj

    # function to parse source defined in sequence components
    def _parse_source_with_seq(source_dict: dict, i: int, nodes: list[Node]) -> Source:
        # get the phases
        phases = {}
        for ph, value in source_dict["phases"].items():
            if value == 0:
                phases[ph] = 0
            else:
                phases[ph] = 1

        if "source_type" not in source_dict:
            raise ValueError("'source_type' needs to be specified")

        if source_dict["source_type"] == "ConstantVoltage":
            terminal = SingleTerminal(
                at_node=get_node_by_id(str(source_dict["Node"]), nodes)
            )
            # get magnitude of negative sequence voltage from a and b and positive seq voltage in the input
            a = source_dict["a"]
            b = source_dict["b"]

            # get magnitude of positive sequence voltage from the input
            V1 = (source_dict["postive_seq_voltage"],)
            # get angle for positive sequence voltage from the input
            theta1 = source_dict["theta1"]
            # get phasor for V1
            V1 = V1 * np.exp(1j * theta1)

            # get magnitude of negative sequence voltage from a and b
            V2 = a * V1 + b  # linear function of pos seq voltage
            # get angle for negative sequence voltage
            theta2 = source_dict["theta2"]
            # get phasor for V2
            V2 = V2 * np.exp(1j * theta2)

            # get magnitude of zero sequence voltage from a and b and positive seq voltage in the input
            V0 = source_dict["zero_seq_voltage"]

            # get abc voltage from the sequence voltages using Fortescue transformation
            Vabc = utils.fortescue_transform(
                V1, V2, V0
            )  # write this function in utils so as it returns a dict

            source_obj = ConstantVoltageSource(
                name=f"source{i + 1}",  # should continue from earlier source
                id=f"source{i + 1}",  # should continue from earlier source
                # at_node=get_node_by_id(str(source_dict["Node"]), nodes),
                terminal=terminal,
                phases=phases,
                nominal_voltage=Vabc,  # this would not be same on all phases now and should be a dict returned by Fortescue transformation
                n_ph=len(phases),
            )
        else:
            raise NotImplementedError

        return source_obj

    # theirs
    @staticmethod
    def _parse_inverters(
        inverter_data: list[dict],
        nodes,
        wnom: float,
        base_voltage: float,
        base_power: float,
    ) -> list[Inverter]:
        result = []
        for i, data in enumerate(inverter_data):
            # model = data["config"]
            # config = inverter_config[model]
            typ = data["source_type"]

            nominal_voltage = {
                ph: 1e3 * v for (ph, v) in data["nominal_voltage"].items()
            }

            # phases = adapter_utils.find_phases(source["nominal_voltage"])
            phases = {}
            for ph, v in nominal_voltage.items():
                if v == 0:
                    phases[ph] = 0
                else:
                    phases[ph] = 1

            match typ:
                case "GFLInverter3Ph":
                    terminal = SingleTerminal(
                        at_node=get_node_by_id(data["Node"], nodes)
                    )

                    inverter = GFLInverter3Ph(
                        name=f"inverter{i}",
                        # equipment_id=f"inverter{i}",
                        id=f"inverter{i}",
                        terminal=terminal,
                        n_ph=data["n_phases"],
                        phases=data["phases"],
                        Vdc=data["Vdc"],
                        Pb=base_power,  # W
                        V_base=base_voltage,
                        delta=data["delta"],
                        # Filter parameters
                        La1=data["La1"],
                        Lb1=data["Lb1"],
                        Lc1=data["Lc1"],
                        raL1=data["raL1"],
                        rbL1=data["rbL1"],
                        rcL1=data["rcL1"],
                        La2=data["La2"],
                        Lb2=data["Lb2"],
                        Lc2=data["Lc2"],
                        raL2=data["raL2"],
                        rbL2=data["rbL2"],
                        rcL2=data["rcL2"],
                        Ca=data["Ca"],
                        Cb=data["Cb"],
                        Cc=data["Cc"],
                        raC=data["raC"],
                        rbC=data["rbC"],
                        rcC=data["rcC"],
                        # switch and diode parameters
                        Vsw=data["Vsw"],  # V
                        Vd=data["Vd"],  # V
                        rsw=data["rsw"],  # ohm
                        rd=data["rd"],  # ohm
                        # controller constants
                        # k=k,
                        Kpc=data["Kpc"],
                        Kic=data["Kic"],
                        # PLL
                        Kppll=data["Kppll"],
                        Kipll=data["Kipll"],
                        nominal_voltage=nominal_voltage,
                    )
                case "GFL_inverter_3ph":
                    inverter = GFL_inverter_3ph(
                        name=f"inverter{i + 1}",
                        equipment_id=f"inverter{i + 1}",
                        from_node=get_node_by_id(data["Node"], nodes),
                        to_node=None,
                        n_ph=config["n_phases"],
                        phases=config["phases"],
                        Vdc=config["Vdc"],
                        Pb=config["Pb"],  # W\
                        delta=config["delta"],
                        # Filter parameters
                        La1=config["La1"],
                        Lb1=config["Lb1"],
                        Lc1=config["Lc1"],
                        raL1=config["raL1"],
                        rbL1=config["rbL1"],
                        rcL1=config["rcL1"],
                        La2=config["La2"],
                        Lb2=config["Lb2"],
                        Lc2=config["Lc2"],
                        raL2=config["raL2"],
                        rbL2=config["rbL2"],
                        rcL2=config["rcL2"],
                        Ca=config["Ca"],
                        Cb=config["Cb"],
                        Cc=config["Cc"],
                        raC=config["raC"],
                        rbC=config["rbC"],
                        rcC=config["rcC"],
                        # switch and diode parameters (from 61016 AVM case study)
                        Vsw=config["Vsw"],  # V
                        Vd=config["Vd"],  # V
                        rsw=config["rsw"],  # ohm
                        rd=config["rd"],  # ohm
                        # controller constants
                        k=config["k"],  # pu
                        Kpc=config["Kpc"],  # pu
                        Kic=config["Kic"],  # pu
                        Pref=config["Pref"],  # pu
                        Qref=config["Qref"],  # pu
                        # PLL
                        Kppll=config["Kppll"],  # pu
                        Kipll=config["Kipll"],  # pu
                    )

                case "GFMInverter3Ph":
                    terminal = SingleTerminal(
                        at_node=get_node_by_id(data["Node"], nodes)
                    )
                    primary_control = InverterPrimaryControl(data["primary_control"])
                    single_dual_control = InverterSingleDualControl(
                        data["single_dual_control"]
                    )
                    controls = InverterControl(primary_control, single_dual_control)

                    # base values

                    # # NOTE: this is temporary hard-coding [REMOVE THIS]
                    # V_nominal = (
                    #     0.8 * V_base / np.sqrt(3)
                    # )  # line-to-neutral voltage (V, rms)
                    # nominal_voltage = {}
                    # for ph in data["nominal_voltage"]:
                    #     v = 1e3 * V_nominal
                    #     nominal_voltage[ph] = v

                    V_base = data["Vdc"] / np.sqrt(3)

                    print(f">> data:{pformat(data)}")

                    assert data["switch_state"] in (SwitchState.Open.value, SwitchState.Closed.value)

                    inverter = GFMInverter3Ph(
                        name=f"inverter{i}",
                        # equipment_id=f"inverter{i}",
                        id=f"inverter{i}",
                        terminal=terminal,
                        n_ph=data["n_phases"],
                        phases=data["phases"],
                        Vdc=data["Vdc"],
                        Pb=data["Pb"],  # W
                        V_base=V_base,
                        delta=data["delta"],
                        # Filter parameters
                        La1=data["La1"],
                        Lb1=data["Lb1"],
                        Lc1=data["Lc1"],
                        raL1=data["raL1"],
                        rbL1=data["rbL1"],
                        rcL1=data["rcL1"],
                        La2=data["La2"],
                        Lb2=data["Lb2"],
                        Lc2=data["Lc2"],
                        raL2=data["raL2"],
                        rbL2=data["rbL2"],
                        rcL2=data["rcL2"],
                        Ca=data["Ca"],
                        Cb=data["Cb"],
                        Cc=data["Cc"],
                        raC=data["raC"],
                        rbC=data["rbC"],
                        rcC=data["rcC"],
                        # switch and diode parameters
                        Vsw=data["Vsw"],  # V
                        Vd=data["Vd"],  # V
                        rsw=data["rsw"],  # ohm
                        rd=data["rd"],  # ohm
                        # droop coefficients
                        kw=data["kw"],
                        kq=data["kq"],
                        # CC constants (from Hugo paper)
                        taus=data["taus"],  # ms
                        # VR
                        Kpv=data["Kpv_pu"],
                        tauiv=data[
                            "tauiv"
                        ],  # Time constant for PI voltage controlelr (s)
                        Imx=data["Imx_pu"],
                        # CC
                        Kpc=data["Kpc_pu"],
                        tauic=data[
                            "tauic"
                        ],  # TIme constant for PI current controlelr (s)
                        Vmx=data["Vmx_pu"],
                        # Inputs
                        # vqcap_star=data["vqcap_star_pu"],
                        # vdcap_star=data["vdcap_star_pu"],
                        nominal_voltage=nominal_voltage,
                        controls=controls,
                        switch_state=dict["switch_state"]
                    )
                case "GFL_inverter_1ph":
                    inverter = GFL_inverter_1ph(
                        name=f"inverter{i}",
                        equipment_id=f"inverter{i}",
                        from_node=get_node_by_id(data["Node"], nodes),
                        to_node=None,
                        n_ph=config["n_phases"],
                        phases=config["phases"],
                        Vdc=config["Vdc"],
                        Pb=config["Pb"],  # W
                        delta=config["delta"],
                        L1=config["L1"],
                        rL1=config["rL1"],
                        L2=config["L2"],
                        rL2=config["rL2"],
                        C=config["C"],
                        rC=config["rC"],
                        Vsw=config["Vsw"],
                        Vd=config["Vd"],
                        rsw=config["rsw"],
                        rd=config["rd"],
                        Kppll=config["Kppll"],
                        Kipll=config["Kipll"],
                        k=config["k"],
                        L=config["L"],
                        Kpc=config["Kpc"],
                        Kic=config["Kic"],
                        Pref=config["Pref"],
                        Qref=config["Qref"],
                        igdi_ref=config["igdi_ref"],
                    )
                case "GFM_inverter_1ph":
                    inverter = GFM_inverter_1ph(
                        name=f"inverter{i}",
                        equipment_id=f"inverter{i}",
                        from_node=get_node_by_id(data["Node"], nodes),
                        to_node=None,
                        n_ph=config["n_phases"],
                        phases=config["phases"],
                        Vdc=config["Vdc"],
                        Pb=config["Pb"],  # W
                        delta=config["delta"],
                        L1=config["L1"],
                        rL1=config["rL1"],
                        L2=config["L2"],
                        rL2=config["rL2"],
                        C=config["C"],
                        rC=config["rC"],
                        Vsw=config["Vsw"],
                        Vd=config["Vd"],
                        rsw=config["rsw"],
                        rd=config["rd"],
                        kw=config["kw"],
                        Pemx=config["Pemx"],
                        taus=config["taus"],
                        Kpv=config["Kpv"],
                        tauiv=config["tauiv"],
                        Imx=config["Imx"],
                    )
                case _:
                    raise ValueError(f"unsupported inverter_type: {typ}")

            result.append(inverter)
        return result

    @staticmethod
    def _parse_generators(generator_data: list[dict], nodes) -> list[Generator]:
        result = []
        for i, data in enumerate(generator_data):
            raise NotImplementedError
            model = data["gen_name"]
            config = generator_config[model]
            typ = config["gen_type"]

            # presently only one generator so typ is not being used-> rewrite when more typ added
            generator = Generator(
                name=f"generator{i}",
                equipment_id=f"generator{i}",
                from_node=None,
                to_node=get_node_by_id(str(data["Node"]), nodes),
                n_ph=config["n_phases"],
                phases=config["phases"],
                VA_rating=config["VA_rating"],
                V_rating_LL=config["V_rating_LL"],
                power_factor=config["power_factor"],
                poles_count=config["poles_count"],
                speed=config["speed"],  # r/min
                gen_turbine_inertia=config["gen_turbine_inertia"],  # Joules.s^2
                # parameters in ohms
                rs=config["rs"],
                rkd=config["rkd"],
                rfd=config["rfd"],
                rkq1=config["rkq1"],
                rkq2=config["rkq2"],
                Xls=config["Xls"],
                Xq=config["Xq"],
                Xd=config["Xd"],
                Xlkq1=config["Xlkq1"],
                Xlfd=config["Xlfd"],
                Xlkq2=config["Xlkq2"],
                Xlkd=config["Xlkd"],
            )
            result.append(generator)
        return result

    @staticmethod
    def init_config(line_config: dict) -> dict[str, ConfigData]:
        """
        This block of code creates a dictionary of configuration.
        key represents configuration no.
        value consists of R_matrix, X_matrix and Phases associated with
        the configuration
        """
        config_dict = dict.fromkeys(line_config.keys())
        for key in line_config:
            R_matrix = _extract_RXY_matrix(line_config[key][0], "R")
            X_matrix = _extract_RXY_matrix(line_config[key][0], "X")
            Y_matrix = _extract_RXY_matrix(line_config[key][1], "Y")

            Phases = line_config[key][2]
            # Phasing = line_config[key][3]
            config_data_tup = ConfigData(
                R_matrix=R_matrix,
                X_matrix=X_matrix,
                Y_matrix=Y_matrix,
                Phases=Phases,
                # Phasing=Phasing,
            )

            config_dict[key] = config_data_tup
        return config_dict
