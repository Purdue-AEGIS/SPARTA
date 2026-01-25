"""
This file is created to enter the IEEE13 feeder data manually.
 system
   - configurations
   - lines
   - equipments eg. transformers, generators(to be added) etc.
   - loads
"""

import numpy as np

############################################################################################################################################
study_of_interest = "Dynamic"
simulation_time = 0.3
include_capacitance_of_lines = False

nominal_frequency = 60

# pu
base_power = 0.15e6  # kVA
base_voltage = 1500 / np.sqrt(3)  # kV


# create a list of lines
"""
    Enter the line segment data as shown in the table "Line Segment Data"
"""
line_seg_data = [
    # {"Node_A": 600, "Node_B": 650, "Length_ft": 0.0, "config": "Substation"},
    {"Node_A": 650, "Node_B": 630, "Length_ft": 500, "config": "601"},
    {"Node_A": 630, "Node_B": 632, "Length_ft": 2000, "config": "601"},
    {"Node_A": 632, "Node_B": 645, "Length_ft": 500, "config": "603"},
    {"Node_A": 632, "Node_B": 633, "Length_ft": 500, "config": "602"},
    {"Node_A": 633, "Node_B": 634, "Length_ft": 0.0, "config": "XFM-1"},
    {"Node_A": 645, "Node_B": 646, "Length_ft": 300, "config": "603"},
    {"Node_A": 684, "Node_B": 652, "Length_ft": 800, "config": "607"},
    {"Node_A": 632, "Node_B": 671, "Length_ft": 2000, "config": "601"},
    {"Node_A": 671, "Node_B": 684, "Length_ft": 300, "config": "604"},
    {"Node_A": 671, "Node_B": 680, "Length_ft": 1000, "config": "601"},
    {"Node_A": 671, "Node_B": 692, "Length_ft": 500, "config": "601"},
    {"Node_A": 684, "Node_B": 611, "Length_ft": 300, "config": "605"},
    {"Node_A": 692, "Node_B": 675, "Length_ft": 500, "config": "606"},
]

# Note: The line containing a switch is replaced by a line of config 601 and
#       length 500 ft.
###########################################################################################################################################

# equipments:

"""
    Create a list of transformers from Transformer Data
        
"""

transformer_data = [
    # {
    #     "Name": "Substation",
    #     "kVA": 5e3,
    #     "kV_high": 115,
    #     "high_winding_connect": "D",
    #     "kV_low": 4.16,
    #     "low_winding_connect": "GrY",
    #     "R_percent_pu": 1,
    #     "X_percent_pu": 8,
    # },
    {
        "Name": "XFM-1",
        "kVA": 500,
        "kV_high": 4.16,
        "high_winding_connect": "GrY",
        "kV_low": 0.48,
        "low_winding_connect": "GrY",
        "R_percent_pu": 1.1,
        "X_percent_pu": 2,
    },
]


##########################################################################################################################################

# Note to self: Capacitor and Regulator data to be added.
"""
Enter the capacitor data as given in the IEEE document

"""
capacitor_data = [
    # {
    #     "Node": 675,
    #     "power": {"A": 200, "B": 200, "C": 200},
    #     "nominal_voltage": {"A": 2.401777, "B": 2.401777, "C": 2.401777},
    #     "connection": "Y",
    # },
    # {
    #     "Node": 611,
    #     "power": {"C": 100},
    #     "nominal_voltage": {"C": 2.401777},
    #     "connection": "Y",
    # },
]

#########################################################################################################################################
"""
Loads:
    There are two types of load:
        1)Spot load        
        2)Distributed load: The load is distributed uniformly over the line.
        -it will be converted to spot load by lumping 2/3 of total load at
         1/4th distance from the source node of line and remaining 1/3
         is added to the spot load of the end node of the line.
        User needs to enter both the loads in separate lists.

"""
spot_load_data = [
    {
        "Node": 634,
        "Model": "Y_Z",
        "active_power": {"A": 160, "B": 120, "C": 120},
        "reactive_power": {"A": 110, "B": 90, "C": 90},
        "nominal_voltage": {
            "A": 0.48 / np.sqrt(3),
            "B": 0.48 / np.sqrt(3),
            "C": 0.48 / np.sqrt(3),
        },
        "step_change": True,
    },
    {
        "Node": 645,
        "Model": "Y_Z",
        "active_power": {"B": 170},
        "reactive_power": {"B": 125},
        "nominal_voltage": {"B": 2.401777},
        "step_change": True,
    },
    # {
    #     "Node": 646,
    #     "Model": "D_Z",
    #     "active_power": {"BC": 230},
    #     "reactive_power": {"BC": 132},
    #     "nominal_voltage": {"B": 2.401777, "C": 2.401777},
    #     "step_change": True,
    # },
    # {
    #     "Node": 646,
    #     "Model": "Y_Z",
    #     "active_power": {"BC": 230},
    #     "reactive_power": {"BC": 132},
    #     "nominal_voltage": {"B": 2.401777, "C": 2.401777},
    #     "step_change": True,
    # },
    {
        "Node": 646,
        "Model": "Y_Z",
        "active_power": {"B": 115, "C": 115},
        "reactive_power": {"B": 43, "C": 43},
        "nominal_voltage": {"B": 2.401777, "C": 2.401777},
        "step_change": True,
    },
    # {
    #     "Node": 671,
    #     "Model": "D_Z",
    #     "active_power": {"AB": 385, "BC": 385, "CA": 385},
    #     "reactive_power": {"AB": 220, "BC": 220, "CA": 220},
    #     "nominal_voltage": {"A": 2.401777, "B": 2.401777, "C": 2.401777},
    #     "step_change": True,
    # },
    {
        "Node": 671,
        "Model": "Y_Z",
        "active_power": {"A": 128, "B": 128, "C": 128},
        "reactive_power": {"A": 73, "B": 73, "C": 73},
        "nominal_voltage": {"A": 2.401777, "B": 2.401777, "C": 2.401777},
        "step_change": True,
    },
    {
        "Node": 675,
        "Model": "Y_Z",
        "active_power": {"A": 485, "B": 68, "C": 290},
        "reactive_power": {"A": 190, "B": 60, "C": 212},
        "nominal_voltage": {"A": 2.401777, "B": 2.401777, "C": 2.401777},
        "step_change": True,
    },
    # {
    #     "Node": 692,
    #     "Model": "D_Z",
    #     "active_power": {"CA": 170},
    #     "reactive_power": {"CA": 151},
    #     "nominal_voltage": {"A": 2.401777, "C": 2.401777},
    #     "step_change": True,
    # },
    {
        "Node": 692,
        "Model": "Y_Z",
        "active_power": {"A": 85, "C": 85},
        "reactive_power": {"A": 75, "C": 75},
        "nominal_voltage": {"A": 2.401777, "C": 2.401777},
        "step_change": True,
    },
    {
        "Node": 652,
        "Model": "Y_Z",
        "active_power": {"A": 128},
        "reactive_power": {"A": 75},
        "nominal_voltage": {"A": 2.401777},
        "step_change": True,
    },
    {
        "Node": 611,
        "Model": "Y_Z",
        "active_power": {"C": 170},
        "reactive_power": {"C": 80},
        "nominal_voltage": {"C": 2.401777},
        "step_change": True,
    },
]

dist_load_data = [
    {
        "NodeA": 632,
        "NodeB": 671,
        "Model": "Y_Z",
        "active_power": {"A": 17, "B": 66, "C": 117},
        "reactive_power": {"A": 10, "B": 38, "C": 68},
        "nominal_voltage": {"A": 2.401777, "B": 2.401777, "C": 2.401777},
    }
]


########################################################################################################################################
"""
Select the generator model from the equipment_catalog file.
Mention the node to which the generator is connected.

"""
generator_data = [
    # {
    #     "GeneratorModel": "Gen1",
    #     "Node": "650",
    # }
]

source_data = [
    # {
    #     "Node": 600,
    #     "nominal_voltage": {
    #         "A": 115 / np.sqrt(3),
    #         "B": 115 / np.sqrt(3),
    #         "C": 115 / np.sqrt(3),
    #     },
    #     "source_type": "ConstantVoltage",
    # }
    # {
    #     "Node": 650,
    #     "nominal_voltage": {"A": 2.401777, "B": 2.401777, "C": 2.401777},
    #     "source_type": "ConstantVoltage",
    # }
]

slack_bus = 650  # bus on which the primary source is connected

volt_reg_data = []

switch_data = []


def inverter_1(node):
    return {
        "Node": node,
        "nominal_voltage": {"A": 2.401777, "B": 2.401777, "C": 2.401777},
        "inverter_name": "GFM_inverter1",
        "inverter_type": "GFM_inverter_3ph",
        "n_phases": 3,
        "phases": {"A": 1, "B": 1, "C": 1},
        "Vdc": 1500,
        "Pb": 0.15e6,  # W
        "delta": 0.0,
        # Filter parameter
        "La1": 275e-6,
        "Lb1": 275e-6,
        "Lc1": 275e-6,
        "raL1": 19e-3,
        "rbL1": 19e-3,
        "rcL1": 19e-3,
        "La2": 275e-6,
        "Lb2": 275e-6,
        "Lc2": 275e-6,
        "raL2": 19e-3,
        "rbL2": 19e-3,
        "rcL2": 19e-3,
        "Ca": 150e-6,
        "Cb": 150e-6,
        "Cc": 150e-6,
        "raC": 2.67e-3,
        "rbC": 2.67e-3,
        "rcC": 2.67e-3,
        # switch and diode parameters
        "Vsw": 1,  # V
        "Vd": 2,  # V
        "rsw": 100e-3,  # ohm
        "rd": 50e-3,  # ohm
        # constants (from Hugo paper)
        "kw": 0.05,  # pu
        "taus": 0.01,  # ms
        "kq": 0.05,  # pu
        # VR
        "Kpv_pu": 0.54,  # 0.1273 #Proportional constant for PI voltage controller (pu)
        "tauiv": 0.02,  # Time constant for PI voltage controlelr (s)
        "Imx_pu": 1,  # pu
        "Vmx_pu": 1,
        "Kpc_pu": 0.1515,
        "tauic": 0.004,
        "source_type": "GFMInverter3Ph",
        "primary_control": "droop",  # Droop/Synchronous
        "single_dual_control": "dual",
        "switch_state": "open",  # open/closed
    }


def inverter_2(node):
    return {
        "Node": node,
        "nominal_voltage": {"A": 2.401777, "B": 2.401777, "C": 2.401777},
        "inverter_name": "GFM_inverter2",
        "inverter_type": "GFM_inverter_3ph",
        "n_phases": 3,
        "phases": {"A": 1, "B": 1, "C": 1},
        # "Vdc": 5883.128419472076,
        "Vdc": 5900 * 1.15,
        # "Vdc": 2000,
        # "Pb": 10e6,  # W
        "Pb": 20e6/2,  # W
        # "Pb": 0.8e6,  # W
        "delta": 0.0,
        # Filter parameter
        "La1": 275e-6,
        "Lb1": 275e-6,
        "Lc1": 275e-6,
        "raL1": 19e-3,
        "rbL1": 19e-3,
        "rcL1": 19e-3,
        "La2": 275e-6,
        "Lb2": 275e-6,
        "Lc2": 275e-6,
        "raL2": 19e-3,
        "rbL2": 19e-3,
        "rcL2": 19e-3,
        "Ca": 150e-6,
        "Cb": 150e-6,
        "Cc": 150e-6,
        "raC": 2.67e-3,
        "rbC": 2.67e-3,
        "rcC": 2.67e-3,
        # switch and diode parameters
        "Vsw": 1,  # V
        "Vd": 2,  # V
        "rsw": 100e-3,  # ohm
        "rd": 50e-3,  # ohm
        # constants (from Hugo paper)
        "kw": 0.05 / 100,  # pu
        "taus": 0.01,  # ms
        "kq": 0.05,  # pu
        # VR
        "Kpv_pu": 0.54
        * 1.25,  # 0.1273 #Proportional constant for PI voltage controller (pu)
        "tauiv": 0.02 / 10,  # Time constant for PI voltage controlelr (s)
        "Imx_pu": 1,  # pu
        "Vmx_pu": 1,
        # CC
        "Kpc_pu": 0.1515 * 1.5,
        "tauic": 0.004 / 10,
        "source_type": "GFMInverter3Ph",
        "primary_control": "droop",  # Droop/Synchronous
        "single_dual_control": "dual",
        "switch_state": "open",  # open/closed
    }


inverter_data = [
    # {"Node": 650, "Inverter_name": "GFM_inverter2"},
    # {"Node": 632, "Inverter_name": "GFM_inverter2"},
    # inverter_2(632),
    # {"Node": 675, "Inverter_name": "GFM_inverter2"},
    # inverter_2(675),
    # {"Node": 680, "Inverter_name": "GFM_inverter1"},
    # inverter_1(680),
    # inverter_2(680),
    # # {"Node": 671, "Inverter_name": "GFM_inverter1"},
    # inverter_1(671)
    # inverter_2(671)
    # inverter_2(645),
    # inverter_2(652),
    # ------ uncomment below
    inverter_2(650),
    inverter_2(675),
    inverter_2(671),
    inverter_2(632),
    inverter_2(680),
    inverter_2(692),
]
