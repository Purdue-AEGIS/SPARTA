import numpy as np

study_of_interest = "PowerFlow"
simulation_time = 0.1 # only for dynamic study
include_capacitance_of_lines = True
nominal_frequency = 60  # Hz


line_seg_data = [
    {"Node_A": 34, "Node_B": 800, "Length_ft": 0.0, "config": "Substation"},
    {"Node_A": 800, "Node_B": 802, "Length_ft": 2580.0, "config": "300"},
    {"Node_A": 802, "Node_B": 806, "Length_ft": 1730.0, "config": "300"},
    {"Node_A": 806, "Node_B": 808, "Length_ft": 32230.0, "config": "300"},
    {"Node_A": 808, "Node_B": 810, "Length_ft": 5804.0, "config": "303"},
    {"Node_A": 808, "Node_B": 812, "Length_ft": 37500.0, "config": "300"},
    {"Node_A": 812, "Node_B": 814, "Length_ft": 29730.0, "config": "300"},
    {"Node_A": 814, "Node_B": 850, "Length_ft": 100.0, "config": "301"},
    {"Node_A": 816, "Node_B": 818, "Length_ft": 1710.0, "config": "302"},
    {"Node_A": 816, "Node_B": 824, "Length_ft": 10210.0, "config": "301"},
    {"Node_A": 818, "Node_B": 820, "Length_ft": 48150.0, "config": "302"},
    {"Node_A": 820, "Node_B": 822, "Length_ft": 13740.0, "config": "302"},
    {"Node_A": 824, "Node_B": 826, "Length_ft": 3030.0, "config": "303"},
    {"Node_A": 824, "Node_B": 828, "Length_ft": 840.0, "config": "301"},
    {"Node_A": 828, "Node_B": 830, "Length_ft": 20440.0, "config": "301"},
    {"Node_A": 830, "Node_B": 854, "Length_ft": 520.0, "config": "301"},
    {"Node_A": 832, "Node_B": 858, "Length_ft": 4900.0, "config": "301"},
    {"Node_A": 832, "Node_B": 888, "Length_ft": 0.0, "config": "XFM-1"},
    {"Node_A": 834, "Node_B": 860, "Length_ft": 2020.0, "config": "301"},
    {"Node_A": 834, "Node_B": 842, "Length_ft": 280.0, "config": "301"},
    {"Node_A": 836, "Node_B": 840, "Length_ft": 860.0, "config": "301"},
    {"Node_A": 836, "Node_B": 862, "Length_ft": 280.0, "config": "301"},
    {"Node_A": 842, "Node_B": 844, "Length_ft": 1350.0, "config": "301"},
    {"Node_A": 844, "Node_B": 846, "Length_ft": 3640.0, "config": "301"},
    {"Node_A": 846, "Node_B": 848, "Length_ft": 530.0, "config": "301"},
    {"Node_A": 850, "Node_B": 816, "Length_ft": 310.0, "config": "301"},
    {"Node_A": 852, "Node_B": 832, "Length_ft": 100.0, "config": "301"},
    {"Node_A": 854, "Node_B": 856, "Length_ft": 23330.0, "config": "303"},
    {"Node_A": 854, "Node_B": 852, "Length_ft": 36830.0, "config": "301"},
    {"Node_A": 858, "Node_B": 864, "Length_ft": 1620.0, "config": "302"},
    {"Node_A": 858, "Node_B": 834, "Length_ft": 5830.0, "config": "301"},
    {"Node_A": 860, "Node_B": 836, "Length_ft": 2680.0, "config": "301"},
    {"Node_A": 862, "Node_B": 838, "Length_ft": 4860.0, "config": "304"},
    {"Node_A": 888, "Node_B": 890, "Length_ft": 10560.0, "config": "300"},
]

nom_V = 24.9 / np.sqrt(3)
nom_V_xmer = 4.16 / np.sqrt(3)

transformer_data = [
    {
        "Name": "Substation",
        "kVA": 2500,
        "kV_high": 69,
        "high_winding_connect": "D",
        "kV_low": 24.9,
        "low_winding_connect": "GrY",
        "R_percent_pu": 1,
        "X_percent_pu": 8,
    },
    {
        "Name": "XFM-1",
        "kVA": 500,
        "kV_high": 24.9,
        "high_winding_connect": "GrY",
        "kV_low": 4.16,
        "low_winding_connect": "GrY",
        "R_percent_pu": 1.9,
        "X_percent_pu": 4.08,
    },
]

capacitor_data = [
    {
        "Node": 844,
        "power": {"A": 100, "B": 100, "C": 100},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
        "connection": "Y",
    },
    {
        "Node": 848,
        "power": {"A": 150, "B": 150, "C": 150},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
        "connection": "Y",
    },
]


dist_load_data = [
    {
        "NodeA": 802,
        "NodeB": 806,
        "Model": "Y_PQ",
        "active_power": {"B": 30, "C": 25},
        "reactive_power": {"B": 15, "C": 14},
        "nominal_voltage": {"B": nom_V, "C": nom_V},
    },
    {
        "NodeA": 808,
        "NodeB": 810,
        "Model": "Y_I",
        "active_power": {"B": 16},
        "reactive_power": {"B": 8},
        "nominal_voltage": {"B": nom_V},
    },
    {
        "NodeA": 818,
        "NodeB": 820,
        "Model": "Y_Z",
        "active_power": {"A": 34},
        "reactive_power": {"A": 17},
        "nominal_voltage": {"A": nom_V},
    },
    {
        "NodeA": 820,
        "NodeB": 822,
        "Model": "Y_PQ",
        "active_power": {"A": 135},
        "reactive_power": {"A": 70},
        "nominal_voltage": {"A": nom_V},
    },
    {
        "NodeA": 816,
        "NodeB": 824,
        "Model": "D_I",
        "active_power": {"BC": 5},
        "reactive_power": {"BC": 2},
        "nominal_voltage": {"B": nom_V, "C": nom_V},
    },
    {
        "NodeA": 824,
        "NodeB": 826,
        "Model": "Y_I",
        "active_power": {"B": 40},
        "reactive_power": {"B": 20},
        "nominal_voltage": {"B": nom_V},
    },
    {
        "NodeA": 824,
        "NodeB": 828,
        "Model": "Y_PQ",
        "active_power": {"C": 4},
        "reactive_power": {"C": 2},
        "nominal_voltage": {"C": nom_V},
    },
    {
        "NodeA": 828,
        "NodeB": 830,
        "Model": "Y_PQ",
        "active_power": {"A": 7},
        "reactive_power": {"A": 3},
        "nominal_voltage": {"A": nom_V},
    },
    {
        "NodeA": 854,
        "NodeB": 856,
        "Model": "Y_PQ",
        "active_power": {"B": 4},
        "reactive_power": {"B": 2},
        "nominal_voltage": {"B": nom_V},
    },
    {
        "NodeA": 832,
        "NodeB": 858,
        "Model": "D_Z",
        "active_power": {"AB": 7, "BC": 2, "CA": 6},
        "reactive_power": {"AB": 3, "BC": 1, "CA": 3},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "NodeA": 858,
        "NodeB": 864,
        "Model": "Y_PQ",
        "active_power": {"A": 2},
        "reactive_power": {"A": 1},
        "nominal_voltage": {"A": nom_V},
    },
    {
        "NodeA": 858,
        "NodeB": 834,
        "Model": "D_PQ",
        "active_power": {"AB": 4, "BC": 15, "CA": 13},
        "reactive_power": {"AB": 2, "BC": 8, "CA": 7},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "NodeA": 834,
        "NodeB": 860,
        "Model": "D_Z",
        "active_power": {"AB": 16, "BC": 20, "CA": 110},
        "reactive_power": {"AB": 8, "BC": 10, "CA": 55},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "NodeA": 860,
        "NodeB": 836,
        "Model": "D_PQ",
        "active_power": {"AB": 30, "BC": 10, "CA": 42},
        "reactive_power": {"AB": 15, "BC": 6, "CA": 22},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "NodeA": 836,
        "NodeB": 840,
        "Model": "D_I",
        "active_power": {"AB": 18, "BC": 22},
        "reactive_power": {"AB": 9, "BC": 11},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "NodeA": 862,
        "NodeB": 838,
        "Model": "Y_PQ",
        "active_power": {"B": 28},
        "reactive_power": {"B": 14},
        "nominal_voltage": {"B": nom_V},
    },
    {
        "NodeA": 842,
        "NodeB": 844,
        "Model": "Y_PQ",
        "active_power": {"A": 9},
        "reactive_power": {"A": 5},
        "nominal_voltage": {"A": nom_V},
    },
    {
        "NodeA": 844,
        "NodeB": 846,
        "Model": "Y_PQ",
        "active_power": {"B": 25, "C": 20},
        "reactive_power": {"B": 12, "C": 11},
        "nominal_voltage": {"B": nom_V, "C": nom_V},
    },
    {
        "NodeA": 846,
        "NodeB": 848,
        "Model": "Y_PQ",
        "active_power": {"B": 23},
        "reactive_power": {"B": 11},
        "nominal_voltage": {"B": nom_V},
    },
]

spot_load_data = [
    {
        "Node": 860,
        "Model": "Y_PQ",
        "active_power": {"A": 20, "B": 20, "C": 20},
        "reactive_power": {"A": 16, "B": 16, "C": 16},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "Node": 840,
        "Model": "Y_I",
        "active_power": {"A": 9, "B": 9, "C": 9},
        "reactive_power": {"A": 7, "B": 7, "C": 7},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "Node": 844,
        "Model": "Y_Z",
        "active_power": {"A": 135, "B": 135, "C": 135},
        "reactive_power": {"A": 105, "B": 105, "C": 105},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "Node": 848,
        "Model": "D_PQ",
        "active_power": {"AB": 20, "BC": 20, "CA": 20},
        "reactive_power": {"AB": 16, "BC": 16, "CA": 16},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
    {
        "Node": 890,
        "Model": "D_I",
        "active_power": {"AB": 150, "BC": 150, "CA": 150},
        "reactive_power": {"AB": 75, "BC": 75, "CA": 75},
        "nominal_voltage": {"A": nom_V_xmer, "B": nom_V_xmer, "C": nom_V_xmer},
    },
    {
        "Node": 830,
        "Model": "D_Z",
        "active_power": {"AB": 10, "BC": 10, "CA": 25},
        "reactive_power": {"AB": 5, "BC": 5, "CA": 10},
        "nominal_voltage": {"A": nom_V, "B": nom_V, "C": nom_V},
    },
]

source_data = [
    {
        "Node": 34,
        "nominal_voltage": {
            "A": 69 / np.sqrt(3),
            "B": 69 / np.sqrt(3),
            "C": 69 / np.sqrt(3),
        },
    }
]

inverter_data = []
generator_data = []
volt_reg_data = []
