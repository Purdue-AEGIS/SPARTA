"""
-This file provides data for Grid following inverters
-The user can also create their own models by just copying the inverter model and modifying the parameters as
 per their requirements for analysis purpose.
-The user can view the model parameters and use the inverter_name to call these model in the input ieee13_input_data file.
"""

import numpy as np

####################################################################################

inverter_config = {}
# Grid following 3ph inverter default values
inverter_config["GFL_inverter1"] = {
    "inverter_name": "GFL_inverter1",
    "inverter_type": "GFL_inverter_3ph",
    "n_phases": 3,
    "phases": ["A", "B", "C"],
    "Vdc": 1500,
    "Pb": 2 * 0.5e6,  # W
    "delta": 0.0,
    # Filter parameters
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
    # switch and diode parameters (from 61016 AVM case study)
    "Vsw": 1,  # V
    "Vd": 2,  # V
    "rsw": 100e-3,  # ohm
    "rd": 50e-3,  # ohm
    # controller constants
    "k_pu": 0.5,  # pu
    "Kpc_pu": 2.5 * 0.05,  # pu
    "Kic_pu": 2.5 * 5,  # pu
    "Pref": 3e5,  # W
    "Qref": 0,  # W
    # PLL
    "Kppll_pu": 2.22,  # pu
    "Kipll_pu": 246.7,  # pu
}

inverter_config["GFL_inverter2"] = {
    "inverter_name": "GFL_inverter2",
    "inverter_type": "GFL_inverter_3ph",
    "n_phases": 3,
    "phases": ["A", "B", "C"],
    "Vdc": 2 * 4200,
    "Pb": 10e6,  # W
    "delta": 0.0,
    # Filter parameters
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
    # switch and diode parameters (from 61016 AVM case study)
    "Vsw": 1,  # V
    "Vd": 2,  # V
    "rsw": 100e-3,  # ohm
    "rd": 50e-3,  # ohm
    # controller constants
    "k_pu": 0.5,  # pu
    "Kpc_pu": 2.5 * 0.05,  # pu
    "Kic_pu": 2.5 * 5,  # pu
    "Pref": 3e5,  # W
    "Qref": 0,  # W
    # PLL
    "Kppll_pu": 2.22,  # pu
    "Kipll_pu": 246.7,  # pu
}

# Grid forming 3ph inverter default values
inverter_config["GFM_inverter1"] = {
    "inverter_name": "GFM_inverter1",
    "inverter_type": "GFM_inverter_3ph",
    "n_phases": 3,
    "phases": ["A", "B", "C"],
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
    # constants CC (from Hugo paper)
    "kw": 0.05,  # pu
    "tauic": 0.01,  # ms
    # VR
    "Kpv": 0.54,  # 0.1273 #Proportional constant for PI voltage controller (pu)
    "tauiv": 0.02,  # Time constant for PI voltage controlelr (s)
    "Imx": 1,  # pu
}

inverter_config["GFM_inverter2"] = {
    "inverter_name": "GFM_inverter2",
    "inverter_type": "GFM_inverter_3ph",
    "n_phases": 3,
    "phases": ["A", "B", "C"],
    "Vdc": 1500,
    "Pb": 0.25e6,  # W
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
    # resume-here
    "ki_pu": 0.05,  # pu
    "taus": 0.01,  # ms
    # VR
    "Kpv_pu": 0.54,  # 0.1273 #Proportional constant for PI voltage controller (pu)
    "tauiv": 0.02,  # Time constant for PI voltage controlelr (s)
    "Imx_pu": 1,  # pu
}

# Grid following 1ph inverter default values
inverter_config["GFL_inverter3"] = {
    "inverter_name": "GFL_inverter3",
    "inverter_type": "GFL_inverter_1ph",
    "n_phases": 1,
    "phases": ["A"],
    # Base values
    "Vdc": 4200 / 2,
    "Pb": 10e6,
    "delta": 0,
    "Pref": 5e5,
    # Filter parameters
    "L1": 275e-6,
    "rL1": 19e-3,
    "L2": 275e-6,
    "rL2": 19e-3,
    "C": 150e-6,
    "rC": 2.67e-3,
    # Inverter switch and diode parameters (From 61016 AVM case study)
    "Vsw": 1,  # V
    "Vd": 2,  # V
    "rsw": 100e-3,  # ohm
    "rd": 50e-3,  # ohm
    # PLL
    "Kppll": -50,  # pu
    "Kipll": -1000,  # pu
    # Current Controller
    "k": 0.5,  # pu
    "L": 2 * 275e-6,  # H
    "Kpc": 20 * 0.05,  # pu
    "Kic": 20 * 5,  # pu
    "Pref": 5e5,  # W
    "Qref": 0,  # W
    "igdi_ref": 0.0,  # pu
}


# Grid forming 1ph inverter default values
inverter_config["GFM_inverter3"] = {
    "inverter_name": "GFM_inverter3",
    "inverter_type": "GFM_inverter_1ph",
    "n_phases": 1,
    "phases": ["A"],
    # Base values
    "Vdc": 4200,
    "Pb": 1e6,
    "delta": 0,
    # Filter parameters
    "L1": 275e-6,
    "rL1": 19e-3,
    "L2": 275e-6,
    "rL2": 19e-3,
    "C": 3 * 150e-6,
    "rC": 2.67e-3,
    # Inverter switch and diode parameters (From 61016 AVM case study)
    "Vsw": 1,  # V
    "Vd": 2,  # V
    "rsw": 100e-3,  # ohm
    "rd": 50e-3,  # ohm
    # constants (from Hugo paper)
    "kw": 0.05,  # pu
    "Pemx": 1,  # pu
    "taus": 0.01,  # ms
    # VR
    "Kpv": 0.54,  # Proportional constant for PI voltage controller (pu)
    "tauiv": 0.02,  # Time constant for PI voltage controlelr (s)
    "Imx": 1,
}

inverter_config["GFM_inverter_thesis"] = {
    "inverter_name": "GFM_inverter_thesis",
    "inverter_type": "GFM_inverter_3ph",
    "nominal_voltage": {
        "A": 0.8 * 1500 / np.sqrt(2 * 3) / 1000,
        "B": 0.8 * 1500 / np.sqrt(2 * 3) / 1000,
        "C": 0.8 * 1500 / np.sqrt(2 * 3) / 1000,
    },
    "n_phases": 3,
    "phases": {"A": 1, "B": 1, "C": 1},
    "Vdc": 1500,
    "Pb": 0.15e6,  # kVA
    "delta": 0.0,
    # Filter parameter
    "La1": 5.73e-6,
    "Lb1": 5.73e-6,
    "Lc1": 5.73e-6,
    "raL1": 19e-3,
    "rbL1": 19e-3,
    "rcL1": 19e-3,
    "La2": 417e-6,
    "Lb2": 417e-6,
    "Lc2": 417e-6,
    "raL2": 19e-3,
    "rbL2": 19e-3,
    "rcL2": 19e-3,
    "Ca": 118e-6,
    "Cb": 118e-6,
    "Cc": 118e-6,
    "raC": 73e-3,
    "rbC": 73e-3,
    "rcC": 73e-3,
    # switch and diode parameters
    "Vsw": 1,  # V
    "Vd": 2,  # V
    "rsw": 1,  # ohm
    "rd": 1,  # ohm
    # P-w droop
    "kw": 0.05,     # 5%
    "taus": 0.005,   # s (from Hugo's paper)
    # Q-v droop
    "kq": 0.02,     # 2%
    # VR
    "Kpv_pu": 0.54,  # 0.1273 #Proportional constant for PI voltage controller (pu)
    "tauiv": 0.02,  # Time constant for PI voltage controlelr (s)
    "Imx_pu": 1,  # pu
    # CC
    "Kpc_pu": 0.1515,   # pu
    "tauic": 0.004, # s
    # misc
    "source_type": "GFMInverter3Ph",
    "primary_control": "droop",  # Droop/Synchronous
    "single_dual_control": "dual",
}