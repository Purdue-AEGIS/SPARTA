"""
-This file provides models of generator and inverters(to be added later)
-The user can also create their own models by just copying the a generator model and modifying the parameters as
 per their requirements for analysis purpose.
-Presently two generator models are provided here "Gen1" and "Gen2"
-The user can view the model parameters and use the gen_model_name to call these model in the input ieee13_input_data file.
"""

####################################################################################
"""
Create a catalog of generators:
This file provides models of generators with default values
The user can also create their own models by just copying the a generator model and modifying the parameters as
per their requirements for analysis purpose.
The user should pick the gen_name from the catalog and use it in the input ieee13_input_data file.
"""
generator_config = {}


generator_config["Gen1"] = {
    "gen_name": "Gen1",
    "MVA_rating": 835.0,
    "kV_rating_LL": 26.0,
    "power_factor": 0.85,
    "poles_count": 2,
    "speed": 3600,  # r/min
    "gen_turbine_inertia": 0.0658e6,  # Joules.s^2
    # parameters in ohms
    "rs": 0.00243,
    "rkd": 0.01080,
    "rfd": 0.00075,
    "rkq1": 0.00144,
    "rkq2": 0.00681,
    "Xls": 0.1538,
    "Xq": 1.457,
    "Xd": 1.457,
    "Xlkq1": 0.6578,
    "Xlfd": 0.1145,
    "Xlkq2": 0.07602,
    "Xlkd": 0.06577,
}

generator_config["Gen2"] = {
    "gen_name": "Gen2",
    "MVA_rating": 325.0,
    "kV_rating_LL": 20.0,
    "power_factor": 0.85,
    "poles_count": 64,
    "speed": 112.5,  # r/min
    "gen_turbine_inertia": 35.1e6,  # Joules.s^2
    # parameters in ohms
    "rs": 0.00234,
    "rkd": 0.01736,
    "rfd": 0.00050,
    "rkq2": 0.00681,
    "Xls": 0.1478,
    "Xq": 0.5911,
    "Xd": 1.0467,
    "Xlfd": 0.2523,
    "Xlkq2": 0.1267,
    "Xlkd": 0.1970,
}
