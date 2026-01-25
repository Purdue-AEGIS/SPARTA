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

# this generator has two field windings and one damper winding
generator_config["Gen1_2fd"] = {
    "gen_name": "Gen1_2fd",
    "gen_type": "Gen_fd2_d1",
    "n_phases": 1,
    "phases": ["A"],
    # Base values
    "VA_rating": 1e6,
    "V_rating_LL": 1.16e3,
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
