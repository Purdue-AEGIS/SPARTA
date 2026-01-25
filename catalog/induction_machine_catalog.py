"""
    -This file provides data for 3-ph and single phase induction machine models with default values
    -The user can also create their own models by just copying the an induction machine model and modifying the parameters as
     per their requirements for analysis purpose.
    -Presently two induction machine models are provided here "Im1_3ph" and "Im2_1ph"
    -The user can view the model parameters and use the induction_mach_name to call these model in the input ieee13_input_data file.
"""

####################################################################################
induction_machine_config = {}

induction_machine_config["IM1_3ph"] = {
    "induction_mach_name": "IM1_3ph",
    "HP_rating": 03.0,
    "kV_rating_LL": 220,  # rated rms line-line voltage, V
    "speed": 1710,  # r/min
    "poles_count": 4,
    "rs": 0.435,  # stator resistance(ohms)
    "Xls": 0.754,  # leakage reactance (ohms)
    "Xm": 26.13,  # magnetizing reactance (ohms)
    "Xlr_prime": 0.754,  # rotor leakage reactance referred to stator (ohms)
    "rr_prime": 0.816,  # rotor resistance referred to stator (ohms)
    "inertia": 0.089,  # kg.m^2
    "t_b": 11.9,  # Nm
}

induction_machine_config["IM2_1ph"] = {
    "induction_mach_name": "IM2_1ph",
    "HP_rating": 0.25,  # HP
    "kV_rating_LL": 110,  # rated rms line-line voltage, V
    "speed": 1450,  # r/min
    "poles_count": 4,
    "rs": 2.02,  # stator resistance(ohms)
    "Xls": 2.79,  # leakage reactance (ohms)
    "Xm": 66.8,  # magnetizing reactance (ohms)
    "Xlr_prime": 2.12,  # rotor leakage reactance referred to stator (ohms)
    "rr_prime": 4.12,  # rotor resistance referred to stator (ohms)
    "inertia": 0.0146,  # kg.m^2
    "t_b": 11.9,  # Nm
}
