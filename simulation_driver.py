from simulation.powerflow import Powerflow
from simulation.dynamic__multistage import Dynamic
from parsers.parser_ieee_doc import IEEEDocParser
from models.system_model import SystemModel


import importlib


def main():     
    filename = "test_inputs.ieee_13_input_data_dyn__multi_gfm_study"
    # filename = "test_inputs.ieee_13_input_data_dyn__multi_gfm_study"
  

    # filename = "test_inputs.ieee_34_input_data_wip" # for power flow analysis
    # filename = "test_inputs.ieee_123_input_data" # for power flow analysis
    system = IEEEDocParser.from_ieee_input_data(filename)    
    model = SystemModel(system)
    # powerflow = Powerflow(model)
    # powerflow.run()
    # powerflow.print_y(powerflow.y_final)
    # powerflow.save_results_xlsx("Powerflow_results.xlsx")

    tsim = load_tsim_from_input(filename)
    dynamic = Dynamic(model, tsim)
    dynamic.run()  

def load_tsim_from_input(filename) -> float:
    input_data = importlib.import_module(filename)
    return input_data.simulation_time


if __name__ == "__main__":
    main()
