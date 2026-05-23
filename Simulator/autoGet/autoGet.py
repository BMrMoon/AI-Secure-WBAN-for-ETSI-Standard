from simulation.simulation import Simulation
import pandas as pd
import json
from icecream import ic
from tqdm import tqdm



def load_simulation_config():
    with open("config/smartBAN_config.json", "r") as file:
        return json.load(file)
    
def get_parameters(simulation_config):
    parameters = {}
    for component in simulation_config:
        try:
            parameters[component] = simulation_config[component]
        except:
            pass
    return parameters



dataset_file_path = '/Users/berkaybey/Code/Python/WBAN/v2.0.0/results/data/dataset.csv'
chunk_list = []
for chunk in tqdm(pd.read_csv(dataset_file_path, low_memory=False, chunksize=2000)):
    chunk_list.append(chunk)
dataset_file = pd.concat(chunk_list)
dataset_file = dataset_file.loc[:, ~dataset_file.columns.str.contains('^Unnamed')]
dataset_file = dataset_file.reset_index(drop=True)
init_row = dataset_file.shape[0]
ic(dataset_file.shape)


simulation_config = load_simulation_config()
simulation_config = get_parameters(simulation_config)
ic(simulation_config)

count_start_simulation = int(init_row/int(simulation_config["Frame Number Entry"])) + 1


count = 0
simulation_config["Eb/N0 Entry"] = str(simulation_config["Eb/N0 Entry"])
for fading in ["off", "on"]:
    simulation_config["Fading Channel Combobox"] = fading

    for slot_length in [1, 2, 4]:
        simulation_config["Lslot Combobox"] = slot_length

        for bch in ["off", "on"]:
            simulation_config["BCH Encoding Combobox"] = bch

            for ppdu in [1, 2, 4]:
                simulation_config["PPDU Repetition Combobox"] = ppdu

                for retransmission in ["off", "on"]:
                    simulation_config["Retransmission Combobox"] = retransmission

                    for packet_inj in ["off", "on"]:
                        simulation_config["Packet Injection Attack Combobox"] = packet_inj

                        for replay in ["off", "on"]:
                            simulation_config["Replay Attack Combobox"] = replay

                            for jamming in ["off", "on"]:
                                simulation_config["Jamming Attack Combobox"] = jamming

                                count += 1
                                if count >= count_start_simulation:
                                    ic(count)
                                    ic(simulation_config)
                                    simulation = Simulation(config_path=simulation_config, logger=None, root=None, dataset_path=dataset_file_path)
                                    simulation.run()




#simulation = Simulation(config_path=self.config_path, logger=self.components["Log Text"], root=self.root, dataset_path=dataset_file)
#simulation.run()