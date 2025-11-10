import json
from network.network import network
from network.channel import channel
from network.receiver import receiver
import tkinter as tk
import random
from icecream import ic
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
from tqdm import tqdm


class AnomalySimulator:
    def jamming_attack(self, signal, intensity):
        """Simulate a jamming attack by adding high-intensity noise."""
        noise = np.random.normal(0, intensity, len(signal))
        return signal + noise

    def inject_packets(self, packet_stream, num_injections):
        """Simulate packet injection by modifying random packets."""
        for _ in range(num_injections):
            idx = np.random.randint(0, len(packet_stream))
            packet_stream[idx] = np.random.choice([0, 1])
        return packet_stream

    def replay_attack(self, packet_stream, replay_rate):
        """Simulate a replay attack by repeating random packets."""
        replay_indices = np.random.choice(
            range(len(packet_stream)), size=int(len(packet_stream) * replay_rate), replace=False
        )
        for idx in replay_indices:
            packet_stream[idx] = packet_stream[np.random.randint(0, len(packet_stream))]
        return packet_stream



class Simulation:
    def __init__(self, config_path, logger, root, dataset_path):
        self.dataset_path = dataset_path
        self.root = root
        self.config_path = config_path
        self.logger = logger
        # Load configuration
        if type(config_path) == str():
            with open(config_path+"/smartBAN_config.json", "r") as file:
                self.smartBAN_config = json.load(file)
        else:
            self.smartBAN_config = config_path

        #self.phy_layer = PHYLayer(self.config["phy"])
        #self.mac_layer = MACLayer(self.config["mac"])

    def count_error(self, array1, array2):
        count_true = 0
        count_false = 0
        comparison = (array1==array2)
        for index in range(len(comparison)):
            if comparison[index]:
                count_true += 1
            else:
                count_false += 1
        ic(ic(), "True count: ", count_true)
        ic(ic(), "False count: ", count_false)
    
    def plotter(self, values):
        keys = list(values.keys())
        fig, axes = plt.subplots(len(keys), 1)
        for index in range(len(keys)):
            init_array = values[keys[index]]
            axes[index].plot(range(len(init_array)), init_array)
            axes[index].set_title(keys[index])
        plt.tight_layout()
        plt.show()

    def generate_SNR(self, num_packets):
        start = eval(self.smartBAN_config["Eb/N0 Entry"])["start"]
        stop = eval(self.smartBAN_config["Eb/N0 Entry"])["stop"]
        step = eval(self.smartBAN_config["Eb/N0 Entry"])["step"]
        #start = self.smartBAN_config["Eb/N0 Entry"]["start"]
        #stop = self.smartBAN_config["Eb/N0 Entry"]["stop"]
        #step = self.smartBAN_config["Eb/N0 Entry"]["step"]
        sample_number = int((stop - start)/step + 1)
        SNR_list = np.linspace(start, stop, sample_number)

        mu = random.choice(SNR_list)
        sigma = 13
        alpha = 0.9

        snr_values = np.zeros(num_packets)
        snr_values[0] = np.random.uniform(mu-20, mu+20)

        for i in range(1, num_packets):
            snr_values[i] = alpha * snr_values[i-1] + (1 - alpha) * mu + sigma * np.random.randn()
            snr_values[i] = np.clip(snr_values[i], mu-20, mu+20)  # Ensure within limits
    
        return snr_values

    def estimated_SNR(self, snr, index, distance, f):
        fspl = 20 * np.log10(distance) + 20 * np.log10(f) + 20 * np.log10(4 * np.pi / 3e8)
        snr_init = snr[index]
        return snr_init
    
    def get_signal(self, packet_type, tx, snr, data_channel):
        if packet_type == "jamming":
            signal = tx.i_signal - 1j*tx.q_signal
            ch = channel(self.smartBAN_config, signal, tx.distance)
            ch.run(snr)
            noisy_signal = ch.channel_output
            signal = self.anomaly.jamming_attack(noisy_signal, intensity=5)
        elif packet_type == "packet_injection":
            injected_data = self.anomaly.inject_packets(tx.data.copy(), num_injections=10)
            m_s, i_s, q_s = tx.gfsk_modulation(data=np.array(injected_data, dtype=np.float64), fc=data_channel)
            signal = i_s - 1j*q_s
            ch = channel(self.smartBAN_config, signal, tx.distance)
            ch.run(snr)
            signal = ch.channel_output
        elif packet_type == "replay":
            replayed_data = self.anomaly.replay_attack(tx.data.copy(), replay_rate=0.2)
            m_s, i_s, q_s = tx.gfsk_modulation(data=np.array(replayed_data, dtype=np.float64), fc=data_channel)
            signal = i_s - 1j*q_s
            ch = channel(self.smartBAN_config, signal, tx.distance)
            ch.run(snr)
            signal = ch.channel_output
        else:
            signal = tx.i_signal - 1j*tx.q_signal
            ch = channel(self.smartBAN_config, signal, tx.distance)
            ch.run(snr)
            noisy_signal = ch.channel_output
            signal = noisy_signal
        return signal
    
    def selected_anomalies(self):
        anomaly_list = []
        if self.smartBAN_config["Jamming Attack Combobox"] == "on":
            anomaly_list.append("jamming")
        if self.smartBAN_config["Replay Attack Combobox"] == "on":
            anomaly_list.append("replay")
        if self.smartBAN_config["Packet Injection Attack Combobox"] == "on":
            anomaly_list.append("packet_injection")
        return anomaly_list
    
    def selected_packet_type(self, anomaly_list):
        if anomaly_list == []:
            return "anomaly_free"
        elif len(anomaly_list) == 1:
            return np.random.choice([anomaly_list[0], "anomaly_free"])
        else:
            return np.random.choice([np.random.choice(anomaly_list), "anomaly_free"])
    def check_path(self, file_path):
        if os.path.exists(file_path):
            return True
        else:
            return False
    def datalist_update(self, datalist, data, current_dataset_length):
        max_length = max([current_dataset_length, datalist.shape[1], len(data)])

        if datalist == []:
            datalist.append(data)
        else:
            length_datalist_elements = len(datalist[0])
            if length_datalist_elements > len(data):
                pass

    def dataframes_update(self):
        pass

    def initial_update(self, initial_dataset_bits, initial_dataset_label, initial_bits, initial_label, target_length):
        pass

    def maximum_length(self, current_dataset_df, initial_dataset_bits, initial_bits):
        return max([current_dataset_df.shape[1]-1, initial_dataset_bits.shape[1], len(initial_bits)])
    


    def log(self, message):
        """Log simulation events."""
        try:
            self.logger.insert(tk.END, f"{message}\n")
            self.logger.insert(tk.END, f"\n")
            self.root.update()
        except:
            pass
        print(message)  # For console debugging

    def run(self):
        """Run the simulation."""
        self.log("Simulation started.")
        networks = list()
        for _ in range(int(self.smartBAN_config["Hub Number Entry"])):
            Network = network(self.smartBAN_config)
            globals()[f"network_{Network.banMAC[0]}"] = Network
            networks.append(f"network_{Network.banMAC[0]}")

        # Data Storage variables
        self.dataset = ()
        if self.check_path(self.dataset_path):
            chunk_list = []
            for chunk in tqdm(pd.read_csv(self.dataset_path, low_memory=False, chunksize=int(self.smartBAN_config["Frame Number Entry"]))):
                chunk_list.append(chunk)
            current_dataset_df = pd.concat(chunk_list)
            current_dataset_df = current_dataset_df.loc[:, ~current_dataset_df.columns.str.contains('^Unnamed')]
        else:
            current_dataset_df = pd.DataFrame()
        



        # Initial variables
        packet_number_total = int(self.smartBAN_config["Frame Number Entry"])
        packet_number_hub = int(packet_number_total/len(networks))
        packet_number_node = int(packet_number_hub/10)
        self.anomaly = AnomalySimulator()
        anomaly_list = self.selected_anomalies()
        anomaly_duration = 0
        start_step = 0
        step = 0
        tic = time.time()
        while step < packet_number_total:
            # Hub, Nodes and random anomalies setup
            if step % packet_number_hub == 0:
                network_selection = networks[int(step/packet_number_hub)]
                snr_channel = self.generate_SNR(packet_number_hub)
            if step % packet_number_node == 0:
                node_selection = random.choice(list(getattr(globals()[network_selection], "transmitters").keys()))
                packet_type = self.selected_packet_type(anomaly_list)
                anomaly_duration = random.randint(1, int(packet_number_node/5))
                start_step = step
            if step-start_step+1 == anomaly_duration:
                packet_type = self.selected_packet_type(anomaly_list)
                anomaly_duration = random.randint(1, int(packet_number_node/5))
                start_step = step

            # Transmitter
            data_channel = random.choice(list(getattr(globals()[network_selection], "data_channels")))
            cm_channel = random.choice(list(getattr(globals()[network_selection], "cm_channels")))
            tx = getattr(globals()[network_selection], "transmitters")[node_selection]
            tx.run(data_channel)

            # Channel
            snr = self.estimated_SNR(snr_channel, step % packet_number_hub, tx.distance, data_channel)
            signal = self.get_signal(packet_type, tx, snr, data_channel)

            # Receiver
            info = ""
            if self.smartBAN_config["Retransmission Combobox"] == "off":
                rx = receiver(signal, data_channel)
                rx.run()
                if rx.error:
                    data = rx.demodulated_bits
                    info = "Lost packet"
                else:
                    data = rx.demodulated_bits
                    info = "Error-free transmission"
                label = packet_type
                
                # Updating numpy arrays
                if 'initial_dataset_bits' in locals():
                    maximum_length = self.maximum_length(current_dataset_df, initial_dataset_bits, data)
                    initial_dataset_bits = np.concatenate((np.pad(initial_dataset_bits, [(0, 0), (0, maximum_length-initial_dataset_bits.shape[1])], 'constant', constant_values=(0, 0)), np.array([np.pad(data, (0, maximum_length-len(data)), 'constant', constant_values=(0, 0))])), axis=0)
                else:
                    initial_dataset_bits = np.array([data])
                if 'initial_dataset_label' in locals():
                    initial_dataset_label = np.concatenate((initial_dataset_label, np.array(label)), axis=None)
                else:
                    initial_dataset_label = np.array(label)

                
                self.dataset = self.dataset + ((data, label), )
                self.log(f"Network: {network_selection} Node: {node_selection} Packet No: {step} Packet: {label} Simulating... Information: {info}")
                
                step += 1
            else:
                attempt = 1
                while (attempt <= 3) and (step < packet_number_total):
                    rx = receiver(signal, data_channel)
                    rx.run()
                    data = rx.demodulated_bits
                    label = packet_type
                    
                    # Updating numpy arrays
                    if 'initial_dataset_bits' in locals():
                        maximum_length = self.maximum_length(current_dataset_df, initial_dataset_bits, data)
                        initial_dataset_bits = np.concatenate((np.pad(initial_dataset_bits, [(0, 0), (0, maximum_length-initial_dataset_bits.shape[1])], 'constant', constant_values=(0, 0)), np.array([np.pad(data, (0, maximum_length-len(data)), 'constant', constant_values=(0, 0))])), axis=0)
                    else:
                        initial_dataset_bits = np.array([data])
                    if 'initial_dataset_label' in locals():
                        initial_dataset_label = np.concatenate((initial_dataset_label, np.array(label)), axis=None)
                    else:
                        initial_dataset_label = np.array(label)

                    self.dataset = self.dataset + ((data, label), )
                    self.log(f"Network: {network_selection} Node: {node_selection} Packet No: {step} Packet: {label} Simulating... Information: {info}")
                    if rx.error:
                        if attempt == 1:
                            if ((step) % packet_number_node == 0) or ((step) % packet_number_hub == 0) or ((step) > packet_number_total):
                                step += 1
                                break
                            else:
                                snr = self.estimated_SNR(snr_channel, step % packet_number_hub, tx.distance, data_channel)
                                signal = self.get_signal(packet_type, tx, snr, data_channel)
                                attempt += 1
                                step += 1
                                info = "Transmition error"
                        else:
                            if ((step) % packet_number_node == 0) or ((step) % packet_number_hub == 0) or ((step) > packet_number_total):
                                step += 1
                                break
                            else:
                                snr = self.estimated_SNR(snr_channel, step % packet_number_hub, tx.distance, data_channel)
                                signal = self.get_signal(packet_type, tx, snr, data_channel)
                                info = "Retransmission"
                                attempt += 1
                                step += 1
                    else:
                        info = "Error-free transmission"
                        step += 1
                        break


        if self.check_path(self.dataset_path):
            current_dataset_df_bits = pd.DataFrame(current_dataset_df.iloc[:,0:-1])
            current_dataset_df_label = pd.DataFrame(current_dataset_df.iloc[:,-1])
            ic(len(current_dataset_df_bits), len(current_dataset_df_label))
            initial_dataset_bits_df = pd.DataFrame(initial_dataset_bits)
            initial_dataset_label_df = pd.DataFrame(initial_dataset_label)

            current_dataset_df_bits.columns = [int(col) for col in current_dataset_df_bits.columns]
            current_dataset_df_label.columns = [int(col) for col in current_dataset_df_label.columns]
            initial_dataset_bits_df.columns = [int(col) for col in initial_dataset_bits_df.columns]
            initial_dataset_label_df.columns = [int(col) for col in initial_dataset_label_df.columns]

            initial_df = pd.concat([initial_dataset_bits_df, initial_dataset_label_df], ignore_index=True, axis=1)
            initial_df.columns = [int(col) for col in initial_df.columns]

            bits_length = current_dataset_df_bits.shape[1]
            if self.maximum_length(current_dataset_df, initial_dataset_bits, data) > bits_length:
                padding_df = pd.DataFrame(np.zeros(shape=(current_dataset_df_bits.shape[0], self.maximum_length(current_dataset_df, initial_dataset_bits, data)-bits_length)))
                ic(padding_df)
                ic(current_dataset_df_label)
                padding_df.columns = [int(col) for col in padding_df.columns]
                padded_df = pd.concat([current_dataset_df_bits, padding_df, current_dataset_df_label], ignore_index=True, axis=1)
                ic(padded_df)
                padded_df.columns = [int(col) for col in padded_df.columns]
                final_df = pd.concat([padded_df, initial_df], ignore_index=True, axis=0)
                ic(final_df)
            else:
                current_dataset_df.columns = [int(col) for col in current_dataset_df.columns]
                final_df = pd.concat([current_dataset_df, initial_df], ignore_index=True, axis=0)
        else:
            initial_dataset_bits_df = pd.DataFrame(initial_dataset_bits)
            initial_dataset_label_df = pd.DataFrame(initial_dataset_label)
            initial_dataset_bits_df.columns = [int(col) for col in initial_dataset_bits_df.columns]
            initial_dataset_label_df.columns = [int(col) for col in initial_dataset_label_df.columns]
            final_df = pd.concat([initial_dataset_bits_df, initial_dataset_label_df], ignore_index=True, axis=1)

        final_df.to_csv(self.dataset_path, index=False)
        ic(final_df.shape)
            #initial_dataset_bits_df.to_csv(self.dataset_path, index=False)
            
        self.log("Simulation completed.")
        toc = time.time()
        print(f"Estimation Time: {toc-tic}")            

