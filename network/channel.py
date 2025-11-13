import numpy as np
import random
from icecream import ic
import matplotlib.pyplot as plt


class channel():
    def __init__(self, smartBAN_config, signal, distance):
        """
            smartBAN_config: selections either from GUI or smartBAN_config file
            signal: modulated signaş from transmitter
            distance: distance between transmitter node and hub
            """

        start = eval(smartBAN_config["Eb/N0 Entry"])["start"]
        stop = eval(smartBAN_config["Eb/N0 Entry"])["stop"]
        step = eval(smartBAN_config["Eb/N0 Entry"])["step"]
        sample_number = int((stop - start)/step + 1)
        self.SNR = np.linspace(start, stop, sample_number)

        self.fading = smartBAN_config["Fading Channel Combobox"]
        self.interference = smartBAN_config["Interference Combobox"]
        self.interference_freq = smartBAN_config["Frequency of Interference Combobox"]
        self.distance = distance
        self.signal = signal
        self.Rsym = int(smartBAN_config["Frame Generation Rate Entry"])
    
    def ricean_fading(self, signal):
        """Adds ricean fading."""
        def calculate_path_loss(d):
            P0 = 10 ** (-25.8 / 10)  # Convert dB to linear scale
            m0 = 2.0  # dB/cm
            P1 = 10 ** (-71.3 / 10)  # Convert dB to linear scale
            sigma_P = 3.6  # dB
            n_P = np.random.normal()  # Random variable for shadowing
            shadowing = sigma_P * n_P
            path_loss_linear = P0 * np.exp(-m0 * d) + P1
            P_LdB = -10 * np.log10(path_loss_linear) + shadowing
            return P_LdB
        def calculate_ricean_k_factor(P_LdB):
            K0 = 30.6
            mK = 0.43
            sigma_K = 3.4
            n_K = np.random.normal()  # Random variable for K-factor variation
            K_dB = K0 - mK * P_LdB + sigma_K * n_K
            return K_dB
        
        P_LdB = calculate_path_loss(self.distance)
        K_dB = calculate_ricean_k_factor(P_LdB)
        
        K_linear = 10 ** (K_dB / 10)  # Convert dB to linear scale
        sigma = 1 / np.sqrt(2 * (K_linear + 1))  # Standard deviation for NLOS
        h_LOS = np.sqrt(K_linear / (K_linear + 1))  # LOS component
        h_NLOS = sigma * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
        h = h_LOS + h_NLOS  # Total Ricean fading coefficient
        self.fading_out = h
        return h*signal

    def channel_interference(self, signal):
        """Adds interference"""
        # Define parameters for each interference level
        fs = self.Rsym
        level = self.interference_freq
        if level == 'low':
            num_interferers = 1
            interference_power_dBm = -50  # Weak interference
            freq_offset_range = (0.95 * fs, 1.05 * fs)  # Narrow range
        elif level == 'moderate':
            num_interferers = 3
            interference_power_dBm = -40  # Moderate interference
            freq_offset_range = (0.9 * fs, 1.1 * fs)  # Medium range
        elif level == 'high':
            num_interferers = 6
            interference_power_dBm = -30  # Strong interference
            freq_offset_range = (0.8 * fs, 1.2 * fs)  # Wide range
        else:
            raise ValueError("Invalid level. Choose 'low', 'moderate', or 'high'.")
    
        # Convert power from dBm to linear scale
        interference_power_linear = 10 ** (interference_power_dBm / 10)
    
        # Initialize interference
        signal_length = len(signal)
        interference = np.zeros(signal_length)
    
        # Generate interference
        for _ in range(num_interferers):
            # Random frequency offset within the specified range
            freq_offset = np.random.uniform(*freq_offset_range)
        
            # Generate sinusoidal interferer
            t = np.arange(signal_length) / fs
            interferer = np.sqrt(interference_power_linear) * np.sin(2 * np.pi * freq_offset * t)
        
            # Add to total interference
            interference += interferer
    
        # Add interference to the signal
        received_signal = signal + interference
        self.interference_out = received_signal
        return received_signal

    def awgn(self, signal, snr):
        """Adds AWGN to the rf signal"""
        snr_dB = snr
        #ic(snr_dB)
        # Calculate signal power
        signal_power = np.mean(np.abs(signal)**2)
    
        # Calculate noise power based on desired SNR
        snr_linear = 10**(snr_dB / 10)
        noise_power = signal_power / snr_linear

        # Generate Gaussian noise with calculated variance
        noise = (noise_power**(1/2)) * np.random.randn(len(signal))
    
        # Add noise to the signal
        noisy_signal = signal + noise
        self.awgn_out = noisy_signal
        return noisy_signal

    def run(self, snr):
        
        if self.fading=="on":
            self.signal = self.ricean_fading(self.signal)
        
        if self.interference=="on":
            self.signal = self.channel_interference(self.signal)

        
        self.signal = self.awgn(self.signal, snr)
        self.channel_output = self.signal


