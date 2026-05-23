import numpy as np
from scipy.signal import lfilter, upfirdn
from sympy import *
from icecream import ic
import matplotlib.pyplot as plt


def plotter(values):
        keys = list(values.keys())
        fig, axes = plt.subplots(len(keys), 1)
        for index in range(len(keys)):
            init_array = values[keys[index]]
            axes[index].plot(range(len(init_array)), init_array)
            axes[index].set_title(keys[index])
        plt.tight_layout()
        plt.show()

def crc_check(data_bits, crc_poly_list):
    x = symbols('x')
    # Define the CRC polynomial 1 + x + x^4 -> x^4 + x + 1
    # In sympy, we represent this polynomial as a Poly object:
    crc_poly = Poly(crc_poly_list, x, modulus=2)  # x^4 + x + 1
    
    # Create a polynomial for the combined data
    data_poly = Poly(data_bits, x, modulus=2)
    
    # Perform the polynomial division to find the remainder (CRC result)
    quotient, remainder = div(data_poly, crc_poly)
    
    return remainder.coeffs()

def binary_partition_sequence(sequence, chunk_size):
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]

def compute_syndromes(received, generator_polynomial, x):
    syndromes = []
    for i in range(generator_polynomial.degree()):
        eval_poly = Poly(received[::-1], x, modulus=2).eval(i)
        syndromes.append(int(eval_poly % 2))
    return syndromes

def correct_errors(received, syndromes, n, k):
    if max(syndromes) == 0:
        return received  # No errors detected
    
    error_positions = []  # Error locator polynomial solution
    for i in range(n):
        eval_syndrome = sum([syndromes[j] * (i ** j) for j in range(len(syndromes))]) % 2
        if eval_syndrome == 0:
            error_positions.append(i)
    
    corrected = received[:]
    for pos in error_positions:
        corrected[pos] ^= 1  # Flip bit at detected error position
    
    return corrected

def bch_decoder(PSDU, n, k, fragment_size):
    x = symbols('x', real=True)
    generator_polynomial = Poly(x**14 + x**9 + x**8 + x**6 + x**5 + x**4 + x**2 + x + 1, x, modulus=2)
    
    MPDU = []
    for fragmented_packet in binary_partition_sequence(PSDU, fragment_size + (fragment_size // k) * (n - k)):
        subpackets = [fragmented_packet[i:i+n] for i in range(0, len(fragmented_packet), n) if len(fragmented_packet[i:i+n]) == n]
        decoded_subpackets = []
        
        for subpacket in subpackets:
            syndromes = compute_syndromes(subpacket, generator_polynomial, x)
            corrected_packet = correct_errors(subpacket, syndromes, n, k)
            decoded_subpackets.append(corrected_packet[:k])
        
        MPDU.extend([bit for subpacket in decoded_subpackets for bit in subpacket])
    
    return MPDU

class GFSK_demodulator():
    def __init__(self, fc):
        self.BT = 0.5
        self.h = 0.5
        self.L = 20
        self.fc = fc

    def gaussianLPF(self, Tb, k, BT, L):
        """
        Generate filter coefficients of Gaussian low pass filter (used in gmsk_mod)
        Parameters:
        BT : BT product - Bandwidth x bit period
        Tb : bit period
        L : oversampling factor (number of samples per bit)
        k : span length of the pulse (bit interval)
        Returns:
        h_norm : normalized filter coefficients of Gaussian LPF
        """
        B = BT/Tb # bandwidth of the filter
        # truncated time limits for the filter
        t = np.arange(start = -k*Tb, stop = k*Tb + Tb/L, step = Tb/L)
        h = B*np.sqrt(2*np.pi/(np.log(2)))*np.exp(-2 * (t*np.pi*B)**2 /(np.log(2)))
        h_norm=h/np.sum(h)
        
        return h_norm
    
    def gfsk_demodulate(self, r_complex):
        I=np.real(r_complex)
        Q = -np.imag(r_complex)
        z1 = Q * np.hstack((np.zeros(self.L), I[0:len(I)-self.L]))
        z2 = I * np.hstack((np.zeros(self.L), Q[0:len(I)-self.L]))
        z = z1 - z2
        a_hat = (z[2*self.L-1:-self.L:self.L] > 0).astype(int)
        return a_hat
    
    def viterbi_decoder(self, modulation_index, pulse_length, M):
        m = modulation_index[0]
        p = modulation_index[1]
        L = pulse_length
        M = M

        if m % 2 == 0:  # Even m
            num_states = p * (M ** (L - 1))
        else:  # Odd m
            num_states = 2 * p * (M ** (L - 1))

        D = int(5 * np.log2(num_states))


    
    
    

class receiver():

    def __init__(self, received_signal, fc):
        self.received_signal = received_signal
        self.fc = fc

    
    def run(self):
        demodulator = GFSK_demodulator(self.fc)
        self.demodulated_bits = demodulator.gfsk_demodulate(self.received_signal)
        error_check = crc_check(self.demodulated_bits, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1])
        if int(1) in error_check:
            self.error = True
        else:
            self.error = False
            #self.demodulated_bits = self.demodulated_bits[:len(self.demodulated_bits)-16]
            #self.preamble = self.demodulate#d_bits[:16]
            #self.PLCP = self.demodulated_bits[16:56]
            #self.PSDU = self.demodulated_bits[56:]


            #self.PLSDU = bch_decoder(self.PSDU, 127, 113, 143)
            
            #plcp_error_check = crc_check(self.PLCP, [1, 0, 1, 0, 1])
            #plsdu_error_check = crc_check(self.PLSDU, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1])
            #ic(plsdu_error_check)
            #psdu_error_check = crc_check(self.PSDU, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1])

        #self.decoded_bits = demodulator.viterbi_decoder(self.demodulated_bits)
        #ic(ic(), self.decoded_bits)










'''
class GFSKDemodulator:
    def __init__(self, modulation_index, pulse_length, M, samples_per_symbol):
        self.m = modulation_index[0]
        self.p = modulation_index[1]
        self.L = pulse_length
        self.M = M
        self.samples_per_symbol = samples_per_symbol
        self.fs=1e6
        
        # Calculate numStates
        if self.m % 2 == 0:  # Even m
            self.num_states = self.p * (self.M ** (self.L - 1))
        else:  # Odd m
            self.num_states = 2 * self.p * (self.M ** (self.L - 1))
        
        # Traceback Depth
        self.traceback_depth = int(5 * np.log2(self.num_states))
    
    def viterbi_demodulate(self, received_signal):
        """
        Perform GFSK demodulation using the Viterbi algorithm.

        Parameters:
            received_signal (np.ndarray): The received signal after channel impairments.

        Returns:
            np.ndarray: Demodulated binary bit sequence.
        """
        
        num_symbols = len(received_signal) // self.samples_per_symbol
        num_states = self.num_states

        # Initialize path metrics and survivor paths
        path_metrics = np.full((num_states, num_symbols), np.inf)
        path_metrics[0, 0] = 0  # Start with state 0
        survivor_paths = np.zeros((num_states, num_symbols), dtype=int)

        # Viterbi forward pass
        for t in range(1, num_symbols):
            for next_state in range(num_states):
                min_metric = float('inf')
                best_prev_state = None

                for prev_state in range(num_states):
                    current_symbol = received_signal[t * self.samples_per_symbol:(t + 1) * self.samples_per_symbol]
                    # Ensure `expected_symbol` is compatible
                    expected_symbol = np.array(self.expected_symbol(next_state))
                    if expected_symbol.ndim == 0:
                        expected_symbol = np.expand_dims(expected_symbol, axis=0)

                    # Compute the Euclidean distance
                    #if len(current_symbol.flatten() - expected_symbol.flatten())==1:
                    #    metric = np.abs((current_symbol.flatten() - expected_symbol.flatten()).item())
                    #else:
                    metric = np.linalg.norm(current_symbol - expected_symbol)
                    total_metric = path_metrics[prev_state, t - 1] + metric

                    if total_metric < min_metric:
                        min_metric = total_metric
                        best_prev_state = prev_state

                path_metrics[next_state, t] = min_metric
                survivor_paths[next_state, t] = best_prev_state

        # Traceback
        decoded_states = np.zeros(num_symbols, dtype=int)
        decoded_states[-1] = np.argmin(path_metrics[:, -1])

        for i in range(num_symbols - 2, max(-1, num_symbols - self.traceback_depth - 1), -1):
            decoded_states[i] = survivor_paths[decoded_states[i + 1], i + 1]

        # Convert states to bits
        demodulated_bits = self.states_to_bits(decoded_states)

        return demodulated_bits
    
    def expected_symbol(self, state):
        """
        Calculate the expected signal for a given state in GFSK.

        Parameters:
            state (int): Current state (related to symbol history in MLSD).

        Returns:
            float: Expected value of the signal (e.g., frequency deviation or phase).
        """
        # Modulation parameters
        h = 0.5  # GFSK modulation index
        samples_per_symbol = self.samples_per_symbol
        frequency_deviation = h / (2 * samples_per_symbol)  # Frequency deviation (Hz)

        # Convert state to phase
        phase = 2 * np.pi * frequency_deviation * state / self.fs

        # Return the expected value (e.g., phase or signal level)
        return np.cos(2 * np.pi * phase)
    
    def states_to_bits(self, states):
        """
        Convert states to binary bits based on the state mapping.
        """
        return np.array([int(bin(state)[2:].zfill(int(np.log2(self.num_states)))[-1]) for state in states])


class receiver():

    def __init__(self, received_signal):
        self.received_signal = received_signal

    def gfsk_demodulate(self):
        samples_per_symbol = 1
        # Frequency discrimination (differentiator)
        diff_signal = np.diff(self.received_signal)

        # Low-pass filter (to recover baseband)
        lpf = np.ones(samples_per_symbol) / samples_per_symbol
        baseband_signal = lfilter(lpf, 1.0, diff_signal)

        # Threshold decision (for binary symbols)
        demodulated_bits = (baseband_signal > 0).astype(int)
        return demodulated_bits
    
    def egc(self, signal):
        return signal / np.abs(signal)
    
    def crc_check(self, data, polynomial):
        x = symbols('x')
        data_poly = Poly(data, x, modulus=2)
        crc_poly = Poly(polynomial, x, modulus=2)
        _, remainder = div(data_poly, crc_poly)
        return all(bit == 0 for bit in remainder.coeffs())
    
    def run(self):
        egc_out = self.egc(self.received_signal)
        #ic(len(egc_out))

        demodulator = GFSKDemodulator(modulation_index=(1, 2),  # m/p
                              pulse_length=1,
                              M=2,  # Binary GFSK
                              samples_per_symbol=20)
        self.demodulated_bits = demodulator.viterbi_demodulate(egc_out)
        #ic((self.demodulated_bits))
        #self.modulator_output = self.gfsk_demodulate().tolist()
        check = self.crc_check(self.demodulated_bits[:-16], [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1])
        #ic(check)




        #######################################################




        def viterbi_decoder(self, received_signal):
        """
        Demodulate the GFSK signal using the Viterbi algorithm with traceback depth.
        """

        def generate_reference_signals():
            """
            Generate the reference signals for 0 and 1 using the Gaussian filter.
            """
            Tb = 1 / self.fc  # Bit period
            L = self.L            # Oversampling factor
            h = self.h        # Modulation index
            BT = self.BT          # BT product for the Gaussian filter

            # Gaussian Low-Pass Filter
            g_lpf = self.gaussianLPF(Tb=Tb, k=1, BT=BT, L=L)

            # Generate reference signals for 0 and 1
            data_0 = np.zeros(L)  # Bit 0
            data_1 = np.ones(L)   # Bit 1

            # Pass through the Gaussian filter and integrate
            c0 = upfirdn(h=[1] * L, x=2 * data_0 - 1, up=L)
            c1 = upfirdn(h=[1] * L, x=2 * data_1 - 1, up=L)

            b0 = np.convolve(g_lpf, c0, mode="full")[:L]
            b1 = np.convolve(g_lpf, c1, mode="full")[:L]

            b0_norm = b0 / np.max(np.abs(b0))
            b1_norm = b1 / np.max(np.abs(b1))

            phi0 = np.cumsum(b0_norm * (h * np.pi / Tb))
            phi1 = np.cumsum(b1_norm * (h * np.pi / Tb))

            ref_0 = np.cos(phi0) - 1j * np.sin(phi0)  # Reference for bit 0
            ref_1 = np.cos(phi1) - 1j * np.sin(phi1)  # Reference for bit 1

            #plotter({"c0": c0, "c1": c1, "b0": b0, "b1": b1, "b0_norm": b0_norm, "b1_norm": b1_norm, "phi0": phi0, "phi1": phi1, "ref_0": ref_0, "ref_1": ref_1})

            return {0: ref_0, 1: ref_1}

        # Generate reference signals
        ref_signals = generate_reference_signals()

        # Extend reference signals to match the received signal
        ref_signals = {key: np.tile(signal, len(received_signal) // len(signal) + 1)[:len(received_signal)]
                       for key, signal in ref_signals.items()}

        # Compute branch metrics
        branch_metrics = {
            0: np.abs(received_signal - ref_signals[0])**2,
            1: np.abs(received_signal - ref_signals[1])**2
        }

        # Initialize Trellis
        num_states = 2  # For GFSK, two states represent bit-0 and bit-1
        traceback_depth = int(5 * np.log2(num_states))  # Traceback depth for reliable decoding
        trellis = np.zeros((len(received_signal), num_states))
        path = np.zeros((len(received_signal), num_states), dtype=int)

        # Build Trellis
        for t in range(1, len(received_signal)):
            for s in range(num_states):
                costs = [
                    trellis[t-1, prev_state] + branch_metrics[prev_state][t]
                    for prev_state in range(num_states)
                ]
                trellis[t, s] = min(costs)
                path[t, s] = np.argmin(costs)

        # Traceback to determine the decoded bits
        state = np.argmin(trellis[-1])
        decoded_bits = []

        # Start traceback from the end of the trellis
        for t in range(len(received_signal) - 1, len(received_signal) - traceback_depth - 1, -1):
            decoded_bits.insert(0, state)
            state = path[t, state]

        return decoded_bits[::-1]  # Reverse the list to get the correct order

'''