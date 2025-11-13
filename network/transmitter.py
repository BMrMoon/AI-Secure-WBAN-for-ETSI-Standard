import random
import numpy as np
from sympy import *
from scipy.signal import lfilter, upfirdn
from scipy.ndimage import gaussian_filter1d
import numpy as np
from icecream import ic
import matplotlib.pyplot as plt
from itertools import chain

def plotter(values):
    """Build to observe GFSK demodulator output.
        
        values: dictionary with key value pairs
        """
    keys = list(values.keys())
    fig, axes = plt.subplots(len(keys), 1)
    for index in range(len(keys)):
        init_array = values[keys[index]]
        axes[index].plot(range(len(init_array)), init_array)
        axes[index].set_title(keys[index])
    plt.tight_layout()
    plt.show()

def binary_partition_sequence(sequence, chunk_size):
    """Partitioner for BCH encoder."""
    return [sequence[i:i + chunk_size] for i in range(0, len(sequence), chunk_size)]

def bch_encoder(MPDU, n, k, fragment_size):
    """n-k: number of parity bits"""
    x = symbols('x', real=True)
    generator_polynomial = Poly(x**14 + x**9 + x**8 + x**6 + x**5 + x**4 + x**2 + x + 1, x, modulus=2)

    PSDU = []
    for fragmented_packet in binary_partition_sequence(MPDU[::-1], fragment_size):
        N_padding = (k - (len(fragmented_packet) % k)) % k
        padding_bits = [0]*N_padding
        padded_fragmented_packet = fragmented_packet
        padded_fragmented_packet.extend(padding_bits)
        subpackets = [padded_fragmented_packet[i:i+k] for i in range(0, len(padded_fragmented_packet), k)]
        parity_bits = []
        for subpacket in subpackets:
            subpacket_poly = Poly(subpacket[::-1], x, modulus=2)
            remainder = subpacket_poly.div(generator_polynomial)[1]  # [1] gives the remainder
            remainder_bits = [int(coef) for coef in remainder.all_coeffs()[::-1]]
            parity_bits_length = n-k
            init_parity = []
            init_parity.extend([0] * (parity_bits_length - len(remainder_bits)))
            init_parity.extend(remainder_bits)
            parity_bits.append(init_parity)
        if N_padding>0:
            subpackets[-1] = subpackets[-1][:-N_padding]
        encoded_subpackets = [subpackets[i] + parity_bits[i] for i in range(len(subpackets))]
        psdu = sum(encoded_subpackets, [])
        PSDU.extend(psdu)
    return PSDU


def crc_encoder(data, key):
    """Cyclic redundancy check"""
    def xor(a, b):
 
        # initialize result
        result = []
 
        # Traverse all bits, if bits are
        # same, then XOR is 0, else 1
        for i in range(1, len(b)):
            if a[i] == b[i]:
                result.append('0')
            else:
                result.append('1')
 
        return ''.join(result)
 
 
    # Performs Modulo-2 division
    def mod2div(dividend, divisor):
 
        # Number of bits to be XORed at a time.
        pick = len(divisor)
 
        # Slicing the dividend to appropriate
        # length for particular step
        tmp = dividend[0 : pick]
 
        while pick < len(dividend):
 
            if tmp[0] == '1':
 
                # replace the dividend by the result
                # of XOR and pull 1 bit down
                tmp = xor(divisor, tmp) + dividend[pick]
 
            else: # If leftmost bit is '0'
 
                # If the leftmost bit of the dividend (or the
                # part used in each step) is 0, the step cannot
                # use the regular divisor; we need to use an
                # all-0s divisor.
                tmp = xor('0'*pick, tmp) + dividend[pick]
 
            # increment pick to move further
            pick += 1
 
        # For the last n bits, we have to carry it out
        # normally as increased value of pick will cause
        # Index Out of Bounds.
        if tmp[0] == '1':
            tmp = xor(divisor, tmp)
        else:
            tmp = xor('0'*pick, tmp)
 
        checkword = tmp
        return checkword
    
    # Function used at the sender side to encode
    # data by appending remainder of modular division
    # at the end of data.
    def encodeData(data, key):
 
        l_key = len(key)
 
        # Appends n-1 zeroes at end of data
        appended_data = data + '0'*(l_key-1)
        remainder = mod2div(appended_data, key)
 
        # Append remainder in the original data
        return remainder
    
    def str2list_conv(message):
        return [int(bit) for bit in message]

    def list2str_conv(message):
        sequence = ''
        for bit in message:
            sequence+=str(bit)
        return sequence
    
    parity = str2list_conv(encodeData(list2str_conv(data),list2str_conv(key)))
    parity = [0] * ((len(key)-1) - len(parity)) + parity
    return parity
        
class transmitter:
    def __init__(self, IDs, smartBAN_config):
        self.RID = IDs["hub"]
        self.SID = IDs["node"]
        self.BID = IDs["BAN"]
        self.preamble = IDs["preamble"]
        self.smartBAN_config = smartBAN_config

        self.distance = round(random.uniform(20, float(self.smartBAN_config["Distance Entry"])), 2)

        

        

    def get_MAC_header(self):
        protocol_verion = [0, 0, 0]
        ack_policy = [1]
        frame_type = [1, 0] #data
        frame_subtype = [0] + [random.randint(0,1) for _ in range(2)] # 4 priority levels, randomly selected
        sequence_number = [0, 0, 0, 0, 0, 0, 0, 0]
        fragment_number = [0, 0, 0]
        non_final_fragment = [0]
        command_ack = [1]
        reserved = [0, 0]
        control_frame = protocol_verion + ack_policy + frame_type + frame_subtype + sequence_number + fragment_number + non_final_fragment + command_ack + reserved
        MAC_header = control_frame #[random.randint(0,1) for _ in range(24)] Control Frame
        MAC_header.extend(self.RID) # Receiver ID
        MAC_header.extend(self.SID) # Sender ID
        MAC_header.extend(self.BID) # BAN ID
        MAC_header.extend(crc_encoder(MAC_header, [1, 1, 0, 0, 0, 1, 1, 1, 1])) # 8 bit FCS of MAC Header
        return MAC_header
    
    def length_macBody(self):
        # PHY and MAC parameters
        Lpreamble = 16  # Preamble length (in bits)
        LPLCPheader = 40  # PLCP Header length (in bits)
        Lheader = 56  # MAC Header length (in bits, including FCS)
        Lparity = 8  # Parity length (in bits, for error detection/correction)
        Rsym = self.Rsym  # Symbol rate (in symbols/second)
        TS = self.L_slot*1e-3  # Time slot duration (in seconds)
        TIFS = 50e-6  # Inter-Frame Spacing (in seconds)
        Nrep = self.PPDUrepetition  # Number of repetitions for PPDU
        TMUA = 0  # Sensing time for Scheduled/Slotted Aloha mode

        # Frame-specific parameters
        BCH_n = 127  # BCH encoding: block size (n)
        BCH_k = 113  # BCH encoding: message size (k)

        # Calculate T_ACK (Acknowledgment time)
        TACK = (Lpreamble + LPLCPheader + Lheader + Lparity) / Rsym

        # Calculate T_TX,max (Maximum permissible time for initial transmission)
        TTX_max = (TS - TMUA - TACK - 2 * TIFS) / Nrep

        # Calculate L_PSDU,max (Maximum PSDU length in bits)
        LPSDU_max = TTX_max * Rsym - (Lpreamble + LPLCPheader)

        if self.FEC_type=="on":
            # Calculate L_MPDU,max (Maximum MPDU length with BCH encoding)
            LMPDU_max = int(LPSDU_max / BCH_n) * BCH_k
        elif self.FEC_type=="off":
            LMPDU_max = LPSDU_max

        # Calculate L_F,max (Maximum length of MAC Frame Body)
        
        return LMPDU_max - Lheader - Lparity
        #LF_max = LMPDU_max - Lheader - Lparity

        # Print results
        #print("Maximum Packet Length Calculations:")
        #print(f"Maximum PSDU Length (L_PSDU,max): {LPSDU_max:.2f} bits")
        #print(f"Maximum MPDU Length (L_MPDU,max): {LMPDU_max} bits")
        #print(f"Maximum MAC Frame Body Length (L_F,max): {LF_max} bits")

    def PLCP_header(self):
        def str2list_conv(message):
            return [int(bit) for bit in message]
        self.plcp_header = bin(len(self.MAC_body))[2:].zfill(15)
        self.plcp_header = str2list_conv(self.plcp_header)
        self.plcp_header.extend(self.phy_scheme)
        self.plcp_header.extend(self.reserved)
        self.plcp_header.extend(bch_encoder(self.plcp_header, 36, 22, len(self.plcp_header))[-14:])
        self.plcp_header.extend(crc_encoder(self.plcp_header, [1, 0, 1, 0, 1]))

    def PLSDU(self):
        self.plsdu.extend(self.MAC_header)
        self.plsdu.extend(self.MAC_body)
        self.plsdu.extend(crc_encoder(self.MAC_body, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1]))

    def PPDU(self):
        self.ppdu = []
        if self.FEC_type=="on":
            self.psdu = bch_encoder(self.plsdu, 127, 113, 143)
        elif self.FEC_type=="off":
            self.psdu = self.plsdu
        self.ppdu.extend(self.preamble)
        self.ppdu.extend(self.plcp_header)
        self.ppdu.extend(self.psdu)
        self.ppdu.extend(crc_encoder(self.ppdu, [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]))

    
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
    
    def gfsk_modulation(self, data, fc, BT=0.5, h=0.5, L=20):
        fs = L*fc
        Ts=1/fs
        Tb = L*Ts

        h_t = self.gaussianLPF(Tb=Tb, k=1, BT=BT, L=L)
        c_t = upfirdn(h=[1]*L, x=2*data-1, up = L)
        b_t = np.convolve(h_t,c_t,'full')
        bnorm_t = b_t/max(abs(b_t))

        phi_t = lfilter(b = [1], a = [1,-1], x = bnorm_t*Ts) * h*np.pi/Tb

        I = np.cos(phi_t)
        Q = np.sin(phi_t) # cross-correlated baseband I/Q signals
        s_complex = I - 1j*Q # complex baseband representation
        t = Ts* np.arange(start = 0, stop = len(I)) # time base for RF carrier
        sI_t = I*np.cos(2*np.pi*fc*t); sQ_t = Q*np.sin(2*np.pi*fc*t)
        s_t = sI_t - sQ_t # s(t) - GMSK with RF carrier
        s_t = s_t#[:len(c_t)]

        #ic(len(data), len(h_t), len(c_t), len(b_t), len(bnorm_t), len(phi_t), len(I), len(Q), len(s_complex), len(s_t))
        #plotter({"h_t": h_t, "c_t": c_t, "b_t": b_t, "phi_t":phi_t, "s_t": s_t})

        return s_t, I, Q

        
    


    def run(self, fc):
        self.PPDUrepetition = int(self.smartBAN_config["PPDU Repetition Combobox"])
        self.FEC_type = self.smartBAN_config["BCH Encoding Combobox"]
        self.frame_size = self.smartBAN_config["Frame Size Entry"]
        self.Rsym = int(self.smartBAN_config["Frame Generation Rate Entry"])
        self.L_slot = int(self.smartBAN_config["Lslot Combobox"])
        
        self.data = []
        self.MAC_header = []
        self.MAC_body = []
        self.phy_scheme = []
        self.reserved = []
        self.plcp_header = []
        self.plsdu = []
        self.psdu = []
        self.ppdu = []

        self.MAC_header = self.get_MAC_header()

        MAC_body_length = random.randint(1, int(self.length_macBody()))
        self.MAC_body = [random.randint(0,1) for _ in range(MAC_body_length)]

        self.phy_scheme = []
        if self.FEC_type == "on":
            self.phy_scheme.extend([0, 1])
        elif self.FEC_type == "off":
            self.phy_scheme.extend([0, 0])

        if self.PPDUrepetition == 1:
            self.phy_scheme.extend([0, 0])
        elif self.PPDUrepetition == 2:
            self.phy_scheme.extend([0, 1])
        elif self.PPDUrepetition == 4:
            self.phy_scheme.extend([1, 0])
        #print("phy_scheme length:",len(self.phy_scheme))

        self.reserved = [0, 0, 0]
        #print("reserved length:", len(self.reserved))

        self.PLCP_header()
        #print("PLCP header length:", len(self.plcp_header))

        self.PLSDU()
        #print("PLSDU length:", len(self.plsdu))

        self.PPDU()

        
        self.data = np.array(self.ppdu*self.PPDUrepetition, dtype=np.float64)

        self.modulated_signal, self.i_signal, self.q_signal = self.gfsk_modulation(data=self.data, fc=fc)

        #test_crc_tx = crc([1, 0, 0, 1, 0, 0], [1, 1, 0, 1])
        #ic(ic(), test_crc_tx)
        #test_crc_tx = crc_encoder([1, 0, 0, 1, 0, 0], [1, 1, 0, 1])
        #ic(ic(), test_crc_tx)

    

        

        #self.PHY_scheme = PHY_scheme
        #self.L_body = L_body
