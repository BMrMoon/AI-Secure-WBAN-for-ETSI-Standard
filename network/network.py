import random
from network.transmitter import transmitter
from network.receiver import receiver
from icecream import ic


class network():

    def __init__(self, config):
        self.smartBAN_config = config
        if self.smartBAN_config["Random Seed Entry"]=='':
            self.devicesMAC, self.hubMAC, self.banMAC = self.generate_MAC()
            self.preamble = [random.randint(0,1) for _ in range(16)]
        else:
            random.seed(int(self.smartBAN_config["Random Seed Entry"]))
            self.devicesMAC, self.hubMAC, self.banMAC = self.generate_MAC()
            self.preamble = [random.randint(0,1) for _ in range(16)]
            random.random()

        self.transmitters = {}
        for device in self.devicesMAC:
            IDs = {"node":self.bitstream_conv(device), "hub":self.bitstream_conv(self.hubMAC[0]), "BAN":self.bitstream_conv(self.banMAC[0]), "preamble":self.preamble}
            self.transmitters["transmitter_"+device] = transmitter(IDs, self.smartBAN_config)
        self.data_channels = [int((2402 + 2*n)*1e6) for n in range(39) if n not in [0, 12, 39]]
        self.cm_channels = [int((2402 + 2*n)*1e6) for n in range(39) if n in [0, 12, 39]]

    def generate_MAC(self):
        def MAC():
            mac = random.randint(0, 255)
            return f'{mac:02x}'
        unique_macs = set() 
        MAC_nodes = []
        while len(MAC_nodes) < int(self.smartBAN_config["Node Number Entry"]):
            mac = MAC()
            if mac not in unique_macs:  # Benzersiz kontrolü
                unique_macs.add(mac)
                MAC_nodes.append(mac)
        MAC_hubs = []
        while len(MAC_hubs) < 1:
            mac = MAC()
            if mac not in unique_macs:  # Benzersiz kontrolü
                unique_macs.add(mac)
                MAC_hubs.append(mac)
        MAC_BAN = []
        while len(MAC_BAN) < 1:
            mac = MAC()
            if mac not in unique_macs:  # Benzersiz kontrolü
                unique_macs.add(mac)
                MAC_BAN.append(mac)
        return MAC_nodes, MAC_hubs, MAC_BAN

    def randomSelectMAC(self):
        return random.choice(self.devicesMAC)
    
    def hex2bit(self, MACs):
        return bin(int(MACs, 16))[2:].zfill(8)
        #return [bin(int(hexCode, 16))[2:].zfill(8) for hexCode in MACs]

    def bit2hex(self, MACs):
        return hex(int(MACs, 2))[2:]
        #return [hex(int(binary, 2))[2:] for binary in MACs]
    
    def str2list_conv(self, message):
        return [int(bit) for bit in message]

    def list2str_conv(self, message):
        sequence = ''
        for bit in message:
            sequence+=str(bit)
        return sequence
    
    def bitstream_conv(self, hex):
        bitstream = self.hex2bit(hex)
        return self.str2list_conv(bitstream)