import random
import tkinter as tk

class PHYLayer:
    preamble = [random.randint(0,1) for _ in range(16)]
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def log(self, message):
        """Log simulation events."""
        self.logger.insert(tk.END, f"{message}\n")
        print(message)  # For console debugging

    def transmit(self):
        """Simulate data transmission."""
        self.log(f"PHY: Transmitting data {self.preamble}.")