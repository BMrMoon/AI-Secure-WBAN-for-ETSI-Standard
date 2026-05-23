class MACLayer:
    def __init__(self, config):
        self.config = config

    def schedule(self):
        """Simulate packet scheduling."""
        print("MAC: Scheduling packets.")