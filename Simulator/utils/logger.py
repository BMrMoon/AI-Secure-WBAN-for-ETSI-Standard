class Logger:
    def __init__(self, filepath):
        self.filepath = filepath

    def write(self, message):
        with open(self.filepath, "a") as file:
            file.write(message + "\n")