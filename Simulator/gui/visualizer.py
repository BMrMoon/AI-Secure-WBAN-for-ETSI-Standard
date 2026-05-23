import matplotlib.pyplot as plt

class Visualization:
    """Handles plotting of simulation results."""
    
    def show(self):
        # Placeholder data
        time = [0, 1, 2, 3, 4, 5]
        throughput = [10, 20, 15, 25, 30, 35]

        # Plot throughput over time
        plt.figure(figsize=(8, 6))
        plt.plot(time, throughput, marker="o", label="Throughput")
        plt.title("SmartBAN Simulation Results")
        plt.xlabel("Time (s)")
        plt.ylabel("Throughput (kbps)")
        plt.legend()
        plt.grid(True)
        plt.show()