import tkinter as tk
from tkinter import ttk
from tkinter import filedialog, messagebox
import yaml
from simulation.simulation import Simulation
import json
import traceback
import pandas as pd
from icecream import ic
import os

class SmartBANApp:
    def __init__(self, root, config_path):
        """
        config_path : path to configurations file
        root : Tkinter root
            """
        self.root = root
        self.config_path = config_path
        self.components = {}
        self.varianles = {}
        self.initial_path = os.getcwd()
        self.dataset_file = self.initial_path+'/results/data/dataset.csv'

    def load_gui_config(self):
        """Load the GUI configuration file."""
        with open(self.config_path+"/gui_config.yaml", "r") as file:
            return yaml.safe_load(file)
        
    def load_simulation_config(self):
        """Load the simulation configuration file."""
        with open(self.config_path+"/smartBAN_config.json", "r") as file:
            return json.load(file)
    
    def set_default(self):
        """Sets GUI selections to the default smartBAN simulation parameters"""
        simulation_config = self.load_simulation_config()
        for component in simulation_config:
            if component.endswith("Entry"):
                #textvar = tk.StringVar()
                #self.components[component]["textvariable"] = textvar
                #textvar.set(simulation_config[component])
                self.components[component].insert(0, simulation_config[component])
                #print(type(self.components[component].get()))
            elif component.endswith("Combobox"):
                value_list = self.components[component]["values"]
                index = value_list.index(str(simulation_config[component]))
                self.components[component].current(index)
                #print(self.components[component].get())
            elif component.endswith("Checkbutton"):
                #self.components[component]["state"]="DISABLED"
                checkbuttonvar = tk.IntVar()
                self.components[component]["variable"] = checkbuttonvar
                checkbuttonvar.set(0)
                #print(self.components[component].cget("variable"))
                #checkbuttonvar.set(1)
    def get_parameters(self, simulation_config):
        """Creates the components dictionary to make it autonome"""
        parameters = {}
        for component in simulation_config:
            try:
                parameters[component] = self.components[component].get()
            except:
                pass
        return parameters
    
    def build_gui(self):
        """Build the GUI from the YAML configuration."""
        config = self.load_gui_config()

        # Set up the main window
        self.root.title(config["window"]["title"])
        self.root.geometry(config["window"]["size"])
        for columnconfigureDict in config["window"]["columnconfigure"]:
            self.root.columnconfigure(columnconfigureDict["index"], weight=columnconfigureDict["weight"])
        for rowconfigureDict in config["window"]["rowconfigure"]:
            self.root.rowconfigure(rowconfigureDict["index"], weight=rowconfigureDict["weight"])

        # Build components
        for component in config["components"]:
            #self.create_component(self.root, component)
            self.create_component(self.root, component)
        
        on_off_list = ["Jamming Attack Combobox", "Replay Attack Combobox", "Packet Injection Attack Combobox", "Retransmission Combobox", "BCH Encoding Combobox", "Fading Channel Combobox", "Interference Combobox"]
        for component in on_off_list:
            self.components[component]["values"] = ("on", "off")
        self.components["PPDU Repetition Combobox"]["values"] = (1, 2, 4)
        self.components["Lslot Combobox"]["values"] = (1, 2, 4, 8, 16, 32)
        self.components["Frequency of Interference Combobox"]["values"] = ("low", "moderate", "high")

        self.set_default()
        #print(self.components["Eb/N0 Entry"].get())

        #self.components["Log Text"]['state'] = 'disabled'
    def create_component(self, parent, config):
        """Creates the component widgets"""
        try:
            widget_class = getattr(ttk, config["type"])
            if "options" in config:
                options = config.get("options", {})
                widget = widget_class(parent, **options)
            else:
                widget = widget_class(parent)
        except:
            widget_class = getattr(tk, config["type"])
            if "options" in config:
                options = config.get("options", {})
                widget = widget_class(parent, **options)
            else:
                widget = widget_class(parent)
            
        # Handle commands for buttons
        if "command" in config:
            widget["command"] = getattr(self, config["command"])

        # Save the widget reference if it has a name
        if "name" in config:
            self.components[config["name"]] = widget

        

        # Handle nested components for containers like Frames
        if "children" in config:
            for child_config in config["children"]:
                self.create_component(widget, child_config)

        
        if "functions" in config:
            for function_name in config["functions"]:
                for function_options in config['functions'][function_name]:
                    if "child" in function_options.keys():
                        child_widget = self.components[function_options["child"]]
                        del function_options["child"]
                        f = getattr(widget, function_name)
                        f(child_widget, **function_options)
                    else:
                        f = getattr(widget, function_name)
                        f(**function_options)

        else:
            widget.pack()

        


    def browse_folder(self):
        """Open file dialog to select a folder."""
        self.dataset_folder_path = filedialog.askdirectory()


    def check_file(self):
        """Checks the path if given filename is exists.
            If exists, it adds the new datas into that dile, and it will creates at the end in vice ersa."""
        if hasattr(self, "dataset_folder_path"):
            self.initial_path = self.dataset_folder_path
            name = self.components["File Name Entry"].get()
            if name == '':
                name = 'dataset.csv'
                self.dataset_file = self.initial_path + name
            else:
                name = self.components["File Name Entry"].get() + '.csv'
                self.dataset_file = self.initial_path + name
        else:
            name = self.components["File Name Entry"].get()
            if name == '':
                name = 'dataset.csv'
                self.dataset_file = self.initial_path + '/results/data/' + name
            else:
                name = self.components["File Name Entry"].get() + '.csv'
                self.dataset_file = self.initial_path + '/results/data/' + name
        
        

    def start_simulation(self):
        """Start the simulation."""
        self.components["Run Button"]["state"] = "disabled"
        simulation_config = self.load_simulation_config()
        self.config_path = self.get_parameters(simulation_config)
        self.check_file()
        #config_file = self.config_path.get()
        #if not config_file:
        #    messagebox.showerror("Error", "Please select a configuration file!")
        #    return
        try:
            self.simulation = Simulation(config_path=self.config_path, logger=self.components["Log Text"], root=self.root, dataset_path=self.dataset_file)
            self.simulation.run()
            self.components['Log Text'].insert(tk.END, "Simulation started...\n")
            #self.components["Dataset Label"]["text"] = self.dataset_path
            self.components["Run Button"]["state"] = "enabled"
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start simulation: {traceback.format_exc()}")


    

if __name__ == "__main__":
    root = tk.Tk()
    app = SmartBANApp(root, "config")
    app.build_gui()
    root.mainloop()