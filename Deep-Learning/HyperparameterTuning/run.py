import pandas as pd
from icecream import ic
import torch
from tqdm import tqdm

from HyperparameterTuning.geneticAlgo import GeneticAlgo
import traceback


dataset_file_path = '/Users/berkaybey/Code/Python/WBAN/v1.0.0/results/data/dataset.csv'

chunk_list = []
for chunk in tqdm(pd.read_csv(dataset_file_path, low_memory=False, chunksize=1000)):
    chunk_list.append(chunk)


dataset = pd.concat(chunk_list)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset_size = dataset.shape[0]
ic(dataset.shape)


parameters_range = {
    "FC_Model": {
        "number_of_layer": list(range(1, 11)),#[range(1,11)],
        "number_of_neuron": [8, 16, 32, 64, 128, 256],#[2**range(1,13)],
        "epochs": list(range(1, 101)),#[10**range(1,5)],
        "kfold": list(range(0, 6)),
        "kmeans": list(range(3, 4)),
        "lr": [0.00001, 0.0001, 0.001, 0.01],#0.001,
        "train_size": [0.8, 0.8],#0.8,
        "val_size": [0.5, 0.5],#0.2,
        "batch_size": list(range(400000, dataset_size, 100000)),#100,#dataset.shape[0], # lower for cnn 1d
        "seq_length": [1, 1],#5, #1 for 1d CNN
        "out_channels": [2, 2],#16,
        "conv1d_kernel_size": [1, 1],#7,
        "maxpool1d_kernel_size": [2, 2],#2
    },
    "CNN1d": {
        "number_of_layer": list(range(1, 11)),#[range(1,11)],
        "number_of_neuron": [8, 16, 32, 64, 128, 256],#[2**range(1,13)],
        "epochs": list(range(1, 101)),#[10**range(1,5)],
        "kfold": list(range(0, 6)),
        "kmeans": list(range(3, 4)),
        "lr": [0.001, 0.01],#0.001,
        "train_size": [0.8, 0.8],#0.8,
        "val_size":  [0.5, 0.5],#0.2,
        "batch_size": list(range(1000, 10001, 1000)),#100,#dataset.shape[0], # lower for cnn 1d
        "seq_length": list(range(1, 21)),#5, #1 for 1d CNN
        "out_channels": list(range(2, 17, 2)),#16,
        "conv1d_kernel_size": list(range(1, 22, 2)),#7,
        "maxpool1d_kernel_size": list(range(2, 21, 2)),#2
    },
    "RNN": {
        "number_of_layer": list(range(1, 11)),#[range(1,11)],
        "number_of_neuron": [8, 16, 32, 64, 128, 256],#[2**range(1,13)],
        "epochs": list(range(1, 101)),#[10**range(1,5)],
        "kfold": list(range(0, 6)),
        "kmeans": list(range(3, 4)),
        "lr": [0.0001, 0.001, 0.01],#0.001,
        "train_size": [0.8, 0.8],#0.8,
        "val_size":  [0.5, 0.5],#0.2,
        "batch_size": list(range(1000, 10001, 1000)),#100,#dataset.shape[0], # lower for cnn 1d
        "seq_length": list(range(1, 21)),#5, #1 for 1d CNN
        "out_channels": list(range(2, 17, 2)),#16,
        "conv1d_kernel_size": list(range(1, 22, 2)),#7,
        "maxpool1d_kernel_size": list(range(2, 21, 2)),#2
    },
    "GRU": {
        "number_of_layer": list(range(1, 11)),#[range(1,11)],
        "number_of_neuron": [8, 16, 32, 64, 128, 256],#[2**range(1,13)],
        "epochs": list(range(1, 101)),#[10**range(1,5)],
        "kfold": list(range(0, 6)),
        "kmeans": list(range(3, 4)),
        "lr": [0.0001, 0.001, 0.01],#0.001,
        "train_size": [0.8, 0.8],#0.8,
        "val_size":  [0.5, 0.5],#0.2,
        "batch_size": list(range(1000, 10001, 1000)),#100,#dataset.shape[0], # lower for cnn 1d
        "seq_length": list(range(1, 21)),#5, #1 for 1d CNN
        "out_channels": list(range(2, 17, 2)),#16,
        "conv1d_kernel_size": list(range(1, 22, 2)),#7,
        "maxpool1d_kernel_size": list(range(2, 21, 2)),#2
    },
    "LSTM": {
        "number_of_layer": list(range(1, 11)),#[range(1,11)],
        "number_of_neuron": [8, 16, 32, 64, 128, 256],#[2**range(1,13)],
        "epochs": list(range(1, 101)),#[10**range(1,5)],
        "kfold": list(range(0, 6)),
        "kmeans": list(range(3, 4)),
        "lr": [0.0001, 0.001, 0.01],#0.001,
        "train_size": [0.8, 0.8],#0.8,
        "val_size":  [0.5, 0.5],#0.2,
        "batch_size": list(range(1000, 10001, 1000)),#100,#dataset.shape[0], # lower for cnn 1d
        "seq_length": list(range(1, 21)),#5, #1 for 1d CNN
        "out_channels": list(range(2, 17, 2)),#16,
        "conv1d_kernel_size": list(range(1, 22, 2)),#7,
        "maxpool1d_kernel_size": list(range(2, 21, 2)),#2
    }
}


ic(f"PyTorch version: {torch.__version__}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
ic(f"Using device: {device}")




GA = GeneticAlgo(dataset, parameters_range)
best_overall, best_per_model = GA.run_all_models(pop_size=20, ngen=5)
ic(best_overall, best_per_model)