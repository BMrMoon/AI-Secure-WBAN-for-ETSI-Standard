import pandas as pd
from icecream import ic
import torch
from torch.utils.data import DataLoader
import numpy as np

from tqdm import tqdm

from Models.models import FC_Model, CNN1d, RNN, GRU, LSTM, CustomDataset
from HyperparameterTuning.geneticAlgo import GeneticAlgo
import traceback


dataset_file_path = '/Users/berkaybey/Code/Python/WBAN/v2.0.0/results/data/dataset.csv'

chunk_list = []
for chunk in tqdm(pd.read_csv(dataset_file_path, low_memory=False, chunksize=1000)):
    chunk_list.append(chunk)


dataset = pd.concat(chunk_list)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]
dataset = dataset.reset_index(drop=True)
dataset_size = dataset.shape[0]


parameters_range = {
    "FC_Model": {
        "number_of_layer": list(range(1, 11)),#[range(1,11)],
        "number_of_neuron": [8, 16, 32, 64, 128, 256],#[2**range(1,13)],
        "epochs": list(range(1, 101)),#[10**range(1,5)],
        "kfold": list(range(0, 6)),
        "kmeans": list(range(1, 10)),
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
        "kmeans": list(range(1, 10)),
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
        "kmeans": list(range(1, 10)),
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
        "kmeans": list(range(1, 10)),
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
        "kmeans": list(range(1, 10)),
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


hyperparameters = {
                
                "number_of_layer": 9,
                "number_of_neuron": 256,
                "epochs": 100,
                "kfold": 5,
                "kmeans": 1,
                "lr": 0.001,
                "train_size": 0.8,
                "val_size": 0.5,
                "batch_size": 9000,#dataset_size,#dataset.shape[0], # lower for cnn 1d
                "seq_length": 6, #1 for 1d CNN
                "out_channels": 6,
                "conv1d_kernel_size": 5,
                "maxpool1d_kernel_size": 6


                }

configurations = {

                "device": device,
                "dataset": dataset,

}
"""def modify_data(dataset):
    dataset = dataset.copy()
    label_map = {
        'anomaly_free': 0.0,
        'packet_injection': 1.0,
        'replay': 2.0,
        'jamming': 3.0
    }
    dataset.iloc[:, -1] = dataset.iloc[:, -1].map(label_map)

    X = dataset.iloc[:, :96].to_numpy(dtype=np.float32)
    y = dataset.iloc[:, -1].to_numpy(dtype=np.int64).reshape(-1, 1)
    return X, y

X, y = modify_data(dataset)
dataset = CustomDataset(X, y)

def main():
    dataloader = DataLoader(dataset, batch_size=10000, shuffle=True, num_workers=0)
    ic(dataloader)
    try:
        for batch_data, batch_labels in tqdm(dataloader):
            pass
    except Exception as e:
        ic(e)

if __name__ == '__main__':
    main()"""


model = RNN(hyperparameters, configurations, parameters_range).to(device)
Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy = model.run(model)
ic(Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy)
