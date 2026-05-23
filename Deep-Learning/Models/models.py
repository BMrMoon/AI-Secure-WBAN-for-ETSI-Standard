from icecream import ic
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

import numpy as np
import math
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import gc
from statistics import mean
import matplotlib.pyplot as plt

enable_plot = True
ic.disable
class CustomDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features  # Convert to NumPy arrays
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        x = torch.tensor(self.features[idx,:], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

def compute_metrics(y_pred, labels):
    unique_classes = torch.unique(labels)
    num_classes = unique_classes.numel()

    total = [(labels == i).sum().item() for i in range(num_classes)]
    total_sum = sum(total)
    w = [t / total_sum for t in total]

    y_preds = torch.argmax(y_pred, dim=1)

    TP = [((y_preds == i) & (labels == i)).sum().item() for i in range(num_classes)]
    FP = [((y_preds == i) & (labels != i)).sum().item() for i in range(num_classes)]
    FN = [((y_preds != i) & (labels == i)).sum().item() for i in range(num_classes)]

    precision = [TP[i] / (TP[i] + FP[i]) if (TP[i] + FP[i]) > 0 else 0 for i in range(num_classes)]
    recall = [TP[i] / (TP[i] + FN[i]) if (TP[i] + FN[i]) > 0 else 0 for i in range(num_classes)]
    f1 = [2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0 for i in range(num_classes)]

    weighted_precision = sum(w[i] * precision[i] for i in range(num_classes))
    weighted_recall = sum(w[i] * recall[i] for i in range(num_classes))
    weighted_f1 = sum(w[i] * f1[i] for i in range(num_classes))
    
    micro_f1 = (2 * weighted_precision * weighted_recall) / (weighted_precision + weighted_recall) if (weighted_precision + weighted_recall) > 0 else 0

    return weighted_precision, weighted_recall, weighted_f1, micro_f1
    
def get_data(dataset, train_size, val_size, batch_size, device, is_sequenced, seq_length, kfold, cnn=None):
    def modify_data(dataset):
        dataset = dataset.copy()
        label_map = {
            'anomaly_free': 0.0,
            'packet_injection': 1.0,
            'replay': 1.0,
            'jamming': 1.0
        }
        dataset.iloc[:, -1] = dataset.iloc[:, -1].map(label_map)

        X = dataset.iloc[:, :96].to_numpy(dtype=np.float32)
        y = dataset.iloc[:, -1].to_numpy(dtype=np.int64).reshape(-1, 1)
        return X, y

    def reshape_to_sequences(X, y, seq_length):
        total_samples = X.shape[0]
        usable_samples = (total_samples // seq_length) * seq_length
        X = X[:usable_samples]
        y = y[:usable_samples]

        X = X.reshape(-1, seq_length, X.shape[1])  # [num_seq, seq_len, feat_dim]
        y = y.reshape(-1, seq_length)              # [num_seq, seq_len]
        return X, y

    # ---- Preprocess ----
    X, y = modify_data(dataset)
    input_features = X.shape[1]
    output_features = len(np.unique(y[:]))
    y = y[:,-1].reshape(-1,)
    if is_sequenced:
        X, y = reshape_to_sequences(X, y, seq_length)
        y = y[:,-1].reshape(-1,)
        if cnn:
            X = X.reshape(X.shape[0], 1, -1)
    

    def splitAndDataload(X, y):
        # ---- Test split ----
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, shuffle=True)
        test_dataset = CustomDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=False)

        if kfold == 0:
            # Normal val split
            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, train_size=(1 - val_size), shuffle=True)

            train_dataset = CustomDataset(X_train, y_train)
            val_dataset = CustomDataset(X_val, y_val)

            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

            return test_loader, [(train_loader, val_loader)]

        else:
            # ---- K-Fold setup ----
            full_dataset = CustomDataset(X_train, y_train)
            skf = StratifiedKFold(n_splits=kfold, shuffle=True)
            fold_loaders = []
            for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
                print(f"[Fold {fold}]")
                print("Train class distribution:", np.bincount(y_train[train_idx]))
                print("Validation class distribution:", np.bincount(y_train[val_idx]))

                train_subset = Subset(full_dataset, train_idx)
                val_subset = Subset(full_dataset, val_idx)

                train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
                val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0, drop_last=False)

                fold_loaders.append((train_loader, val_loader))

            return test_loader, fold_loaders  # ⬅️ Return all folds
        
    test_loader, fold_loaders = splitAndDataload(X, y)
    return test_loader, fold_loaders, input_features, output_features

def cross_validation(fold_loaders, device, epochs, model, optimizer, criterion):
    Average_Loss = 0.0
    Average_Accuracy = 0.0
    Average_Validation_Loss = 0.0
    Average_Validation_Accuracy = 0.0

    for fold_idx, (train_loader, val_loader) in enumerate(fold_loaders):
        print(f"\n🌀 Fold {fold_idx}")
        Accuracies = []
        Losses = []
        for epoch in tqdm(range(epochs)):
            Loss, Accuracy = train_one_epoch(train_loader, device, model, optimizer, criterion)
            Losses.append(Loss)
            Accuracies.append(Accuracy)
            if early_stop(Losses):
                break

        Average_Loss += Loss
        Average_Accuracy += Accuracy

        # Validation
        Validation_Loss, Validation_Accuracy = validation(val_loader, device, model, criterion)
        Average_Validation_Loss += Validation_Loss
        Average_Validation_Accuracy += Validation_Accuracy

    kfold = len(fold_loaders)

    Average_Loss = Average_Loss / kfold
    Average_Accuracy = Average_Accuracy / kfold
    Average_Validation_Loss = Average_Validation_Loss / kfold
    Average_Validation_Accuracy = Average_Validation_Accuracy / kfold

    return Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy, Accuracies, Losses
    
def early_stop(losses, patience=5):
    """
    Stop if validation loss has increased for 'patience' consecutive epochs.
    """
    if len(losses) < patience + 1:
        return False

    for i in range(1, patience + 1):
        if losses[-i] <= losses[-i - 1]:
            return False
    return True

def validation(val_loader, device, model, criterion):
    Loss = 0.0
    Accuracy = 0.0
    count = 0
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            inputs, labels = batch_data.to(device), batch_labels.squeeze().to(device)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            Loss += loss.item()
            Accuracy += get_accuracy(y_pred, labels)
            count += 1
    if count == 0:
        print("⚠️ Warning: validation loader returned no batches.")
        return 0.0, 0.0
    Loss = Loss / count
    Accuracy = Accuracy / count
    return Loss, Accuracy
        
def train_one_epoch(train_loader, device, model, optimizer, criterion):
    Loss = 0.0
    Accuracy = 0.0
    total_samples = 0
    total_correct = 0
    total_loss = 0.0

    for batch_data, batch_labels in train_loader:
        inputs, labels = batch_data.to(device), batch_labels.squeeze().to(device)
        optimizer.zero_grad()
        y_pred = model.forward(inputs)
        loss = criterion(y_pred, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        total_correct += (torch.argmax(y_pred, dim=1) == labels).sum().item()
        total_samples += inputs.size(0)

    Loss = total_loss / total_samples
    Accuracy = total_correct / total_samples

            

    return Loss, Accuracy

def test(test_loader, device, model, criterion):
    Loss = 0.0
    Accuracy = 0.0
    average_precision = 0.0
    average_recall = 0.0
    average_f1 = 0.0
    average_micro_f1 = 0.0
    count = 0
    with torch.no_grad():
        for batch_data, batch_labels in test_loader:
            inputs, labels = batch_data.to(device), batch_labels.squeeze().to(device)
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            Loss += loss.item()
            Accuracy += get_accuracy(y_pred, labels)

            weighted_precision, weighted_recall, weighted_f1, micro_f1 = compute_metrics(y_pred, labels)
            average_precision += weighted_precision
            average_recall += weighted_recall
            average_f1 += weighted_f1
            average_micro_f1 += micro_f1
            count += 1

    if count == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    Loss = Loss / count
    Accuracy = Accuracy / count
    average_precision = average_precision / count
    average_recall = average_recall / count
    average_f1 = average_f1 / count
    average_micro_f1 = average_micro_f1 / count

    return Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1

def get_accuracy(y_pred, labels):
    preds = torch.argmax(y_pred, dim=1)
    accuracy = (preds == labels).sum().item() / preds.shape[0]
    return accuracy

def plot(Accuracies, Losses):
    if enable_plot == True:
        X = range(1,len(Accuracies)+1)
        y1 = Accuracies
        y2 = Losses
        plt.plot(X, y1, color='g', label='Accuracy')
        plt.plot(X, y2, color='r', label='Loss')
        plt.ylim(0, 1)
        plt.xlim(0, 110)
        plt.xlabel("Epoch")
        plt.ylabel("Percentage")
        plt.title("Train Loss and Accuracy")

        plt.legend()
        plt.show()


class FC_Model(nn.Module):
    
    def __init__(self, hyperparameters, configurations, models_ranges):
        super(FC_Model, self).__init__()
        self.hyperparameters = hyperparameters
        self.configurations = configurations

        self.dataset = self.configurations["dataset"] 
        self.device = configurations["device"]
        self.epochs = max(models_ranges["FC_Model"]["epochs"][0], min(models_ranges["FC_Model"]["epochs"][-1], int(hyperparameters["epochs"])))
        self.lr = max(models_ranges["FC_Model"]["lr"][0], min(models_ranges["FC_Model"]["lr"][-1], hyperparameters["lr"]))
        train_size = max(models_ranges["FC_Model"]["train_size"][0], min(models_ranges["FC_Model"]["train_size"][-1], hyperparameters["train_size"]))
        kfold = max(models_ranges["FC_Model"]["kfold"][0], min(models_ranges["FC_Model"]["kfold"][-1], hyperparameters["kfold"]))
        self.kmeans = max(models_ranges["FC_Model"]["kmeans"][0], min(models_ranges["FC_Model"]["kmeans"][-1], hyperparameters["kmeans"]))
        val_size = max(models_ranges["FC_Model"]["val_size"][0], min(models_ranges["FC_Model"]["val_size"][-1], hyperparameters["val_size"]))
        number_of_layer = max(models_ranges["FC_Model"]["number_of_layer"][0], min(models_ranges["FC_Model"]["number_of_layer"][-1], int(hyperparameters["number_of_layer"])))
        number_of_neuron = max(models_ranges["FC_Model"]["number_of_neuron"][0], min(models_ranges["FC_Model"]["number_of_neuron"][-1], int(hyperparameters["number_of_neuron"])))
        self.batch_size = max(models_ranges["FC_Model"]["batch_size"][0], min(models_ranges["FC_Model"]["batch_size"][-1], int(hyperparameters["batch_size"])))
        is_sequenced = False
        seq_length = int(hyperparameters["seq_length"])

        self.test_loader, self.fold_loaders, in_features, out_features = get_data(self.dataset, train_size, val_size, self.batch_size, self.device, is_sequenced, seq_length, kfold)
        #ic(hyperparameters)

        
        #ic(self.train_data[0], self.train_data[1], self.val_data[0], self.val_data[1], self.test_data[0], self.test_data[1])
        out_features = 4
        #ic(self.train_data[0].shape, self.train_data[1].shape, self.val_data[0].shape, self.val_data[1].shape, self.test_data[0].shape, self.test_data[1].shape)
        #ic(in_features, out_features)
        
        self.layers = nn.ModuleList()

        # First layer
        self.layers.append(nn.Linear(in_features, number_of_neuron))

        # Hidden layers
        for _ in range(1, number_of_layer):
            self.layers.append(nn.Linear(number_of_neuron, number_of_neuron))

        # Output layer
        self.out = nn.Linear(number_of_neuron, out_features)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.out(x)

    def test(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1 = test(self.test_loader, self.device, model, criterion)
        return Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1

    def train(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        l = []
        acc = []
        l_v = []
        acc_v = []
        for k in range(self.kmeans):
            Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy, Accuracies, Losses = cross_validation(self.fold_loaders, self.device, self.epochs, model, optimizer, criterion)
            l.append(Average_Loss)
            acc.append(Average_Accuracy)
            l_v.append(Average_Validation_Loss)
            acc_v.append(Average_Validation_Accuracy)
        l = mean(l)
        acc = mean(acc)
        l_v = mean(l_v)
        acc_v = mean(acc_v)
        plot(Accuracies, Losses)
        return l, acc, l_v, acc_v
        #ic(self.train_data[0], self.train_data[1])
                
    def run(self, model):
        Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy = self.train(model)
        ic(f"Train Loss:{Average_Loss:.4f} Train Accuracy:{Average_Accuracy:.4f} Validation Loss:{Average_Validation_Loss:.4f} and Validation Accuracy:{Average_Validation_Accuracy:.4f}")
        average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1 = self.test(model)
        ic(average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1)
        return Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy
    
class CNN1d(nn.Module):
    def __init__(self, hyperparameters, configurations, models_ranges):
        super(CNN1d, self).__init__()
        def cnn_data_shaper(seq_length):
            for fold_idx, (train_loader, val_loader) in enumerate(self.fold_loaders):
                print(f"\n🌀 Fold {fold_idx}")
                for batch_data, batch_labels in train_loader:
                    ic(batch_data.shape, batch_labels.shape)
                print("-*-*-*-*-*-")
                for batch_data, batch_labels in val_loader:
                    ic(batch_data.shape, batch_labels.shape)
            print("-*-*-*-*-*-*-*-*-*-*-")
            for batch_data, batch_labels in self.test_loader:
                ic(batch_data.shape, batch_labels.shape)
            exit()


            self.train_data[0] = torch.reshape(self.train_data[0], (self.train_data[0].shape[0], self.train_data[0].shape[1], 1, -1 ))
            self.val_data[0] = torch.reshape(self.val_data[0], (self.val_data[0].shape[0], self.val_data[0].shape[1], 1, -1 ))
            self.test_data[0] = torch.reshape(self.test_data[0], (self.test_data[0].shape[0], self.test_data[0].shape[1], 1, -1 ))
            self.train_data[1] = torch.reshape(self.train_data[1][:,:,-1], (self.train_data[1].shape[0], -1, 1))
            self.val_data[1] = torch.reshape(self.val_data[1][:,:,-1], (self.val_data[1].shape[0], -1, 1))
            self.test_data[1] = torch.reshape(self.test_data[1][:,:,-1], (self.test_data[1].shape[0], -1, 1))
            ic(self.train_data[0].shape, self.train_data[1].shape, self.val_data[0].shape, self.val_data[1].shape, self.test_data[0].shape, self.test_data[1].shape)
        self.hyperparameters = hyperparameters
        self.configurations = configurations

        self.dataset = self.configurations["dataset"]
        self.device = configurations["device"]
        self.epochs = max(models_ranges["CNN1d"]["epochs"][0], min(models_ranges["CNN1d"]["epochs"][-1], int(hyperparameters["epochs"])))
        self.lr = max(models_ranges["CNN1d"]["lr"][0], min(models_ranges["CNN1d"]["lr"][-1], hyperparameters["lr"]))
        self.batch_size = max(models_ranges["CNN1d"]["batch_size"][0], min(models_ranges["CNN1d"]["batch_size"][-1], int(hyperparameters["batch_size"])))
        train_size = max(models_ranges["CNN1d"]["train_size"][0], min(models_ranges["CNN1d"]["train_size"][-1], hyperparameters["train_size"]))
        kfold = max(models_ranges["CNN1d"]["kfold"][0], min(models_ranges["CNN1d"]["kfold"][-1], hyperparameters["kfold"]))
        self.kmeans = max(models_ranges["CNN1d"]["kmeans"][0], min(models_ranges["CNN1d"]["kmeans"][-1], hyperparameters["kmeans"]))
        val_size = max(models_ranges["CNN1d"]["val_size"][0], min(models_ranges["CNN1d"]["val_size"][-1], hyperparameters["val_size"]))
        number_of_layer = max(models_ranges["CNN1d"]["number_of_layer"][0], min(models_ranges["CNN1d"]["number_of_layer"][-1], int(hyperparameters["number_of_layer"])))
        number_of_neuron = max(models_ranges["CNN1d"]["number_of_neuron"][0], min(models_ranges["CNN1d"]["number_of_neuron"][-1], int(hyperparameters["number_of_neuron"])))
        out_channels = max(models_ranges["CNN1d"]["out_channels"][0], min(models_ranges["CNN1d"]["out_channels"][-1], int(hyperparameters["out_channels"])))
        conv1d_kernel_size = max(models_ranges["CNN1d"]["conv1d_kernel_size"][0], min(models_ranges["CNN1d"]["conv1d_kernel_size"][-1], int(hyperparameters["conv1d_kernel_size"])))
        maxpool1d_kernel_size = max(models_ranges["CNN1d"]["maxpool1d_kernel_size"][0], min(models_ranges["CNN1d"]["maxpool1d_kernel_size"][-1], int(hyperparameters["maxpool1d_kernel_size"])))
        is_sequenced = True
        seq_length = max(models_ranges["CNN1d"]["seq_length"][0], min(models_ranges["CNN1d"]["seq_length"][-1], int(hyperparameters["seq_length"])))

        self.test_loader, self.fold_loaders, in_features, out_features = get_data(self.dataset, train_size, val_size, self.batch_size, self.device, is_sequenced, seq_length, kfold, cnn=True)
        #ic(in_features, out_features)

        self.layers = nn.ModuleList()

        padding = (conv1d_kernel_size - 1) // 2

        # First layer
        self.layers.append(nn.Conv1d(1, out_channels, conv1d_kernel_size, padding=padding))

        # Hidden layers
        current_channels = out_channels
        for _ in range(1, number_of_layer):
            next_channels = current_channels * 2
            self.layers.append(nn.Conv1d(current_channels, next_channels, conv1d_kernel_size, padding=padding))
            self.layers.append(nn.MaxPool1d(maxpool1d_kernel_size))
            current_channels = next_channels

        # Dummy input to calculate fc1 input size
        x = torch.randn(1, 1, (int(seq_length)*int(in_features)))
        for layer in self.layers:
            x = layer(x) if not isinstance(layer, nn.MaxPool1d) else layer(F.relu(x))
        fc1_input = x.view(1, -1).size(1)

        self.fc1 = nn.Linear(fc1_input, number_of_neuron)
        self.fc2 = nn.Linear(number_of_neuron, out_features)

    def forward(self, x):
        x = F.relu(self.layers[0](x))
        for index in range(1, len(self.layers), 2):
            x = F.relu(self.layers[index](x))
            x = self.layers[index + 1](x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def test(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1 = test(self.test_loader, self.device, model, criterion)
        return Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1

    def train(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        l = []
        acc = []
        l_v = []
        acc_v = []
        for k in range(self.kmeans):
            Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy, Accuracies, Losses = cross_validation(self.fold_loaders, self.device, self.epochs, model, optimizer, criterion)
            l.append(Average_Loss)
            acc.append(Average_Accuracy)
            l_v.append(Average_Validation_Loss)
            acc_v.append(Average_Validation_Accuracy)
        l = mean(l)
        acc = mean(acc)
        l_v = mean(l_v)
        acc_v = mean(acc_v)
        plot(Accuracies, Losses)
        return l, acc, l_v, acc_v
        #ic(self.train_data[0], self.train_data[1])
                
    def run(self, model):
        Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy = self.train(model)
        ic(f"Train Loss:{Average_Loss:.4f} Train Accuracy:{Average_Accuracy:.4f} Validation Loss:{Average_Validation_Loss:.4f} and Validation Accuracy:{Average_Validation_Accuracy:.4f}")
        average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1 = self.test(model)
        ic(average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1)
        return Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy

class RNN(nn.Module):
    def __init__(self, hyperparameters, configurations, models_ranges):
        super(RNN, self).__init__()
        self.hyperparameters = hyperparameters
        self.configurations = configurations

        self.dataset = self.configurations["dataset"] 
        self.device = configurations["device"]
        self.epochs = max(models_ranges["RNN"]["epochs"][0], min(models_ranges["RNN"]["epochs"][-1], int(hyperparameters["epochs"])))
        self.lr = max(models_ranges["RNN"]["lr"][0], min(models_ranges["RNN"]["lr"][-1], hyperparameters["lr"]))
        self.number_of_layer = max(models_ranges["RNN"]["number_of_layer"][0], min(models_ranges["RNN"]["number_of_layer"][-1], int(hyperparameters["number_of_layer"])))
        self.number_of_neuron = max(models_ranges["RNN"]["number_of_neuron"][0], min(models_ranges["RNN"]["number_of_neuron"][-1], int(hyperparameters["number_of_neuron"])))
        train_size = max(models_ranges["RNN"]["train_size"][0], min(models_ranges["RNN"]["train_size"][-1], hyperparameters["train_size"]))
        kfold = max(models_ranges["RNN"]["kfold"][0], min(models_ranges["RNN"]["kfold"][-1], hyperparameters["kfold"]))
        self.kmeans = max(models_ranges["RNN"]["kmeans"][0], min(models_ranges["RNN"]["kmeans"][-1], hyperparameters["kmeans"]))
        val_size = max(models_ranges["RNN"]["val_size"][0], min(models_ranges["RNN"]["val_size"][-1], hyperparameters["val_size"]))
        self.batch_size = max(models_ranges["RNN"]["batch_size"][0], min(models_ranges["RNN"]["batch_size"][-1], int(hyperparameters["batch_size"])))
        is_sequenced = True
        seq_length = max(models_ranges["RNN"]["seq_length"][0], min(models_ranges["RNN"]["seq_length"][-1], int(hyperparameters["seq_length"])))

        self.test_loader, self.fold_loaders, in_features, out_features = get_data(self.dataset, train_size, val_size, self.batch_size, self.device, is_sequenced, seq_length, kfold)
        #ic(self.train_data[0].shape, self.train_data[1].shape, self.val_data[0].shape, self.val_data[1].shape, self.test_data[0].shape, self.test_data[1].shape)
        #ic(self.train_data[0].shape, self.train_data[1].shape, self.val_data[0].shape, self.val_data[1].shape, self.test_data[0].shape, self.test_data[1].shape)
        #ic(in_features, out_features)

        # First layer
        self.rnn = nn.RNN(in_features, self.number_of_neuron, self.number_of_layer, batch_first=True)

        # Output layer
        self.fc = nn.Linear(self.number_of_neuron, out_features)

    def forward(self, x):
        h0 = torch.zeros(self.number_of_layer, x.size(0), self.number_of_neuron).to(self.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def test(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1 = test(self.test_loader, self.device, model, criterion)
        return Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1

    def train(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        l = []
        acc = []
        l_v = []
        acc_v = []
        for k in range(self.kmeans):
            Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy, Accuracies, Losses = cross_validation(self.fold_loaders, self.device, self.epochs, model, optimizer, criterion)
            l.append(Average_Loss)
            acc.append(Average_Accuracy)
            l_v.append(Average_Validation_Loss)
            acc_v.append(Average_Validation_Accuracy)
        l = mean(l)
        acc = mean(acc)
        l_v = mean(l_v)
        acc_v = mean(acc_v)
        plot(Accuracies, Losses)
        return l, acc, l_v, acc_v
        #ic(self.train_data[0], self.train_data[1])
                
    def run(self, model):
        Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy = self.train(model)
        ic(f"Train Loss:{Average_Loss:.4f} Train Accuracy:{Average_Accuracy:.4f} Validation Loss:{Average_Validation_Loss:.4f} and Validation Accuracy:{Average_Validation_Accuracy:.4f}")
        average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1 = self.test(model)
        ic(average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1)
        return Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy

class GRU(nn.Module):
    def __init__(self, hyperparameters, configurations, models_ranges):
        super(GRU, self).__init__()
        self.hyperparameters = hyperparameters
        self.configurations = configurations

        self.dataset = self.configurations["dataset"] 
        self.device = configurations["device"]
        self.epochs = max(models_ranges["GRU"]["epochs"][0], min(models_ranges["GRU"]["epochs"][-1], int(hyperparameters["epochs"])))
        self.lr = max(models_ranges["GRU"]["lr"][0], min(models_ranges["GRU"]["lr"][-1], hyperparameters["lr"]))
        self.number_of_layer = max(models_ranges["GRU"]["number_of_layer"][0], min(models_ranges["GRU"]["number_of_layer"][-1], int(hyperparameters["number_of_layer"])))
        self.number_of_neuron = max(models_ranges["GRU"]["number_of_neuron"][0], min(models_ranges["GRU"]["number_of_neuron"][-1], int(hyperparameters["number_of_neuron"])))
        train_size = max(models_ranges["GRU"]["train_size"][0], min(models_ranges["GRU"]["train_size"][-1], hyperparameters["train_size"]))
        kfold = max(models_ranges["GRU"]["kfold"][0], min(models_ranges["GRU"]["kfold"][-1], hyperparameters["kfold"]))
        self.kmeans = max(models_ranges["GRU"]["kmeans"][0], min(models_ranges["GRU"]["kmeans"][-1], hyperparameters["kmeans"]))
        val_size = max(models_ranges["GRU"]["val_size"][0], min(models_ranges["GRU"]["val_size"][-1], hyperparameters["val_size"]))
        self.batch_size = max(models_ranges["GRU"]["batch_size"][0], min(models_ranges["GRU"]["batch_size"][-1], int(hyperparameters["batch_size"])))
        is_sequenced = True
        seq_length = max(models_ranges["GRU"]["seq_length"][0], min(models_ranges["GRU"]["seq_length"][-1], int(hyperparameters["seq_length"])))

        self.test_loader, self.fold_loaders, in_features, out_features = get_data(self.dataset, train_size, val_size, self.batch_size, self.device, is_sequenced, seq_length, kfold)
        #ic(self.train_data[0].shape, self.train_data[1].shape, self.val_data[0].shape, self.val_data[1].shape, self.test_data[0].shape, self.test_data[1].shape)
        #ic(self.train_data[0].shape, self.train_data[1].shape, self.val_data[0].shape, self.val_data[1].shape, self.test_data[0].shape, self.test_data[1].shape)
        #ic(in_features, out_features)

        

        # First layer
        self.gru = nn.GRU(in_features, self.number_of_neuron, self.number_of_layer, batch_first=True)

        # Output layer
        self.fc = nn.Linear(self.number_of_neuron, out_features)

    def forward(self, x):
        h0 = torch.zeros(self.number_of_layer, x.size(0), self.number_of_neuron).to(self.device)
        out, _ = self.gru(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def test(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1 = test(self.test_loader, self.device, model, criterion)
        return Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1

    def train(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        l = []
        acc = []
        l_v = []
        acc_v = []
        for k in range(self.kmeans):
            Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy, Accuracies, Losses = cross_validation(self.fold_loaders, self.device, self.epochs, model, optimizer, criterion)
            l.append(Average_Loss)
            acc.append(Average_Accuracy)
            l_v.append(Average_Validation_Loss)
            acc_v.append(Average_Validation_Accuracy)
        l = mean(l)
        acc = mean(acc)
        l_v = mean(l_v)
        acc_v = mean(acc_v)
        plot(Accuracies, Losses)
        return l, acc, l_v, acc_v
        #ic(self.train_data[0], self.train_data[1])
                
    def run(self, model):
        Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy = self.train(model)
        ic(f"Train Loss:{Average_Loss:.4f} Train Accuracy:{Average_Accuracy:.4f} Validation Loss:{Average_Validation_Loss:.4f} and Validation Accuracy:{Average_Validation_Accuracy:.4f}")
        average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1 = self.test(model)
        ic(average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1)
        return Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy

class LSTM(nn.Module):
    def __init__(self, hyperparameters, configurations, models_ranges):
        super(LSTM, self).__init__()
        self.hyperparameters = hyperparameters
        self.configurations = configurations

        self.dataset = self.configurations["dataset"] 
        self.device = configurations["device"]
        self.epochs = max(models_ranges["LSTM"]["epochs"][0], min(models_ranges["LSTM"]["epochs"][-1], int(hyperparameters["epochs"])))
        self.lr = max(models_ranges["LSTM"]["lr"][0], min(models_ranges["LSTM"]["lr"][-1], hyperparameters["lr"]))
        self.number_of_layer = max(models_ranges["LSTM"]["number_of_layer"][0], min(models_ranges["LSTM"]["number_of_layer"][-1], int(hyperparameters["number_of_layer"])))
        self.number_of_neuron = max(models_ranges["LSTM"]["number_of_neuron"][0], min(models_ranges["LSTM"]["number_of_neuron"][-1], int(hyperparameters["number_of_neuron"])))
        train_size = max(models_ranges["LSTM"]["train_size"][0], min(models_ranges["LSTM"]["train_size"][-1], hyperparameters["train_size"]))
        kfold = max(models_ranges["LSTM"]["kfold"][0], min(models_ranges["LSTM"]["kfold"][-1], hyperparameters["kfold"]))
        self.kmeans = max(models_ranges["LSTM"]["kmeans"][0], min(models_ranges["LSTM"]["kmeans"][-1], hyperparameters["kmeans"]))
        val_size = max(models_ranges["LSTM"]["val_size"][0], min(models_ranges["LSTM"]["val_size"][-1], hyperparameters["val_size"]))
        self.batch_size = max(models_ranges["LSTM"]["batch_size"][0], min(models_ranges["LSTM"]["batch_size"][-1], int(hyperparameters["batch_size"])))
        is_sequenced = True
        seq_length = max(models_ranges["LSTM"]["seq_length"][0], min(models_ranges["LSTM"]["seq_length"][-1], int(hyperparameters["seq_length"])))

        self.test_loader, self.fold_loaders, in_features, out_features = get_data(self.dataset, train_size, val_size, self.batch_size, self.device, is_sequenced, seq_length, kfold)
        #ic(self.train_data[0].shape, self.train_data[1].shape, self.val_data[0].shape, self.val_data[1].shape, self.test_data[0].shape, self.test_data[1].shape)
        #ic(self.train_data[0].shape, self.train_data[1].shape, self.val_data[0].shape, self.val_data[1].shape, self.test_data[0].shape, self.test_data[1].shape)
        #ic(in_features, out_features)

        # First layer
        self.lstm = nn.LSTM(in_features, self.number_of_neuron, self.number_of_layer, batch_first=True)

        # Output layer
        self.fc = nn.Linear(self.number_of_neuron, out_features)

    def forward(self, x):
        h0 = torch.zeros(self.number_of_layer, x.size(0), self.number_of_neuron).to(self.device)
        c0 = torch.zeros(self.number_of_layer, x.size(0), self.number_of_neuron).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out

    def test(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1 = test(self.test_loader, self.device, model, criterion)
        return Loss, Accuracy, average_precision, average_recall, average_f1, average_micro_f1

    def train(self, model):
        criterion = nn.CrossEntropyLoss().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        l = []
        acc = []
        l_v = []
        acc_v = []
        for k in range(self.kmeans):
            Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy, Accuracies, Losses = cross_validation(self.fold_loaders, self.device, self.epochs, model, optimizer, criterion)
            l.append(Average_Loss)
            acc.append(Average_Accuracy)
            l_v.append(Average_Validation_Loss)
            acc_v.append(Average_Validation_Accuracy)
        l = mean(l)
        acc = mean(acc)
        l_v = mean(l_v)
        acc_v = mean(acc_v)
        plot(Accuracies, Losses)
        return l, acc, l_v, acc_v
        #ic(self.train_data[0], self.train_data[1])
                
    def run(self, model):
        Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy = self.train(model)
        ic(f"Train Loss:{Average_Loss:.4f} Train Accuracy:{Average_Accuracy:.4f} Validation Loss:{Average_Validation_Loss:.4f} and Validation Accuracy:{Average_Validation_Accuracy:.4f}")
        average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1 = self.test(model)
        ic(average_loss, average_accuracy, average_precision, average_recall, average_f1, average_micro_f1)
        return Average_Loss, Average_Accuracy, Average_Validation_Loss, Average_Validation_Accuracy



ic.enable