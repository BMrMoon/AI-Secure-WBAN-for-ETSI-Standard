import pandas as pd
from icecream import ic
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


dataset_file_path = '/Users/berkaybey/Code/Python/WBAN/v2.0.0/results/data/dataset2.csv'

chunk_list1 = []
for chunk in tqdm(pd.read_csv(dataset_file_path, low_memory=False, chunksize=1000)):
    chunk_list1.append(chunk)
dataset1 = pd.concat(chunk_list1)
dataset1 = dataset1.loc[:, ~dataset1.columns.str.contains('^Unnamed')]


dataset_file_path = '/Users/berkaybey/Code/Python/WBAN/v2.0.0/results/data/dataset1.csv'

chunk_list2 = []
for chunk in tqdm(pd.read_csv(dataset_file_path, low_memory=False, chunksize=1000)):
    chunk_list2.append(chunk)
dataset2 = pd.concat(chunk_list2)
dataset = dataset2.loc[:, ~dataset2.columns.str.contains('^Unnamed')]

ic(dataset1.shape, dataset2.shape)

final_df = pd.concat([dataset1, dataset2], ignore_index=True, axis=0)
ic(final_df.shape)
dataset_path = '/Users/berkaybey/Code/Python/WBAN/v2.0.0/results/data/dataset.csv'
final_df.to_csv(dataset_path, index=False)
exit()





dataset = pd.concat(chunk_list)
dataset = dataset.loc[:, ~dataset.columns.str.contains('^Unnamed')]

#sns.heatmap(dataset.iloc[:, :-1])
#plt.hist(dataset.iloc[:, -1])
#plt.show()