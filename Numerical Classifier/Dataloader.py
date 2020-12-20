# data loader

import torch
import pandas as pd


# define a customise torch dataset
class DataFrameDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.data_tensor = torch.Tensor(df.values)

    # a function to get items by index
    def __getitem__(self, index):
        obj = self.data_tensor[index]
        input = self.data_tensor[index][1:]
        target = self.data_tensor[index][0] - 1

        return input, target

    # a function to count samples
    def __len__(self):
        n, _ = self.data_tensor.shape
        return n


# normalize the dataset (min-max norm)
def normalize(Dataframe):
    # features = ['value_1', 'value_2', 'value_3', 'value_4', 'value_5']
    features = ['value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6', 'value_7', 'value_8', 'value_9', 'value_10']
    for column in features:
        Dataframe[column] = Dataframe.loc[:, [column]].apply(lambda x: (x - x.min()) / x.std())
    return None
