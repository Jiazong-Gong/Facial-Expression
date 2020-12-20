# Stratified 5-fold Cross Validation
from torch.utils import data
import torch
from sklearn.model_selection import *
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np


# Stratified K-fold Cross Validation
def KCV(dataset, batch_size):
    # folders
    folders = []

    image_size = len(dataset)
    indices = list(range(image_size))
    targets = dataset.targets

    # 5 folds
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=13)
    # skf = KFold(n_splits=5, shuffle=True, random_state=0)
    # perform stratified 5-fold cross validation on the dataset
    for train_idx, test_idx in skf.split(indices, targets):

        np.random.shuffle(train_idx)
        np.random.shuffle(test_idx)

        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        trainloader = torch.utils.data.DataLoader(dataset,
                                                  sampler=train_sampler, batch_size=batch_size)
        testloader = torch.utils.data.DataLoader(dataset,
                                                 sampler=test_sampler, batch_size=batch_size)

        # append each pair to the folders
        folders.append((trainloader, testloader))
    return folders
