#!/usr/bin/env python
# coding: utf-8
# classifier
# To make it easy for testing, only 10-feature classifier is kept and the 5-feature evaluation are all commented

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import torch.utils.data
from Dataloader import *
from Model import *
from Evaluation import *
from Logger import *


# device configuration
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# print('CUDA is available!' if torch.cuda.is_available() else 'No CUDA!')

# define a board writer
logger = Logger('./logs')

# define the pca model
pca = PCA(n_components=7)

# read data from dataset
file = pd.read_excel ('SFEW.xlsx')
file.columns = ['name', 'target', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6', 'value_7', 'value_8', 'value_9', 'value_10']
file.iloc[:, 2:] = StandardScaler().fit_transform(file.iloc[:, 2:])

# there is nan data and need to replace it with 0
# print(file.isnull().sum())
file = file.fillna(0)

# apply pca on the data
file.iloc[:, 2:9] = pca.fit_transform(file.iloc[:, 2:])

# specify the parameters
input_size = 7
hidden_size = 30
embedded_size = 5
num_classes = 7
num_epochs = 200
batch_size = 10
learning_rate = 0.007
# dropout_rate = 0.3

# generate the training set and testing set
train_file = file.sample(frac=0.8)
test_file = pd.DataFrame(file.merge(train_file, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']).iloc[:, :-1]

# split the features and targets within each set respectively
training = train_file.iloc[:, 1:9]
testing = test_file.iloc[:, 1:9]

# load the dataset
# training = pd.read_csv('training.csv')
# testing = pd.read_csv('testing.csv')
# lpq_training = pd.read_csv('lpq_training.csv')
# lpq_testing = pd.read_csv('lpq_testing.csv')
# phog_training = pd.read_csv('phog_training.csv')
# phog_testing = pd.read_csv('phog_testing.csv')

# normalization is not required as in the paper
# print(training)
# normalize all
# normalize(training)
# normalize(testing)
# normalize(lpq_training)
# normalize(lpq_testing)
# normalize(phog_training)
# normalize(phog_testing)

train_dataset = DataFrameDataset(training)
test_dataset = DataFrameDataset(testing)
# lpq_train_dataset = DataFrameDataset(lpq_training)
# lpq_test_dataset = DataFrameDataset(lpq_testing)
# phog_train_dataset = DataFrameDataset(phog_training)
# phog_test_dataset = DataFrameDataset(phog_testing)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
# lpq_train_loader = torch.utils.data.DataLoader(lpq_train_dataset, batch_size=batch_size, shuffle=False)
# lpq_test_loader = torch.utils.data.DataLoader(lpq_test_dataset, batch_size=batch_size, shuffle=False)
# phog_train_loader = torch.utils.data.DataLoader(phog_train_dataset, batch_size=batch_size, shuffle=False)
# phog_test_loader = torch.utils.data.DataLoader(phog_test_dataset, batch_size=batch_size, shuffle=False)


# construct the network
net = Net(input_size, hidden_size, num_classes)
# net = BDNN(input_size, embedded_size, hidden_size, num_classes)
# net = AE(input_size, embedded_size)

# Loss and Optimizer
# criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

# criterion are for BDNN and Auto Encoder
criterion = nn.CrossEntropyLoss()
# criterion1 = nn.MSELoss()
# criterion2 = nn.CrossEntropyLoss()

# optimizer = torch.optim.SGD(params=net.parameters(), lr=learning_rate, momentum=0.9)

# contain the loss and accuracy
all_losses = []

# train the model by batch
for epoch in range(num_epochs):
    total = 0
    correct = 0
    total_loss1 = 0
    total_loss2 = 0
    total_loss = 0

    # mini-batch training
    for step, (batch_x, batch_y) in enumerate(train_loader):
        X = batch_x
        Y = batch_y.long()

        # Forward + Backward + Optimize
        optimizer.zero_grad()  # zero the gradient buffer
        outputs = net(X)
        # print(outputs)

        # for BDNN and Auto Encoder
        # loss1 = criterion1(decode, X)
        # loss2 = criterion2(outputs, Y)
        # loss = loss1 + loss2

        loss = criterion(outputs, Y)
        # print(loss)
        all_losses.append(loss.item())
        loss.backward()
        optimizer.step()
        # # if (epoch % 50 == 0):
        _, predicted = torch.max(outputs, 1)
        # calculate and print accuracy
        total = total + predicted.size(0)
        correct = correct + sum(predicted.data.cpu().numpy() == Y.data.cpu().numpy())
        total_loss = total_loss + loss
        # total_loss1 = total_loss1 + loss1
        # total_loss2 = total_loss2 + loss2
        accuracy = 100 * correct/total

    # ================================================================== #
    #                        Tensorboard Logging                         #
    # ================================================================== #

    # 1. Log scalar values (scalar summary)
    # info = {'loss': total_loss1, 'accuracy': accuracy}
    info = {'loss': total_loss, 'accuracy': accuracy}
    for tag, value in info.items():
        logger.scalar_summary(tag, value, epoch + 1)

    # 2. Log values and gradients of the parameters (histogram summary)
    # for tag, value in net.named_parameters():
    #     tag = tag.replace('.', '/')
    #     logger.histo_summary(tag, value.data.cpu().numpy(), epoch + 1)
    #     logger.histo_summary(tag + '/grad', value.grad.data.cpu().numpy(), epoch + 1)
    # writer.add_scalar('Training loss',
    #                   total_loss,
    #                   global_step=epoch + 1)
    #
    # writer.add_scalar('Training accuracy',
    #                   accuracy,
    #                  global_step=epoch + 1)

    if (epoch % 50 == 0):
        print('Epoch [%d/%d], Loss: %.4f, Accuracy: %.2f %%'
              % (epoch + 1, num_epochs,
                 total_loss, accuracy))


# accuracy on testing data
test_prediction, targets = prediction(net, testing)
total = test_prediction.size(0)
correct = test_prediction.data.numpy() == targets.data.numpy()

print('Testing Accuracy: %.2f %%' % (100 * sum(correct)/total))


# confusion matrix on training data
train_input = training.iloc[:, 1:]
train_prediction, targets = prediction(net, training)
print('Confusion matrix for training:')
print(plot_confusion(train_input.shape[0], num_classes, train_prediction.long().data, targets.data))

torch.save(net.state_dict(), 'classifier.pth')

