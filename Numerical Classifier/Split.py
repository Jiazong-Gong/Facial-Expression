# generate LPQ and PHOG and combination files

import pandas as pd

# read data from dataset
file = pd.read_excel ('SFEW.xlsx')
file.columns = ['name', 'target', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6', 'value_7', 'value_8', 'value_9', 'value_10']

# generate the training set and testing set
train_file = file.sample(n=400)
test_file = pd.DataFrame(file.merge(train_file, how='outer', indicator=True).loc[lambda x: x['_merge'] == 'left_only']).iloc[:, :-1]

# split the features and targets within each set respectively
train_file = train_file.iloc[:, 1:]
test_file = test_file.iloc[:, 1:]
lpq_training = train_file.iloc[:, :6]
lpq_testing = test_file.iloc[:, :6]
phog_training = train_file.iloc[:, [0, 6, 7, 8, 9, 10]]
phog_testing = test_file.iloc[:, [0, 6, 7, 8, 9, 10]]

# # rename the column names
# train_file.columns = ['target', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6', 'value_7', 'value_8', 'value_9', 'value_10']
# test_file.columns = ['target', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5', 'value_6', 'value_7', 'value_8', 'value_9', 'value_10']
#
# lpq_training.columns = ['target', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5']
# lpq_testing.columns = ['target', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5']
# phog_training.columns = ['target', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5']
# phog_testing.columns = ['target', 'value_1', 'value_2', 'value_3', 'value_4', 'value_5']

# reset all the index
train_file.reset_index(inplace=True)
test_file.reset_index(inplace=True)
lpq_training.reset_index(inplace=True)
lpq_testing.reset_index(inplace=True)
phog_training.reset_index(inplace=True)
phog_testing.reset_index(inplace=True)

# remove the previous index
train_file = train_file.iloc[:, 1:]
test_file = test_file.iloc[:, 1:]
lpq_training = lpq_training.iloc[:, 1:]
lpq_testing = lpq_testing.iloc[:, 1:]
phog_training = phog_training.iloc[:, 1:]
phog_testing = phog_testing.iloc[:, 1:]

# save these files for further processing
train_file.to_csv('training.csv', index=False)
test_file.to_csv('testing.csv', index=False)
lpq_training.to_csv('lpq_training.csv', index=False)
lpq_testing.to_csv('lpq_testing.csv', index=False)
phog_training.to_csv('phog_training.csv', index=False)
phog_testing.to_csv('phog_testing.csv', index=False)
