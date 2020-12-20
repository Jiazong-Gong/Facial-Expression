import torch


# obtain prediction\
def prediction(network, data):
    train_input = data.iloc[:, 1:]
    train_target = data.iloc[:, 0]

    inputs = torch.Tensor(train_input.values).float()
    targets = torch.Tensor(train_target.values - 1).long()

    # outputs = network(inputs)
    outputs = network(inputs)
    _, predicted = torch.max(outputs, 1)
    return predicted, targets


# define a function to plot confusion matrix
def plot_confusion(input_sample, num_classes, des_output, actual_output):
    confusion = torch.zeros(num_classes, num_classes)
    for i in range(input_sample):
        actual_class = actual_output[i]
        predicted_class = des_output[i]

        confusion[actual_class][predicted_class] += 1

    return confusion