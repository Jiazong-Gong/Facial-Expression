# total training and testing process with stratified 5-fold cross validation
from torch import nn as nn
import torchvision
from torch.autograd import Variable
from torchvision import datasets
from torchvision.transforms import transforms
from Loader import *
from Model import *


my_device = ('cuda' if torch.cuda.is_available() else 'cpu')

EPOCH = 50
batch_size = 256
lr = 0.01

# Since the backbone is Resent, the input size should be 224
# And there is standardization

# original dataset
# original_data = datasets.ImageFolder(root='./SFEW/',
#                                      transform=transforms.Compose(
#                                          [transforms.Resize(224),
#                                           transforms.ToTensor(),
#                                           transforms.Normalize([0.485, 0.456, 0.406],
#                                                                [0.229, 0.224, 0.225])]))

# face dataset
face_data = datasets.ImageFolder(root='./Processed/',
                                 transform=transforms.Compose(
                                     [transforms.Resize(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])]))

# Stratified cross validation on each dataset
# original_cv = KCV(original_data, batch_size)
face_cv = KCV(face_data, batch_size)
criterion = torch.nn.CrossEntropyLoss()

# indicate which cross validation
cv_counter = 0
# record the accuracy for each cross validation
cv_acc = []

# training process
for i in face_cv:
    cv_counter += 1
    net = Net(my_device)
    optimizer = torch.optim.Adam(net.fc.parameters(), lr=lr, weight_decay=1e-3)
    for epoch in range(EPOCH):
        train_loss = 0.
        train_acc = 0.
        for step, (batch_x, batch_y) in enumerate(i[0]):

            batch_x, batch_y = Variable(batch_x.to(device=my_device)), Variable(batch_y.to(device=my_device))

            out = net(batch_x)
            loss = criterion(out, batch_y)
            train_loss += loss.item()

            pred = torch.max(out, 1)[1]
            train_correct = (pred == batch_y).sum()
            train_acc += train_correct.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print('Epoch: ', epoch, 'Step', step,
                  'Train_loss: ', train_loss / ((step + 1) * batch_size), 'Train acc: ',
                  train_acc / ((step + 1) * batch_size))

    # evaluate for current cross validation
    net.eval()
    eval_loss = 0
    eval_acc = 0
    for step, (batch_x, batch_y) in enumerate(i[1]):

        batch_x, batch_y = Variable(batch_x.to(device=my_device)), Variable(batch_y.to(device=my_device))

        out = net(batch_x)

        pred = torch.max(out, 1)[1]
        test_correct = (pred == batch_y).sum()
        eval_acc += test_correct.item()

    # the accuracy for current cross validation
    accuracy = float(eval_acc) / 135.0
    print('Cross Validation ' + str(cv_counter) + '    Test acc: ', accuracy)
    torch.save(net.state_dict(), 'model_' + str(cv_counter) + '.pth')
    cv_acc.append(accuracy)

# overall average accuracy
average_acc = sum(cv_acc) / len(cv_acc)
print('The average accuracy of ' + str(batch_size) + ' ' + str(lr) + ' ' + str(EPOCH) + ' is: ' + str(average_acc))
