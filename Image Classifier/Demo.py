# One time training and testing and save the trained model
# without cross validation

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

# processed face dataset
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

# get the first split
i = face_cv[0]
net = Net(my_device)
optimizer = torch.optim.Adam(net.fc.parameters(), lr=lr, weight_decay=1e-3)

# training process
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

# testing process
net.eval()
eval_loss = 0
eval_acc = 0
for step, (batch_x, batch_y) in enumerate(i[1]):

    batch_x, batch_y = Variable(batch_x.to(device=my_device)), Variable(batch_y.to(device=my_device))
    out = net(batch_x)

    pred = torch.max(out, 1)[1]
    test_correct = (pred == batch_y).sum()
    eval_acc += test_correct.item()

# testing accuracy
accuracy = float(eval_acc) / 135.0
print('Test acc: ', accuracy)

# save the trained model (change the file name if necessary)
torch.save(net.state_dict(), 'face.pth')
