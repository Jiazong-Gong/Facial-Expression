# model structure

import torch.nn as nn


# define the basic model
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(input_size),
            nn.Linear(input_size, hidden_size),
            nn.Sigmoid(),
            # nn.BatchNorm1d(num_features=hidden_size),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out


# define an auto encoder
class AE(nn.Module):
    def __init__(self, input_size, embedded_size):
        super(AE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, embedded_size),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedded_size, input_size),
            nn.Sigmoid()
            
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        return decoded


# define the model with technique
class BDNN(nn.Module):
    def __init__(self, input_size, embedded_size, hidden_size, num_classes):
        super(BDNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, embedded_size),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(embedded_size, input_size),
            nn.Sigmoid()
        )
        self.classifier = nn.Sequential(
            nn.Linear(embedded_size, hidden_size),
            nn.Sigmoid(),
            nn.Linear(hidden_size, num_classes)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        out = self.classifier(encoded)
        return decoded, out
