# IDL EX2 Hillel Avreimi
import pandas as pd
import time
from utiltis import import_MNIST_dataset
import numpy as np
import torch
from torch import nn
from tqdm import  tqdm
import plotly.express as px

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER_PATH = './encoder.pth'
class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),  # (batch, 16, 14, 14)
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),  # (batch, 32, 7, 7)
            nn.ReLU(),
            nn.Conv2d(8, 12, 7)  # (batch, , 1, 1)
        )
    def forward(self,x):
        return self.encoder(x)

class ConvAutoDecoder(nn.Module):
    # Decoder
    def __init__(self):
        super(ConvAutoDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12, 8, 7),  # (batch, 32, 7, 7)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),  # (batch, 16, 14, 14)
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),  # (batch, 1, 28, 28)
            nn.Tanh()
        )

    def forward(self, x):
        x = self.decoder(x)
        return x

class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()
        self.encoder = ConvAutoEncoder()
        self.decoder = ConvAutoDecoder()
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x


def trainAE(model,train_loader,test_loader, epochs, device=DEVICE):
    model = model.to(device)
    optimazer = torch.optim.Adam(model.parameters())
    loss_func = nn.L1Loss()

    record_data = pd.DataFrame({"train_loss":float(),"test_loss":float()},index=range(epochs))

    for epoch in range(epochs):
        train_loss = 0
        test_loss = 0
        for x, _ in tqdm(train_loader):
            model.train()
            x = x.to(device)
            predict = model(x)
            loss = loss_func(x,predict)
            loss.backward()
            optimazer.step()
            optimazer.zero_grad()
            train_loss += loss.item()
        for x, _ in tqdm(test_loader):
            model.eval()
            x, _ = x.to(device)
            predict = model(x)
            loss = loss_func(x, predict)
            test_loss += loss.item()
        record_data.iloc[epoch] = [train_loss, test_loss]

    torch.save(model.encoder.state_dict(), ENCODER_PATH)
    px.line(record_data).show()




if __name__ == '__main__':
    train_loader, test_loader = import_MNIST_dataset()
    model = AE()
    trainAE(model, train_loader, test_loader, 10)