# IDL EX2 Hillel Avreimi
import pandas as pd
import time
from utiltis import import_MNIST_dataset
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import plotly.express as px

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER_PATH = './encoder.pth'
ENCODER_PATH_CLASSIFIER = './encoder_classifier.pth'


class ConvAutoEncoder(nn.Module):
    def __init__(self):
        super(ConvAutoEncoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(4, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 12, 7)  # (batch, , 1, 1)
        )

    def forward(self, x):
        return self.encoder(x)


class ConvAutoDecoder(nn.Module):
    # Decoder
    def __init__(self):
        super(ConvAutoDecoder, self).__init__()

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(12, 8, 7),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(4, 1, 3, stride=2, padding=1, output_padding=1),
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

class AE_freeze_encoder(nn.Module):
    def __init__(self, encoder):
        super(AE_freeze_encoder, self).__init__()
        self.encoder = encoder
        self.decoder = ConvAutoDecoder()

    def forward(self, x):
        with torch.no_grad():
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


class AE_MLP(nn.Module):
    def __init__(self, pre_trained_encoder, input_size=12, hidden_size=50, output_size=10):
        super(AE_MLP, self).__init__()
        self.encoder = pre_trained_encoder
        self.mlp = MLP(input_size, hidden_size, output_size)

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp(x)
        return x


def train_classifier_encoder(model, train_loader, test_loader, epochs, loss_func=nn.CrossEntropyLoss(), device=DEVICE):
    model = model.to(device)
    optimazer = torch.optim.Adam(model.parameters())

    record_data = pd.DataFrame({"train_loss": float(), "test_loss": float(), "accuracy": float()},
                               index=range(epochs))

    for epoch in range(epochs):
        train_loss = 0
        test_loss = 0
        accuracy = 0
        for x, y in tqdm(train_loader):
            model.train()
            x, y = x.to(device), y.to(device)
            predict = model(x)
            loss = loss_func(predict, y)
            loss.backward()
            optimazer.step()
            optimazer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        for x, y in tqdm(test_loader):
            model.eval()
            x, y = x.to(device), y.to(device)
            predict = model(x)
            loss = loss_func(predict, y)
            test_loss += loss.item()
            accuracy += (predict.argmax(dim=1) == y).float().mean()
        test_loss /= len(test_loader)
        accuracy /= len(test_loader)
        record_data.iloc[epoch] = [train_loss, test_loss, accuracy]

    if save_model:
        torch.save(model.encoder.state_dict(), ENCODER_PATH_CLASSIFIER)
    px.line(record_data).to_image(save_path)
    px.line(record_data).show()

def train_AE(model, train_loader, test_loader, epochs, loss_func=nn.L1Loss(), device=DEVICE):
    model = model.to(device)
    optimazer = torch.optim.Adam(model.parameters())

    record_data = pd.DataFrame({"train_loss": float(), "test_loss": float()}, index=range(epochs))

    for epoch in range(epochs):
        train_loss = 0
        test_loss = 0
        for x, _ in tqdm(train_loader):
            model.train()
            x = x.to(device)
            predict = model(x)
            loss = loss_func(x, predict)
            loss.backward()
            optimazer.step()
            optimazer.zero_grad()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        for x, _ in tqdm(test_loader):
            model.eval()
            x = x.to(device)
            predict = model(x)
            loss = loss_func(x, predict)
            test_loss += loss.item()
        test_loss /= len(test_loader)
        record_data.iloc[epoch] = [train_loss, test_loss]

    torch.save(model.encoder.state_dict(), ENCODER_PATH)
    px.line(record_data).show()


if __name__ == '__main__':
    # data
    train_loader, test_loader = import_MNIST_dataset()

    # q1
    model = AE()
    train_AE(model, train_loader, test_loader, 10, save_path='./q1.png')

    # q2
    ae_mlp = AE_MLP(ConvAutoEncoder())
    train_classifier_encoder(ae_mlp, train_loader, test_loader, 10, save_path='./q2.png')

    # q3
    encoder = ConvAutoEncoder()
    encoder.load_state_dict(torch.load(ENCODER_PATH_CLASSIFIER))  # load pre-trained encoder

    ae_freeze = AE_freeze_encoder(encoder)
    train_AE(ae_freeze, train_loader, test_loader, 10, save_model=False, save_path='./q3.png')

    # q4
    mini_train_loader, mini_test_loader = import_MNIST_dataset(mini_data=True)
    model = AE()
    train_AE(model, mini_train_loader, mini_test_loader, 20, save_path='./q4_AE.png')
    model2 = AE_MLP(ConvAutoEncoder())
    train_classifier_encoder(model2, mini_train_loader, mini_test_loader, 20, save_path='./q4_MLP.png')

    # q5
    encoder = ConvAutoEncoder()
    encoder.load_state_dict(torch.load(ENCODER_PATH))  # load pre-trained encoder
    pre_trained_encoder = AE_MLP(encoder)
    train_classifier_encoder(pre_trained_encoder, mini_train_loader, mini_test_loader, 20, save_model=False,
                             save_path='./q5.png')
