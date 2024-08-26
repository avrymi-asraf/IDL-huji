# IDL EX2 Hillel Avreimi
import cv2
import pandas as pd
import time
from utiltis import import_MNIST_dataset
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import plotly.express as px
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
ENCODER_PATH = './encoder.pth'
ENCODER_PATH_CLASSIFIER = './encoder_classifier.pth'
AE_Q3_PATH = './ae_q3.pth'
AE_Q1_PATH = './ae_q1.pth'


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


def train_classifier_encoder(model, train_loader, test_loader, epochs, save_path, loss_func=nn.CrossEntropyLoss(), device=DEVICE,save_model=True):
    model = model.to(device)
    optimazer = torch.optim.Adam(model.parameters())

    record_data = pd.DataFrame({"train_loss": float(), "test_loss": float(), "accuracy": float()},
                               index=range(epochs))

    for i, epoch in enumerate(range(epochs)):
        print(f'epoch {i}')
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
    fig = px.line(record_data)
    # fig.write_image(save_path)
    fig.show()

def train_AE(model, train_loader, test_loader, epochs, save_path, loss_func=nn.L1Loss(), device=DEVICE, save_mode=None):
    model = model.to(device)
    optimazer = torch.optim.Adam(model.parameters())

    record_data = pd.DataFrame({"train_loss": float(), "test_loss": float()}, index=range(epochs))

    for i, epoch in enumerate(range(epochs)):
        print(f'epoch {i}')
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

    print('after epochs')
    if save_mode == 'encoder':
        torch.save(model.encoder.state_dict(), ENCODER_PATH)
        print('encoder saved')
        torch.save(model.state_dict(), AE_Q1_PATH)
        print('model saved')
    elif save_mode == 'all':
        torch.save(model.state_dict(), AE_Q3_PATH)
        print('model saved')

    fig = px.line(record_data)
    print('fig created')
    # fig.write_image(save_path)
    # print('fig saved')
    fig.show()

def plot_AE_img(img, outputs):
    for i in range(100):
        ax = plt.subplot(10,10,i+1)
        im1 = (img[i][0].numpy()*255).astype(float)
        im2 = (outputs[i][0].detach().numpy()*255).astype(float)
        im = cv2.hconcat([im1,im2])
        plt.imshow(im,cmap='Greys')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()

def q1():
    # q1
    print('q1')
    model = AE()
    train_AE(model, train_loader, test_loader, 15, save_path='ex2/data/imgs/q1.png', save_mode='encoder')

    # q1.2 show 50 images, 5x10, original and reconstructed
    print('q1.2')
    encoder = ConvAutoEncoder()
    encoder.load_state_dict(torch.load(ENCODER_PATH))
    model = AE()
    model.encoder = encoder
    model.eval()
    images = []
    for i, (x, _) in enumerate(test_loader):
        if i == 5:
            break
        predict = model(x)
        images.append(x)
        images.append(predict)
    images = torch.cat(images)
    images = images.detach().cpu().numpy()
    images = np.transpose(images, (0, 2, 3, 1))
    images = (images + 1) / 2
    fig, axs = plt.subplots(10, 10, figsize=(15, 15))  # Create a 10x10 grid for 50 pairs of images

    for i in range(50):  # Loop over the first 50 images - todo: fix.
        # Display the original image
        axs[i // 5, (i % 5) * 2].imshow(images[i * 2].squeeze(), cmap='gray')
        axs[i // 5, (i % 5) * 2].axis('off')
        axs[i // 5, (i % 5) * 2].set_title('Original')

        # Display the reconstructed image
        axs[i // 5, (i % 5) * 2 + 1].imshow(images[i * 2 + 1].squeeze(), cmap='gray')
        axs[i // 5, (i % 5) * 2 + 1].axis('off')
        axs[i // 5, (i % 5) * 2 + 1].set_title('Reconstructed')

    plt.tight_layout()
    plt.show()

def q2():
    # q2
    print('q2')
    ae_mlp = AE_MLP(ConvAutoEncoder())
    train_classifier_encoder(ae_mlp, train_loader, test_loader, 15, save_path='./q2.png')

def q3():
    # q3
    print('q3')
    encoder = ConvAutoEncoder()
    encoder.load_state_dict(torch.load(ENCODER_PATH_CLASSIFIER))  # load pre-trained encoder

    ae_freeze = AE_freeze_encoder(encoder)
    train_AE(ae_freeze, train_loader, test_loader, 15, save_mode='all', save_path='./q3.png')

def q3_2():
    print('q3.2')
    model_q3 = AE()
    model_q3.load_state_dict(torch.load(AE_Q3_PATH))
    model_q3.eval()
    images_q3 = torch.stack([mini_train_loader.dataset[i][0] for i in range(100)])
    outputs_q3 = model_q3(images_q3)
    plot_AE_img(images_q3, outputs_q3)

    # inference on AE of q1
    model_q1 = AE()
    model_q1.load_state_dict(torch.load(AE_Q1_PATH))
    model_q1.eval()
    images = torch.stack([mini_train_loader.dataset[i][0] for i in range(100)])
    outputs = model_q1(images)
    plot_AE_img(images, outputs)

def q4():
    # q4
    print('q4')
    model = AE()
    train_AE(model, mini_train_loader, mini_test_loader, 15, save_path='./q4_AE.png', save_mode='dont')
    print('q4_MLP')
    model2 = AE_MLP(ConvAutoEncoder())
    train_classifier_encoder(model2, mini_train_loader, mini_test_loader, 15, save_path='./q4_MLP.png')

def q5():
    # q5
    print('q5')
    encoder = ConvAutoEncoder()
    encoder.load_state_dict(torch.load(ENCODER_PATH))  # load pre-trained encoder
    pre_trained_encoder = AE_MLP(encoder)
    train_classifier_encoder(pre_trained_encoder, mini_train_loader, mini_test_loader, 15, save_model=False,
                             save_path='./q5.png')

if __name__ == '__main__':
    # data
    train_loader, test_loader = import_MNIST_dataset()
    mini_train_loader, mini_test_loader = import_MNIST_dataset(mini_data=True)

    # q1()
    # q2()
    # q3()
    q3_2()
    # q4()
    q5()







