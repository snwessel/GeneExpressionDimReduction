import numpy as np
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

LEARNING_RATE = 1e-3
BATCH_SIZE = 128

class Autoencoder(nn.Module):
  def __init__(self):
    super(Autoencoder, self).__init__()
    # TODO: update the number of input features

    # encoder
    self.enc0 = nn.Linear(in_features=56602, out_features=784)
    self.enc1 = nn.Linear(in_features=784, out_features=256)
    self.enc2 = nn.Linear(in_features=256, out_features=128)
    self.enc3 = nn.Linear(in_features=128, out_features=64)
    self.enc4 = nn.Linear(in_features=64, out_features=32)
    self.enc5 = nn.Linear(in_features=32, out_features=16)

    # decoder 
    self.dec1 = nn.Linear(in_features=16, out_features=32)
    self.dec2 = nn.Linear(in_features=32, out_features=64)
    self.dec3 = nn.Linear(in_features=64, out_features=128)
    self.dec4 = nn.Linear(in_features=128, out_features=256)
    self.dec5 = nn.Linear(in_features=256, out_features=784)
    self.enc0 = nn.Linear(in_features=784, out_features=56602)

  def forward(self, x, return_encoded=False):
    x = F.relu(self.enc1(x))
    x = F.relu(self.enc2(x))
    x = F.relu(self.enc3(x))
    x = F.relu(self.enc4(x))
    x = F.relu(self.enc5(x))
    if return_encoded:
      return x

    x = F.relu(self.dec1(x))
    x = F.relu(self.dec2(x))
    x = F.relu(self.dec3(x))
    x = F.relu(self.dec4(x))
    x = F.relu(self.dec5(x))
    return x

net = Autoencoder()

def train_encoder(net, trainloader, NUM_EPOCHS):
  criterion = nn.MSELoss()
  optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
  train_loss = []
  for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
    for data in trainloader:
      img, _ = data
      img = img.view(img.size(0), -1)
      optimizer.zero_grad()
      outputs = net(img)
      loss = criterion(outputs, img)
      loss.backward()
      optimizer.step()
      running_loss += loss.item()
  
    loss = running_loss / len(trainloader)
    train_loss.append(loss)
    print('Epoch {} of {}, Train Loss: {:.3f}'.format(
      epoch+1, NUM_EPOCHS, loss))
  return train_loss

# load gene expression data as a numpy array
print("Loading data...")
X = np.genfromtxt("data/train-test-data/X.csv", delimiter=",")
print("X shape:", X.shape)

# train
trainloader = DataLoader(
    X, 
    batch_size=BATCH_SIZE,
    shuffle=True
)
loss = train_encoder(net, trainloader, 50)
print("train loss:", loss)