from app.simple_autoencoder import AE, train_AE

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import time
from torch import nn
import torch.optim as optim
import torch

num_points = 300

# load gene expression data as a numpy array
print("Loading data...")
X = np.genfromtxt("data/train-test-data/X.csv", delimiter=",")
print("X shape:", X.shape)

# load the diagnosis labels
labels = np.genfromtxt("data/train-test-data/y.csv", delimiter=",", dtype=int)
label_name_dict = {
  0: "Acute myeloid leukemia",
  1: "Clear cell adenocarcinoma",
  2: "Pheochromocytoma",
  3: "Squamous cell carcinoma"
}


# embed using t-SNE & visualize
print("Reducing dimensions...")
X_embedded = TSNE(n_components=2).fit_transform(X)
print("Reduced from", X.shape, "to", X_embedded.shape)

fig, ax = plt.subplots()
for label in np.unique(labels):
  i = np.where(labels == label)
  label_name = label_name_dict[label]
  ax.scatter(X_embedded.T[0][i], X_embedded.T[1][i], label=label_name)
ax.legend(title="Tumor Diagnosis")
plt.title("t-SNE")
plt.show()

# embed using PCA & visualize
pca = PCA(n_components = 2)
fit = pca.fit(X.T)
pca_components = fit.components_

fig, ax = plt.subplots()
for label in np.unique(labels):
  i = np.where(labels == label)
  label_name = label_name_dict[label]
  ax.scatter(pca_components[0][i], pca_components[1][i], label=label_name)
ax.legend(title="Tumor Diagnosis")
plt.title("PCA")
plt.show()

# embed using autoencoder & visualize
print("Training Autoencoder...")
X = torch.tensor(X)
loss_function = nn.L1Loss()
auto = AE()
optimizer = optim.SGD(auto.parameters(), lr=0.000001, momentum=0.9)
start_time = time.perf_counter()
train_AE(X, X, auto, optimizer, loss_function, EPOCHS=75)
print("\tFinished after", time.perf_counter()-start_time, "seconds.")

# encode the data
print("Encoding the data...")
X_embedded = auto.forward(X.float(), return_z=True).detach().numpy()
print("\tembedded shape:", X_embedded.shape)

# create the plots
fig, ax = plt.subplots()
for label in np.unique(labels):
  print("graphing label", label)
  i = np.where(labels == label)
  label_name = label_name_dict[label]
  ax.scatter(X_embedded.T[0][i], X_embedded.T[1][i], label=label_name)
ax.legend(title="Tumor Diagnosis")
plt.title("Auto-Encoder")
plt.show()