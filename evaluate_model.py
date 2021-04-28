from app.simple_autoencoder import AE, train_AE

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch import nn
import torch.optim as optim
import torch


# Load data
print("Loading data from file...")
X = np.genfromtxt("data/train-test-data/X.csv", delimiter=",")
y = np.genfromtxt("data/train-test-data/y.csv", delimiter=",", dtype=int)
print("X shape:", X.shape)
print("y shape:", y.shape)


encoding_methods = ["pca", "tsne", "ae"]
n_dimensions = [2, 3, 8, 16]
for method in encoding_methods:
  for n_dim in n_dimensions:
    # Divide training and testing data and encode X
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    print("Encoding to ", n_dim, "dimensions using ", method, "...")
    if method == "tsne" and n_dim < 4:
      tsne = TSNE(n_components=n_dim)
      tsne.fit(X_train)
      X_train = tsne.transform(X_train)
      X_test = tsne.transform(X_test)
    if method == "pca":
      pca = PCA(n_components=n_dim)
      pca.fit(X_train)
      X_train = pca.transform(X_train)
      X_test = pca.transform(X_test)
    if method == "ae":
      X_train_t = torch.tensor(X_train)
      X_test_t = torch.tensor(X_test)
      loss_function = nn.L1Loss()
      auto = AE(encoded_size=n_dim)
      optimizer = optim.SGD(auto.parameters(), lr=0.000001, momentum=0.9)
      train_AE(X_train_t, X_train_t, auto, optimizer, loss_function, EPOCHS=75)
      X_train = auto.forward(X_train_t.float(), return_z=True).detach().numpy()
      X_test = auto.forward(X_test_t.float(), return_z=True).detach().numpy()

    # Train the MLPClassifier
    print("Training the classifier...")
    classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    classifier.fit(X_train, y_train)

    # Evaluate performance 
    target_names = ['class 0', 'class 1', 'class 2', 'class 3']
    print("Training performance...")
    print(classification_report(y_train, classifier.predict(X_train), target_names=target_names))

    print("Testing performance...")
    print(classification_report(y_test, classifier.predict(X_test), target_names=target_names))
