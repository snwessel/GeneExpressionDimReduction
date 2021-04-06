import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

num_points = 300

# load gene expression data as a numpy array
print("Loading data...")
X = np.genfromtxt("data/processed-data/gene-expression.csv", delimiter=",")
X = X[:num_points]
print("X head:", X[:5, :5])
print("X shape:", X.shape)

# embed using t-SNE
print("Reducing dimensions...")
X_embedded = TSNE(n_components=2).fit_transform(X)
print("Reduced from", X.shape, "to", X_embedded.shape)

# embed using PCA
pca = PCA(n_components = 2)
fit = pca.fit(X.T)
pca_components = fit.components_

# load the diagnosis labels
labels = np.genfromtxt("data/processed-data/ordered-diagnoses.csv", delimiter=",", dtype=str)
labels = labels[:num_points]

# create the plots
fig, ax = plt.subplots()
for label in np.unique(labels):
  ix = np.where(labels == label)
  ax.scatter(X_embedded.T[0][ix], X_embedded.T[1][ix], label=label)
ax.legend()
plt.title("t-SNE")
plt.show()

fig, ax = plt.subplots()
for label in np.unique(labels):
  ix = np.where(labels == label)
  ax.scatter(pca_components.T[0][ix], pca_components.T[1][ix], label=label)
ax.legend()
plt.title("PCA")
plt.show()
