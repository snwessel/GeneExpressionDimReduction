import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

num_points = 300

# load gene expression data as a numpy array
print("Loading data...")
X = np.genfromtxt("data/processed-data/gene-expression.csv", delimiter=",")
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

# create the plots
fig, ax = plt.subplots()
for label in np.unique(labels):
  print("graphing label", label)
  i = np.where(labels == label)
  ax.scatter(X_embedded.T[0][i], X_embedded.T[1][i], label=label)
ax.legend(title="Tumor Diagnosis")
plt.title("t-SNE")
plt.show()

fig, ax = plt.subplots()
for label in np.unique(labels):
  i = np.where(labels == label)
  ax.scatter(pca_components[0][i], pca_components[1][i], label=label)
ax.legend(title="Tumor Diagnosis")
plt.title("PCA")
plt.show()
