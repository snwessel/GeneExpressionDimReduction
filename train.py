from app import data_loader

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# load gene expression data as a numpy array
print("Loading data...")
dl = data_loader.DataLoader()
X = np.genfromtxt("processed-data\\gene-expression.csv", delimiter=",")
X = X[:300]
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

plt.plot(X_embedded.T[0], X_embedded.T[1], 'ro')
plt.title('t-SNE')
plt.show()

print("Used PCA to reduce to:", pca_components.shape)
plt.plot(pca_components[0], pca_components[1], 'ro')
plt.title('PCA')
plt.show()