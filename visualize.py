import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

num_points = 300

# load gene expression data as a numpy array
print("Loading data...")
X = np.genfromtxt("data/train-test-data/X.csv", delimiter=",")
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
labels = np.genfromtxt("data/train-test-data/y.csv", delimiter=",", dtype=int)
label_name_dict = {
  0: "Acute myeloid leukemia",
  1: "Clear cell adenocarcinoma",
  2: "Squamous cell carcinoma"
}


# create the plots
fig, ax = plt.subplots()
for label in np.unique(labels):
  print("graphing label", label)
  i = np.where(labels == label)
  label_name = label_name_dict[label]
  ax.scatter(X_embedded.T[0][i], X_embedded.T[1][i], label=label_name)
ax.legend(title="Tumor Diagnosis")
plt.title("t-SNE")
plt.show()

fig, ax = plt.subplots()
for label in np.unique(labels):
  i = np.where(labels == label)
  label_name = label_name_dict[label]
  ax.scatter(pca_components[0][i], pca_components[1][i], label=label_name)
ax.legend(title="Tumor Diagnosis")
plt.title("PCA")
plt.show()
