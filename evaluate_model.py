import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import time

start_time = time.perf_counter()

# Load data
print("Loading data from file...")
X = np.genfromtxt("data/train-test-data/X.csv", delimiter=",")
y = np.genfromtxt("data/train-test-data/y.csv", delimiter=",", dtype=int)

print("X shape:", X.shape)
print("y shape:", y.shape)

# Divide training and testing data and encode X
# TODO: figure out if encoding should happen after dividing data
print("Encoding Gene expression values...")
#X = TSNE(n_components=3).fit_transform(X) # TODO: experiment with this number and try PCA
pca = PCA(n_components = 3)
fit = pca.fit(X.T)
X = fit.components_.T
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train the MLPClassifier
print("Training the classifier...")
classifier = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
classifier.fit(X_train, y_train)

# Evaluate performance 
print("Evaluating performance...")
y_pred = classifier.predict(X_test)
target_names = ['class 0', 'class 1', 'class 2']
print(classification_report(y_test, y_pred, target_names=target_names))

# log time taken
print("Operation took", int(time.perf_counter() - start_time), "seconds.")
