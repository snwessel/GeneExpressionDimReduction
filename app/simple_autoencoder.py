
import matplotlib.pyplot as plt
import numpy as np
import time
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch import autograd



class AE(nn.Module):
    
    def __init__(self, input_size=56602, encoded_size=2):
        '''
        In the initializer we setup model parameters/layers.
        '''
        super(AE, self).__init__() 
        self.input_size = input_size
        self.encoded_size = encoded_size
       
        # input layer; from x -> z
        self.i = nn.Linear(self.input_size, self.encoded_size)
        
        # output layer
        self.o = nn.Linear(self.encoded_size, self.input_size)
        

    def forward(self, X, return_z=False):
        ### REMOVE BELOW
        z = self.i(X)
        if return_z:
            return z
        return self.o(z)


def train_AE(X_in, X_target, model, optimizer, loss_function, EPOCHS):
    for epoch in range(EPOCHS):  
        idx, batch_num = 0, 0
        batch_size = 500

        while idx < 453: # not really using batches here
            # zero the parameter gradients
            model.zero_grad() # added this
            optimizer.zero_grad()

            X_batch = X_in[idx: idx + batch_size].float()
            X_target_batch = X_target[idx: idx + batch_size].float()
            idx += batch_size

            # now run our X's forward, get preds, incur
            # loss, backprop, and step the optimizer.
            X_tilde_batch = model(X_batch)
            loss = loss_function(X_tilde_batch, X_target_batch)
            loss.backward()
            optimizer.step()

            batch_num += 1

        # print out loss
        if epoch % 5 == 0:
            print("\tepoch: {}, loss: {:.3f}".format(epoch, loss.item()))


# load data from file
start_time = time.perf_counter()
print("Loading data...")
X = np.genfromtxt("data/train-test-data/X.csv", delimiter=",")
print("\tFinished after", time.perf_counter()-start_time, "seconds.")
print("\tX shape:", X.shape)
X = torch.tensor(X)

loss_function = nn.L1Loss()
auto = AE()
optimizer = optim.SGD(auto.parameters(), lr=0.000001, momentum=0.9)

print("Training...")
start_time = time.perf_counter()
train_AE(X, X, auto, optimizer, loss_function, EPOCHS=75)
print("\tFinished after", time.perf_counter()-start_time, "seconds.")


print("Encoding the data...")
X_embedded = auto.forward(X.float(), return_z=True).detach().numpy()
print("\tembedded shape:", X_embedded.shape)

# load the diagnosis labels
labels = np.genfromtxt("data/train-test-data/y.csv", delimiter=",", dtype=int)
label_name_dict = {
  0: "Acute myeloid leukemia",
  1: "Clear cell adenocarcinoma",
  2: "Pheochromocytoma",
  3: "Squamous cell carcinoma"
}

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