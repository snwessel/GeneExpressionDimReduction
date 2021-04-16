
import matplotlib.pyplot as plt
import numpy as np
import time
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torch
from torch import autograd



class AE(nn.Module):
    
    def __init__(self, input_size=56602, hidden_size=2):
        '''
        In the initializer we setup model parameters/layers.
        '''
        super(AE, self).__init__() 

        ### REMOVE BELOW
        self.input_size = input_size
        self.hidden_size = hidden_size
       
        # input layer; from x -> z
        self.i = nn.Linear(self.input_size, self.hidden_size)
        
        # output layer
        self.o = nn.Linear(self.hidden_size, self.input_size)
        

    def forward(self, X, return_z=False):
        ### REMOVE BELOW
        z = self.i(X)
        if return_z:
            return z
        return self.o(z)


def train_AE(X_in, X_target, model, optimizer, loss_function, EPOCHS=10):
    for epoch in range(EPOCHS):  
        idx, batch_num = 0, 0
        batch_size = 200

        print("Training epoch...")
        while idx < 60000:
            # zero the parameter gradients
            optimizer.zero_grad()

            X_batch = X_in[idx: idx + batch_size].float()
            X_target_batch = X_target[idx: idx + batch_size].float()
            idx += batch_size

            # now run our X's forward, get preds, incur
            # loss, backprop, and step the optimizer.
            with autograd.detect_anomaly():
              X_tilde_batch = model(X_batch)
              loss = loss_function(X_tilde_batch, X_target_batch)
              loss.backward()
              optimizer.step()

            # print out loss
            if batch_num % 10 == 0:
                print("\tepoch: {}, batch: {} // loss: {:.3f}".format(epoch, batch_num, loss.item()))
            
            # print("\t\tAny NAN:", np.isnan(X).any())
            # print("\t\tmax encoder param:", torch.max(model.i.weight.grad))
            # print("\t\tmin encoder param:", torch.min(model.i.weight.grad))

            batch_num += 1


# load gene expression data as a numpy array
start_time = time.perf_counter()
print("Loading data...")
X = np.genfromtxt("data/train-test-data/X.csv", delimiter=",")
print("\tFinished after", time.perf_counter()-start_time, "seconds.")
print("\tX shape:", X.shape)

print("Converting to tensor and normalizing...")
start_time = time.perf_counter()
#X = F.normalize(torch.tensor(X)) # TODO: check that this is normalizing across the correct dimension
X = torch.tensor(X)
auto = AE(hidden_size=100)
print("\tFinished after", time.perf_counter()-start_time, "seconds.")

loss_function = nn.L1Loss()
auto = AE()
optimizer = optim.SGD(auto.parameters(), lr=0.00001, momentum=0.9)

print("Training...")
start_time = time.perf_counter()
train_AE(X, X, auto, optimizer, loss_function, EPOCHS=5)
print("\tFinished after", time.perf_counter()-start_time, "seconds.")


# print("Encoding the data...")
# X_embedded = auto.forward(X.float(), return_z=True)
# print("\tembedded shape:", X_embedded.shape)

# # load the diagnosis labels
# labels = np.genfromtxt("data/train-test-data/y.csv", delimiter=",", dtype=int)
# label_name_dict = {
#   0: "Acute myeloid leukemia",
#   1: "Clear cell adenocarcinoma",
#   2: "Squamous cell carcinoma"
# }

# # create the plots
# fig, ax = plt.subplots()
# for label in np.unique(labels):
#   print("graphing label", label)
#   i = np.where(labels == label)
#   label_name = label_name_dict[label]
#   ax.scatter(X_embedded.T[0][i], X_embedded.T[1][i], label=label_name)
# ax.legend(title="Tumor Diagnosis")
# plt.title("Auto-Encoder")
# plt.show()