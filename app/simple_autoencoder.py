from torch import nn

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
