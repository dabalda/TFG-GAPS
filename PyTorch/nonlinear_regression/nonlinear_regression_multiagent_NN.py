
import torch
from torch.autograd import Variable
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

start = time.clock()

# Hyperparameters ----------------------------------
N = 4000                # Batch size
noise_level = 0.05
H1 = 16                  # Hidden layers dimensions
H2 = 8
train_centralized_net = False
train_net_of_nets = True
n_nets = 2             # Number of nets
diffusion = True
same_distr = False
data_overlap = 0        
adjacency = 'FC'        # 'FC', 'chain' 
learning_rate = 5e-3
max_loss_central = N*n_nets*noise_level/8
max_loss_net = N*noise_level/8
testset_size = N*n_nets    # e.g. int(N*n_nets/2)
test_on_trainset = True
plot_period = 10
height = 7
width = 15
# --------------------------------------------------

#%% Regression data

# Generate train set
if same_distr:
    x_tensor = torch.rand(n_nets, N, 1)
else:
    x_tensor = torch.zeros(n_nets, N, 1)
    for i in range(n_nets):
        x_tensor[i] = torch.rand(N, 1)*(1+data_overlap)/n_nets + (i - data_overlap/2)/n_nets 
  
# f(x) = sin(13x)*sin(27x) + 1/2 + Noise
y_tensor = 1/2*torch.ones(n_nets, N, 1)                         # 1/2
for i in range(n_nets):                 
    t1 = torch.sin(13*x_tensor[i])                              # sin(13x)
    t2 = torch.sin(27*x_tensor[i])                              # sin(27x)
    y_tensor[i] = torch.addcmul(y_tensor[i], t1, t2)
    y_tensor[i] = torch.add(y_tensor[i], noise_level, torch.randn(N,1)) # Add noise
 
x = Variable(x_tensor)    
y = Variable(y_tensor, requires_grad=False)

x_flat = x.view(-1,1)
y_flat = y.view(-1,1)

ranges = np.zeros([n_nets,2])
for i in range(n_nets):
    ranges[i] = [np.min(x_tensor[i].numpy()), np.max(x_tensor[i].numpy())]
print(ranges) 

#%% Generate test set
x_test_tensor = torch.rand(testset_size, 1)
y_test_tensor = 1/2*torch.ones(testset_size, 1)

t1_test = torch.sin(13*x_test_tensor)
t2_test = torch.sin(27*x_test_tensor) 
y_test_tensor = torch.addcmul(y_test_tensor, t1_test, t2_test) 
y_test_tensor = torch.add(y_test_tensor, noise_level, torch.randn(testset_size,1))
    
x_test = Variable(x_test_tensor, requires_grad=False)
y_test = Variable(y_test_tensor, requires_grad=False)

range_test = [np.min(x_test_tensor.numpy()), np.max(x_test_tensor.numpy())]
print(range_test)

#%% Nets
models = [None]*(n_nets+1);
for i in range(n_nets+1):
    models[i] = torch.nn.Sequential(
    torch.nn.Linear(1, H1),
    torch.nn.Sigmoid(), # ReLU
    torch.nn.Linear(H1, H2),
    torch.nn.Sigmoid(), # ReLU
    torch.nn.Linear(H2, 1),
)
     
#%% Neighbours definition

if adjacency == 'FC':
    neighbours = 1/(n_nets)*torch.ones(n_nets,n_nets)
elif adjacency == 'chain':
    # TODO
    neighbours = None

#%%
loss_fn = torch.nn.MSELoss(size_average=False)

# Optimizers
optimizers = [None]*(n_nets+1);
for i in range(n_nets+1):
    optimizers[i]= torch.optim.Adam(models[i].parameters(), lr=learning_rate)
    
#%% Local copy of parameters for diffusion
param_lists_copy = [None]*(n_nets+1)

# Parameters                         
param_lists = [None]*(n_nets+1)
for i in range(n_nets+1):
    param_lists[i] = list(models[i].parameters())

#%% Training a neural network with all the training data
if train_centralized_net:
    start_central = time.clock()
    last_plot = time.clock()
    it_central = 0
#%%
    while True:
    # for it in range(n_it):
    
        y_pred = models[n_nets](x_flat) 
        loss_central = loss_fn(y_pred, y_flat)
        
        now = time.clock()
        if now-last_plot >= plot_period:
            # Plot
            plt.figure(figsize=(width,height))
            plt.plot(x_flat.data.numpy(), y_pred.data.numpy(), 'ro', x_flat.data.numpy(), y_flat.data.numpy(), 'g^')
            plt.xlim([0, 1])
            plt.ylim([-1, 2])
            plt.show()
            last_plot = now
            # Print      
            print("Iteration",'{:3}'.format(it_central+1),", net", '{:2}'.format(n_nets+1),": loss:", loss_central.data[0],", objective:", max_loss_central)
        
        optimizers[n_nets].zero_grad()
        loss_central.backward()
        optimizers[n_nets].step()
    
        it_central+=1
        if loss_central.data[0] <= max_loss_central:
            break
        
    end_central = time.clock()
#%% Training n nets, each with a subset of the training set, 
#   with or without diffusion of parameters between nets
if train_net_of_nets:
    start_net = time.clock()
    last_plot = time.clock()*np.ones(n_nets)
    it_net = 0 
    loss = [None]*n_nets
#%%
    while True:
    # for it in range(n_it):
        for i in range(n_nets):
            y_pred = models[i](x[i])  
            loss[i] = loss_fn(y_pred, y[i])
                      
            now = time.clock()
            if now-last_plot[i] >= plot_period:
                # Plot
                plt.figure(figsize=(width,height))
                plt.plot(x[i].data.numpy(), y_pred.data.numpy(), 'ro', x[i].data.numpy(), y[i].data.numpy(), 'g^')
                plt.xlim([0, 1])
                plt.ylim([-1, 2])
                plt.show()
                last_plot[i] = now
                # Print      
                print("Iteration",'{:3}'.format(it_net+1),", net", '{:2}'.format(i+1),": loss:", loss[i].data[0],", objective:", max_loss_net)
            
            optimizers[i].zero_grad()
            loss[i].backward() 
            optimizers[i].step()
          
        if diffusion:
            # Diffusion step
            for i in range(n_nets):
                param_lists_copy[i] = copy.deepcopy(param_lists[i])
                
            for i in range(n_nets):    
                for k in range(len(param_lists_copy[i])):
                    param_lists[i][k].data = param_lists[i][k].data*0
                    for j in range(n_nets):
                        param_lists[i][k].data += param_lists_copy[j][k].data*neighbours[i][j]
        
        it_net+=1
        if np.max(loss).data[0] <= max_loss_net:
            break
                    
    end_net = time.clock()
#%%
range_test = []
if train_net_of_nets:
    for i in range(n_nets):
        range_test += list(range(n_nets))
if train_centralized_net:
    range_test.append(n_nets)
#%% Testing loss with training data
if test_on_trainset:
    for i in range(n_nets):                             # Training data sets
        for j in range_test:                            # Nets
            y_pred_train = models[j](x[i]) 
            loss = loss_fn(y_pred_train, y[i])
            
            plt.figure(figsize=(width,height))
            plt.plot(x[i].data.numpy(), y_pred_train.data.numpy(), 'ro', x[i].data.numpy(), y[i].data.numpy(), 'g^')
            plt.xlim([0, 1])
            plt.ylim([-1, 2])
            plt.show()
            
            print("Net ", '{:2}'.format(j+1),", train data ", '{:2}'.format(i+1),": loss: ", loss.data[0])
        
#%% Testing loss with test data
for i in range_test:
    y_pred_test = models[i](x_test) 
    loss = loss_fn(y_pred_test, y_test)
    
    plt.figure(figsize=(width,height))
    plt.plot(x_test.data.numpy(), y_pred_test.data.numpy(), 'ro', x_test.data.numpy(), y_test.data.numpy(), 'g^')
    plt.xlim([0, 1])
    plt.ylim([-1, 2])
    plt.show()
    
    print("Net ", '{:2}'.format(i+1),", test data, loss: ", loss.data[0])

#%%      
end = time.clock()
print('Script: ',end - start, ' seconds.')
if train_centralized_net:
    print(end_central - start_central, ' seconds to train centralized model.')
if train_net_of_nets:
    print(end_net - start_net, ' seconds to train distributed model.')