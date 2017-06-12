
import torch
from torch.autograd import Variable
import time
import numpy as np
import matplotlib.pyplot as plt

start = time.clock()

# Hyperparameters -----------------------------------
N = 1000                # Batch size
noise_level = 0.05
H1 = 8               # Hidden layers dimensions
H2 = 8
learning_rate = 5e-3
# n_it = 100000
max_loss = N*noise_level/15
testset_size = N #int(N*n_nets/2)
test_on_trainset = True
plot_period = 2
height = 7
width = 15
# ---------------------------------------------------

#%% Regression data

# Generate train set
x_tensor = torch.rand(N, 1)

# f(x) = sin(13x)*sin(27x) + 1/2 + Noise
y_tensor = 1/2*torch.ones(N, 1)                         # 1/2               
t1 = torch.sin(13*x_tensor)                              # sin(13x)
t2 = torch.sin(27*x_tensor)                              # sin(27x)
y_tensor = torch.addcmul(y_tensor, t1, t2)
y_tensor = torch.add(y_tensor, noise_level, torch.randn(N,1)) # Add noise
 
x = Variable(x_tensor)    
y = Variable(y_tensor, requires_grad=False)

range_train = [np.min(x_tensor.numpy()), np.max(x_tensor.numpy())]
print(range_train) 

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

#%% Net
model = torch.nn.Sequential(
    torch.nn.Linear(1, H1),
    torch.nn.Sigmoid(), # ReLU
    torch.nn.Linear(H1, H2),
    torch.nn.Sigmoid(), # ReLU
    torch.nn.Linear(H2, 1),
)

#%% Loss function
loss_fn = torch.nn.MSELoss(size_average=False)

#%% Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
         
#%% Training a new neural network with all the training data 
last_plot = time.clock()
it = 0
#%%
while True:
# for t in range(n_it):

    # Forward
    y_pred = model(x)
    loss = loss_fn(y_pred, y)  
    
    now = time.clock()
    if now-last_plot >= plot_period:
        # Plot
        plt.figure(figsize=(width,height))
        plt.plot(x.data.numpy(), y_pred.data.numpy(), 'ro', x.data.numpy(), y.data.numpy(), 'g^')
        plt.xlim([0, 1])
        plt.ylim([-1, 2])
        plt.show()
        last_plot = now
        # Print
        print("Iteration",'{:3}'.format(it+1),": loss:", loss.data[0],", objective:",max_loss) 
    
    # Backward    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    it+=1
    if loss.data[0] <= max_loss:
        break
        
#%% Test loss with training data
if test_on_trainset:
    y_pred_train = model(x) 
    loss = loss_fn(y_pred_train, y)
    
    plt.figure(figsize=(width,height))
    plt.plot(x.data.numpy(), y_pred_train.data.numpy(), 'ro', x.data.numpy(), y.data.numpy(), 'g^')
    plt.xlim([0, 1])
    plt.ylim([-1, 2])
    plt.show()
    
    print("Train data, loss:", loss.data[0])
        
#%% Test loss with test data
y_pred_test = model(x_test) 
loss = loss_fn(y_pred_test, y_test)

plt.figure(figsize=(width,height))
plt.plot(x_test.data.numpy(), y_pred_test.data.numpy(), 'ro', x_test.data.numpy(), y_test.data.numpy(), 'g^')
plt.xlim([0, 1])
plt.ylim([-1, 2])
plt.show()

print("Test data, loss:", loss.data[0])
      
end = time.clock()
print('After', it, 'iterations,', '{:3}'.format(end - start), 'seconds')