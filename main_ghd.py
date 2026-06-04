from datetime import datetime, date
import time
from sklearn.datasets import make_spd_matrix as spd_matrix
import numpy as np
import importlib # needed for testing only to reload scripts importlib.reload(Lossfct)
import matplotlib
# matplotlib.use("Qt5Agg") #otherwise plots not shown with error --- has to be removed for computations on server
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torchinfo import summary
# import cmath
import Lossfct_ghd as Lossfct
import model_generator
from torch.utils.tensorboard import SummaryWriter
from collections.abc import Mapping


# set the device for computation on cpu or gpu
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("Cuda available?", torch.cuda.is_available(), "device: ",device)


# set seed for reproducibility
np.random.seed(771994)
torch.manual_seed(771994)

# set default datatype to make conversion from numpy easier
torch.set_default_dtype(torch.float32)

######################### set hyperparameters
widths=[300,50] #Widhts of the hidden layers
iterations=6000
learning_rate_Adam=0.01
batchsize=6000
s_size_approx=6000 #sample size of the W vectors to approximate kernel evaluations
testsize=20000  #number of random variables used to estimate loss on test set
plotsize=100000
k=2 #factor for input dimension

print("batchsize:",batchsize)
#########################


########################## Define the characteristic function that is supposed to be simulated
dim=2

# User-set GHD parameters.
lam=-0.5
alpha=1.0
delta=1.0
mu=np.array([-1.0,1.0])
beta = np.zeros(dim)
beta[0] = 0.4 * alpha
Lambda = (1.0 - 0.25) * np.eye(dim) + 0.25 * np.ones((dim, dim))

# Validate the GHD parameters before training
gamma2=alpha**2-beta @ Lambda @ beta
if gamma2<=0:
    raise ValueError("Invalid GHD parameters: need alpha^2 > beta^T Lambda beta.")
if delta<=0:
    raise ValueError("Invalid GHD parameters: this script assumes delta > 0.")
if np.any(np.linalg.eigvalsh(Lambda)<=0):
    raise ValueError("Invalid GHD parameters: Lambda must be positive definite.")

beta_torch=torch.tensor(beta,dtype=torch.float32).to(device)
mu_torch=torch.tensor(mu,dtype=torch.float32).to(device)
Lambda_torch=torch.tensor(Lambda,dtype=torch.float32).to(device)
beta_cpu=torch.tensor(beta,dtype=torch.float32).to("cpu")
mu_cpu=torch.tensor(mu,dtype=torch.float32).to("cpu")
Lambda_cpu=torch.tensor(Lambda,dtype=torch.float32).to("cpu")

print("GHD model","lambda:",lam,"alpha:",alpha,"beta:",beta,"delta:",delta,"mu:",mu,"Lambda:",Lambda)


def charfct_vec(W):
    return Lossfct.ghd_char_fct_vec(W,lam,alpha,beta_torch,delta,mu_torch,Lambda_torch,device)

def charfct_vec_cpu(W):
    return Lossfct.ghd_char_fct_vec(W,lam,alpha,beta_cpu,delta,mu_cpu,Lambda_cpu,"cpu")

################################


################################# Define the kernelmatrix corresponding to the kernel evaluations and the samples to approximate the kernel evaluations
bandwidth=[0.02,0.5,1,5,100] #must be a list
def kernelmatrix(X):
    return Lossfct.gaussian_kernelmatrix(X,device,bandwidth) #needs to be compatible with the distribution that is sampled in HERE
def kernelmatrix_cpu(X):
    return Lossfct.gaussian_kernelmatrix(X,"cpu",bandwidth) #needed to conduct evaluation on cpu
def sample_kernel(size):
    return Lossfct.kernelsample_gaussian(size,device,dim,bandwidth) # HERE
def sample_kernel_cpu(size):
    return Lossfct.kernelsample_gaussian(size,"cpu",dim,bandwidth) #needed to conduct evaluation on cpu
################################

################################# Define the model
hidden_architecture=[]
for w in widths:
    hidden_architecture.append( (w,torch.nn.ReLU()) )

model=model_generator.MLP(k*dim,dim,hidden_architecture).to(device)
summary(model,input_data=torch.tensor(np.zeros(shape=(1,k*dim)),dtype=torch.float32),device=device)

# set optimizer
# optimizer=torch.optim.SGD(model.parameters(),lr=learning_rate_SGD )
optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate_Adam )
scheduler=torch.optim.lr_scheduler.MultiStepLR(optimizer,[2000,4000],gamma=0.1)


# create summarywriter for tensorboard
writer = SummaryWriter()
###############################



################################ Training loop

# ADDED: to make sure that the same random samples of both W and Z are used for calculating loss in every epoch

# Creates fixed random input samples (noted by Z in the thesis)
U_eval=np.matrix(np.random.normal(size=k*dim * batchsize)).reshape(batchsize,k* dim)
U_eval=torch.tensor(U_eval,dtype=torch.float32).to(device)

# Creates fixed random samples of W used in loss-graph
W_eval = sample_kernel(s_size_approx)

# END OF ADDED

now=time.time()
loss_history=[]
eval_loss_history=[]
for epoch in np.arange(iterations):

    # Create the random input data
    U=np.matrix(np.random.normal(size=k*dim * batchsize)).reshape(batchsize,k* dim)
    U=torch.tensor(U,dtype=torch.float32).to(device)
    # Sample to approximation the dim-dimensional kernel (every 10th iteration)
    if epoch%20==0:
        W = sample_kernel(s_size_approx)
    # Make predictions
    samples=model(U)

    # Empty gradient
    optimizer.zero_grad()
    # Calculate Loss
    loss = Lossfct.lossfunction_vec(samples, kernelmatrix, charfct_vec, W, device, False) #True means using exact representation of kernel where possible
    # Calculate gradient
    loss.backward()
    # Take one SGD step
    optimizer.step()
    scheduler.step() 

 
    # logging for tensorboard
    with torch.no_grad():
        samples_eval=model(U_eval)
        eval_loss = Lossfct.lossfunction_vec(samples_eval, kernelmatrix, charfct_vec, W_eval, device, False)
        elapsed_time=time.time()-now
        writer.add_scalar('Loss', loss.item(), epoch) 
        writer.add_scalar('Loss', loss.item(), elapsed_time) 
        loss_history.append(loss.item())
        eval_loss_history.append(eval_loss.item())

    if epoch%25==0:
        print("epoch: ",epoch," loss=",loss)

print("Training time: ",time.time()-now)      
################################## end of training loop



# Save the model
model.eval()
model.cpu()
widthstring='_'.join([str(w) for w in widths])
filenamemodel="saved_models/GHD_"+str(dim)+"-dim_"+widthstring+"widths_"+date.today().strftime('%d-%m-%Y')+".pth"
filenamemetadata="saved_models/GHD_"+str(dim)+"-dim_"+widthstring+"widths_"+date.today().strftime('%d-%m-%Y')+"_metadata.npz"
torch.save([model.kwargs,model.state_dict()], filenamemodel) 
np.savez(filenamemetadata,lam=lam,alpha=alpha,beta=beta,delta=delta,mu=mu,Lambda=Lambda,bandwidth=np.array(bandwidth),widths=np.array(widths),k=k,dim=dim,iterations=iterations,batchsize=batchsize,s_size_approx=s_size_approx,learning_rate_Adam=learning_rate_Adam,loss_history=np.array(loss_history),eval_loss_history=np.array(eval_loss_history),seed=771994)
print("Saved model to",filenamemodel)
print("Saved metadata to",filenamemetadata)

