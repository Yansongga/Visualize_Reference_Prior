#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torchvision
import torchvision.transforms as transforms
from model import *
import torch.optim as optim
#import autograd_hacks
from utils import * 
import torch.nn.functional as F
from itertools import chain
from tqdm import trange
import time
from collections import defaultdict
import math
from torch.nn.parameter import Parameter
import itertools
import numpy as np


import numpy as np
#import torch
from matplotlib import cm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
#from visualization import ANN

args = {
    'J': 3000, #J represents number of modes in our experiments
    'x_dim': 196, 
    'h_dim': 32, 
    'dev': torch.device('cuda' ),
   # 'shot': 5, 
    'batch_size': 64,
    'test_batch_size': 1000, 
    'epochs': 2000,
    'y_dim':2,
    'target': [3,5],
    'source': [8,9],
    'beta': 0.2, 
    'order': 2, 
    'train_iteration': 1024, 
    'lr':1e-3, 
}
args['y_dim'] = len( args['source'] )
args['num_classes'] = args['y_dim']
#args['num_label'] = 1

#args['num_classes'] = len 
##transformor
transf = transforms.Compose(
        [
            transforms.Resize(14), 
            transforms.ToTensor()]
    )

#saving model path
save_models_path = './models_MNIST'
check_mkdir(save_models_path)

#saving data path
save_results_path = './results_MNIST'
check_mkdir(save_results_path)
##loading MNIST
trainset_all = torchvision.datasets.MNIST('./', transform=transf, download=True, train=True)
testset = torchvision.datasets.MNIST('./', transform=transf, download=True, train=False)

#split doamin
s_trainset, t_testset = data_split( trainset_all, args[ 'source' ], 0 ), data_split( testset, args[ 'target'], 0 )

trainset_all = torchvision.datasets.MNIST('./', transform=transf, download=True, train=True)
t_trainset = data_split( trainset_all, args[ 'target' ], 0 )

all_testset = torchvision.datasets.MNIST('./', transform=transf, download=True, train=False)
s_test = data_split( all_testset, args[ 'source'], 0 )

## train set subsets
#indices = shots(args, trainset_all, args['shot'] )    
#labelled_indices = shots(args, trainset_all, args['num_label'] )  

#labelled_indices = shots(args, trainset_all, 2 )  
#labelled_indices = labelled_indices[2:4]

#trainset = torch.utils.data.Subset(trainset_all, indices)
#trainset = trainset_all
#labelled_set = torch.utils.data.Subset(trainset_all, labelled_indices )
print(len(s_trainset), 'lenth of trainset')
print(len(t_trainset), 'lenth of trainset')
#print(len( labelled_set ), 'lenth of laebelled set')

#dataloaders
s_trainloader = torch.utils.data.DataLoader(s_trainset, batch_size= args['batch_size'],
                                          shuffle= True, num_workers=2, drop_last=True)

t_trainloader = torch.utils.data.DataLoader(t_trainset, batch_size= args['batch_size'],
                                          shuffle= True, num_workers=2, drop_last=True)


testloader = torch.utils.data.DataLoader(t_testset, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=2)

s_testloader = torch.utils.data.DataLoader(s_test, batch_size=args['test_batch_size'],
                                         shuffle=False, num_workers=2)


# In[2]:


import matplotlib.pyplot as plt

get_ipython().run_line_magic('matplotlib', 'notebook')


# In[3]:


all_shots = [1, 2, 5, 10,  500 ]
args['order'] * math.log(5)


# In[ ]:





# In[4]:


from torch.nn.functional import log_softmax, softmax
import torch.cuda.amp as amp

class WeightEMA(object):
    def __init__(self, model_param, ema_model_param, alpha=0.999):
        #self.model = model
        #self.ema_model = ema_model
        self.alpha = alpha
        #self.params = list(model.state_dict().values())
        #self.ema_params = list(ema_model.state_dict().values())
        self.params = model_param
        self.ema_params = ema_model_param
        #self.wd = 0.02 * args['lr']
        #self.wd = 0.5 * args['lr']
        self.wd = 0.

        for param, ema_param in zip(self.params, self.ema_params):
            param.data.copy_(ema_param.data)

    def step(self):
        one_minus_alpha = 1.0 - self.alpha
        for param, ema_param in zip(self.params, self.ema_params):
            if ema_param.dtype==torch.float32:
                ema_param.mul_(self.alpha)
                ema_param.add_(param.data * one_minus_alpha)
                # customized weight decay
                #param.mul_(1 - self.wd)
                param.data.mul_(1 - self.wd)
                
def train_epoch(args, optimizer, ema_optimizer, net, ema_net, s_loader, t_loader, s_shapes, all_shapes):
    scaler = amp.GradScaler(enabled=True)
    run_s_H_yx = 0. 
    run_t_H_yx = 0. 
    run_s_H_y = 0. 
    run_all_H_y = 0.
    run_entropy = 0. 
    run_sloss = 0.
    
    net.train()
    dev = args['dev']
    order = args['order']
    #s_train_iter = iter( s_loader)
    t_train_iter = iter(t_loader)
    for idx in range( args['train_iteration'] ):
    #for batch_idx in range(args.train_iteration):
        optimizer.zero_grad(set_to_none=True)
        #optimizer.zero_grad()
        
        try:
            inputs_t, targets_t = t_train_iter.next()
        except:
            t_train_iter = iter( t_loader)
            inputs_t, targets_t = t_train_iter.next()

        #try:
        #    inputs_s, targets_s = s_train_iter.next()
        #except:
       #     s_train_iter = iter( s_loader)
        #    inputs_s, targets_s = s_train_iter.next()

        #inputs_s, inputs_t = inputs_s.to(dev), inputs_t.to(dev)
        inputs_t = inputs_t.to(dev)
        batch_size = inputs_t.size(0)
        #targets_s =  targets_s.to(dev)
        #targets_s = torch.zeros(batch_size,  args['y_dim']  ).to(
        #    dev).scatter_(1, targets_s.view(-1,1).long(), 1).unsqueeze(2).repeat(1,1, args['J'] ) 
        
        with amp.autocast(enabled=True):
            #_, logits_t, _, _ = net( inputs_t, inputs_t )
            _, _, _, logits_t = net( inputs_t, inputs_t )
            
            #logits_s, logits_t, _, _ = net( inputs_s, inputs_t )
            #log_probs_s = torch.log_softmax(logits_s, dim=1)
            #source_loss = -torch.mean(torch.sum(log_probs_s * targets_s, dim=1))
            #if idx == 1:
             #   print( torch.sum(log_probs_s * targets_s, dim=1).shape )
            #run_sloss += source_loss.item()
            
            num_J = logits_t.size(2)
            logprior = torch.zeros(num_J).to( dev)
            prior = softmax( logprior, dim=0)
            
            #1) Compute entropy of particles P(Y|X, W)
            #s_ln_p_y = log_softmax(logits_s, dim=1)
            t_ln_p_y = log_softmax(logits_t, dim=1)
            all_ln_p_y = t_ln_p_y
            #all_ln_p_y = torch.cat( (s_ln_p_y, t_ln_p_y), dim =0 )  ### no permutation here
            #permu_id = torch.randperm(all_ln_p_y.size(0))
            #all_ln_p_y = all_ln_p_y[permu_id]
            
            margin = ( all_ln_p_y.exp() ).mean(0)
            entropy = (- margin * ( margin.log() )).sum(0)     
            run_entropy += entropy.mean(0).item()
            bar = 0.
            entropy_bar = entropy[entropy < bar]
            num_trivial = max( 1, len( entropy_bar ) )
            reg = entropy_bar.sum() / num_trivial
            
            #s_H_yx = torch.sum(torch.exp(s_ln_p_y) * s_ln_p_y, dim=[1]).mean(0)
            t_H_yx = torch.sum(torch.exp(t_ln_p_y) * t_ln_p_y, dim=[1]).mean(0)
            
            #s_H_yx = - torch.mean(torch.sum(prior * s_H_yx)) * order
            t_H_yx = - torch.mean(torch.sum(prior * t_H_yx))  * order
            #prior = net.get_prior() 
            
            #2) Compute other term in mutual information P(Y|X)
            bs = len(targets_t) // order
            #s_ln_p_y = torch.transpose(s_ln_p_y, 1, 2)
            all_ln_p_y = torch.transpose(all_ln_p_y, 1, 2)
            
            
            #s_ln_p_y = torch.reshape(s_ln_p_y, ( order, bs, num_J, args['y_dim']))
            #print( s_ln_p_y.shape, 'why' )
            #s_ln_p_yn = [s_ln_p_y[i].view(s_shapes[i]) for i in range(order)]
            #s_ln_p_yn = sum(s_ln_p_yn).view(bs, num_J, -1)
            #s_pi_ln_p = (s_ln_p_yn.exp() * prior.view(1, -1, 1)).sum(dim=1)
            #s_H_y = - (s_pi_ln_p * torch.log(s_pi_ln_p)).sum(1).mean()
            
            
            all_ln_p_y = torch.reshape(all_ln_p_y, ( order, bs, num_J, args['y_dim']))
            all_ln_p_yn = [all_ln_p_y[i].view(all_shapes[i]) for i in range( order)]
            all_ln_p_yn = sum(all_ln_p_yn).view(bs, num_J, -1)
            all_pi_ln_p = (all_ln_p_yn.exp() * prior.view(1, -1, 1)).sum(dim=1)
            all_H_y = -(all_pi_ln_p * torch.log(all_pi_ln_p)).sum(1).mean()
            
            #run_s_H_yx += s_H_yx.item()
            run_t_H_yx += t_H_yx.item()
            
            #run_s_H_y += s_H_y.item()
            run_all_H_y += all_H_y.item()
            
            b = args['beta']
           # loss = (t_H_yx - all_H_y) / order + source_loss- 10.* reg
            loss = (t_H_yx - all_H_y) / order - 10.* reg
            #loss = ( t_H_yx + ( 1-b ) * s_H_yx -  all_H_y + b * s_H_y ) / ( 2*order - b * order ) - 10.* reg
            #loss = source_loss
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()          
        ema_optimizer.step()
            
   # run_s_H_yx /= args['train_iteration']
    run_t_H_yx /= args['train_iteration']  
    #run_s_H_y /= args['train_iteration']
    run_all_H_y /= args['train_iteration']   
    run_entropy /= args['train_iteration']  
    run_sloss /= args['train_iteration']  
    return run_sloss, run_t_H_yx/order, run_all_H_y / (order), run_entropy    


# In[5]:


def evaluate_model(net, dataloader, dev):
    """
    Evaluate the network
    """
    acc = 0.0
    loss = 0.0
    count = 0.0
    acc2 = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    net.eval()
    # with torch.inference_mode():
    with torch.no_grad():
        for dat, labels in dataloader:
            batch_size = int(labels.size()[0])
            labels = labels.long()

            dat = dat.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            #_, out  = net(dat, dat)
            _, _, _, out  = net(dat, dat)
            #ensemble = torch.mean(out, axis=2)
            #loss += (criterion(ensemble, labels).item())

            labels = labels.cpu().numpy()
            out = out.cpu().detach().numpy()
            #ensemble = ensemble.cpu().detach().numpy()
            #acc += np.sum(labels == (np.argmax(ensemble, axis=1)))
            acc2 += np.sum(labels[:, None] == (np.argmax(out, axis=1)), axis=0)

            count += batch_size

    ret = (         
           np.round((acc2/count) * 100, 3))
    return ret


# In[6]:


def evaluate_source(net, dataloader, dev):
    """
    Evaluate the network
    """
    acc = 0.0
    loss = 0.0
    count = 0.0
    acc2 = 0.0
    criterion = nn.CrossEntropyLoss(reduction='sum')

    net.eval()
    # with torch.inference_mode():
    with torch.no_grad():
        for dat, labels in dataloader:
            batch_size = int(labels.size()[0])
            labels = labels.long()

            dat = dat.to(dev, non_blocking=True)
            labels = labels.to(dev, non_blocking=True)

            #_, out  = net(dat, dat)
            out, _, _, _  = net(dat, dat)
            #ensemble = torch.mean(out, axis=2)
            #loss += (criterion(ensemble, labels).item())

            labels = labels.cpu().numpy()
            out = out.cpu().detach().numpy()
            #ensemble = ensemble.cpu().detach().numpy()
            #acc += np.sum(labels == (np.argmax(ensemble, axis=1)))
            acc2 += np.sum(labels[:, None] == (np.argmax(out, axis=1)), axis=0)

            count += batch_size

    #ret = (         
    #       np.round((acc2/count) * 100, 3))
    #return ret
    print( 100 * (acc2/count).mean().item(), acc2.shape, 'accuracy on source task' )
    
    
def data_space( args, net, loader):
    dev = args['dev']
    net.eval()
    #all_data = []
    with torch.no_grad():
        idx =0
        for dat, labels in loader:
            batch_size = int(labels.size()[0])
            dat = dat.to(dev, non_blocking=True)
            _, _, _, out  = net(dat, dat)
            data_point = softmax(out, dim=1).sqrt()
            #data_point = data_point[:,0,:]
            #data_point = data_point.view( data_point.size(0), -1 )
            data_point = data_point.view( -1, data_point.size(2) )
            if idx == 0:
                all_data = data_point
            else:
                all_data = torch.cat( ( all_data, data_point ), dim =0 )
            idx +=1
            #all_data.append( data_point )
        
    #all_data_point = torch.stack( all_data, dim=0 )   
    return all_data


# In[7]:


#statistics
all_H, all_H_th,  all_MI, all_Obj, all_bay_accu, all_bay_loss = [], [], [], [], [], [] 
all_per = full_permute( args, args['y_dim'] )

net = TNN_trans_oct( y_dim = args['y_dim'], J = args['J'], dev = args['dev'], permu = all_per )

ema_net = TNN_trans_oct( y_dim = args['y_dim'], J = args['J'], dev = args['dev'], permu = all_per )
ema_params = [ ema_net.W1, ema_net.b1, ema_net.W2, ema_net.b2, ema_net.W3, ema_net.b3 ]
for p in ema_params:
    p.detach_()
    
params = [ net.W1, net.b1, net.W2, net.b2, net.W3, net.b3 ]

ema_optimizer = WeightEMA(params, ema_params, alpha=0.999)

#optimizer = optim.Adam(params, lr= 1e-3,  weight_decay=0.)

#loading = torch.load(
#                 './results_MNIST/MODEL_N={}.t7'.format( 
#                ( args['N'] )
#        )
 #           )
#all_p = loading['params'] 
#net.W1.data = all_p[0] +0.
#net.b1.data = all_p[1] +0.
#net.W2.data = all_p[2] +0.
#net.b2.data = all_p[3] +0.
#id_list = torch.tensor([0, 1]).to(args['dev'])
def get_shapes(order, bs, num_J, y_dim):
    shapes = []
    #bs = cfg.unlab_bs // order
    shp = [bs, num_J] + ([1] * (order))
    for i in range(order):
        shp[2+i] = y_dim
        if i != 0:
            shp[1+i] = 1
        shapes.append(list(shp))
    return shapes

num_J = args['J'] * all_per.size(0)
#num_J = args['J'] 
bs = args['batch_size'] // args['order']
s_shapes = get_shapes(args['order'], bs, num_J, args['y_dim'])
all_shapes = get_shapes( 2 * args['order'], bs, num_J, args['y_dim'])


# In[8]:


all_accu = evaluate_model(ema_net, testloader, args['dev'])
max(all_accu)


# In[9]:



sum_p2 = 0.
for pm in ema_params:
    sum_p2 += ((pm.data)**2).sum().item()
    
print(sum_p2, 'sum_p2')

optimizer = optim.SGD(params, lr= 5e-2, momentum=0.9, weight_decay=1e-5, nesterov=True)


# In[10]:


#optimizer = optim.SGD(params, lr= 1., momentum=0.9, weight_decay=0., nesterov=True)  args['lr']

#optimizer = optim.Adam(params, lr=1e-4, weight_decay=0.)
traj = []
data_point = data_space( args, ema_net, testloader)
traj.append( data_point )
for epoch in trange( 60 ): 
    #time.sleep(1)
    sloss, t_H_yx, all_H_y, entropy = train_epoch(args, optimizer, ema_optimizer, 
                                                          net, ema_net, s_trainloader, t_trainloader, s_shapes, all_shapes)
    
    if (epoch + 1)%5 == 0:
        data_point = data_space( args, ema_net, testloader)
        traj.append( data_point )
        all_accu = evaluate_model(ema_net, testloader, args['dev'])
        evaluate_source(ema_net, s_testloader, args['dev'])
        print(sloss, t_H_yx, all_H_y, entropy )
        print( max(all_accu), 'accu' )
        ##compute norm
        sum_p2 = 0.
        for pm in ema_params:
            sum_p2 += ((pm.data)**2).sum().item()
        print(sum_p2, 'sum_p2')


# In[ ]:





# In[ ]:





# In[12]:


import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sub_traj = [ traj[0], traj[1], traj[10] ]
#sub_traj = [ traj[0], traj[4] ]
my_data = torch.cat( (sub_traj), dim = 1 )
embedding = my_data.T.cpu().numpy()
print(embedding.shape)

my_pca = PCA(3)
pca_proj = my_pca.fit_transform(embedding)
#pca_proj = my_pca.fit(embedding)
print(pca_proj.shape)


# In[ ]:





# In[15]:


#saving data path
save_results_path = './results'
check_mkdir(save_results_path)

##save data
data = {
    'pca_mnist': pca_proj, 
}

torch.save( data, './results_syn/syn={}.t7'.format( 
        5.0
))


# In[ ]:





# In[4]:


###loading the indices
data = torch.load(
                 './results_syn/syn={}.t7'.format(  5.0 )
            )
#sub_indices = all_indices[ args['1_task'] ]
pca_proj = data['pca_mnist']


# In[5]:


(pca_proj[2] **2).mean()


# In[22]:


pca_proj.shape


# In[27]:


max( pca_proj[2] )


# In[28]:


min(pca_proj[2])


# In[18]:


#fig, ax = plt.subplots(figsize=(8,8))
fsz = 25
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=fsz)
plt.rc('figure', titlesize=fsz)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(projection='3d')

ax.set_xlim3d(-25., 24.)  
ax.set_ylim3d(-2, 6.5)  
ax.set_zlim3d(-2.0, 1.6) 

#ax.axes.xaxis.set_ticklabels([])
#ax.axes.yaxis.set_ticklabels([])
#ax.axes.zaxis.set_ticklabels(labelsize =16)
ax.tick_params(axis='z', labelsize= 16) 
ax.set_yticks([-1,2, 5])
ax.set_xticks([-20, 0, 20])
ax.set_zticks([-1, 1])

plt.yticks( fontsize = 16)
plt.xticks( fontsize = 16)
#plt.zticks( fontsize = 30)
#plt.axis('off')
plt.xlabel(" $p_{1}$", fontsize= 20)
#plt.xlabel(" $p_{1} $")
plt.ylabel("$ p_{2} $", fontsize= 20)
#plt.zlabel("Y Label")
ax.set_zlabel('$ p_{3} $', fontsize= 20)

#time_line = [ 'itr$= 0$', 'itr$= 1e4$',  'itr$= 5e4$']
time_line = [ 'Beginning',  'Middle', 'End']
#time_line = [ 'itr$= 0$']
alpha_list = [ 1., 0.1, 0.2 ]
for k, lab in enumerate(time_line):
    #indices = test_predictions==lab
    indices = list( range( 6000 *( k), 6000*(k+1) ) )
    ax.scatter(pca_proj[indices,0],
               pca_proj[indices,1], 
               pca_proj[indices,2],
               #c=np.array(cmap(lab)).reshape(1,4), 
               label = lab,
               marker = "o", 
               s = 5., 
               alpha=alpha_list[k],
               rasterized=True
              )
ax.legend(fontsize=20, markerscale=4)

#ax.scatter(                            
#             pca_proj[:, 0],
 #             pca_proj[:, 1],
#             pca_proj[:, 2],
               #c=np.array(cmap(lab)).reshape(1, 4),
              # label=lab,
#               marker = ".", 
 #              s = 2., 
               #alpha=0.1    
  #        )
#ax.set_xticks([])
#plt.show()
plt.tight_layout()
plt.savefig('Manifold_boundary_rasterized.pdf', bbox_inches='tight' )


# In[ ]:


tsne = TSNE(3, verbose=1)
tsne_proj = tsne.fit_transform(test_embeddings)
cmap = cm.get_cmap('tab20')
num_categories = 10
for lab in range(num_categories):
    indices = test_predictions == lab
    ax.scatter(tsne_proj[indices, 0],
               tsne_proj[indices, 1],
               tsne_proj[indices, 2],
               c=np.array(cmap(lab)).reshape(1, 4),
               label=lab,
               alpha=0.5)
ax.legend(fontsize='large', markerscale=2)
plt.show()


# In[ ]:


base = torch.einsum( 'ij,jk->ik', traj[0], traj[0].T )
e, v = torch.symeig(base, eigenvectors=True)
p123 = v[:, 3801:3804 ]

sub_traj = [ traj[0] ]
#sub_traj = [ traj[0], traj[4] ]
my_data = torch.cat( (sub_traj), dim = 1 )

embedding = torch.einsum( 'ij, ik->jk', my_data, p123 ).cpu().numpy()
pca_proj = embedding


# In[ ]:




