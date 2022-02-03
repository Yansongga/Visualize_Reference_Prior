import os, pdb, sys, json, subprocess,        time, logging, argparse,        pickle, math, gzip, numpy as np,        glob
       #pandas as pd,        
from functools import partial, reduce
from pprint import pprint
from copy import deepcopy

import  torch as th, torch.nn as nn,         torch.backends.cudnn as cudnn,         torchvision as thv,         torch.nn.functional as F,         torch.optim as optim
from torchvision import transforms
cudnn.benchmark = True
import torch

from collections import defaultdict
from torch._six import container_abcs
import torch
from copy import deepcopy
from itertools import chain
from model import *
from torch.utils.data import DataLoader
import math

#import ot
#import ot.plot
#from ot.datasets import make_1D_gauss as gauss
import random
import torchvision.models as models_t
import matplotlib.pyplot as plt  
import torch.optim as optim

from tqdm import trange
from collections import defaultdict

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import itertools


import time

def imshow(img):
    #img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# save model
def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)

#train
def train_epoch(args, network, optimizer, dl):
    network.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate( dl ):
        optimizer.zero_grad()
        data, target = data.to(args['dev']), target.to(args['dev'])
        output =  network(data)
        #loss = criterion(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        
#####
def train_syn(args, network, optimizer, data, target):
    network.train()
    criterion = nn.CrossEntropyLoss()
    #for batch_idx, (data, target) in enumerate( dl ):
    optimizer.zero_grad()
    #data, target = data.to(args['dev']), target.to(args['dev'])
    output =  network(data)
    #loss = criterion(output, target)
    loss = F.cross_entropy(output, target)
    loss.backward()
    optimizer.step()

#test function
def test(args, network, testloader):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(args['dev']), target.to(args['dev'])
            output = F.log_softmax(  network(data) )
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(testloader.dataset)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(testloader.dataset), 
        100. * correct / len(testloader.dataset)))
    
    
    
    
    
    
    
def mul_test(args, network, testloader):
    network.eval()
    test_loss = defaultdict(lambda: 0. )
    correct = defaultdict(lambda: 0 )
    J = args['J']
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(args['dev']), target.to(args['dev'])     
            output = F.log_softmax(  network(data), dim = 2 )
            for k in range(J):
                test_loss[k] += F.nll_loss(output[k], target, size_average=False).item()
                pred = output[k].data.max(1, keepdim=True)[1]
                correct[k] += pred.eq(target.data.view_as(pred)).sum()
    for k in range(J):
        test_loss[k] /= len(testloader.dataset)
        correct[k] =  100. * correct[k] / len(testloader.dataset)
    
    return test_loss, correct 

####test 
def test_syn(args, network, data, target):
    network.eval()
    test_loss = defaultdict(lambda: 0. )
    correct = defaultdict(lambda: 0 )
    J = args['J']
    with torch.no_grad():
        output = F.log_softmax(  network(data), dim = 2 )
        for k in range(J):
            test_loss[k] += F.nll_loss(output[k], target, size_average=False).item()
            pred = output[k].data.max(1, keepdim=True)[1]
            correct[k] += pred.eq(target.data.view_as(pred)).sum()
    for k in range(J):
        test_loss[k] /= len(target)
        correct[k] =  100. * correct[k] / len(target)
    
    return test_loss, correct 

  

#### zero per sample gradient
def zero_grad1(net):
    for p in net.parameters():
        p.grad1.zero_()

###
def weighted_loss_grad(net, weight):
    for p in net.parameters():
        if p.data.dim() == 2:
            p.grad.data = torch.einsum('i, ijk -> jk', weight, (p.grad1.data + 0.) ) + 0.
        if p.data.dim() == 1:
            p.grad.data = torch.einsum('i, ij -> j', weight, (p.grad1.data + 0.) ) + 0.
            
####
#split
def data_split( dataset, labels, shift):
    #dataset.targets = torch.tensor( dataset.targets )
    for k in range( len(labels) ):
        if k == 0:
            idx = dataset.targets ==  labels[k]
        else:
            idx += dataset.targets ==  labels[k] 
    dataset.targets= dataset.targets[idx]
    dataset.data = dataset.data[idx.numpy().astype(np.bool)]
    
    ####re-labelling images
    for k in range( len(labels) ):
        for i in range( len(dataset) ):
            if dataset.targets[i] == torch.tensor(labels[k]):
                dataset.targets[i] = torch.tensor( int( k + shift ) )
            #train[nk] = (x, y)      
    return dataset

##
def shots(args, dataset, num):
    indices = []
    un_indices = []
    shots = defaultdict(lambda: 0)
    finished = []
    for k, d in enumerate(dataset):
        x, y = d
        if shots[y] < num :
            indices.append( k )
            shots[y] += 1
        else:
            un_indices.append( k )
            if y not in finished:
                finished.append( y )
        #elif y not in finished:
         #   finished.append( y )
            #print( shots )
        #if len(finished) == args['y_dim']:
         #   break
    return indices, un_indices


def shots2(args, dataset, num):
    indices = []
    shots = defaultdict(lambda: 0)
    finished = []
    for k, d in enumerate(dataset):
        x, target = d
        y = int( target )
        if shots[y] < num :
            indices.append( k )
            shots[y] += 1
        elif y not in finished:
            finished.append( y )
            #print( shots )
        if len(finished) == args['y_dim']:
            break
    return indices


def compute_p_y(args, nets, dl, prior):
    #margin distrubution p(Y^M)
    outputs = []
    for data, target in dl:
        data, target = data.to(args['dev']), target.to(args['dev'])
        for k in range(args['num_theta']):
            nets[k].eval()
            with torch.no_grad():   
                output = F.softmax(  nets[k](data) )

                #这里的连续eisum是否可以优化？？
                p_y_th = output[0,:] + 0.
                for km in range(len(target) - 1):
                    p_y_th = torch.einsum('i,j->ij', p_y_th, output[km + 1,:]).view(-1)

                outputs.append(p_y_th.unsqueeze(0))

    p_y = torch.cat( outputs, 0 )
    p_y = torch.matmul( prior.T, p_y  )
    
    return p_y


def target_gen(args, real, data):
    target = []
    logits = real(data) 
    for k in range( data.size(0) ):
        pred = logits[k]
        if pred[0] > pred[1]:
            target.append(0)
        else:
            target.append(1)

    target = torch.tensor(target).to(args['dev'])
    return target


def target_trans(args, real, xs, xt):
    ys, yt = [], []
    out_s, out_t = real(xs, xt) 
    for k in range( xs.size(0) ):
        pred = out_s[k]
        if pred[0] > pred[1]:
            ys.append(0)
        else:
            ys.append(1)
            
    for k in range( xt.size(0) ):
        pred = out_t[k]
        if pred[0] > pred[1]:
            yt.append(0)
        else:
            yt.append(1)

    ys, yt = torch.tensor(ys).to(args['dev']), torch.tensor(yt).to(args['dev'])
    return ys, yt 




#test function
def test_sy(args, network, data, target):
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        #for data, target in testloader:
        #data, target = data.to(args['dev']), target.to(args['dev'])
        #output = F.log_softmax(  network(data)  )
        output = F.log_softmax(  network(data), dim =1  )
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(target)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format( 
        test_loss, correct, len(target), 
        100. * correct / len(target)))




def compute_mode(args, datasize):
    y = {}
    for i in range(args['y_dim']):
        y[i] = []
        for j in range(args['y_dim']):
            if j == i:
                y[i].append(1.)
            else:
                y[i].append(0.)

    num_ind = args['y_dim'] ** datasize

    mode = [] 
    for k in range( num_ind ):
        k1 = int(k + 0 )
        targets = []
        for idx in range( datasize ):
            dig = k1 % args['y_dim']  
            k1 = int( (k1 - dig)/ args['y_dim']   )
            targets.append( y[dig] )
        mode.append(targets)

    mode = torch.tensor(mode).to(args['dev'])
    return mode



def bayesian_test(args, net, data, target, test_data, test_target):
    net.eval()
    posterior = []
    with torch.no_grad():
        output = F.log_softmax(  net(data), dim = 2 )

        for k in range( output.size(0) ):
            log_p = - F.nll_loss(output[k], target, size_average=False).item()
            posterior.append(log_p)
        #beta = 2. 
        posterior = torch.tensor(posterior, device = args['dev']  ) 
        posterior = posterior.exp()
        #posterior = posterior ** beta
        posterior = posterior / posterior.sum()
    
        output1 = F.softmax(  net(test_data), dim = 2 )
        output = torch.einsum('i,ijk->jk', posterior, output1.data) + 0. 
        loss = F.nll_loss(output.log(), test_target, size_average=True).item()
        pred = (output.log()).data.max(1, keepdim=True)[1]
        correct = pred.eq(test_target.data.view_as(pred)).sum()
        #print(correct)
    return loss, 100. * correct/ len(test_target)


def bayesian_trans_test(args, net, out_s, out_t, ys, yt, test_data, test_target):
    net.eval()
    posterior = []
    with torch.no_grad():
        #output = F.log_softmax(  net(data), dim = 2 )

        for k in range( out_s.size(0) ):
            log_ps = - F.nll_loss(out_s[k], ys, size_average=False).item()
            log_pt = - F.nll_loss(out_t[k], yt, size_average=False).item()
            log_p = log_ps + log_pt
            posterior.append( log_p )
        #beta = 2. 
        posterior = torch.tensor(posterior, device = args['dev']  ) 
        posterior = posterior.exp()
        #posterior = posterior ** beta
        posterior = posterior / posterior.sum()
        
        _, output2 = net(test_data, test_data)
        output1 = F.softmax(  output2, dim = 2 )
        output = torch.einsum('i,ijk->jk', posterior, output1.data) + 0. 
        loss = F.nll_loss(output.log(), test_target, size_average=True).item()
        pred = (output.log()).data.max(1, keepdim=True)[1]
        correct = pred.eq(test_target.data.view_as(pred)).sum()
        
        loss2 = 0. 
        correct2 = 0
        output2 = F.log_softmax(  output2, dim = 2 )
        for k in range( output2.size(0) ):
            loss_k = F.nll_loss(output2[k], test_target, size_average=True).item()
            pred2 = (output2[k]).data.max(1, keepdim=True)[1]
            correct_k = pred2.eq(test_target.data.view_as(pred2)).sum()
            
            loss2 +=( loss_k * posterior[k]).item()
            correct2 += posterior[k] * 1.0 * correct_k
        #print(correct)
    return loss, 100. * correct/ len(test_target), loss2,  100. * correct2/  len(test_target)


def bayesian_test_MNIST_old(args, net, data, target, testloader):
    net.eval()
    posterior = []
    with torch.no_grad():
        val_output = F.log_softmax(  net(data), dim = 2 )
        max_k = 0
        k_accu = 0 
        for k in range( val_output.size(0) ):
            val_pred = (val_output[k]).data.max(1, keepdim=True)[1]
            val_correct = val_pred.eq(target.data.view_as(val_pred)).sum()
            
            log_p = - F.nll_loss(val_output[k], target, size_average=False).item() 
            posterior.append(log_p)
            if k == 0:
                likeli = log_p
                max_accu = val_correct + 0
            if log_p > likeli:
                max_k = k 
                likeli = max(likeli, log_p)
            if val_correct > max_accu:
                k_accu = k 
                max_accu = val_correct + 0
        #beta = 2. 
        posterior = torch.tensor(posterior, device = args['dev']  ) 
        mm = posterior.mean() + 0.
        posterior += mm
        posterior = posterior.exp()
        #posterior = posterior ** beta
        posterior = posterior / posterior.sum()
    
        #output1 = F.softmax(  net(test_data), dim = 2 )
        #output = torch.einsum('i,ijk->jk', posterior, output1.data) + 0. 
        #loss = F.nll_loss(output.log(), test_target, size_average=True).item()
        #pred = (output.log()).data.max(1, keepdim=True)[1]
        #correct = pred.eq(test_target.data.view_as(pred)).sum()
        
        net.eval()
        test_loss = 0.
        correct, correct_max, correct_accu = 0, 0, 0
        #test_loss = defaultdict(lambda: 0. )
        #correct = defaultdict(lambda: 0 )
        
        J = args['J']
        with torch.no_grad():
            for test_data, test_target in testloader:
                test_data, test_target = test_data.to(args['dev']), test_target.to(args['dev'])     
                #output1 = F.log_softmax(  network(data), dim = 2 )
                output1 = F.softmax(  net( test_data), dim = 2 )
                output = torch.einsum('i,ijk->jk', posterior, output1.data) + 0. 
                test_loss += F.nll_loss(output.log(), test_target, size_average=False).item()
                pred = (output.log()).data.max(1, keepdim=True)[1]
                correct += pred.eq(test_target.data.view_as(pred)).sum()
                
                pred_max = (output1[max_k].log()).data.max(1, keepdim=True)[1]
                correct_max += pred_max.eq(test_target.data.view_as(pred_max)).sum()     
                
                pred_accu = (output1[k_accu].log()).data.max(1, keepdim=True)[1]
                correct_accu += pred_accu.eq(test_target.data.view_as(pred_accu)).sum()
        #for k in range(J):
        test_loss /= len(testloader.dataset)
        #correct =  100. * correct / len(testloader.dataset)
             
    return test_loss, 100. * correct/ len(testloader.dataset), 100. * correct_max/ len(testloader.dataset), 100. * correct_accu/ len(testloader.dataset)


def train_TNN(args, net, dl):
    net.train()
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, target) in enumerate( dl ):
        #optimizer.zero_grad()
        data, target = data.to(args['dev']), target.to(args['dev'])
        output =  net(data)
        #loss = criterion(output, target)
        loss = F.cross_entropy(output, target)
        loss.backward()
        #optimizer.step()
        dW1, db1 = net.W1.grad.data, net.b1.grad.data
        dW2, db2 = net.W2.grad.data, net.b2.grad.data
        
        lr = args['lr']
        net.W1.data -= lr * dW1  
        #/torch.norm(dW1)
        net.b1.data -= lr * db1 
        #/torch.norm(db1)
        net.W2.data -= lr * dW2 
        #/torch.norm(dW2)
        net.b2.data -= lr * db2 
        
        net.W1.grad = None
        net.b1.grad = None
        net.W2.grad = None
        net.b2.grad = None


def pca( args, output, output1 = None, plot0 = True ):
    u, s, _ = torch.svd( output.T )
    e1, e2 = u[:, 0], u[:, 1]
    e = torch.cat((e1.unsqueeze(0), e2.unsqueeze(0)), 0)
    cd = torch.einsum('ik,jk->ij', output, e)
    x, y = cd[:, 0].cpu().numpy(), cd[:, 1].cpu().numpy()

    # convert to pandas dataframe
    d = {'x': x, 'y': y }
    pdnumsqr = pd.DataFrame(d)
    if output1 is not None:
        cd1 = torch.einsum('ik,jk->ij', output1, e)
        x1, y1 = cd1[:, 0].cpu().numpy(), cd1[:, 1].cpu().numpy()
        d1 = {'x': x1, 'y': y1 }
        pdnumsqr1 = pd.DataFrame(d1)
        
    sns.set(style='darkgrid')
    
    if plot0 == True:
        sns.scatterplot(x='x', y='y', data=pdnumsqr)     
    else:
        sns.scatterplot(x='x', y='y', data=pdnumsqr1)     
    
###coordinate
def coordinate( args, output, output1 = None, plot0 = True ):
    u, s, _ = torch.svd( output.T )
    e1, e2 = u[:, 0], u[:, 1]
    e = torch.cat((e1.unsqueeze(0), e2.unsqueeze(0)), 0)
    cd = torch.einsum('ik,jk->ij', output, e)
    x, y = cd[:, 0].cpu().numpy(), cd[:, 1].cpu().numpy()

    # convert to pandas dataframe
    d = {'x': x, 'y': y }
    pdnumsqr = pd.DataFrame(d)
    if output1 is not None:
        cd1 = torch.einsum('ik,jk->ij', output1, e)
        x1, y1 = cd1[:, 0].cpu().numpy(), cd1[:, 1].cpu().numpy()
        d1 = {'x': x1, 'y': y1 }
        pdnumsqr1 = pd.DataFrame(d1)
        
    
    if plot0 == True: 
        return x, y   
    else:
        return x1, y1
    
    
def full_permute( args, num_classes ):
    all_label = list(range(num_classes))
    all_permute = list(itertools.permutations( all_label ))
    
    all_id = torch.tensor( list(range(num_classes)) ).to(args['dev'])
    for k, per in enumerate(all_permute):
        s = torch.tensor( list(per) ).to(args['dev'])
        per_matrix = ( all_id.unsqueeze(1) ==s.unsqueeze(0) ).float().unsqueeze(0)
        
        if k == 0:
            all_per = per_matrix + 0. 
        else:
            all_per = torch.cat( (all_per, per_matrix), 0 )
    return all_per




def shots_binary(binary_task, dataset, n1, n0):
    ###there are 10 classes in a domain. For Binary classification, we sample n1 images for task-1 classes, we sample n0 images for each task_0 class. 
    indices = []
    shots = defaultdict(lambda: 0)
    finished = []
    for k, d in enumerate(dataset):
        x, y = d
        y = int(y + 0 )
        if y not in finished:
            if y == binary_task:
                if shots[y] < n1:
                    indices.append( k )
                    shots[y] += 1
                else:            
                    finished.append( y )
            else:
                if shots[y] < n0 :
                    indices.append( k )
                    shots[y] += 1
                else:            
                    finished.append( y )
   
        if len(finished) == 10:
            break
    return indices



def shots_binary2(dataset, n1, n0):
    ###there are 10 classes in a domain. For Binary classification, we sample n1 images for task-1 classes, we sample n0 images for each task_0 class. 
    indices = []
    shots = defaultdict(lambda: 0)
    finished = []
    
    all_id = list(range( len(dataset) ))
    random.shuffle(all_id)
    
    #for k, d in enumerate(dataset):
    for k in all_id:
        _, y = dataset[k]
        y = int(y + 0 )
        if y not in finished:
            if y == 1:
                if shots[y] < n1:
                    indices.append( k )
                    shots[y] += 1
                else:            
                    finished.append( y )
            else:
                if shots[y] < n0 :
                    indices.append( k )
                    shots[y] += 1
                else:            
                    finished.append( y )
   
        if len(finished) == 2:
            break
    return indices



def data_split_binary( Binary_task, dataset, labels):
    dataset.targets = torch.tensor( dataset.targets )
    for k in range( len(labels) ):
        if k == 0:
            idx = dataset.targets == labels[k]
        else:
            idx += dataset.targets == labels[k]
    dataset.targets= dataset.targets[idx]
    dataset.data = dataset.data[idx.numpy().astype(np.bool)]
    
    ####re-labelling images
    for i in range( len(dataset) ):
        if dataset.targets[i] == torch.tensor( Binary_task ):
            dataset.targets[i] = torch.tensor( 1 )
        else:
            dataset.targets[i] = torch.tensor( 0 )
            #train[nk] = (x, y)      
    return dataset



def bayesian_test_MNIST2(args, net, data, target, testloader):
    net.eval()
    posterior = []
    with torch.no_grad():
        val_output = F.log_softmax(  net(data), dim = 2 )
        max_k = 0
        k_accu = 0 
        max_accu = 0
        for k in range( val_output.size(0) ):
            val_pred = (val_output[k]).data.max(1, keepdim=True)[1]
            val_correct = val_pred.eq(target.data.view_as(val_pred)).sum().item()
            
            log_p = - F.nll_loss(val_output[k], target, size_average=False).item() 
            posterior.append(log_p)
            if k == 0:
                likeli = log_p
            if log_p > likeli:
                max_k = k 
                likeli = max(likeli, log_p)
            if val_correct > max_accu:
                k_accu = k 
                max_accu = max( val_correct, max_accu )
           # if (k+1) %200 == 0:
           #     print( val_correct,  max_accu)

        posterior = torch.tensor(posterior, device = args['dev']  ) 
        max_log = torch.max( posterior ).item() + 0. 
        posterior = posterior - max_log + 50. 
        posterior_logits = posterior+ 0.
        
        
        #log_prob = posterior + 0. 
        #mean_log = posterior.mean() + 0.
        #posterior -= mean_log
       # bar = 30. 
        #log_probability = posterior - ( posterior - bar ).relu() + ( - bar - posterior ).relu()
        #log_probability = posterior - ( posterior - bar ).relu() 
        
        posterior = posterior.exp()
        #posterior = posterior ** beta
        posterior = posterior / posterior.sum()
        print( posterior.sum().item(), 'Is this a well-defined measure?', max_log)
      #  print( max_accu, 'max_accu', k_accu, 'k_accu', max_k, 'max_k')
    
        #output1 = F.softmax(  net(test_data), dim = 2 )
        #output = torch.einsum('i,ijk->jk', posterior, output1.data) + 0. 
        #loss = F.nll_loss(output.log(), test_target, size_average=True).item()
        #pred = (output.log()).data.max(1, keepdim=True)[1]
        #correct = pred.eq(test_target.data.view_as(pred)).sum()
        
        net.eval()
        test_loss = 0.
        correct, correct_max, correct_accu = 0, 0, 0
        #test_loss = defaultdict(lambda: 0. )
        #correct = defaultdict(lambda: 0 )
        
      #  J = args['J']
        with torch.no_grad():
            for test_data, test_target in testloader:
                test_data, test_target = test_data.to(args['dev']), test_target.to(args['dev'])     
                #output1 = F.log_softmax(  network(data), dim = 2 )
                output1 = F.softmax(  net( test_data), dim = 2 )
                output = torch.einsum('i,ijk->jk', posterior, output1.data) + 0. 
                test_loss += F.nll_loss(output.log(), test_target, size_average=False).item()
                pred = (output.log()).data.max(1, keepdim=True)[1]
                correct += pred.eq(test_target.data.view_as(pred)).sum()
                
                pred_max = (output1[max_k].log()).data.max(1, keepdim=True)[1]
                correct_max += pred_max.eq(test_target.data.view_as(pred_max)).sum()     
                
                pred_accu = (output1[k_accu].log()).data.max(1, keepdim=True)[1]
                correct_accu += pred_accu.eq(test_target.data.view_as(pred_accu)).sum()
        #for k in range(J):
        test_loss /= len(testloader.dataset)
        #correct =  100. * correct / len(testloader.dataset)
             
    return  posterior_logits, 100. * correct/ len(testloader.dataset), 100. * correct_max/ len(testloader.dataset), 100. * correct_accu/ len(testloader.dataset)


def data_split_CIFAR( dataset, labels, shift):
    dataset.targets = torch.tensor( dataset.targets )
    for k in range( len(labels) ):
        if k == 0:
            idx = dataset.targets == labels[k]
        else:
            idx += dataset.targets == labels[k]
    #print(idx)
    dataset.targets= dataset.targets[idx]
    dataset.data = dataset.data[idx.numpy().astype(np.bool)]
    
    ####re-labelling images
    for k in range( len(labels) ):
        for i in range( len(dataset) ):
            if dataset.targets[i] == torch.tensor(labels[k]):
                dataset.targets[i] = torch.tensor( int( k + shift ) )
            #train[nk] = (x, y)      
    return dataset