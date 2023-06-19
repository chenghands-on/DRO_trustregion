import torch
import time
import glob
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Subset


def sample_idx(max_num, split=np.array((7, 2, 1,)), is_val=False, random_seed=42):
    np.random.seed(random_seed)
    train_idx, test_idx, val_idx = None, None, None
    split = split / np.sum(split)
    split_idx = np.zeros_like(split, dtype=int)
    subset_idx = np.array([idx for idx in range(max_num)])
    np.random.shuffle(subset_idx)
    for idx, weight in enumerate(split):
        split_idx[idx] = int(max_num * weight)
    if is_val:
        train_idx, test_idx, val_idx = subset_idx[0:split_idx[0]], \
                                       subset_idx[split_idx[0]:split_idx[0] + split_idx[1]], \
                                       subset_idx[split_idx[0] + split_idx[1]:]
    else:
        train_idx, test_idx = subset_idx[0:split_idx[0]], \
                              subset_idx[split_idx[0]:]
    return train_idx, test_idx, val_idx


def split_dataset(full_dataset, split=np.array((7, 2, 1,)), is_val=False):
    train_dataset = None
    test_dataset = None
    val_dataset = None
    train_idx, test_idx, val_idx = sample_idx(max_num=full_dataset.__len__(), split=split, is_val=is_val)
    if not is_val:
        train_dataset = Subset(full_dataset, train_idx)
        train_dataset.dataset.is_train = True
        train_dataset.dataset.Image_Transform()
        test_dataset = Subset(full_dataset, test_idx)
        test_dataset.dataset.is_train = False
        test_dataset.dataset.Image_Transform()
    else:
        train_dataset = Subset(full_dataset, train_idx)
        train_dataset.dataset.is_train = True
        train_dataset.dataset.Image_Transform()
        test_dataset = Subset(full_dataset, test_idx)
        test_dataset.dataset.is_train = False
        test_dataset.dataset.Image_Transform()
        val_dataset = Subset(full_dataset, val_idx)
        val_dataset.dataset.is_train = False
        val_dataset.dataset.Image_Transform()
    return train_dataset, test_dataset, val_dataset



def save_model(model,args,path,is_best):
    torch.save(model.state_dict(),args.save_path+path)
    if is_best:
        torch.save(model.state_dict(),args.save_path+'best.pth')

def MAE(predict,age):
    # predict[predict>=0.5] = 1
    # predict[predict<0.5] = 0
    # predict_age = torch.sum(predict,dim=1)[:,0] + 15
    # abs_error = torch.sum(torch.abs(predict_age - age))
    abs_error=torch.sum(torch.abs(predict-age.reshape(-1,1)))
    # print(torch.abs(predict-age.reshape(-1,1)).shape)
    mean_abs_error = abs_error/len(predict)
    return mean_abs_error

def make_task_importance(data_path):
    lambda_t = []
    age_list = glob.glob(data_path+'/*/*')
    for age in age_list:
        # print(age)
        temp_list = glob.glob(age+'/*')
        lambda_t.append(len(temp_list))
    lambda_t = np.sqrt(lambda_t)
    summary = np.sum(lambda_t)
    lambda_t = lambda_t / summary
    fin_lambda_t=[]
    for i in range(len(lambda_t)):
        if 2*i <len(lambda_t)+1:
            tem =(lambda_t[i]+lambda_t[i+1])/2
            fin_lambda_t.append(tem)
    return torch.tensor(tem)

def importance_cross_entropy(predict,label,importance):
    predict = torch.log(predict)
    entropy = torch.sum(-1*predict*label,dim=2)
    entropy = entropy * importance
    loss = torch.sum(entropy) / label.shape[0]
    return loss

def get_eta(inner_loss,lbda,init_eta=0.002,lr=0.01):
    eta=init_eta
    iter=0
    # objective=lbda*(-1+1/4*(max(0,inner_loss/lbda-eta+2))^2)+eta
    while abs(1-(max(0,(inner_loss-eta)/lbda+2))/2) >1e-5 and iter <1000:
        gradient=1-(max(0,(inner_loss-eta)/lbda+2))/2
        eta=eta-lr*gradient
        # print('balba'+str(iter))
        iter+=1
    return eta
def DRO_cross_entropy(predict,label,importance,lbda):
    predict = torch.log(predict)
    entropy = torch.sum(-1*predict*label,dim=2)
    entropy = entropy * importance
    inner_loss = torch.sum(entropy) / label.shape[0]
    eta=get_eta(inner_loss,lbda,0,0.01)
    loss=lbda*(-1+1/4*(max(0,(inner_loss-eta)/lbda+2))^2)+eta
    return loss

def DRO_MSE(predict,label,importance,lbda):
    # predict = torch.log(predict)
    # entropy = torch.sum(-1*predict*label,dim=2)
    # entropy = entropy * importance
    inner_loss = F.mse_loss(predict,label)/len(label)
    eta=get_eta(inner_loss,lbda,0,0.01)
    if (inner_loss-eta)/lbda+2>0:
        loss=torch.tensor(lbda)*(-1+1/4*torch.pow((inner_loss-torch.tensor(eta))/torch.tensor(lbda)+2,2))+torch.tensor(eta)
    else:loss=torch.tensor(lbda*(-1)+eta)

    return loss
    
def make_label(age):
    batch_size = age.shape[0]
    k = torch.tensor(torch.arange(15,72).repeat(batch_size).reshape(batch_size,-1))
    label = k-age.reshape(batch_size,-1)
    label[label>=0], label[label<0] = 0, 1
    label = label.reshape(batch_size,72-15,1)
    true = torch.cat((label,1-label),dim=2)
    return true