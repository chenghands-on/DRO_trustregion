import torch
import time
import glob
import numpy as np
import torch.nn.functional as F

def save_model(model,args,path,is_best):
    # torch.save(model.state_dict(),args.save_path+path)
    if is_best:
        torch.save(model.state_dict(),args.save_path+'best.pth')

def MAE(predict,age):
    '''
    Calculate mean_abs_error
    '''
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
    '''
    The obejctive function used in DRO-regressive
    '''
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