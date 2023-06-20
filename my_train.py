## This is the only py I use. Another .py I revise is dataset.py; Also utils.py, plot.py

import torch
import csv
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
import argparse
from dataset import AgeDataset
# from network import AgeNet
from utils import *
from torchvision.models import resnet18
from pydrsom.pydrsom.drsom import DRSOMB as DRSOM
from pydrsom.pydrsom.drsom_utils import *
import matplotlib.pyplot as plt
import glob

def make_args():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument("--optim",
                    required=False,
                    type=str,
                    default='nsgd',
                    choices=[
                      'adam',
                      'sgd',
                      'drsom',
                      'nsgd'
                    ])
    parser.add_argument('--epoch',default=100,type=int)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--train_path',default=r'/nfsshare/home/xiechenghan/DRO_trustregion/tarball/AFAD-Full')
    parser.add_argument('--val_path',default=r'/nfsshare/home/xiechenghan/DRO_trustregion/tarball/AFAD-Full')
    parser.add_argument('--trained_model',default=None,help='the path to the saved trained model')
    parser.add_argument('--lr',default=1e-1,type=float)
    parser.add_argument('--save_path',default=r'/nfsshare/home/xiechenghan/DRO_trustregion/results/620')
    parser.add_argument('--out_dim',default=1)
    ## penalty size
    parser.add_argument('--gamma',default=1e-3)
    parser.add_argument('--momentum', default=0.9, type=float)
    # parser.add_argument('--gamma', default=1e-5, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)  # WEIGHT DECAY  
    parser.add_argument('--epoch_list', default='50,80,100', type=str) 
    add_parser_options(parser)
    args = parser.parse_args()
    return args

def train_loop(model,loader,optimizer,loss_func,device,importance):
    total = len(loader.dataset)
    importance = importance.to(device)
    avg_loss=0;amount=100;avg_mae=0
    for step,batch in enumerate(loader):
        x,label,age = batch
        x = x.to(device)
        label = label.to(device)
        age = age.to(device)
        def closure(backward=True):
            optimizer.zero_grad()
            predict = model(x)
            loss = loss_func(predict, label,importance,lbda=0.1)
            loss.float()
            if not backward:
                return loss
            if DRSOM_MODE_QP == 0 or DRSOM_VERBOSE == 1:
                # only need for hvp
                loss.backward(create_graph=True)
            else:
                loss.backward()
            return loss, predict   
        if args.optim=='drsom':
            loss,predict= optimizer.step(closure=closure)
        else:
            predict = model(x)
            loss = loss_func(predict,label,importance,lbda=0.1).to(device)
            loss.float()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        ## 计算均方差的时候用真是年龄而不是以前的年龄。
        predict_real=predict*75
        mae = MAE(predict_real,age)
        avg_loss+=loss.item()
        avg_mae+=mae
        if step % 100 == 0:
            print('training || loss:{:.7f} MAE:{:.5f} [{}/{}]'.format(loss.item(),mae,len(x)*(step+1),total))
        ## 如果算力不够
        # if step>amount:
        #     avg_loss=avg_loss/amount
        #     avg_mae=avg_mae/amount
        #     break
    avg_loss=avg_loss/total
    avg_mae=avg_mae/amount
    return avg_loss,avg_mae

def val_loop(model,loader,device):
    total = len(loader.dataset)
    amount=20
    mae = 0
    for step,batch in enumerate(loader):
        # if step < amount:
        x,label,age = batch
        x = x.to(device)
        label = label.to(device)
        age = age.to(device)
        predict = model(x)
        # mae += MAE(predict,age)*len(age)
        mae+=MAE(predict,age)
        # else:
        #     break
    # mae = mae/(total*len(age))
    mae=mae/total
    print('validate|| MAE:{:.5f}'.format(mae))
    return mae
# resnet model
def get_model(out_dim):
    model=resnet18(pretrained=False)
    fc_features=model.fc.in_features
    model.fc=nn.Linear(fc_features,out_dim)
    return model


## SGD and NSGD
def parse_algo(args, model, **kwargs):
    """
        args: a string containing algorithm type and paras
            sgd
            sgd_clip([layer])
            normalized_sgd([layer])
            adagrad([element], [layer])
            qhm_clip([layer])
        Note: if [layer], grad normalization is applied layerwise.
            Here every submodel with direct parameters is considered a layer.
    """
    from algorithm import Algorithm, SGD, NormalizedSGD
    net_paras = model.parameters()
    
    if 'normalized_sgd' in args.lower():
        algo = NormalizedSGD
        para = ('wd', 'lr', 'momentum')
    elif 'sgd' in args.lower():
        algo = SGD
        para = ('wd', 'lr', 'momentum')
    else:
        raise NotImplementedError
    return Algorithm(net_paras, algo, **{key: kwargs[key] for key in para})


def adjust_lr(lr, epochs, epoch, optimizer):
    import bisect
    index = bisect.bisect_right(epochs, epoch)
    lr_now = lr / (10 ** index)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_now

def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Using [{device}] for the work')
    
    # torch.cuda.manual_seed_all(2000)
    model = get_model(args.out_dim)
    if args.trained_model is not None:
        dict = torch.load(args.trained_model)
        model.load_state_dict(dict)
        print('model loaded successfully!')
    else:
        print('train from scratch!')
    model.to(device)

    

# 该字符串应匹配你的数据目录结构，'*'代表任意字符或目录
    img_paths = glob.glob('/nfsshare/home/xiechenghan/DRO_trustregion/tarball/AFAD-Full/*/*/*.jpg')

    # 然后，你可以根据每个路径生成对应的年龄标签
    ages = [int(path.split('/')[-3]) for path in img_paths]

    # 这里，img_paths, ages 是所有图像的路径和对应的年龄
    train_img_paths, val_img_paths, train_ages, val_ages = train_test_split(img_paths, ages, test_size=0.2, stratify=ages)

    train_dataset = AgeDataset(train_img_paths, train_ages, train=True)
    val_dataset = AgeDataset(val_img_paths, val_ages, train=False)

    # train_dataset = AgeDataset(args.train_path,train=True)
    # val_dataset = AgeDataset(args.val_path)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size)

    if args.optim=='drsom':
        func_kwargs=render_args(args)
        optimizer = DRSOM(model.parameters(),gamma=args.gamma,**func_kwargs)
    else:
        # optimizer = torch.optim.SGD(model.parameters(),lr=args.lr,weight_decay=0.0001,momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,30,gamma=0.5,last_epoch=-1,verbose=False)
        optimizer = parse_algo(args.optim, model, wd=args.wd,
                           lr=args.lr, momentum=args.momentum)
        
    epoch_list = [int(epoch) for epoch in args.epoch_list.split(',')]
    importance = make_task_importance(args.train_path)

    best_MAE = 72. - 15. 
    is_best = 0
    all_avg_loss=[];all_avg_mae=[];all_val_mae=[]
    for i in range(args.epoch):
        print('-----------------------epoch {}-----------------------'.format(i+1))
        if args.optim=='drsom':
            print('-----------current methods: drsom, initial region:{:.6f}-----------.'.format(args.gamma))
        if args.optim=='sgd':
            adjust_lr(args.lr, epoch_list, i, optimizer)
            print('-----------current methods: sgd, learning rate: {:.6f}-----------'.format(args.lr))
        if args.optim=='nsgd':
            adjust_lr(args.lr, epoch_list, i, optimizer)
            print('-----------current methods: nsgd, learning rate: {:.6f}-----------'.format(args.lr))
        model.train()
        train_loss,train_mae=train_loop(model,train_loader,optimizer,DRO_MSE,device,importance)
        with torch.no_grad():
            model.eval()
            val_mae = val_loop(model,val_loader,device)
        if val_mae < best_MAE:
            best_MAE = val_mae
            is_best = 1
        save_model(model,args,'epoch_{}.pth'.format(i+1),is_best)
        # if (i+1) % 5 == 0 and not is_best:
        #     dict = torch.load('E:\PKU\cv_learning\ordinal-regression\model\\best.pth')
        #     model.load_state_dict(dict)
        #     print('early stop and go back')
        # if args.optim=='sgd':
        #     scheduler.step()
        is_best = 0
        all_avg_loss.append(train_loss)
        all_avg_mae.append(val_mae.item())
        # all_val_mae.append(mae_val.item())
    header = ['all_train_loss', 'all_val_mae']
    # data=[all_avg_loss,all_avg_mae,all_val_mae]
    with open(args.save_path+f'/{args.optim}.csv', 'w', encoding='utf-8', newline='') as file_obj:
    # 创建writer对象
        writer = csv.writer(file_obj)
        # 写表头
        writer.writerow(header)
        # 一次写入多行
        for i in range(args.epoch):
            writer.writerow((all_avg_loss[i],all_avg_mae[i]))

    # plt.figure(dpi=300,figsize=(8,4))
    # plt.title('Result on PointENV')
    # plt.plot(np.arange(1, epsoids + 1), vpg_1[col_name], label='pure pg, a=1e-2')
    # plt.plot(np.arange(1, epsoids + 1), vpg_2[col_name], label='pure pg, a=1e-1')
    # plt.legend()
    # plt.ylabel('AVGReturn')
    # plt.xlabel('Episode #')
    # plt.savefig('PointENV_result.jpg')
    # plt.show()
        


if __name__ == '__main__':
    args = make_args()
    main(args)