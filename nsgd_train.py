import torch
import csv
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
from dataset import AgeDataset
from utils import *
from torchvision.models import resnet18
from pydrsom.pydrsom.drsom import DRSOMB as DRSOM
from pydrsom.pydrsom.drsom_utils import *
import matplotlib.pyplot as plt


def make_args():
    parser = argparse.ArgumentParser(description='argument parser')
    parser.add_argument("--optim",
                    required=False,
                    type=str,
                    default='nsgd',
                    choices=[
                      'sgd',
                      'nsgd',
                    ])
    parser.add_argument('--epoch',default=15,type=int)
    parser.add_argument('--epoch_list', default='50,80,100', type=str)
    parser.add_argument('--batch_size',default=32,type=int)
    parser.add_argument('--train_path',default='/nfsshare/home/lichenxi/DRO/AFAD-Full/AFAD-Full')
    parser.add_argument('--val_path',default='/nfsshare/home/lichenxi/DRO/AFAD-Full/AFAD-Full')
    parser.add_argument('--trained_model',default=None,help='the path to the saved trained model')
    parser.add_argument('--lr',default=0.1,type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    # parser.add_argument('--gamma', default=1e-5, type=float)
    parser.add_argument('--wd', default=1e-4, type=float)  # WEIGHT DECAY   
    parser.add_argument('--save_path',default='/nfsshare/home/lichenxi/DRO/results')
    parser.add_argument('--out_dim',default=1)
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

        predict = model(x)
        predict_real=predict*75
        loss = loss_func(predict,label,importance,0.1).to(device)
        loss.float()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predict_real=predict*75
        mae = MAE(predict_real,age)
        avg_loss+=loss.item()
        avg_mae+=mae
        if step % 100 == 0:
            print('training || loss:{:.7f} MAE:{:.5f} [{}/{}]'.format(loss.item(),mae,len(x)*(step+1),total))
        if step>amount:
            avg_loss=avg_loss/amount
            avg_mae=avg_mae/amount
            break
    return avg_loss,avg_mae


def val_loop(model,loader,device):
    total = len(loader.dataset)
    amount=20
    mae = 0
    for step,batch in enumerate(loader):
        if step < amount:
            x,label,age = batch
            x = x.to(device)
            label = label.to(device)
            age = age.to(device)
            predict = model(x)
            mae += MAE(predict,age)*len(age)
        else:
            break
    mae = mae/(amount*len(age))
    print('validate|| MAE:{:.5f}'.format(mae))
    return mae

# resnet model
def get_model(out_dim):
    model=resnet18(pretrained=False)
    fc_features=model.fc.in_features
    model.fc=nn.Linear(fc_features,out_dim)
    return model


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

    model = get_model(args.out_dim)
    if args.trained_model is not None:
        dict = torch.load(args.trained_model)
        model.load_state_dict(dict)
        print('model loaded successfully!')
    else:
        print('train from scratch!')
    model.to(device)

    train_dataset = AgeDataset(args.train_path,train=True)
    val_dataset = AgeDataset(args.val_path)

    train_loader = DataLoader(train_dataset,batch_size=args.batch_size,shuffle=False)
    val_loader = DataLoader(val_dataset,batch_size=args.batch_size)

    optimizer = parse_algo(args.optim, model, wd=args.wd,
                           lr=args.lr, momentum=args.momentum)

    importance = make_task_importance(args.train_path)

    best_MAE = 72. - 15. 
    is_best = 0

    epoch_list = [int(epoch) for epoch in args.epoch_list.split(',')]

    all_avg_loss=[];all_avg_mae=[];all_val_mae=[]
    for i in range(args.epoch):
        # print('-----------------------epoch {}-----------------------'.format(i+1))
        # if args.optim=='sgd':
        #    print('-----------current learning rate: {:.6f}-----------'.format(optimizer.state_dict()['param_groups'][0]['lr']))
        
        adjust_lr(args.lr, epoch_list, i, optimizer)
        model.train()
        avg_loss,avg_mae=train_loop(model,train_loader,optimizer,DRO_MSE,device,importance)
        # print(optimizer.param_groups[0]['lr'])

        save_model(model,args,'epoch_{}.pth'.format(i+1),is_best)

        is_best = 0

        all_avg_loss.append(avg_loss)
        all_avg_mae.append(avg_mae.item())
        # all_avg_mae.append(avg_mae)
        # all_val_mae.append(mae_val.item())

    header = ['all_avg_loss', 'all_avg_mae', 'all_val_mae']
    
    file_name = args.optim + '_lr_'+str(args.lr)+'_clipping.csv'
    with open(file_name, 'w', encoding='utf-8', newline='') as file_obj:
    # 创建writer对象
        writer = csv.writer(file_obj)
        # 写表头
        writer.writerow(header)
        # 一次写入多行
        for i in range(args.epoch):
            writer.writerow((all_avg_loss[i],all_avg_mae[i]))


if __name__ == '__main__':
    args = make_args()
    main(args)