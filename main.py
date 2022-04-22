'''
程序主文件
'''


from load_data import unetdataset
from unet_model import Unet
import torch
import os 
import time
from torch.backends import cudnn
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

def plot(data):
    fig, ax = plt.subplots()
    iteration = len(data)
    x = np.linspace(1,iteration,num=iteration,endpoint=True)
    ax.plot(x,data)
    ax.set(xlabel=r"Iterations",ylabel='RMSE(Gy)')
    ax.ticklabel_format(style='sci',scilimits=[0,0],axis='x')
    ax.grid()

    fig.savefig('train_loss.png')




def trainer(net,device,data_path, epochs=100, batch_size=1):
    #batch_size过大貌似训练效果也不是很好，比如55就有点大了
    dataset = unetdataset(data_path)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    #感觉医学数据集没什么很多相同的数据，batchsize还是设置为1比较好，不然老不收敛，然后epoch设置大点
    best_loss = float('inf')
    count = 0
    all_loss = []
    for epoch in range(epochs):
        net.train()
        if epoch % 10 == 0:
            count += 1
            lr = net.lr/(2*count)
            net.optimizer = torch.optim.Adam(net.parameters(),lr=lr)
        for image, label in train_loader:
            image = image.to(device=device)
            label = label.to(device=device)
            image = image.float()
            label = label.float()
            predy = net(image)
            net.set_input(predy, label)
            net.optimize_params()
            print('Loss:',net.loss.item())
            all_loss.append(net.loss.item())
            if net.loss.item() < best_loss:
                best_loss = net.loss.item()
        torch.save(net.state_dict(),'trained_model.pth')
        plot(all_loss)


        #用于记录的玩意
        with open("log.txt","a") as f:
            f.write(f"已完成第{epoch+1}次epoch，min_loss={best_loss}.{time.asctime()}\n")
            if epoch+1 == epochs:
                f.write(f"\n本次训练已完成\n")

if __name__ == '__main__':
    #限定能使用哪个GPU
    #os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #固定随机种子
    my_seed = 10
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    random.seed(my_seed)              ##
    cudnn.benchmark = True             ##
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False #关闭cudnn，否则BN层会在eval的时候出问题
    device = torch.device('cuda')
    torch.cuda.empty_cache() 
    unet = Unet(in_ch=6,out_ch=1)
    unet = unet.to(device=device)
    data_path = '/home/zhangyuxiang/project/unet/'
    trainer(unet,device,data_path)