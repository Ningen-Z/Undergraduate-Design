import torch 
import glob
import numpy as np
import os
import shutil #用于删除非空文件夹
import cv2 
from torch.backends import cudnn 
import random   #用于设定随机种子
from unet_model import Unet 
from load_data import *

def get_img(folder,PTV=False):
    #获取目标病人某通道里的全部png图片
    image = []
    img_path = sorted(glob.glob(os.path.join(folder,'*.png')),key=os.path.getmtime)
    for path in range(len(img_path)):
        img = cv2.imread(img_path[path])
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        if PTV == True:
            img = img/255
        image.append(img)
    return image

def split2d(list,tick):
    for i in range(len(list)):
        for j in range(len(list[i])):
            if list[i][j] != tick:
                list[i][j] = 0
            if list[i][j] == tick:
                list[i][j] = 1

    return list         
#sorted控制glob函数返回的数组列表顺序

def prediction(net,device,test_path,label_path,num_patients):
    #准备数据部分
    #获取测试数据中各个病人每个通道的全部数据路径
    folder_patients = []
    for patient in range(num_patients):
        num = str(patient+1).zfill(3)
        if not os.path.exists(test_path+f"img_{num}Al_0000_slice"):
            continue
        folder_patients.append(sorted(glob.glob(os.path.join(test_path,f"*{num}Al*")),key=os.path.getmtime))
    #读取数据为数组，folder_patients为各个病人的数据路径
    '''patient_data = []
    for patient in range(len(folder_patients)):
        img_1 = get_img(folder_patients[patient][0],PTV=True)
        img_2 = get_img(folder_patients[patient][1])
        img_3 = get_img(folder_patients[patient][2])
        temp = []
        #把得到的数据转化为以z轴为分割的形式，即unet读取数据要求的形式
        for i in range(len(img_1)):
            z = i % 96
            height = np.ones(shape=(192,192))*z*0.01
            oar_1 = split2d(copy.deepcopy(img_3[i]),1)
            oar_2 = split2d(copy.deepcopy(img_3[i]),3)
            oar_3 = split2d(copy.deepcopy(img_3[i]),5)
            temp.append([img_1[i],img_2[i],oar_1,oar_2,oar_3,height])
        #最后全部储存在patient_data中，index为每一个病人(即temp)
        patient_data.append(temp)
    patient_data = np.asarray(patient_data)
    patient_data_tensor = torch.from_numpy(patient_data)'''
    #patient_data_tensor为8维tensor，最外层是patient，然后96个切片，之后是6个通道
    #为每一个patient创建一个文件夹，并保留pred_folder为文件夹目录数组
    pred_folder = sorted(glob.glob(os.path.join(label_path,'*')),key=os.path.getmtime)
    #以下为获取标签数据以计算loss
    '''label_folder =pred_folder
    labels = []
    for i in range(len(label_folder)):
        label = get_img(label_folder[i])
        labels.append(label)
    labels = np.asarray(labels)
    labels = torch.from_numpy(labels)'''
    for i in range(len(pred_folder)):
        pred_folder[i] = pred_folder[i].replace('label','prediction')
        if os.path.exists(pred_folder[i]):
            shutil.rmtree(pred_folder[i])
        os.mkdir(pred_folder[i])
    #读取网络，开始预测
    Loss = []
    net = net.to(device=device)
    net.load_state_dict(torch.load('./network/trained_model1.pth',map_location=device))
    for seq in range(len(pred_folder)):
    #利用labels每个病人只有一个文件夹的特性确定病人的个数
        save_path = pred_folder[seq]
        loss_fn = torch.nn.MSELoss()
        setting = False
        if setting:
            for image in range(len(patient_data_tensor[seq])):
                #对每个病人的每个切片进行预测
                save_image_path = os.path.join(save_path,f"{image}.png")
                label = labels[seq][image]
                image = patient_data_tensor[seq][image]
                image = torch.unsqueeze(image,dim=0)
                image = image.float()
                image = image.to(device=device)
                label = label.float()
                label = label.to(device=device)
                net.train()
                with torch.no_grad():
                    pred = net(image)
                    pred = pred.squeeze(0)
                    loss = torch.sqrt(loss_fn(pred,label))
                    print(f"loss:{loss.item()}")
                    pred = np.array(pred.cpu().detach().squeeze(0))
                    pred[pred < 0] = 0
                    pred = pred*255
                    cv2.imwrite(save_image_path,pred)
        else:
            if seq == 0:
                data_path = 'F:/design/code/unet_42-master/'
                dataset = unetdataset(data_path,test=True)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
                images = []
                for image,label in train_loader:
                    net.train()
                    with torch.no_grad():
                        image = image.to(device=device)
                        label = label.to(device=device)
                        image = image.float()
                        label = label.float()
                        pred = net(image)
                        #pred = pred.squeeze(0)
                        loss = torch.sqrt(loss_fn(pred,label))
                        print('Loss:',loss.item())
                        pred = np.array(pred.cpu().detach().squeeze(0))
                        Loss.append(loss.item())
                        pred[pred < 0] = 0
                        pred = pred*255
                        images.append(pred)
                for i in range(96):
                    save_image_path = os.path.join(save_path,f"{i}.png")
                    cv2.imwrite(save_image_path,images[96*seq+i])
            for i in range(96):
                save_image_path = os.path.join(save_path,f"{i}.png")
                cv2.imwrite(save_image_path,images[96*seq+i])
                
    np.save('predict_loss.npy',Loss)

if __name__ == '__main__':
    device = torch.device('cuda')
    my_seed = 10
    np.random.seed(my_seed)
    torch.manual_seed(my_seed)
    torch.cuda.manual_seed(my_seed)
    torch.cuda.manual_seed_all(my_seed)
    random.seed(my_seed)              ##
    cudnn.benchmark = True            ##
    torch.backends.cudnn.deterministic = True
    test_path = 'F:/design/code/unet_42-master/test_images/'
    test_label_path = test_path.replace('images','labels')
    unet = Unet(in_ch=6,out_ch=1)
    prediction(net=unet,device=device,test_path=test_path,label_path=test_label_path,num_patients=86)
