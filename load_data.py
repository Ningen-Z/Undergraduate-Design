'''
用于预处理数据，我的想法是把nii.gz文件转换成npy文件再训练,npy文件已经有了
本文件意在把npy文件载入dataset
'''


import torch
from torch.utils.data import Dataset 
import os 
import cv2
import copy
import glob
import numpy as np


class unetdataset(Dataset):
    '''现在这个dataset还是只能读png格式文件的，
    想知道npy的怎么搞（不过最后可能也没必要读npy格式的就是了）'''
    def __init__(self,data_path,test=False):
        self.data_path = data_path
        if test:
             self.folder0_path = sorted(glob.glob(os.path.join(data_path,'test_images/*0000*')),key=os.path.getmtime)
             self.folder1_path = sorted(glob.glob(os.path.join(data_path,'test_images/*0001*')),key=os.path.getmtime)
             self.folder2_path = sorted(glob.glob(os.path.join(data_path,'test_images/*0002*')),key=os.path.getmtime)
             self.label_folder = sorted(glob.glob(os.path.join(data_path,'test_labels/*')),key=os.path.getmtime)
        else:
            self.folder0_path = sorted(glob.glob(os.path.join(data_path,'images/*0000*')),key=os.path.getmtime)
            self.folder1_path = sorted(glob.glob(os.path.join(data_path,'images/*0001*')),key=os.path.getmtime)
            self.folder2_path = sorted(glob.glob(os.path.join(data_path,'images/*0002*')),key=os.path.getmtime)
            self.label_folder = sorted(glob.glob(os.path.join(data_path,'labels/*')),key=os.path.getmtime)
        self.img0_path = []
        self.img1_path = []
        self.img2_path = []
        self.label_path = []
        self.set_data()
        self.set_label()

    def __getitem__(self,index):
        image1_path = self.img0_path[index] 
        image2_path = self.img1_path[index]
        image3_path = self.img2_path[index]
        label_path = self.label_path[index]
        img1 = self.read_image(image1_path)/255
        img2 = self.read_image(image2_path)
        img3 = self.read_image(image3_path)
        oar1 = copy.deepcopy(img3)
        oar2 = copy.deepcopy(img3)
        oar3 = copy.deepcopy(img3)
        oar1 = self.split2d(oar1,1)
        oar2 = self.split2d(oar2,3)
        oar3 = self.split2d(oar3,5)
        height = self.z_axis(index)
        image = np.asarray([img1,img2,oar1,oar2,oar3,height])
        label = self.read_image(label_path)/255
        image = torch.from_numpy(image)
        label = torch.from_numpy(label)

        return image, label

    def __len__(self):
        return len(self.label_path)

    def set_data(self):
        for i in range(len(self.folder0_path)):
                self.img0_path.extend(sorted(glob.glob(os.path.join(self.folder0_path[i],'*.png')),key=os.path.getmtime))
                self.img1_path.extend(sorted(glob.glob(os.path.join(self.folder1_path[i],'*.png')),key=os.path.getmtime))
                self.img2_path.extend(sorted(glob.glob(os.path.join(self.folder2_path[i],'*.png')),key=os.path.getmtime))


    def set_label(self):
        for i in range(len(self.label_folder)):
            self.label_path.extend(sorted(glob.glob(os.path.join(self.label_folder[i],'*.png')),key=os.path.getmtime))

    def read_image(self,path):
        image = cv2.imread(str(path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


        return image
    
    def split2d(self,list,tick):
        for i in range(len(list)):
            for j in range(len(list[i])):
                if list[i][j] != tick:
                    list[i][j] = 0
                if list[i][j] == tick:
                    list[i][j] = 1

        return list
    
    def z_axis(self,order):
        ones = np.ones(shape=(192,192))
        z = order % 96
        height = ones*z*0.01
        
        return(height)



if __name__ == '__main__': 
    example = unetdataset('F:/design/code/unet_42-master/')
    print(example.__len__())
    img, label = example.__getitem__(0)
    print(img.shape)
