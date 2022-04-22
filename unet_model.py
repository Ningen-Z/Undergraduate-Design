'''
整体的unet网络结构
'''

from basic_net import *

class Unet(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet,self).__init__()
        self.display_names = ['loss stack', 'matrix_iou_stack']
        self.loss_fn = nn.MSELoss()
        self.device = torch.device('cuda')  if torch.cuda.is_available() else torch.device('cpu')
        self.inc = DoubleConv(in_ch, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        #self.drop3 = nn.Dropout2d(p=0.5)
        self.down4 = Down(512, 1024)
        #self.drop4 = nn.Dropout2d(p=0.5)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, 1)
        self.lr = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.lr)
        #self.optimizer = torch.optim.SGD(self.parameters(),lr=0.1,momentum=0.9,weight_decay=0.0005)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        #x4 = self.drop3(x4)
        x5 = self.down4(x4)
        #x5 = self.drop4(x5)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = x.squeeze(1)
        return x

    def set_input(self, x, y):
        self.pred_y = x.clone().requires_grad_(True).to(device=self.device)
        self.y = y.to(device=self.device)

    def optimize_params(self):
        self.loss = torch.sqrt(self.loss_fn(self.pred_y,self.y)).requires_grad_()
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()




if __name__ == '__main__':
    net = Unet(3, 1)
    print(net)