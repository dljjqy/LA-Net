from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
class DoubleConv(nn.Module):
    def __init__(self, inc, outc, padding='same'):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=(3, 3), stride=(1,1), padding=padding, bias=True),
            nn.BatchNorm2d(outc),
            nn.ReLU(),
            nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=(3, 3), stride=(1,1), padding='same', bias=True),
            nn.BatchNorm2d(outc),
            nn.ReLU()]
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class Attention_block(nn.Module):
    
    def __init__(self, F_g, F_l, F_init):
        super(Attention_block, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(F_g, F_init, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_init),
        )

        self.w_x = nn.Sequential(
            nn.Conv2d(F_l, F_init, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_init),
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_init, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        shape = x.shape[-2:]
        g1 = F.interpolate(self.w_g(g), shape, mode='bilinear')
        x1 = self.w_x(x)
        psi = self.psi(self.relu(g1 + x1))
        out = x * psi
        return out

class AttUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, features=16, boundary_type = ''):
        super().__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dconv0 = DoubleConv(in_c, features)
        self.dconv1 = DoubleConv(features    , features * 2)
        self.dconv2 = DoubleConv(features * 2, features * 4)
        self.dconv3 = DoubleConv(features * 4, features * 8)
        self.dconv4 = DoubleConv(features * 8, features * 16)

        self.uconv4 = DoubleConv(features *16, features * 8)
        self.uconv3 = DoubleConv(features * 8, features * 4)
        self.uconv2 = DoubleConv(features * 4, features * 2)
        self.uconv1 = DoubleConv(features * 2, features)

        if boundary_type == 'D':            
            self.final = nn.Conv2d(features, out_c, 3, 1, padding='valid')
        elif boundary_type == 'N':
            self.final = nn.Conv2d(features, out_c, 3, 1, padding=(0, 1))

        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, (2, 2), (2, 2))
        self.up3 = nn.ConvTranspose2d(features * 8,  features * 4, (2, 2), (2, 2))
        self.up2 = nn.ConvTranspose2d(features * 4,  features * 2, (2, 2), (2, 2))
        self.up1 = nn.ConvTranspose2d(features * 2,  features * 1, (3, 3), (2, 2))

        self.ag1 = Attention_block(features *16, features * 8, features * 8)
        self.ag2 = Attention_block(features * 8, features * 4, features * 4)
        self.ag3 = Attention_block(features * 4, features * 2, features * 2)
        self.ag4 = Attention_block(features * 2, features    , features)


    def forward(self, x):
        x1 = self.dconv0(x)
        x2 = self.dconv1(self.maxpool(x1))
        x3 = self.dconv2(self.maxpool(x2)) 
        x4 = self.dconv3(self.maxpool(x3)) 
        x5 = self.dconv4(self.maxpool(x4)) 
        
        g4 = self.ag1(g = x5, x = x4)
        y4 = self.up4(x5)
        y4 = self.uconv4(torch.cat([g4, y4], 1))

        g3 = self.ag2(g = y4, x = x3)
        y3 = self.up3(y4)
        y3 = self.uconv3(torch.cat([g3, y3], 1))

        g2 = self.ag3(g = y3, x = x2)
        y2 = self.up2(y3)
        y2 = self.uconv2(torch.cat([g2, y2], 1))

        g1 = self.ag4(g = y2, x = x1)
        y1 = self.up1(y2)
        y1 = self.uconv1(torch.cat([g1, y1], 1))

        y = self.final(y1)
        return y


class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, features=16, boundary_type='D'):
        super().__init__()
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))

        self.dconv0 = DoubleConv(in_c, features)
        self.dconv1 = DoubleConv(features, features * 2)
        self.dconv2 = DoubleConv(features * 2, features * 4)
        self.dconv3 = DoubleConv(features * 4, features * 8)
        self.dconv4 = DoubleConv(features * 8, features * 16)
        
        self.up4 = nn.ConvTranspose2d(features * 16, features * 8, (2, 2), (2, 2))
        self.up3 = nn.ConvTranspose2d(features * 8,  features * 4, (2, 2), (2, 2))
        self.up2 = nn.ConvTranspose2d(features * 4,  features * 2, (2, 2), (2, 2))
        self.up1 = nn.ConvTranspose2d(features * 2,  features * 1, (3, 3), (2, 2))

        self.uconv3 = DoubleConv(features *16, features * 8)
        self.uconv2 = DoubleConv(features * 8, features * 4)
        self.uconv1 = DoubleConv(features * 4, features * 2)
        self.uconv0 = DoubleConv(features * 2, features * 1)

        if boundary_type == 'D':            
            self.final = nn.Conv2d(features, out_c, 3, 1, padding='valid')
        elif boundary_type == 'N':
            self.final = nn.Conv2d(features, out_c, 3, 1, padding=(0, 1))


    def forward(self , x):
        x0 = self.dconv0(x)
        x1 = self.dconv1(self.maxpool(x0))
        x2 = self.dconv2(self.maxpool(x1))
        x3 = self.dconv3(self.maxpool(x2))
        x4 = self.dconv4(self.maxpool(x3))

        y = self.uconv3(torch.cat([self.up4(x4),x3], 1))
        y = self.uconv2(torch.cat([self.up3(y), x2], 1))
        y = self.uconv1(torch.cat([self.up2(y), x1], 1))
        y = self.uconv0(torch.cat([self.up1(y), x0], 1))
        y = self.final(y)
        return y


class myUNet(nn.Module):
    def __init__(self, in_c=3, out_c=1, features=16, layers=[1, 2, 4, 8, 16, 32], boundary_type='D'):
        super().__init__()
        self.features = features
        self.layers = layers
        self.maxpool = nn.MaxPool2d((2, 2), (2, 2))
        
        if boundary_type == 'D':            
            self.final = nn.Conv2d(features, out_c, 3, 1, padding='valid')
        elif boundary_type == 'N':
            self.final = nn.Conv2d(features, out_c, 3, 1, padding=(0, 1))
        
        self.dconvs = [DoubleConv(in_c, features)]
        self.ups = []
        self.uconvs = []
        for i in range(1, len(layers)):
            self.dconvs.append(DoubleConv(layers[i-1] * features, layers[i] * features))
            self.uconvs.append(DoubleConv(layers[i] * features, layers[i-1] * features))
            self.ups.append(nn.ConvTranspose2d(layers[i] * features, layers[i-1] * features, (2, 2), (2, 2)))
        self.ups[0] = nn.ConvTranspose2d(features * layers[1],  features * layers[0], (3, 3), (2, 2))
        self.uconvs = self.uconvs[::-1]
        self.ups = self.ups[::-1]

    def forward(self, x):
        xs = [self.dconvs[0](x)]
        for i in range(1, len(self.layers)):
            xs.append(self.dconvs[i](self.maxpool(xs[i-1])))

        # for x in xs:    print(x.shape)

        y = self.uconvs[0](torch.cat([self.ups[0](xs[-1]), xs[-2]], 1))
        for i in range(1, len(self.ups)):
            y = self.ups[i](y)
            y = self.uconvs[i](torch.cat([y, xs[-i-2]], 1))

        y = self.final(y)
        return y
        
# Do not delete it
model_names = {'UNet': UNet, 'AttUNet': AttUNet}

if __name__ == '__main__':
    x = torch.rand(4, 3, 65, 65)
    net1 = myUNet(3, 1, 1)
    # net2 = AttUNet(3, 1, 2)

    print(net1(x).shape)
    # print(net1.dconvs)
    # print(net2(x).shape)
