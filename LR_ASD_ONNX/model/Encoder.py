import torch
import torch.nn as nn


class Audio_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2):
        super(Audio_Block, self).__init__()

        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1-1)/2)
        self.padding_2 = int((kernel_2-1)/2)

        self.m_1 = nn.Conv2d(in_channels, out_channels//2, kernel_size = (kernel_1, 1), padding = (self.padding_1, 0), bias = False)
        self.m_norm_1 = nn.BatchNorm2d(out_channels//2, momentum = 0.01, eps = 0.001)
        self.m_2 = nn.Conv2d(out_channels//2, out_channels, kernel_size = (kernel_2, 1), padding = (self.padding_2, 0), bias = False)
        self.m_norm_2 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        
        self.t_1 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, kernel_1), padding = (0, self.padding_1), bias = False)
        self.t_norm_1 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_2 = nn.Conv2d(out_channels, out_channels, kernel_size = (1, kernel_2), padding = (0, self.padding_2), bias = False)
        self.t_norm_2 = nn.BatchNorm2d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):

        x = self.relu(self.m_norm_1(self.m_1(x)))
        x = self.relu(self.m_norm_2(self.m_2(x)))

        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))

        return x


class Visual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_1, kernel_2, is_down = False):
        super(Visual_Block, self).__init__()

        self.relu = nn.ReLU()
        self.padding_1 = int((kernel_1-1)/2)
        self.padding_2 = int((kernel_2-1)/2)

        if is_down:
            self.s_1 = nn.Conv3d(in_channels, out_channels//2, kernel_size = (1, kernel_1, kernel_1), stride = (1, 2, 2), padding = (0, self.padding_1, self.padding_1), bias = False)
        else:
            self.s_1 = nn.Conv3d(in_channels, out_channels//2, kernel_size = (1, kernel_1, kernel_1), padding = (0, self.padding_1, self.padding_1), bias = False)
        
        self.s_norm_1 = nn.BatchNorm3d(out_channels//2, momentum = 0.01, eps = 0.001)
        
        self.s_2 = nn.Conv3d(out_channels//2, out_channels, kernel_size = (1, kernel_2, kernel_2), padding = (0, self.padding_2, self.padding_2), bias = False)
        self.s_norm_2 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

        self.t_1 = nn.Conv3d(out_channels, out_channels, kernel_size = (kernel_1, 1, 1), padding = (self.padding_1, 0, 0), bias = False)
        self.t_norm_1 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)
        self.t_2 = nn.Conv3d(out_channels, out_channels, kernel_size = (kernel_2, 1, 1), padding = (self.padding_2, 0, 0), bias = False)
        self.t_norm_2 = nn.BatchNorm3d(out_channels, momentum = 0.01, eps = 0.001)

    def forward(self, x):

        x = self.relu(self.s_norm_1(self.s_1(x)))
        x = self.relu(self.s_norm_2(self.s_2(x)))
        
        x = self.relu(self.t_norm_1(self.t_1(x)))
        x = self.relu(self.t_norm_2(self.t_2(x)))

        return x


class visual_encoder(nn.Module):
    def __init__(self):
        super(visual_encoder, self).__init__()

        self.block1 = Visual_Block(1, 32, 5, 3, is_down = True)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))

        self.block2 = Visual_Block(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 3, 3), stride = (1, 2, 2), padding = (0, 1, 1))
        
        self.block3 = Visual_Block(64, 128, 5, 3)

        self.maxpool = nn.AdaptiveMaxPool2d((1, 1))

        self.__init_weight()     

    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)
        x = x.transpose(1,2)
        B, T, C, W, H = x.shape  
        x = x.reshape(B*T, C, W, H)

        x = self.maxpool(x)

        x = x.view(B, T, C)  
        
        return x

    def __init_weight(self):

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class audio_encoder(nn.Module):
    def __init__(self):
        super(audio_encoder, self).__init__()
        
        self.block1 = Audio_Block(1, 32, 5, 3)
        self.pool1 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))

        self.block2 = Audio_Block(32, 64, 5, 3)
        self.pool2 = nn.MaxPool3d(kernel_size = (1, 1, 3), stride = (1, 1, 2), padding = (0, 0, 1))
        
        self.block3 = Audio_Block(64, 128, 5, 3)

        self.__init_weight()
            
    def forward(self, x):

        x = self.block1(x)
        x = self.pool1(x)

        x = self.block2(x)
        x = self.pool2(x)

        x = self.block3(x)

        x = torch.mean(x, dim = 2, keepdim = True)
        x = x.squeeze(2).transpose(1, 2)
        
        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()