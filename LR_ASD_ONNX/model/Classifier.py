import torch
from torch import nn


class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.sigmoid = nn.Sigmoid()
        self.attention = nn.Conv1d(channel, channel, kernel_size = 1, padding = 0, bias = False)
        self.bn = nn.BatchNorm1d(channel, momentum = 0.01, eps = 0.001)
    
    def forward(self, x1, x2):
        x = torch.cat((x1, x2),2)
        identity = x.transpose(1, 2)
        w = self.sigmoid(self.bn(self.attention(identity)))
        x = (identity * w).transpose(1, 2) 
        return x


class Detector(nn.Module):
    def __init__(self, channel):
        super(Detector, self).__init__()

        self.gru_forward = nn.GRU(input_size = channel, hidden_size = channel//4, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.gru_backward = nn.GRU(input_size = channel, hidden_size = channel//4, num_layers = 1, bidirectional = False, bias = True, batch_first = True)
        self.drop = nn.Dropout(0.5)
        self.attention = Fusion(channel//2)
        self.__init_weight()

    def forward(self, x):
        x1, _ = self.gru_forward(self.drop(x))
        x = torch.flip(x, dims=[1])
        x2, _ = self.gru_backward(self.drop(x))
        x2 = torch.flip(x2, dims=[1])
        x = self.attention(x1, x2)

        return x

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.GRU):
                torch.nn.init.kaiming_normal_(m.weight_ih_l0)
                torch.nn.init.kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()