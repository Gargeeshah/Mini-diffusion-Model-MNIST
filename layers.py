import torch
import torch.nn as nn
import math
from tqdm import tqdm

class ChannelShuffle(nn.Module):
    def __init__(self,groups):
        super().__init__()
        self.groups=groups
    def forward(self, x):
        N, C, H, W = x.shape
        x = x.view(N, self.groups, C//self.groups, H, W)
        x = x.transpose(1, 2).contiguous().view(N, -1, H, W)
        return x
    
class ConvBnSiLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.module = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                                  nn.BatchNorm2d(out_channels),
                                  nn.SiLU(inplace=True))
    def forward(self, x):
        return self.module(x)

class ResidualBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.branch1 = nn.Sequential(nn.Conv2d(in_channels//2, in_channels//2,3,1,1, groups=in_channels//2),
                                    nn.BatchNorm2d(in_channels//2),
                                    ConvBnSiLu(in_channels//2, out_channels//2,1,1,0))
        self.branch2 = nn.Sequential(ConvBnSiLu(in_channels//2, in_channels//2,1,1,0),
                                    nn.Conv2d(in_channels//2, in_channels//2,3,1,1, groups=in_channels//2),
                                    nn.BatchNorm2d(in_channels//2),
                                    ConvBnSiLu(in_channels//2, out_channels//2,1,1,0))
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x = torch.cat([self.branch1(x1),self.branch2(x2)],dim=1)
        x = self.channel_shuffle(x)
        return x
    
class ResidualDownsample(nn.Module):
    def __init__(self,in_channels,out_channels):
        super().__init__()
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels,in_channels,3,2,1,groups=in_channels),
                                    nn.BatchNorm2d(in_channels),
                                    ConvBnSiLu(in_channels,out_channels//2,1,1,0))
        self.branch2 = nn.Sequential(ConvBnSiLu(in_channels,out_channels//2,1,1,0),
                                    nn.Conv2d(out_channels//2,out_channels//2,3,2,1,groups=out_channels//2),
                                    nn.BatchNorm2d(out_channels//2),
                                    ConvBnSiLu(out_channels//2,out_channels//2,1,1,0))
        self.channel_shuffle = ChannelShuffle(2)

    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        x=self.channel_shuffle(x)
        return x
    
class TimeMLP(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(embedding_dim, hidden_dim),
                                nn.SiLU(),
                               nn.Linear(hidden_dim, out_dim))
        self.act = nn.SiLU()
    def forward(self, x, t):
        t_emb = self.mlp(t).unsqueeze(-1).unsqueeze(-1)
        x = x + t_emb
        return self.act(x)
    
class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_embedding_dim):
        super().__init__()
        self.conv0=nn.Sequential(*[ResidualBottleneck(in_channels, in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels, out_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim, hidden_dim=out_channels, out_dim=out_channels//2)
        self.conv1=ResidualDownsample(out_channels//2, out_channels)

    def forward(self, x, t=None):
        x_shortcut = self.conv0(x)
        if t is not None:
            x = self.time_mlp(x_shortcut,t)
        x = self.conv1(x)
        return [x, x_shortcut]
    
class DecoderBlock(nn.Module):
    def __init__(self,in_channels,out_channels,time_embedding_dim):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2,mode='bilinear',align_corners=False)
        self.conv0 = nn.Sequential(*[ResidualBottleneck(in_channels,in_channels) for i in range(3)],
                                    ResidualBottleneck(in_channels,in_channels//2))

        self.time_mlp=TimeMLP(embedding_dim=time_embedding_dim,hidden_dim=in_channels,out_dim=in_channels//2)
        self.conv1=ResidualBottleneck(in_channels//2,out_channels//2)

    def forward(self,x,x_shortcut,t=None):
        x = self.upsample(x)
        x = torch.cat([x,x_shortcut],dim=1)
        x=self.conv0(x)
        if t is not None:
            x = self.time_mlp(x,t)
        x = self.conv1(x)

        return x