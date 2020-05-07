import os
import torch.nn.functional as F
from collections import OrderedDict
from pretrainedmodels import se_resnext50_32x4d, se_resnext101_32x4d
from lib.net.scg_gcn import *


def load_model(name='MSCG-Rx50', classes=7, node_size=(32,32)):
    if name == 'MSCG-Rx50':
        net = rx50_gcn_3head_4channel(out_channels=classes)
    elif name == 'MSCG-Rx101':
        net = rx101_gcn_3head_4channel(out_channels=classes)
    else:
        print('not found the net')
        return -1

    return net


class rx50_gcn_3head_4channel(nn.Module):
    def __init__(self, out_channels=7, pretrained=True,
                 nodes=(32, 32), dropout=0,
                 enhance_diag=True, aux_pred=True):
        super(rx50_gcn_3head_4channel, self).__init__()  # same with  res_fdcs_v5

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = se_resnext50_32x4d()
        self.layer0, self.layer1, self.layer2, self.layer3, = \
            resnet.layer0, resnet.layer1, resnet.layer2, resnet.layer3

        self.conv0 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for child in self.layer0.children():
            for param in child.parameters():
                par = param
                break
            break

        self.conv0.parameters = torch.cat([par[:, 0, :, :].unsqueeze(1), par], 1)
        self.layer0 = torch.nn.Sequential(self.conv0, *list(self.layer0)[1:4])

        self.graph_layers1 = GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        self.graph_layers2 = GCN_Layer(128, out_channels, bnorm=False, activation=None)

        self.scg = SCG_block(in_ch=1024,
                             hidden_ch=out_channels,
                             node_size=nodes,
                             add_diag=enhance_diag,
                             dropout=dropout)

        weight_xavier_init(self.graph_layers1, self.graph_layers2, self.scg)

    def forward(self, x):
        x_size = x.size()

        gx = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        gx90 = gx.permute(0, 1, 3, 2)
        gx180 = gx.flip(3)
        B, C, H, W = gx.size()

        A, gx, loss, z_hat = self.scg(gx)
        gx, _ = self.graph_layers2(
            self.graph_layers1((gx.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx += z_hat
        gx = gx.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])

        A, gx90, loss2, z_hat = self.scg(gx90)
        gx90, _ = self.graph_layers2(
            self.graph_layers1((gx90.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx90 += z_hat
        gx90 = gx90.reshape(B, self.num_cluster, self.node_size[1], self.node_size[0])
        gx90 = gx90.permute(0, 1, 3, 2)
        gx += gx90

        A, gx180, loss3, z_hat = self.scg(gx180)
        gx180, _ = self.graph_layers2(
            self.graph_layers1((gx180.reshape(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx180 += z_hat
        gx180 = gx180.reshape(B, self.num_cluster, self.node_size[0], self.node_size[1])
        gx180 = gx180.flip(3)
        gx += gx180

        gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        if self.training:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False), loss + loss2 + loss3
        else:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)


class rx101_gcn_3head_4channel(nn.Module):
    def __init__(self, out_channels=7, pretrained=True,
                 nodes=(32, 32), dropout=0,
                 enhance_diag=True, aux_pred=True):
        super(rx101_gcn_3head_4channel, self).__init__()  # same with  res_fdcs_v5

        self.aux_pred = aux_pred
        self.node_size = nodes
        self.num_cluster = out_channels

        resnet = se_resnext101_32x4d()
        self.layer0, self.layer1, self.layer2, self.layer3, = \
            resnet.layer0, resnet.layer1, resnet.layer2, resnet.layer3

        self.conv0 = torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for child in self.layer0.children():
            for param in child.parameters():
                par = param
                break
            break

        self.conv0.parameters = torch.cat([par[:, 0, :, :].unsqueeze(1), par], 1)
        self.layer0 = torch.nn.Sequential(self.conv0, *list(self.layer0)[1:4])

        self.graph_layers1 = GCN_Layer(1024, 128, bnorm=True, activation=nn.ReLU(True), dropout=dropout)

        self.graph_layers2 = GCN_Layer(128, out_channels, bnorm=False, activation=None)

        self.scg = SCG_block(in_ch=1024,
                             hidden_ch=out_channels,
                             node_size=nodes,
                             add_diag=enhance_diag,
                             dropout=dropout)

        weight_xavier_init(self.graph_layers1, self.graph_layers2, self.scg)

    def forward(self, x):
        x_size = x.size()

        gx = self.layer3(self.layer2(self.layer1(self.layer0(x))))
        gx90 = gx.permute(0, 1, 3, 2)
        gx180 = gx.flip(3)

        B, C, H, W = gx.size()

        A, gx, loss, z_hat = self.scg(gx)

        gx, _ = self.graph_layers2(
            self.graph_layers1((gx.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx += z_hat
        gx = gx.view(B, self.num_cluster, self.node_size[0], self.node_size[1])

        A, gx90, loss2, z_hat = self.scg(gx90)
        gx90, _ = self.graph_layers2(
            self.graph_layers1((gx90.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx90 += z_hat
        gx90 = gx90.view(B, self.num_cluster, self.node_size[1], self.node_size[0])
        gx90 = gx90.permute(0, 1, 3, 2)
        gx += gx90

        A, gx180, loss3, z_hat = self.scg(gx180)
        gx180, _ = self.graph_layers2(
            self.graph_layers1((gx180.view(B, -1, C), A)))  # + gx.reshape(B, -1, C)
        if self.aux_pred:
            gx180 += z_hat
        gx180 = gx180.view(B, self.num_cluster, self.node_size[0], self.node_size[1])
        gx180 = gx180.flip(3)
        gx += gx180

        gx = F.interpolate(gx, (H, W), mode='bilinear', align_corners=False)

        if self.training:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False), loss + loss2 + loss3
        else:
            return F.interpolate(gx, x_size[2:], mode='bilinear', align_corners=False)

