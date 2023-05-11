import warnings

import torch
from torch import nn
from torchvision.ops import RoIPool

warnings.filterwarnings("ignore")

class VGG16RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
        super(VGG16RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc    = nn.Linear(4096, n_class * 4)
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score      = nn.Linear(4096, n_class)
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)
        #---------SSP（spatial pyramid pooling）--roi pooling
        self.roi = RoIPool((roi_size, roi_size), spatial_scale)

        
    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        rois        = torch.flatten(rois, 0, 1)
        roi_indices = torch.flatten(roi_indices, 0, 1)

        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)
        #-----------------------------------#
        #   利用classifier网络进行特征提取
        #-----------------------------------#
        pool = pool.view(pool.size(0), -1)
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 4096]
        #--------------------------------------------------------------#
        fc7 = self.classifier(pool)

        roi_cls_locs    = self.cls_loc(fc7)
        roi_scores      = self.score(fc7)

        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

class Resnet50RoIHead(nn.Module):
    def __init__(self, n_class, roi_size, spatial_scale, classifier):
    #  self.head = Resnet50RoIHead(
    #             n_class         = num_classes + 1,
    #             roi_size        = 14,
    #             spatial_scale   = 1,
    #             classifier      = classifier)
        super(Resnet50RoIHead, self).__init__()
        self.classifier = classifier
        #--------------------------------------#
        #   对ROIPooling后的的结果进行回归预测
        #--------------------------------------#
        self.cls_loc = nn.Linear(2048, n_class * 4)#分类建议框的调整参数，每个框4个调整参数
        # nn.Linear(in_features, out_features) ，输入为2048，输出为分类的结果
        #-----------------------------------#
        #   对ROIPooling后的的结果进行分类
        #-----------------------------------#
        self.score = nn.Linear(2048, n_class)#分类
        #-----------------------------------#
        #   权值初始化
        #-----------------------------------#
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        self.roi = RoIPool((roi_size, roi_size), spatial_scale)#resize为14*14大小
        #官方函数，

    def forward(self, x, rois, roi_indices, img_size):
        n, _, _, _ = x.shape
        #x是共享特征层
        if x.is_cuda:
            roi_indices = roi_indices.cuda()
            rois = rois.cuda()
        #print('Base_layers : ',x.size())#--------（1,1024,38,38）---feature map数据维度大小
        # print('roi_indices : ',roi_indices.size())
        # print('rois : ',rois.size())   
        # 打印共享特征层经过roipooling层的数据维度变化
        rois        = torch.flatten(rois, 0, 1)#建议框-----300*4
        roi_indices = torch.flatten(roi_indices, 0, 1)#建议框的序号，300
        #img_size=600*600
        rois_feature_map = torch.zeros_like(rois)
        rois_feature_map[:, [0,2]] = rois[:, [0,2]] / img_size[1] * x.size()[3]
        rois_feature_map[:, [1,3]] = rois[:, [1,3]] / img_size[0] * x.size()[2]

        indices_and_rois = torch.cat([roi_indices[:, None], rois_feature_map], dim=1)
        #利用cat函数将建议框的内容与建议框的序号进行堆叠，将堆叠后结果传入RoIPool层中
        #-----------------------------------#
        #   利用建议框对公用特征层进行截取
        #-----------------------------------#
        pool = self.roi(x, indices_and_rois)#x为共享特征层，
        # roipooling层输出为300*1024*14*14，其中1024是通道数，300是建议框
        #相当于是300个调整大小后的局部特征层，调整数据维度300*14*14*1024，
        #-----------------------------------#
        #   利用classifier网络进行特征提取----即resnet50主干特征提取网络中的第五次压缩，留到这里使用，里面包含依次
        #-----------------------------------#
        fc7 = self.classifier(pool)#classifier包括全局池化，因此不再进行全局池化，假如不进行全局池化，则
        #数据维度为300*2048*7*7
        #--------------------------------------------------------------#
        #   当输入为一张图片的时候，这里获得的f7的shape为[300, 2048，1,1]
        #--------------------------------------------------------------#
        fc7 = fc7.view(fc7.size(0), -1)#reshape,将矩阵后两维度1,压缩-------300*2048

        roi_cls_locs    = self.cls_loc(fc7)#建议框参数--回归预测
        roi_scores      = self.score(fc7)#分类预测---判断种类

        roi_cls_locs    = roi_cls_locs.view(n, -1, roi_cls_locs.size(1))
        roi_scores      = roi_scores.view(n, -1, roi_scores.size(1))
        return roi_cls_locs, roi_scores

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
