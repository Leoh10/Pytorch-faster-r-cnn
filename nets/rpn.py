
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import nms
from utils.anchors import _enumerate_shifted_anchor, generate_anchor_base
from utils.utils_bbox import loc2bbox


class ProposalCreator():
    def __init__(
        self, 
        mode, 
        nms_iou             = 0.7,#IOU筛选阈值
        n_train_pre_nms     = 12000,
        n_train_post_nms    = 600,
        n_test_pre_nms      = 3000,
        n_test_post_nms     = 300,
        min_size            = 16
    
    ):#预定义参数，元组数据
        #-----------------------------------#
        #   设置预测还是训练
        #-----------------------------------#
        self.mode               = mode
        #-----------------------------------#
        #   建议框非极大抑制的iou大小--筛选阈值
        #-----------------------------------#
        self.nms_iou            = nms_iou
        #-----------------------------------#
        #   训练用到的建议框数量
        #-----------------------------------#
        self.n_train_pre_nms    = n_train_pre_nms
        self.n_train_post_nms   = n_train_post_nms
        #-----------------------------------#
        #   测试用到的建议框数量
        #-----------------------------------#
        self.n_test_pre_nms     = n_test_pre_nms
        self.n_test_post_nms    = n_test_post_nms
        self.min_size           = min_size
        #给所有创建的实例都赋予属性self的所有参数

    def __call__(self, loc, score, anchor, img_size, scale=1.):#重载()运算符，为了将类的实例对象变为可调用对象
        #将所有实例化ProposalCreator类的对象变为可调用对象，实现二次调用
        if self.mode == "training":
            n_pre_nms   = self.n_train_pre_nms
            n_post_nms  = self.n_train_post_nms
        else:
            n_pre_nms   = self.n_test_pre_nms
            n_post_nms  = self.n_test_post_nms

        #-----------------------------------#
        #   将先验框转换成tensor
        #   利用建议框网络对先验框进行调整
        #   
        #-----------------------------------#
        anchor = torch.from_numpy(anchor).type_as(loc)#数据类型转换，将numpy转为张量
        #anchor输入为1444*9*4,loc为输入数据为1*12996*4

        #-----------------------------------#
        #   将RPN网络预测结果转化成建议框
        #-----------------------------------#
        roi = loc2bbox(anchor, loc)#得到的roi为调整后的建议框
        #-----------------------------------#
        #   防止建议框超出图像边缘------------anchor坐标有负数
        #-----------------------------------#
        roi[:, [0, 2]] = torch.clamp(roi[:, [0, 2]], min = 0, max = img_size[1])
        roi[:, [1, 3]] = torch.clamp(roi[:, [1, 3]], min = 0, max = img_size[0])
        
        #-----------------------------------#
        #   建议框的宽高的最小值不可以小于16
        #-----------------------------------#
        min_size    = self.min_size * scale
        keep        = torch.where(((roi[:, 2] - roi[:, 0]) >= min_size) & ((roi[:, 3] - roi[:, 1]) >= min_size))[0]
        #-----------------------------------#
        #   将对应的建议框保留下来
        #-----------------------------------#
        roi         = roi[keep, :]
        score       = score[keep]

        #-----------------------------------#
        #   根据得分进行排序，取出建议框
        #-----------------------------------#
        order       = torch.argsort(score, descending=True) #argsort与sort均为排序函数，（input,dim,descending）输入矩阵，排序维度，默认为1，对行排序
                                                            #返回排序后数据和在原矩阵中坐标indices
                                                            #descending默认为false,从小到大，反之从大到小
        if n_pre_nms > 0:
            order   = order[:n_pre_nms]
        roi     = roi[order, :]
        score   = score[order]

        #-----------------------------------#
        #   对建议框进行非极大抑制---------防止建议框过多
        #   工作原理，根据得分，将一定区域内得分最多的建议框拿出来
        #   使用官方的非极大抑制会快非常多
        #-----------------------------------#
        keep    = nms(roi, score, self.nms_iou)
        if len(keep) < n_post_nms:
            index_extra = np.random.choice(range(len(keep)), size=(n_post_nms - len(keep)), replace=True)
            keep        = torch.cat([keep, keep[index_extra]])
        keep    = keep[:n_post_nms]
        roi     = roi[keep]
        return roi#将筛选后的建议框返回


class RegionProposalNetwork(nn.Module):
    def __init__(
        self, 
        in_channels     = 512, 
        mid_channels    = 512, 
        ratios          = [0.5, 1, 2],
        anchor_scales   = [8, 16, 32], 
        feat_stride     = 16,
        mode            = "training",
    ):
        super(RegionProposalNetwork, self).__init__()#初始化父类，否则继承的子类没有相关属性
        #-----------------------------------------#
        #   生成基础先验框，anchor_base数据维度为9*4，n_anchor为9
        #-----------------------------------------#
        self.anchor_base    = generate_anchor_base(anchor_scales = anchor_scales, ratios = ratios)

        n_anchor            = self.anchor_base.shape[0]

        #-----------------------------------------#
        #   先进行一个3x3的卷积，可理解为特征整合输入38*38*1024
        #-----------------------------------------#
        self.conv1  = nn.Conv2d(in_channels, mid_channels, 3, 1, 1)
        #卷积核为3相当于整合一下周围特征点信息，滑动窗口
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)函数输入
        #----38*38*512
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体，18通道的卷积，9*2，一个是背景的概率，一个是物体的概率#38*38*18
        #-----------------------------------------#
        self.score  = nn.Conv2d(mid_channels, n_anchor * 2, 1, 1, 0)#输出通道就是18,18也就是卷积核个数，再利用softmax就得到每个框
        #是前景还是背景的概率值，二者和为1

        #-----------------------------------------#
        #   回归预测对先验框进行调整,36通道的卷积，9*4，先验框的参数；38*38*36
        #-----------------------------------------#
        self.loc    = nn.Conv2d(mid_channels, n_anchor * 4, 1, 1, 0)#输出通道就是36,36也就是卷积核个数
        #每四个值就是1个anchor的4个需要调整的偏移量36=4*9
        #-----------------------------------------#
        #   特征点间距步长
        #-----------------------------------------#
        self.feat_stride    = feat_stride
        #-----------------------------------------#
        #   用于对建议框解码并进行非极大抑制
        #-----------------------------------------#
        self.proposal_layer = ProposalCreator(mode)
        #--------------------------------------#
        #   对RPN的网络部分进行权值初始化
        #--------------------------------------#
        normal_init(self.conv1, 0, 0.01)
        normal_init(self.score, 0, 0.01)
        normal_init(self.loc, 0, 0.01)

    def forward(self, x, img_size, scale=1.):
        n, _, h, w = x.shape
        #-----------------------------------------#
        #   输入x为共享特征层，feature map 38*38*1024,先进行一个3x3的卷积，可理解为特征整合，
        #   则n为1,batchsize
        #-----------------------------------------#
        x = F.relu(self.conv1(x))#激活函数的应用-3*3卷积后38*38*512再经过一个激活函数
        #-----------------------------------------#
        #   回归预测对先验框进行调整
        #-----------------------------------------#
        rpn_locs = self.loc(x)#-------x维度为1*36*38*38，reshape layer
        rpn_locs = rpn_locs.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        #view结果为第0维度为batchsize，第1维度为每个先验框，最后一个维度为对先验框进行调整获得建议框
        #调整维数，将tensor的维度换位，调整为1*38*38*36------>1*（38*38*9)*4----1*12996*4
        #-----------------------------------------#
        #   分类预测先验框内部是否包含物体
        #-----------------------------------------#
        rpn_scores = self.score(x)#-------x维度为1*18*38*38，reshape layer
        rpn_scores = rpn_scores.permute(0, 2, 3, 1).contiguous().view(n, -1, 2)
        #                                   -->1*（38*38*9)*2
        #----permute调整矩阵维度，但pytorch中需要使用ncontigous方法，确保张量连续
        #---这块将通道内容调整到最后一维度，然后reshape,第0维度为batchsize,第1维度为每个先验框，最后一维度
        #----在scores里是用来判断是否包含物体，在locs里调整先验框大小，获得建议框
        #--------------------------------------------------------------------------------------#
        #   进行softmax概率计算，每个先验框只有两个判别结果
        #   内部包含物体或者内部不包含物体，rpn_softmax_scores[:, :, 1]的内容为包含物体的概率
        #--------------------------------------------------------------------------------------#
        rpn_softmax_scores  = F.softmax(rpn_scores, dim=-1)#对最后一维数据2个进行softmax处理
        #softmax（x,dim）;dim通常为0,1,2，-1；
        # #0时表示对每一维度相同维度的数值进行softmax运算
        # dim为1时，对某一维度的列进行softmax运算；
        # dim为2时对某一维度的行进行softmax运算
        # dim为-1时，对某一维度的行进行softmax运算
        #----对scores进行softmax,将其变成一个概率的概念，将第一维度的数据拿出来处理，根据概率判断包含物体的置信度
        rpn_fg_scores       = rpn_softmax_scores[:, :, 1].contiguous()
        #总的数据为1*12996*1，取出最后一维数据，判断每个框的背景概率值
        rpn_fg_scores       = rpn_fg_scores.view(n, -1)#再进行一个reshape为1*12996

        #再对矩阵进行reshape
        #将数据再转为1*38*38*（2*9），此时是经过softmax的，可以通过概率值区别前景和背景
        #------------------------------------------------------------------------------------------------#
        #   生成先验框，此时获得的anchor是布满网格点的，当输入图片为600,600,3的时候，shape为(12996, 4)
        #------------------------------------------------------------------------------------------------#
        anchor = _enumerate_shifted_anchor(np.array(self.anchor_base), self.feat_stride, h, w)#数据维度为1444*9*4
        rois        = list()#建立list数据格式
        roi_indices = list()
        for i in range(n):#batchsize为1，输入为一张图片，self.proposal_layer实例化了ProposalCreator这个类，调用了call函数
            roi         = self.proposal_layer(rpn_locs[i], rpn_fg_scores[i], anchor, img_size, scale = scale)
            #利用输入的得分和先验框参数对先验框进行处理
            batch_index = i * torch.ones((len(roi),))
            rois.append(roi.unsqueeze(0))
            roi_indices.append(batch_index.unsqueeze(0))

        rois        = torch.cat(rois, dim=0).type_as(x)
        #x为共享特征层1*1024*38*38,cat(seq,dim,out)沿着dim维度连接seq所有tensor
        roi_indices = torch.cat(roi_indices, dim=0).type_as(x)
        anchor      = torch.from_numpy(anchor).unsqueeze(0).float().to(x.device)
        #将anchor变为tensor张量，在第0维后加一维，然后发送到GPU上
        
        return rpn_locs, rpn_scores, rois, roi_indices, anchor

def normal_init(m, mean, stddev, truncated=False):
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
