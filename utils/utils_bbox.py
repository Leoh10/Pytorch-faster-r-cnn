import numpy as np
import torch
from torch.nn import functional as F
from torchvision.ops import nms


def loc2bbox(src_bbox, loc):#对输入src_bbox进行loc参数的调整
    #src_bbox是先验框，loc是建议框网络的预测偏移结果，建议框也称为先验框
    #roi = loc2bbox(anchor, loc)
    #anchor输入为1444*9*4,loc为输入数据为1*12996*4
    if src_bbox.size()[0] == 0:#若先验框没有数据，则调整参数为0，生成一个1*4的零矩阵
        return torch.zeros((0, 4), dtype=loc.dtype)
    #---首先计算了先验框的中心，其次计算了先验框的宽高
    #---torch.unsqueeze(input,dim,out=None),扩展维度，对输入的既定位置插入维度1
    #---若dim为负，转化为dim+input.dim()+1,数据维度变为1444*9*4*1
    #  src_bbox[:, 2] - src_bbox[:, 0]-----数据维度为1444*4
    src_width   = torch.unsqueeze(src_bbox[:, 2] - src_bbox[:, 0], -1)#宽--先验框的坐标从左往右为x1,y1左下角坐标,x2,y2右上角坐标--x2-x1
    #扩展矩阵后为1444*4*1----因为9个框共用一个中心坐标
    src_height  = torch.unsqueeze(src_bbox[:, 3] - src_bbox[:, 1], -1)#高y2-y1
    src_ctr_x   = torch.unsqueeze(src_bbox[:, 0], -1) + 0.5 * src_width#中心x
    src_ctr_y   = torch.unsqueeze(src_bbox[:, 1], -1) + 0.5 * src_height#中心y
#   对获得先验框的参数进行分割，dx,dy,dw,dh，取出调整参数，中心坐标和宽高的调整参数
    dx          = loc[:, 0::4]#从哪个数据开始取，间隔为4，二维数据1*1444
    dy          = loc[:, 1::4]
    dw          = loc[:, 2::4]
    dh          = loc[:, 3::4]
    #------------------数据维度有误--产生了尚未经过筛选的建议框
    ctr_x = dx * src_width + src_ctr_x#调整中心坐标----调整参数*先验框宽+先验框的中心坐标x
    ctr_y = dy * src_height + src_ctr_y#调整中心坐标----调整参数*先验框宽+先验框的中心坐标Y
    w = torch.exp(dw) * src_width#利用指数形式调整宽高
    h = torch.exp(dh) * src_height

    dst_bbox = torch.zeros_like(loc)#对尚未经过筛选的建议框，对数据格式进行转变
    #   转为左上角到右下角的格式
    dst_bbox[:, 0::4] = ctr_x - 0.5 * w
    dst_bbox[:, 1::4] = ctr_y - 0.5 * h
    dst_bbox[:, 2::4] = ctr_x + 0.5 * w
    dst_bbox[:, 3::4] = ctr_y + 0.5 * h

    return dst_bbox

class DecodeBox():#预测框的解码过程-建议框，传入数据为1*84（0.1,0.1,0.2，0.2）*21次，20个种类
    def __init__(self, std, num_classes):
        self.std            = std
        self.num_classes    = num_classes + 1    #20+1

    def frcnn_correct_boxes(self, box_xy, box_wh, input_shape, image_shape):
        #-----------------------------------------------------------------#
        #   把y轴放前面是因为方便预测框和图像的宽高进行相乘
        #-----------------------------------------------------------------#
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        input_shape = np.array(input_shape)
        image_shape = np.array(image_shape)

        box_mins    = box_yx - (box_hw / 2.)
        box_maxes   = box_yx + (box_hw / 2.)
        boxes  = np.concatenate([box_mins[..., 0:1], box_mins[..., 1:2], box_maxes[..., 0:1], box_maxes[..., 1:2]], axis=-1)
        boxes *= np.concatenate([image_shape, image_shape], axis=-1)
        return boxes

    def forward(self, roi_cls_locs, roi_scores, rois, image_shape, input_shape, nms_iou = 0.3, confidence = 0.5):
        results = []
        bs      = len(roi_cls_locs)
        #--------------------------------#
        #   对获取的建议框进行reshape,将其调整为维度依次为batch_size, num_rois-建议框的数量, 4-建议框的坐标
        #--------------------------------#
        rois    = rois.view((bs, -1, 4))
        #----------------------------------------------------------------------------------------------------------------#
        #   对每一张图片进行处理，循环次数即图片个数，由于在predict.py的时候，我们只输入一张图片，所以for i in range(len(mbox_loc))只进行一次，
        #   bs为1
        #----------------------------------------------------------------------------------------------------------------#
        for i in range(bs):
            #----------------------------------------------------------#
            #   对回归参数进行reshape，乘std，作用是改变数据的数量级
            #   std是指定的修改系数，暂不修改
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_locs[i] * self.std
            #----------------------------------------------------------#
            #   第一维度是建议框的数量，第二维度是每个种类
            #   第三维度是对应种类的调整参数
            #----------------------------------------------------------#
            roi_cls_loc = roi_cls_loc.view([-1, self.num_classes, 4])
            #-------------------------------------------------------------#
            #   利用classifier网络的预测结果对建议框进行调整获得预测框
            #   num_rois, 4 -> num_rois, 1, 4 -> num_rois, num_classes, 4
            #   对数据进行reshape及自动扩张第二维度，变为num_classes
            #-------------------------------------------------------------#
            roi         = rois[i].view((-1, 1, 4)).expand_as(roi_cls_loc)
            #   loc2bbox(roi.contiguous().view((-1, 4)建议框，roi_cls_loc.contiguous().view((-1, 4))建议框调整参数)
            #   后者对前者进行一个调整，获得预测框
            cls_bbox    = loc2bbox(roi.contiguous().view((-1, 4)), roi_cls_loc.contiguous().view((-1, 4)))
            #   对调整后结果进行reshape,维度相对于输入图片一致，num_rois, num_classes, 4
            cls_bbox    = cls_bbox.view([-1, (self.num_classes), 4])
            #-------------------------------------------------------------#
            #   对预测框进行归一化，调整到0-1之间
            #-------------------------------------------------------------#
            cls_bbox[..., [0, 2]] = (cls_bbox[..., [0, 2]]) / input_shape[1]
            cls_bbox[..., [1, 3]] = (cls_bbox[..., [1, 3]]) / input_shape[0]
            #   取出建议框的种类得分
            roi_score   = roi_scores[i]
            #   对种类得分取一个softmax，相当于获得建议框属于每一个种类的概率
            prob        = F.softmax(roi_score, dim=-1)

            results.append([])
            for c in range(1, self.num_classes):#对种类进行循环，循环从1开始，序号0维度数据表示该框为背景的概率
                #--------------------------------#
                #   取出单张图片里面属于该类的所有框的置信度
                #   判断是否大于门限，然后会保留大于的框
                #--------------------------------#
                c_confs     = prob[:, c]
                c_confs_m   = c_confs > confidence

                if len(c_confs[c_confs_m]) > 0:
                    #-----------------------------------------#
                    #   取出得分高于confidence的框，看该类所有框大于门限数值
                    #   数量是否大于0，若大于0则执行下述操作，
                    #   取出预测框坐标与预测框置信度，赋值到boxes_to_process,confs_to_process
                    #-----------------------------------------#
                    boxes_to_process = cls_bbox[c_confs_m, c]#取出预测框坐标
                    confs_to_process = c_confs[c_confs_m]#取出预测框置信度
                    #   进行非极大抑制处理
                    keep = nms(
                        boxes_to_process,
                        confs_to_process,
                        nms_iou
                    )
                    #-----------------------------------------#
                    #   取出在非极大抑制中效果较好的内容，存放在keep变量中，
                    #-----------------------------------------#
                    good_boxes  = boxes_to_process[keep]
                    confs       = confs_to_process[keep][:, None]
                    labels      = (c - 1) * torch.ones((len(keep), 1)).cuda() if confs.is_cuda else (c - 1) * torch.ones((len(keep), 1))
                    #-----------------------------------------#
                    #   将预测框的label、置信度、框的位置进行堆叠。
                    #-----------------------------------------#
                    c_pred      = torch.cat((good_boxes, confs, labels), dim=1).cpu().numpy()
                    # 添加进result里
                    results[-1].extend(c_pred)
            #   进行一个形式的调整，调整为相对于原图的形式
            if len(results[-1]) > 0:
                results[-1] = np.array(results[-1])
                box_xy, box_wh = (results[-1][:, 0:2] + results[-1][:, 2:4])/2, results[-1][:, 2:4] - results[-1][:, 0:2]
                results[-1][:, :4] = self.frcnn_correct_boxes(box_xy, box_wh, input_shape, image_shape)

        return results
        
