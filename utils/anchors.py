import numpy as np

#--------------------------------------------#
#   生成基础的先验框
#--------------------------------------------#
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32]):
    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4), dtype=np.float32)#基本框数据维度9*4
    for i in range(len(ratios)):#包含两个循环，长度都为3，获得9个基础的先验框
        #9个anchor的大小按3种长宽比（1:2,1:1,2:1）
        for j in range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = - h / 2.
            anchor_base[index, 1] = - w / 2.
            anchor_base[index, 2] = h / 2.
            anchor_base[index, 3] = w / 2.#先验框的坐标依次为左下角到右上角，每两个为一组目标；
    return anchor_base

#--------------------------------------------#
#   对基础先验框进行拓展对应到所有特征点上
#--------------------------------------------#
def _enumerate_shifted_anchor(anchor_base, feat_stride, height, width):
    #前导单下划线不影响调用方法名，但是需要导入该方法时，单独导入该方法
    #---------------------------------#输入为9*4,16,38,38
    #   计算网格中心点，feat_stride数值为16,16可以理解为原始图像上每16个像素为一个网格
    #----width、height是共享特征层的宽和高，即38,38
    #---------------------------------#
    shift_x             = np.arange(0, width * feat_stride, feat_stride)#在0与width*feat_stride间隔feat_stride生成一个向量
    shift_y             = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y    = np.meshgrid(shift_x, shift_y)#组合网格中心
    shift               = np.stack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel(),), axis=1)#1444*4,meiyi列都是相同的值
    #x.ravel()将数组维度拉成一维数组
    #np.stack()
    #将网格中心堆叠

    #---------------------------------#
    #   每个网格点上的9个先验框
    #---------------------------------#
    A       = anchor_base.shape[0]#维度的第一个值9*4
    K       = shift.shape[0]#维度的第一个值1444*4
    anchor  = anchor_base.reshape((1, A, 4)) + shift.reshape((K, 1, 4))
    #1*9*4+1444*1*4
    # 通过加减可以获得每个网格上的9个先验框
    #---------------------------------#
    #   所有的先验框
    #---------------------------------#
    anchor  = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor#返回的anchor是每个网格上的9个先验框1444*9*4数据维度
    
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    nine_anchors = generate_anchor_base()#首先利用generate_anchor_base获得基础的先验框
    print(nine_anchors)

    height, width, feat_stride  = 38,38,16
    anchors_all                 = _enumerate_shifted_anchor(nine_anchors, feat_stride, height, width)
    print(np.shape(anchors_all))
    
    fig     = plt.figure()
    ax      = fig.add_subplot(111)
    plt.ylim(-300,900)
    plt.xlim(-300,900)
    shift_x = np.arange(0, width * feat_stride, feat_stride)
    shift_y = np.arange(0, height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    plt.scatter(shift_x,shift_y)
    box_widths  = anchors_all[:,2]-anchors_all[:,0]
    box_heights = anchors_all[:,3]-anchors_all[:,1]
    
    for i in [108, 109, 110, 111, 112, 113, 114, 115, 116]:
        rect = plt.Rectangle([anchors_all[i, 0],anchors_all[i, 1]],box_widths[i],box_heights[i],color="r",fill=False)
        ax.add_patch(rect)
    plt.show()
