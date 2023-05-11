import math

import torch.nn as nn
from torch.hub import load_state_dict_from_url


class Bottleneck(nn.Module):         
    #瓶颈结构，当输入输出维度是否一致对应identity block 与 conv block,
    #torch中神经网络的构建需要创建类来实现，通常需要继承pytorch中的神经网络模型，torch.nn.Module
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):#重载初始化函数
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)#1*1卷积，压缩通道数，步长1
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)函数输入
        self.bn1 = nn.BatchNorm2d(planes)#标准化

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)#3*3卷积，提取特征
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)#1*1卷积扩张通道数，具体看笔记本
        self.bn3 = nn.BatchNorm2d(planes * 4)

        self.relu = nn.ReLU(inplace=True)#激活函数，输出替换输入
        self.downsample = downsample#降采样
        self.stride = stride

    def forward(self, x):#forward函数用于构建前向传递的过程
        residual = x#残差矩阵初始值为输入x矩阵

        out = self.conv1(x)#第一次卷积
        out = self.bn1(out)#第一次标准化，归一化
        out = self.relu(out)#第一个激活函数

        out = self.conv2(out)#第二次卷积
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)#第三次卷积
        out = self.bn3(out)
        if self.downsample is not None:#判断残差边是否有卷积，若有，则相加，为conv block ,若无卷积则为identity block
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        #-----------------------------------#
        #   假设输入进来的图片是600,600,3
        #-----------------------------------#
        self.inplanes = 64
        super(ResNet, self).__init__()

        # 600,600,3 -> 300,300,64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)#对输入图片进行卷积，通道数64
        #从输入层3层转为输出层，64层
        #Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)函数输入
        self.bn1 = nn.BatchNorm2d(64)#标准化
        #torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)#归一化函数输入
        self.relu = nn.ReLU(inplace=True)#激活函数

        # 300,300,64 -> 150,150,64
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True)#最大池化，卷积核2，步长2
        #pooling即下采样，特征维度下降
        #MaxPool2d(kernel_size, stride=None, padding=0, dilation=1池化核间隔大小, return_indices=False记录池化像素索引, ceil_mode=False尺寸向上取整)


        # 150,150,64 -> 150,150,256
        self.layer1 = self._make_layer(block, 64, layers[0])#利用_make_layer实现conv block & idnentity block,第一层通道数64，layers[0]=3
        # 150,150,256 -> 75,75,512
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)#利用_make_layer实现conv block & idnentity block,第二层通道数128,layers[0]=4
        # 相比较第一层conv block & identity block,仅在步长发生了变化
        # 75,75,512 -> 38,38,1024 到这里可以获得一个38,38,1024的共享特征层,layers[0]=6
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # self.layer4被用在classifier模型中,layers[0]=3
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #第五次压缩，将特征提取分开，
        
        self.avgpool = nn.AvgPool2d(7)
        #全连接层，512*4=2048输出神经元数量，作用是对最后获得的框进行分类，
        #num_classes*4全连接层用于对相应的建议框进行调整，至此，建议框就是ROI的先验框
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        #nn.Linear(in_features:输入节点数，out_features:输出节点数，即算法分类的种类，bias:是否需要偏置)
        #nn.linear对信号进行线性组合

        for m in self.modules():#--------------------未理解，判断网络中每层网络是否是卷积层或标准化层
            #来计算n、m值
            if isinstance(m, nn.Conv2d):#判断两个对象类型是否一致，相同返回true,不相同返回false
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None#定义残差边-卷积
        #-------------------------------------------------------------------#
        #   当模型需要进行高和宽的压缩的时候，就需要用到残差边的downsample
        #-------------------------------------------------------------------#
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,kernel_size=1, stride=stride, bias=False),#步长为1，长宽为1
                #通道数发生变化为64*4=256，卷积核1，步长1，则卷积为1*1
                nn.BatchNorm2d(planes * block.expansion),#标准化
            )
        layers = []#数据结构list
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):#循环，第一个conv block，由1:3----identity block
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) #
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def resnet50(pretrained = False):
    #加载了预训练权重的话，pretrained参数无意义
    #创建主干特征提取网络
    model = ResNet(Bottleneck, [3, 4, 6, 3])#对照上面的代码，block即Bottleneck layers为[3,4,6，3]
    
    if pretrained:#由于该参数为false,则不执行该命令
        state_dict = load_state_dict_from_url("https://download.pytorch.org/models/resnet50-19c8e357.pth", model_dir="./model_data")
        model.load_state_dict(state_dict)
    #----------------------------------------------------------------------------#
    #   获取特征提取部分，从conv1到model.layer3，最终获得一个38,38,1024的特征层，赋值到features,
    #   主干特征提取网络部分的网络结构，输入600*600*3
    #----------------------------------------------------------------------------#
    features    = list([model.conv1, model.bn1, model.relu, model.maxpool, model.layer1, model.layer2, model.layer3])
    #----------------------------------------------------------------------------#
    #   获取分类部分，从model.layer4第五次压缩内容和平均池化矩阵赋值model.avgpool到classifer,分类模型
    #----------------------------------------------------------------------------#
    classifier  = list([model.layer4, model.avgpool])
    
    features    = nn.Sequential(*features)
    #序贯模型---通过索引获得第几层，输入可以是orderdict，或者是一系列的模型如
    #-----一系列模型----送入sequential---------------------------------------------
    # conv1=nn.FractionalMaxPool2d(2,output_ratio=(scaled,scaled))
    # conv2=nn.Conv2d(D,D,kernel_size= 3,stride=1,padding=1,bias=True)
    # conv3=nn.Upsample(size=(inputRes,inputRes),mode='bilinear')
    #-----return nn.Sequential(conv1,conv2,conv3)----------------------------------------
    #定义ordrdick来送入sequential
    # model = nn.Sequential(OrderedDict([           #orderdict按照建造时候的顺序进行存储
    #         ('conv1', nn.Conv2d(1,20,5)),
    #         ('relu1', nn.ReLU()),
    #         ('conv2', nn.Conv2d(20,64,5)),
    #         ('relu2', nn.ReLU())
    #         ]))
    #-------若是以list创建，输入时需加入*引用
    classifier  = nn.Sequential(*classifier)
    return features, classifier
