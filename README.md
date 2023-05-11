#   faster_rcnn_v1
#   ywh
#   faster rcnn pytorch实现目标检测，源码来源：https://github.com/bubbliiiing/faster-rcnn-pytorch.git
#   实现计划：
'''
（1） 学习原理、读懂代码、添加注释，根据相应的文件夹创建并逐步在服务器上传代码；
（2） 利用提供数据集运行成功算法，得到算法检测结果；
（3）利用自己数据集实现算法；
（4）总结算法，为下一步学习yolo算法做准备；
#源代码测试运行步骤：
训练阶段
（1）训练数据--针对下载的源码提供的数据集；--所需要的的主干提取网络的权重利用源码提供的voc_weights_resnet.pth
（2）数据集预处理：利用voc_annotation.py
（3）faster_rcnn网络的训练，train.py代码
（4）训练结果预测：frcnn.py和predict.py，运行完成进行过输入图片预测结果
预测阶段
（1）使用源码提供的预训练权重进行预测，下载权重，执行predict.py对输入图片进行预测
算法评估
（1）评估源码提供的数据集，算法运行无误的情况下,执行get_map.py可以获得评估结果，评估结果会在map_out文件夹中
'''
