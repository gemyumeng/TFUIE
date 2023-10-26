import os
from ssd_test.utils.utils import cvtColor, preprocess_input
import torchvision.transforms.functional as ttf
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class MyTrainDataSet(Dataset):  # 训练数据集
    def __init__(self, inputPathTrain, targetPathTrain, annotation_lines, input_shape, anchors, num_classes, train, overlap_threshold = 0.5):
        super(MyTrainDataSet, self).__init__()

        self.inputPath = inputPathTrain
        self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表
        self.annotation_lines   = annotation_lines
        self.targetPath = targetPathTrain
        self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表
        self.input_shape = input_shape
        self.train = train
        self.num_classes = num_classes
        self.anchors            = anchors
        self.num_anchors        = len(anchors)
        self.overlap_threshold  = overlap_threshold

    def __len__(self):
        return len(self.targetImages)

    def __getitem__(self, index):

        index = index % len(self.targetImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片

        targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
        targetImage = Image.open(targetImagePath).convert('RGB')

        inputImage = ttf.to_tensor(inputImage)  # 将图片转为张量
        targetImage = ttf.to_tensor(targetImage)

        box  = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)

        if len(box)!=0:
            boxes               = np.array(box[:,:4] , dtype=np.float32)
            # 进行归一化，调整到0-1之间
            boxes[:, [0, 2]]    = boxes[:,[0, 2]] / self.input_shape[1]
            boxes[:, [1, 3]]    = boxes[:,[1, 3]] / self.input_shape[0]
            # 对真实框的种类进行one hot处理
            one_hot_label   = np.eye(self.num_classes - 1)[np.array(box[:,4], np.int32)]
            box             = np.concatenate([boxes, one_hot_label], axis=-1)
        box = self.assign_boxes(box)
        box1 = np.array(box)

        # aug = random.randint(0, 8)  # 随机数，对应对图片进行的操作

        input_ = inputImage  # 裁剪 patch ，输入和目标 patch 要对应相同
        target = targetImage

        return input_, target, np.array(box)

    def get_random_data(self, annotation_line, input_shape,  random = False):
        line = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            # #---------------------------------#
            # #   将图像多余的部分加上灰条
            # #---------------------------------#
            # image       = image.resize((nw,nh), Image.BICUBIC)
            # new_image   = Image.new('RGB', (w,h), (128,128,128))
            # new_image.paste(image, (dx, dy))
            # image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return box

    def assign_boxes(self, boxes):
        # ---------------------------------------------------#
        #   assignment分为3个部分
        #   :4      的内容为网络应该有的回归预测结果
        #   4:-1    的内容为先验框所对应的种类，默认为背景
        #   -1      的内容为当前先验框是否包含目标
        # ---------------------------------------------------#
        assignment = np.zeros((self.num_anchors, 4 + self.num_classes + 1))
        assignment[:, 4] = 1.0
        if len(boxes) == 0:
            return assignment

        # 对每一个真实框都进行iou计算
        encoded_boxes = np.apply_along_axis(self.encode_box, 1, boxes[:, :4])
        # ---------------------------------------------------#
        #   在reshape后，获得的encoded_boxes的shape为：
        #   [num_true_box, num_anchors, 4 + 1]
        #   4是编码后的结果，1为iou
        # ---------------------------------------------------#
        encoded_boxes = encoded_boxes.reshape(-1, self.num_anchors, 5)

        # ---------------------------------------------------#
        #   [num_anchors]求取每一个先验框重合度最大的真实框
        # ---------------------------------------------------#
        best_iou = encoded_boxes[:, :, -1].max(axis=0)
        best_iou_idx = encoded_boxes[:, :, -1].argmax(axis=0)
        best_iou_mask = best_iou > 0
        best_iou_idx = best_iou_idx[best_iou_mask]

        # ---------------------------------------------------#
        #   计算一共有多少先验框满足需求
        # ---------------------------------------------------#
        assign_num = len(best_iou_idx)

        # 将编码后的真实框取出
        encoded_boxes = encoded_boxes[:, best_iou_mask, :]
        # ---------------------------------------------------#
        #   编码后的真实框的赋值
        # ---------------------------------------------------#
        assignment[:, :4][best_iou_mask] = encoded_boxes[best_iou_idx, np.arange(assign_num), :4]
        # ----------------------------------------------------------#
        #   4代表为背景的概率，设定为0，因为这些先验框有对应的物体
        # ----------------------------------------------------------#
        assignment[:, 4][best_iou_mask] = 0
        assignment[:, 5:-1][best_iou_mask] = boxes[best_iou_idx, 4:]
        # ----------------------------------------------------------#
        #   -1表示先验框是否有对应的物体
        # ----------------------------------------------------------#
        assignment[:, -1][best_iou_mask] = 1
        # 通过assign_boxes我们就获得了，输入进来的这张图片，应该有的预测结果是什么样子的
        return assignment

    def encode_box(self, box, return_iou=True, variances=[0.1, 0.1, 0.2, 0.2]):
        # ---------------------------------------------#
        #   计算当前真实框和先验框的重合情况
        #   iou [self.num_anchors]
        #   encoded_box [self.num_anchors, 5]
        # ---------------------------------------------#
        iou = self.iou(box)
        encoded_box = np.zeros((self.num_anchors, 4 + return_iou))

        # ---------------------------------------------#
        #   找到每一个真实框，重合程度较高的先验框
        #   真实框可以由这个先验框来负责预测
        # ---------------------------------------------#
        assign_mask = iou > self.overlap_threshold

        # ---------------------------------------------#
        #   如果没有一个先验框重合度大于self.overlap_threshold
        #   则选择重合度最大的为正样本
        # ---------------------------------------------#
        if not assign_mask.any():
            assign_mask[iou.argmax()] = True

        # ---------------------------------------------#
        #   利用iou进行赋值
        # ---------------------------------------------#
        if return_iou:
            encoded_box[:, -1][assign_mask] = iou[assign_mask]

        # ---------------------------------------------#
        #   找到对应的先验框
        # ---------------------------------------------#
        assigned_anchors = self.anchors[assign_mask]

        # ---------------------------------------------#
        #   逆向编码，将真实框转化为ssd预测结果的格式
        #   先计算真实框的中心与长宽
        # ---------------------------------------------#
        box_center = 0.5 * (box[:2] + box[2:])
        box_wh = box[2:] - box[:2]
        # ---------------------------------------------#
        #   再计算重合度较高的先验框的中心与长宽
        # ---------------------------------------------#
        assigned_anchors_center = (assigned_anchors[:, 0:2] + assigned_anchors[:, 2:4]) * 0.5
        assigned_anchors_wh = (assigned_anchors[:, 2:4] - assigned_anchors[:, 0:2])

        # ------------------------------------------------#
        #   逆向求取ssd应该有的预测结果
        #   先求取中心的预测结果，再求取宽高的预测结果
        #   存在改变数量级的参数，默认为[0.1,0.1,0.2,0.2]
        # ------------------------------------------------#
        encoded_box[:, :2][assign_mask] = box_center - assigned_anchors_center
        encoded_box[:, :2][assign_mask] /= assigned_anchors_wh
        encoded_box[:, :2][assign_mask] /= np.array(variances)[:2]

        encoded_box[:, 2:4][assign_mask] = np.log(box_wh / assigned_anchors_wh)
        encoded_box[:, 2:4][assign_mask] /= np.array(variances)[2:4]
        return encoded_box.ravel()

    def iou(self, box):
        #---------------------------------------------#
        #   计算出每个真实框与所有的先验框的iou
        #   判断真实框与先验框的重合情况
        #---------------------------------------------#
        inter_upleft    = np.maximum(self.anchors[:, :2], box[:2])
        inter_botright  = np.minimum(self.anchors[:, 2:4], box[2:])

        inter_wh    = inter_botright - inter_upleft
        inter_wh    = np.maximum(inter_wh, 0)
        inter       = inter_wh[:, 0] * inter_wh[:, 1]
        #---------------------------------------------#
        #   真实框的面积
        #---------------------------------------------#
        area_true = (box[2] - box[0]) * (box[3] - box[1])
        #---------------------------------------------#
        #   先验框的面积
        #---------------------------------------------#
        area_gt = (self.anchors[:, 2] - self.anchors[:, 0])*(self.anchors[:, 3] - self.anchors[:, 1])
        #---------------------------------------------#
        #   计算iou
        #---------------------------------------------#
        union = area_true + area_gt - inter

        iou = inter / union
        return iou

# class MyValueDataSet(Dataset):  # 评估数据集
#     def __init__(self, inputPathTrain, targetPathTrain, annotation_lines, input_shape, train, patch_size=256):
#         super(MyValueDataSet, self).__init__()
#
#         self.inputPath = inputPathTrain
#         self.inputImages = os.listdir(inputPathTrain)  # 输入图片路径下的所有文件名列表
#
#         self.targetPath = targetPathTrain
#         self.targetImages = os.listdir(targetPathTrain)  # 目标图片路径下的所有文件名列表
#
#         self.ps = patch_size
#
#     def __len__(self):
#         return len(self.targetImages)
#
#     def __getitem__(self, index):
#
#         ps = self.ps
#         index = index % len(self.targetImages)
#
#         inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
#         inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片,灰度图
#
#         targetImagePath = os.path.join(self.targetPath, self.targetImages[index])
#         targetImage = Image.open(targetImagePath).convert('RGB')
#         #
#         inputImage = ttf.to_tensor(inputImage)  # 将图片转为张量
#         targetImage = ttf.to_tensor(targetImage)
#         return  inputImage, targetImage

class MyTestDataSet(Dataset):  # 测试数据集
    def __init__(self, inputPathTest):
        super(MyTestDataSet, self).__init__()

        self.inputPath = inputPathTest
        self.inputImages = os.listdir(inputPathTest)  # 输入图片路径下的所有文件名列表

    def __len__(self):
        return len(self.inputImages)  # 路径里的图片数量

    def __getitem__(self, index):
        index = index % len(self.inputImages)

        inputImagePath = os.path.join(self.inputPath, self.inputImages[index])  # 图片完整路径
        inputImage = Image.open(inputImagePath).convert('RGB')  # 读取图片

        input_ = ttf.to_tensor(inputImage)  # 将图片转为张量

        return input_