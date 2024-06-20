import torch
import torch.nn as nn
import numpy as np
from keras import backend as K
import tensorflow as tf
import copy

class MultiTaskLoss(nn.Module):
    def __init__(self, num_classes=4, lmbda=10, epsilon=10e-6, mode="train"):
        super(MultiTaskLoss, self).__init__()
        self.epsilon = epsilon
        self.lmbda = lmbda
        self.channels = num_classes
        self.CELoss = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 4.0, 5.0, 1.0])).cuda()
        self.l2loss = nn.MSELoss()
        self.CELoss = LabelSmoothSoftmaxCE()
        self.bce2d = nn.BCELoss().cuda()
        self.epoch = 1
        self.alpha = 1.0
        if mode == "train":
            self.seg_loss = ImageBasedCrossEntropyLoss2d(
                        classes=num_classes, upper_bound=1.0).cuda()
        else:
            self.seg_loss = self.seg_loss = CrossEntropyLoss2d(size_average=True).cuda()

    def Edge_Extracted(self,y_pred):
        min_x = tf.constant(0, tf.float32)
        max_x = tf.constant(1, tf.float32)
        sobel_x = tf.constant([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], tf.float32)

        sobel_x_filter = tf.reshape(sobel_x, [3, 3, 1, 1])
        sobel_y_filter = tf.transpose(sobel_x_filter, [1, 0, 2, 3])
        filters_x = tf.nn.conv2d(y_pred.cpu().detach().numpy(), sobel_x_filter,strides=[1, 1, 1, 1], data_format="NCHW", padding='SAME')
        filters_y = tf.nn.conv2d(y_pred.cpu().detach().numpy(), sobel_y_filter,
                                 strides=[1, 1, 1, 1], data_format="NCHW", padding='SAME')

        edge = tf.sqrt(filters_x * filters_x + filters_y * filters_y + 1e-16)

        edge = tf.clip_by_value(edge, min_x, max_x)

        return edge

    def Dist_Loss(self,y_true, y_pred):
        Dist = 0.
        for i in range(self.channels):
            y_pred_tmp = y_pred[:, i, :, :]
            y_pred_tmp = y_pred_tmp.unsqueeze(dim=1)
            edge = self.Edge_Extracted(y_pred_tmp)
            edge = K.flatten(edge)
            y_true_f = K.flatten(y_true)
            edge_loss = K.sum(edge * tf.cast(y_true_f, tf.float32))

            sess = tf.Session()

            result = sess.run(edge_loss)

            Dist+=result
        return Dist

    def Get_edge(self,img):
        new = copy.deepcopy(img)
        row, col = img.shape
        for i in range(row - 1):
            for j in range(col - 1):
                if i - 1 > 0 & i + 1 < row - 1 & j - 1 > 0 & j + 1 < col - 1:
                    if img[i][j - 1] & img[i][j + 1] & img[i - 1][j] & img[i + 1][j]:
                        new[i][j] = 0

        return new

    def Get_Edge_position(self,img):
        row, col = img.shape
        row -= 1
        col -= 1
        new = copy.deepcopy(img)
        x = []
        y = []
        while (new == 0).all() == False:
            for i in range(row):
                for j in range(col):
                    if new[i][j]:
                        start_x = i
                        start_y = j
                        x.append(start_x)
                        y.append(start_y)
                        print("start_x,start_y=", start_x, start_y)
                        new[start_x][start_y] = 0
                        break

            flage = 1
            while (flage):
                print("start_x,start_y=",start_x,start_y)
                if new[start_x][start_y - 1]:
                    start_x = start_x
                    start_y = start_y - 1
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                #            flage=1
                elif new[start_x + 1][start_y - 1]:
                    start_x = start_x + 1
                    start_y = start_y - 1
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                elif new[start_x + 1][start_y]:
                    start_x = start_x + 1
                    start_y = start_y
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                elif new[start_x + 1][start_y + 1]:
                    start_x = start_x + 1
                    start_y = start_y + 1
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                elif new[start_x - 1][start_y - 1]:
                    start_x = start_x - 1
                    start_y = start_y - 1
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                elif new[start_x - 1][start_y]:
                    start_x = start_x - 1
                    start_y = start_y
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                elif new[start_x - 1][start_y + 1]:
                    start_x = start_x - 1
                    start_y = start_y + 1
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                elif new[start_x][start_y + 1]:
                    start_y = start_y + 1
                    start_x = start_x
                    x.append(start_x)
                    y.append(start_y)
                    new[start_x][start_y] = 0
                else:
                    flage = 0
        x = np.array(x)
        y = np.array(y)
        return x, y

    def Distance_Map(self,array):
        chan, col, row = np.shape(array)
        DisMap = np.zeros_like(array)
        TempMap = np.zeros_like(array[0])
        TempMap[TempMap == 0] = 1000
        for i in range(chan):
            img = array[i]
            print(img)
            if np.sum(img):
                print("img.astype('uint8')=",img.astype('uint8'))
                img = self.Get_edge(img.astype('uint8'))
                edgex, edgey = self.Get_Edge_position(img)
                for c in range(col):
                    for r in range(row):
                        DisMap[i, c, r] = np.min(np.sqrt(np.square(edgex - c) + np.square(edgey - r)))

            else:
                DisMap[i] = TempMap
        return DisMap

class ImageBasedCrossEntropyLoss2d(nn.Module):

    def __init__(self, classes, weight=None, size_average=True, ignore_index=255,
                 norm=False, upper_bound=1.0):
        super(ImageBasedCrossEntropyLoss2d, self).__init__()
        self.num_classes = classes
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)
        self.norm = norm
        self.upper_bound = upper_bound
        self.batch_weights = False

    def calculateWeights(self, target):
        hist = np.histogram(target.flatten(), range(
            self.num_classes + 1), normed=True)[0]
        if self.norm:
            hist = ((hist != 0) * self.upper_bound * (1 / hist)) + 1
        else:
            hist = ((hist != 0) * self.upper_bound * (1 - hist)) + 1
        return hist

    def forward(self, inputs, targets):
        target_cpu = targets.data.cpu().numpy()
        if self.batch_weights:
            weights = self.calculateWeights(target_cpu)
            self.nll_loss.weight = torch.Tensor(weights).cuda()

        loss = 0.0
        for i in range(0, inputs.shape[0]):
            if not self.batch_weights:
                weights = self.calculateWeights(target_cpu[i])
                self.nll_loss.weight = torch.Tensor(weights).cuda()
            
            loss += self.nll_loss(nn.functional.log_softmax(inputs[i].unsqueeze(0)),
                                          targets[i].unsqueeze(0))
        return loss

class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True, ignore_index=255):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average, ignore_index)

    def forward(self, inputs, targets):
        return self.nll_loss(F.log_softmax(inputs), targets)

def dice_loss(true, logits, eps=1e-7):
    num_classes = logits.shape[1]
    if len(true.shape) == 2:
        true = true.unsqueeze(0).unsqueeze(0)
    elif len(true.shape) == 3:
        true = true.unsqueeze(1)
        
    if num_classes == 1:
        true_1_hot = torch.eye(num_classes + 1)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        true_1_hot_f = true_1_hot[:, 0:1, :, :]
        true_1_hot_s = true_1_hot[:, 1:2, :, :]
        true_1_hot = torch.cat([true_1_hot_s, true_1_hot_f], dim=1)
        pos_prob = torch.sigmoid(logits)
        neg_prob = 1 - pos_prob
        probas = torch.cat([pos_prob, neg_prob], dim=1)
    else:
        true_1_hot = torch.eye(num_classes)[true.squeeze(1)]
        true_1_hot = true_1_hot.permute(0, 3, 1, 2).float()
        probas = nn.functional.softmax(logits, dim=1)
    true_1_hot = true_1_hot.type(logits.type())
    dims = (0,) + tuple(range(2, true.ndimension()))
    intersection = torch.sum(probas * true_1_hot, dims)
    cardinality = torch.sum(probas + true_1_hot, dims)
    dice_loss = (2. * intersection / (cardinality + eps)).mean()
    return (1 - dice_loss)

class LabelSmoothSoftmaxCE(nn.Module):
    def __init__(self,
                 lb_pos=0.9,
                 lb_neg=0.005,
                 reduction='mean',
                 lb_ignore=-1,
                 ):
        super(LabelSmoothSoftmaxCE, self).__init__()
        self.lb_pos = lb_pos
        self.lb_neg = lb_neg
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(1)

    def forward(self, logits, label):
        logs = self.log_softmax(logits)
        ignore = label.data.cpu() == self.lb_ignore
        n_valid = (ignore == 0).sum()
        label = label.clone()
        # label[ignore] = 0
        lb_one_hot = logits.data.clone().zero_().scatter_(1, label.unsqueeze(1), 1)
        label = self.lb_pos * lb_one_hot + self.lb_neg * (1-lb_one_hot)

        loss = -torch.sum(logs*label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        elif self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'none':
            loss = loss
        return loss





