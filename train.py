import os
import re
import time
from medpy.metric.binary import hd, hd95
import random
import argparse
import math
import torch

import torch.nn as nn
import torch.utils.data as data
from data.augmentations import Compose, RandomVerticallyFlip, RandomHorizontallyFlip, RandomRotate, PaddingCenterCrop
from data.ac17_dataloader import AC17Data as AC17, AC17_2DLoad as load2D
from models import ModelBuilder, SegmentationModule
from utils import AverageMeter, parse_devices, accuracy, intersectionAndUnion
from lib.nn import UserScatteredDataParallel, async_copy_to, user_scattered_collate, patch_replication_callback
from lib.utils import as_numpy
import numpy as np
from loss import MultiTaskLoss
from radam import RAdam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device_ids = [0, 1]

np.set_printoptions(threshold=np.inf)

def eval(loader_val, segmentation_module, args, crit):
    intersection_meter = [AverageMeter(), AverageMeter()]
    union_meter = [AverageMeter(), AverageMeter()]
    loss_meter = AverageMeter()
    sum_acc_iou_all = 0.
    sum_acc_dice_all = 0.
    sum_acc1 = ""
    phaseFlag = 1
    iou = np.zeros((3, 4))
    countED = 0
    countES = 0
    countall = 0
    iou_all_list = []
    hded95 = [0., 0., 0.]
    hdes95 = [0., 0., 0.]
    hded_count = [0, 0, 0]
    hdes_count = [0, 0, 0]
    segmentation_module.eval()
    for batch_data in loader_val:
        batch_data = batch_data[0]
        if batch_data["phase"] == "ED":
            phaseFlag = 0
            countED += 1
        elif batch_data["phase"] == "ES":
            phaseFlag = 1
            countES += 1
        countall += 1
        seg_label = as_numpy(batch_data["mask"][0])
        torch.cuda.synchronize()
        batch_data["image"] = batch_data["image"].unsqueeze(0).cuda()

        with torch.no_grad():
            segSize = (seg_label.shape[0], seg_label.shape[1])
            scores = torch.zeros(1, args.num_class, segSize[0], segSize[1])
            scores = async_copy_to(scores, args.gpu)
            feed_dict = batch_data.copy()
            scores_tmp, loss = segmentation_module(feed_dict, epoch=0, segSize=segSize)
            scores = scores + scores_tmp
            loss_meter.update(loss)

            _, pred = torch.max(scores, dim=1)
            pred = as_numpy(pred.squeeze(0).cpu())

        torch.cuda.synchronize()

        intersection, union = intersectionAndUnion(pred, seg_label, args.num_class)
        for i in range(1, 4):
            predh = np.copy(pred)
            seg_labelh = np.copy(seg_label)
            if i == 1:
                predh[predh == 2] = 0
                predh[predh == 3] = 0
                seg_labelh[seg_labelh == 2] = 0
                seg_labelh[seg_labelh == 3] = 0
            if i == 2:
                predh[predh == 1] = 0
                predh[predh == 3] = 0
                seg_labelh[seg_labelh == 1] = 0
                seg_labelh[seg_labelh == 3] = 0
            if i == 3:
                predh[predh == 2] = 0
                predh[predh == 1] = 0
                seg_labelh[seg_labelh == 2] = 0
                seg_labelh[seg_labelh == 1] = 0
            if (predh[predh == 1].size != 0 or predh[predh == 2].size != 0 or predh[predh == 3].size != 0) and (
                    seg_labelh[seg_labelh == 1].size != 0 or seg_labelh[seg_labelh == 2].size != 0 or seg_labelh[
                seg_labelh == 3].size != 0):
                hau95 = hd95(predh, seg_labelh)
                if phaseFlag == 0:
                    hded95[i - 1] += hau95
                    hded_count[i - 1] += 1
                if phaseFlag == 1:
                    hdes95[i - 1] += hau95
                    hdes_count[i - 1] += 1
        if phaseFlag:
            intersection_meter[1].update(intersection)
            union_meter[1].update(union)
        else:
            intersection_meter[0].update(intersection)
            union_meter[0].update(union)
        acc1 = accuracy(pred, seg_label)
        sum_acc1 += str(acc1)
    iou[0] = intersection_meter[0].sum / (union_meter[0].sum + 1e-10)
    iou[1] = intersection_meter[1].sum / (union_meter[1].sum + 1e-10)
    iou[2] = (intersection_meter[0].sum * countED + intersection_meter[1].sum * countES) / (
            union_meter[0].sum * countED + union_meter[1].sum * countES + 1e-10)
    dice = DSC(pred, seg_label, args.num_class)
    for i in range(len(iou[0])):
        if i >= 1:
            iou_all = (iou[0][i] * countED + iou[1][i] * countES) / countall
            dice_all = (dice[0][i] * countED + dice[1][i] * countES) / countall
            print('class [{}], IoU: {:.4f},{:.4f},{:.4f} dice: {:.4f},{:.4f},{:.4f}'.format(i, iou[0][i], iou[1][i],
                                                                                            iou_all, dice[0][i],
                                                                                            dice[1][i], dice_all))
            sum_acc_iou_all = sum_acc_iou_all + iou_all
            sum_acc_dice_all = sum_acc_dice_all + dice_all
            iou_all_list.append(iou_all)
    print('Accuracy: {:.4f}'.format(PerPixelAccuracy(sum_acc1)), end=" ")
    print('Sum_acc: {:.4f}, {:.4f}'.format(sum_acc_iou_all, sum_acc_dice_all), end=" ")
    print('loss: {:.4f}'.format(loss_meter.average()))
    print("hded95/hded_count,hdes95/hdes_count=", hded95[0] / hded_count[0], hded95[1] / hded_count[1],
          hded95[2] / hded_count[2], hdes95[0] / hdes_count[0], hdes95[1] / hdes_count[1], hdes95[2] / hdes_count[2])

    return iou_all_list, loss_meter.average()


def train(segmentation_module, loader_train, optimizers, history, epoch, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    ave_total_loss = AverageMeter()
    ave_acc = AverageMeter()
    ave_jaccards = []

    for i in range(args.num_class - 1):
        ave_jaccards.append(AverageMeter())

    segmentation_module.train(not args.fix_bn)

    tic = time.time()
    iter_count = 0

    if epoch == args.start_epoch and args.start_epoch > 1:
        scale_running_lr = ((1. - float(epoch - 1) / (args.num_epoch)) ** args.lr_pow)
        args.running_lr_encoder = args.lr_encoder * scale_running_lr
        for param_group in optimizers[0].param_groups:
            param_group['lr'] = args.running_lr_encoder

    for batch_data in loader_train:
        data_time.update(time.time() - tic)
        batch_data["image"] = batch_data["image"].cuda()
        segmentation_module.zero_grad()
        loss, acc = segmentation_module(batch_data, epoch)
        loss = loss.mean()

        jaccard = acc[1]
        for j in jaccard:
            j = j.float().mean()
        acc = acc[0].float().mean()

        loss.backward()
        for optimizer in optimizers:
            optimizer.step()

        batch_time.update(time.time() - tic)
        tic = time.time()
        iter_count += args.batch_size_per_gpu

        ave_total_loss.update(loss.data.item())
        ave_acc.update(acc.data.item() * 100)

        for n, j in enumerate(ave_jaccards):
            j.update(jaccard[n].data.item() * 100)

        if iter_count % 240 == 0:
            pass
            print('Epoch: [{}/{}], Iter: [{}], Time: {:.2f}, Data: {:.2f},'
                  ' lr_unet: {:.6f}, Accuracy: {:4.2f}, '
                  'Loss: {:.6f}, Jaccard: '
                  .format(epoch, args.max_iters, iter_count,
                          batch_time.average(), data_time.average(),
                          args.running_lr_encoder, ave_acc.average(),
                          ave_total_loss.average()), end=" ")

            for i in range(len(ave_jaccards)):
                if i == 0:
                    print("[", end=" ")
                print('{:4.2f}'.format(ave_jaccards[i].average()), end=" ")
                if i == len(ave_jaccards) - 1:
                    print("]")

    print("epoch:", epoch)
    j_avg = 0
    for j in ave_jaccards:
        j_avg += j.average()
    j_avg /= len(ave_jaccards)

    history['train']['epoch'].append(epoch)
    history['train']['loss'].append(loss.data.item())
    history['train']['acc'].append(acc.data.item())
    history['train']['jaccard'].append(j_avg)
    adjust_learning_rate(optimizers, epoch, args)


def PerPixelAccuracy(i):
    accList = re.findall("\((.*?)\)", i)
    sum_num = 0.0
    sum_accnum = 0.0
    for j in accList:
        if ", 0" not in j:
            num = int(j.split(",")[1])
            accnum = num * float(j.split(",")[0])
            sum_num += num
            sum_accnum += accnum
    return sum_accnum / sum_num


def checkpoint(nets, history, args, epoch_num):
    print('Saving checkpoints...')
    (unet, crit) = nets
    suffix_latest = 'epoch_{}.pth'.format(epoch_num)

    torch.save(history,
               '{}/history_{}'.format(args.ckpt, suffix_latest))

    dict_unet = unet.state_dict()
    torch.save(dict_unet,
               '{}/unet_{}'.format(args.ckpt, suffix_latest))


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.conv._ConvNd):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.modules.batchnorm._BatchNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def create_optimizers(nets, args):
    (unet, crit) = nets
    if args.optimizer.lower() == 'sgd':
        optimizer_unet = torch.optim.SGD(
            group_weight(unet),
            lr=args.lr_encoder,
            momentum=args.beta1,
            weight_decay=args.weight_decay,
            nesterov=False)
    elif args.optimizer.lower() == 'adam':
        optimizer_unet = torch.optim.Adam(
            group_weight(unet),
            lr=args.lr_encoder,
            betas=(0.9, 0.999))
    elif args.optimizer.lower() == 'radam':
        optimizer_unet = RAdam(
            group_weight(unet),
            lr=args.lr_encoder,
            betas=(0.9, 0.999))
    return [optimizer_unet]


def adjust_learning_rate(optimizers, cur_iter, args):
    scale_running_lr = 0.5 * (1 + math.cos(3.14159 * (cur_iter) / args.num_epoch))
    args.running_lr_encoder = args.lr_encoder * scale_running_lr

    optimizer_unet = optimizers[0]
    for param_group in optimizer_unet.param_groups:
        param_group['lr'] = args.running_lr_encoder


def main(args):
    builder = ModelBuilder()

    unet = builder.build_unet(num_class=args.num_class,
                              arch="", )

    crit = MultiTaskLoss(mode="train")

    segmentation_module = SegmentationModule(crit, unet, args.num_class)
    train_augs = Compose([PaddingCenterCrop(256), RandomHorizontallyFlip(), RandomVerticallyFlip(), RandomRotate(180)])
    test_augs = Compose([PaddingCenterCrop(256)])

    dataset_train = AC17(
        root=args.data_root,
        split='train',
        k_split=args.k_split,
        augmentations=train_augs)
    ac17_train = load2D(dataset_train, split='train', deform=True)

    loader_train = data.DataLoader(
        ac17_train,
        batch_size=args.batch_size_per_gpu,
        shuffle=True,
        num_workers=int(args.workers),
        drop_last=True,
        pin_memory=True)

    dataset_val = AC17(
        root=args.data_root,
        split='val',
        k_split=args.k_split,
        augmentations=test_augs)

    ac17_val = load2D(dataset_val, split='val', deform=False)

    loader_val = data.DataLoader(
        ac17_val,
        batch_size=1,
        shuffle=False,
        collate_fn=user_scattered_collate,
        num_workers=5,
        drop_last=True)

    if len(args.gpus) > 1:
        print("len(args.gpus) > 1")
        segmentation_module = UserScatteredDataParallel(
            segmentation_module,
            device_ids=device_ids)
        patch_replication_callback(segmentation_module)
    segmentation_module.cuda()

    nets = (net_encoder, net_decoder, crit) if args.unet == False else (unet, crit)
    optimizers = create_optimizers(nets, args)

    history = {'train': {'epoch': [], 'loss': [], 'acc': [], 'jaccard': []}}
    best_val = {'epoch_1': 0, 'mIoU_1': 0,
                'epoch_2': 0, 'mIoU_2': 0,
                'epoch_3': 0, 'mIoU_3': 0,
                'epoch': 0, 'mIoU': 0}

    for epoch in range(args.start_epoch, args.num_epoch + 1):
        train(segmentation_module, loader_train, optimizers, history, epoch, args)
        iou, loss = eval(loader_val, segmentation_module, args, crit)
        ckpted = False
        if iou[0] > best_val['mIoU_1']:
            best_val['epoch_1'] = epoch
            best_val['mIoU_1'] = iou[0]
            ckpted = True

        if iou[1] > best_val['mIoU_2']:
            best_val['epoch_2'] = epoch
            best_val['mIoU_2'] = iou[1]
            ckpted = True

        if iou[2] > best_val['mIoU_3']:
            best_val['epoch_3'] = epoch
            best_val['mIoU_3'] = iou[2]
            ckpted = True

        if (iou[0] + iou[1] + iou[2]) / 3 > best_val['mIoU']:
            best_val['epoch'] = epoch
            best_val['mIoU'] = (iou[0] + iou[1] + iou[2]) / 3
            ckpted = True

        if epoch % 50 == 0:
            checkpoint(nets, history, args, epoch)
            continue

        if epoch == args.num_epoch:
            checkpoint(nets, history, args, epoch)
            continue
        if epoch < 15:
            ckpted = False
        if ckpted == False:
            continue
        else:
            checkpoint(nets, history, args, epoch)
            continue
        ckpted = False
        print("ckpted =", ckpted)

    print('Training Done!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--id', default='',
                        help="a name for identifying the model")
    parser.add_argument('--unet', default=True,
                        help="use unet?")
    parser.add_argument('--data-root', type=str, default=r'/DATAPATH')
    parser.add_argument('--gpus', default='0',
                        help='gpus to use, e.g. 0-3 or 0,1,2,3')
    parser.add_argument('--batch_size_per_gpu', default=2, type=int,
                        help='input batch size')
    parser.add_argument('--num_epoch', default=800, type=int,
                        help='epochs to train for')
    parser.add_argument('--start_epoch', default=1, type=int,
                        help='epoch to start training. useful if continue from a checkpoint')
    parser.add_argument('--epoch_iters', default=0, type=int,
                        help='iterations of each epoch (irrelevant to batch size)')
    parser.add_argument('--optim', default='Adam', help='optimizer')
    parser.add_argument('--lr_encoder', default=0.0005, type=float, help='LR')
    parser.add_argument('--lr_pow', default=0.9, type=float,
                        help='power in poly to drop LR')
    parser.add_argument('--beta1', default=0.9, type=float,
                        help='momentum for sgd, beta1 for adam')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weights regularizer')
    parser.add_argument('--fix_bn', action='store_true',
                        help='fix bn params')

    parser.add_argument('--num_class', default=4, type=int,
                        help='number of classes')
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loading workers')
    parser.add_argument('--k_split', default=1)

    parser.add_argument('--seed', default=1042, type=int, help='manual seed')
    parser.add_argument('--ckpt', default='',
                        help='folder to output checkpoints')

    parser.add_argument('--optimizer', default='radam')

    args = parser.parse_args()
    print("Input arguments:")
    for key, val in vars(args).items():
        print("{:16} {}".format(key, val))

    if args.optimizer.lower() in ['sgd', 'adam', 'radam']:
        all_gpus = parse_devices(args.gpus)
        all_gpus = [x.replace('gpu', '') for x in all_gpus]
        args.gpus = [int(x) for x in all_gpus]
        num_gpus = len(args.gpus)
        args.batch_size = num_gpus * args.batch_size_per_gpu
        args.gpu = 0

        args.max_iters = args.num_epoch
        args.running_lr_encoder = args.lr_encoder
        args.id += '-'

        args.id += '-ngpus' + str(num_gpus)
        args.id += '-batchSize' + str(args.batch_size)

        args.id += '-LR_unet' + str(args.lr_encoder)

        args.id += '-epoch' + str(args.num_epoch)

        print('Model ID: {}'.format(args.id))

        args.ckpt = os.path.join(args.ckpt, args.id)
        if not os.path.isdir(args.ckpt):
            os.makedirs(args.ckpt)

        random.seed(args.seed)
        torch.manual_seed(args.seed)

        main(args)

    else:
        print("Invalid optimizer")
