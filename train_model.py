# -*- coding: utf-8 -*-

__author__ = 'kohou.wang'
__time__ = '19-11-20'
__email__ = 'oukohou@outlook.com'

# If this runs wrong, don't ask me, I don't know why;
# If this runs right, thank god, and I don't know why.
# Maybe the answer, my friend, is blowing in the wind.
# Well, I'm kidding... Always, Welcome to contact me.

"""Description for the script:
train models.
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import time
import copy
from sklearn.model_selection import train_test_split
import pandas as pd
from datasets.read_data import ReadImageDataset
# from models.MixNet import MixNet
from torch.nn import init
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import Tensor
# import torch.nn.functional as F
from torch.autograd import Variable
# from torchvision import transforms as T
# from models.EfficientNet import EfficientNet
# from models.DPN import dpns
from models.mixup_utils import mixup_criterion, mixup_data
from models.GhostNet import ghost_net
from models.Resnext import resnext101_32x16d_swsl, resnext101_32x4d_swsl, resnext50_32x4d_swsl
import numpy as np
from models.EfficientNet import EfficientNet
from models.CBAM_Resnet import ResNetCBAM

print('torch.cuda.device_count : {}'.format(torch.cuda.device_count()))
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# TODO: check detach
class CircleLoss(nn.Module):
    def __init__(self, m: float, gamma: float) -> None:
        super(CircleLoss, self).__init__()
        self.m = m
        self.gamma = gamma
        self.loss = nn.CrossEntropyLoss()
    
    def forward(self, inp: Tensor, label: Tensor) -> Tensor:
        a = torch.clamp_min(inp + self.m, min=0).detach()
        a[label] = torch.clamp_min(- inp[label] + 1 + self.m, min=0).detach()
        sigma = torch.ones_like(inp, device=inp.device, dtype=inp.dtype) * self.m
        sigma[label] = 1 - self.m
        return self.loss(a * (inp - sigma) * self.gamma, label)


class LabelSmoothSoftmaxCEV1(nn.Module):
    
    def __init__(self, lb_smooth=0.1, reduction='mean', lb_ignore=-100, weights=None):
        super(LabelSmoothSoftmaxCEV1, self).__init__()
        self.lb_smooth = lb_smooth
        self.reduction = reduction
        self.lb_ignore = lb_ignore
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.weights = None
        if weights is not None:
            self.weights = weights
    
    def forward(self, logits, label):
        '''
        args: logits: tensor of shape (N, C, H, W)
        args: label: tensor of shape(N, H, W)
        '''
        # overcome ignored label
        with torch.no_grad():
            num_classes = logits.size(1)
            label = label.clone().detach()
            ignore = label == self.lb_ignore
            n_valid = (ignore == 0).sum()
            label[ignore] = 0
            lb_pos, lb_neg = 1. - self.lb_smooth, self.lb_smooth / num_classes
            label = torch.empty_like(logits).fill_(
                lb_neg).scatter_(1, label.unsqueeze(1), lb_pos).detach()
            if self.weights is not None:
                label = label * self.weights
        
        # import ipdb
        # ipdb.set_trace()
        logs = self.log_softmax(logits)
        loss = -torch.sum(logs * label, dim=1)
        loss[ignore] = 0
        if self.reduction == 'mean':
            loss = loss.sum() / n_valid
        if self.reduction == 'sum':
            loss = loss.sum()
        
        return loss


def train_model(model_, dataloaders_, Loss_criterion_, feat_criterion_, optimizer_, num_epochs_=25, Use_Mixnet=False,
                tensorboard_writer=None):
    global lr_scheduler
    
    since = time.time()
    val_acc_history = []
    
    best_model_wts = copy.deepcopy(model_.state_dict())
    best_acc = 0.0
    for epoch in range(num_epochs_):
        print('\nEpoch {}/{}'.format(epoch, num_epochs_ - 1))
        print('-' * 10)
        
        # for phase in ['train', 'val']:
        phase_list = sorted(dataloaders_.keys())
        if len(phase_list) == 3:
            phase_list = ['train', 'val', 'test']
        for phase in phase_list:
            if phase == 'train':
                model_.train()  # Set model to training mode
                print('in train mode...')
            else:
                print('in {} mode...'.format(phase))
                model_.eval()  # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects_1 = 0
            running_corrects_2 = 0
            # groud_truth = []
            # predictions = []
            for i, (inputs, labels_5, label_lists_54) in enumerate(dataloaders_[phase]):
                inputs = inputs.to(device)
                labels_5 = labels_5.to(device)
                label_lists_54 = label_lists_54.to(device)
                
                # zero the parameter gradients
                optimizer_.zero_grad()
                
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # feats_54, preds_5 = model_(inputs)
                    feats_54 = model_(inputs)
                    # import ipdb
                    # ipdb.set_trace()
                    # if MixNet:
                    #     End_loss = Loss_criterion_(preds_5, labels_5)
                    feat_loss = feat_criterion_(feats_54, label_lists_54)
                    # feat_loss = feat_criterion_(F.log_softmax(feats_54 + + 1e-10, dim=1), label_lists)
                    _, feats_54 = torch.max(feats_54, 1)
                    # _, preds_5 = torch.max(preds_5, 1)
                    
                    # if Use_Mixnet:
                    #     feat_loss = 20 * feat_loss + End_loss
                    # total_loss = 20 * feat_loss + End_loss
                    if phase == 'train':
                        feat_loss.backward()
                        optimizer_.step()
                
                # statistics
                running_loss += feat_loss.item() * inputs.size(0)
                running_corrects_1 += torch.sum(feats_54 == label_lists_54.data)  # .long())
                # running_corrects_2 += torch.sum(preds_5 == labels_5.data)  # .long())
            
            # print("feat preds_5:{}".format(feats_preds))
            print("label_54 list:{}".format(label_lists_54))
            print("preds_54 list:{}".format(feats_54))
            epoch_loss = running_loss / len(dataloaders_[phase].dataset)
            epoch_acc = running_corrects_1.double() / len(dataloaders_[phase].dataset)
            
            if Use_Mixnet:
                pred_ = running_corrects_2.double() / len(dataloaders_[phase].dataset)
                print('{} Loss: {:.4f} pred_54_acc: {:.4f}, preds_5 acc:{}'.format(phase, epoch_loss, epoch_acc, pred_))
            else:
                print('{} Loss: {:.4f} pred_54_acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            time_elapsed = time.time() - since
            print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model_.state_dict())
            if phase == 'val':
                val_acc_history.append(epoch_acc)
        
        lr_scheduler.step(epoch)
        if tensorboard_writer:
            tensorboard_writer.add_graph(model_, inputs)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val epoch_acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model_.load_state_dict(best_model_wts)
    return model_, val_acc_history


def train_model_without_valid(model_, dataloaders_, Loss_criterion_, feat_criterion_, optimizer_, num_epochs_=25, Use_Two_outputs=False,
                              save_path_format_=''):
    global lr_scheduler
    
    since = time.time()
    val_acc_history = []
    best_epoch_acc = 0.0
    best_model_weights = copy.deepcopy(model_.state_dict())
    for epoch in range(num_epochs_):
        print('\nEpoch {}/{}'.format(epoch, num_epochs_ - 1))
        print('-' * 10)
        
        model_.train()  # Set model to training mode
        print('in train mode...')
        
        running_loss = 0.0
        running_corrects_1 = 0
        running_corrects_2 = 0
        for i, (inputs, labels_10) in enumerate(dataloaders_['train']):
            inputs = inputs.to(device)
            labels_10 = labels_10.to(device)
            
            # zero the parameter gradients
            optimizer_.zero_grad()
            
            # track history if only in train
            with torch.set_grad_enabled(True):
                # _, feats_54 = model_(inputs)
                feats_10 = model_(inputs)
                # if MixNet:
                #     End_loss = Loss_criterion_(preds_5, labels_10)
                # import ipdb
                # ipdb.set_trace()
                feat_loss = feat_criterion_(feats_10, labels_10)
                # feat_loss = feat_criterion_(F.log_softmax(feats_54 + + 1e-10, dim=1), label_lists)
                _, feats_10 = torch.max(feats_10, 1)
                
                # if Use_Mixnet:
                #     feat_loss = 20 * feat_loss + End_loss
                feat_loss.backward()
                optimizer_.step()
            
            # statistics
            running_loss += feat_loss.item() * inputs.size(0)
            running_corrects_1 += torch.sum(feats_10 == labels_10.data)  # .long())
        
        # print("feat preds_5:{}".format(feats_preds))
        print("labels_10 list:{}".format(labels_10))
        print("preds_10 list:{}".format(feats_10))
        epoch_loss = running_loss / len(dataloaders_['train'].dataset)
        epoch_acc = running_corrects_1.double() / len(dataloaders_['train'].dataset)
        val_acc_history.append(epoch_acc)
        
        if Use_Two_outputs:
            pred_ = running_corrects_2.double() / len(dataloaders_['train'].dataset)
            print('{} Loss: {:.4f} pred_10_acc: {:.4f}, preds_5 acc:{}'.format('train', epoch_loss, epoch_acc, pred_))
        else:
            print('{} Loss: {:.4f} pred_10_acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
        time_elapsed = time.time() - since
        print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        lr_scheduler.step(epoch)
        
        # if epoch % (num_epochs_//3) == 0 and save_path_format_ != '':
        if epoch % 10 == 0 and save_path_format_ != '':
            save_path_ = save_path_format_.format(
                "input{}".format(input_size), use_different_lr, batch_size,
                learning_rate,
                epoch
            )
            torch.save(
                #     {
                #     'epoch': num_epochs,
                #     'state_dict': model_to_train.state_dict(),
                #     'optimizer_state_dict': optimizer_ft.state_dict(),
                # },
                model_to_train.state_dict()
                ,
                save_path_
            )
        
        # save best acc's model weights
        if epoch_acc > best_epoch_acc:
            best_epoch_acc = epoch_acc
            best_model_weights = copy.deepcopy(model_.state_dict())
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Directly use last train epoch_acc: {:4f}'.format(epoch_acc))
    print("epoch acc list:{}".format(val_acc_history))
    print("epoch acc list:{}, \nbest epoch at:{}".format(val_acc_history, val_acc_history.index(max(val_acc_history))))
    
    return model_, best_model_weights


def train_model_withoutValid_withMixUp(model_, dataloaders_, Loss_criterion_, feat_criterion_, optimizer_, num_epochs_=25,
                                       Use_Two_outputs=False,
                                       save_path_format_=''):
    global lr_scheduler
    global mixup_alpha
    since = time.time()
    val_acc_history = []
    
    best_epoch_acc = 0.0
    best_model_weights = copy.deepcopy(model_.state_dict())
    for epoch in range(num_epochs_):
        print('\nEpoch {}/{}'.format(epoch, num_epochs_ - 1))
        print('-' * 10)
        
        model_.train()  # Set model to training mode
        print('in train mode...')
        
        running_loss = 0.0
        running_corrects_1 = 0
        running_corrects_2 = 0
        for i, (inputs, labels_10) in enumerate(dataloaders_['train']):
            inputs = inputs.to(device)
            # labels_5 = labels_5.to(device)
            label_lists_10 = labels_10.to(device)
            
            inputs, targets_a, targets_b, lam = mixup_data(inputs, label_lists_10, mixup_alpha, True)
            # zero the parameter gradients
            optimizer_.zero_grad()
            inputs, targets_a, targets_b = Variable(inputs), Variable(targets_a), Variable(targets_b)
            # track history if only in train
            with torch.set_grad_enabled(True):
                # _, feats_54 = model_(inputs)
                feats_10 = model_(inputs)
                # import ipdb
                # ipdb.set_trace()
                loss_func = mixup_criterion(targets_a, targets_b, lam)
                feat_loss = loss_func(feat_criterion_, feats_10)
                # feat_loss = feat_criterion_(F.log_softmax(feats_54 + + 1e-10, dim=1), label_lists)
                _, feats_10 = torch.max(feats_10, 1)
                
                # if Use_Mixnet:
                #     feat_loss = 20 * feat_loss + End_loss
                feat_loss.backward()
                optimizer_.step()
            
            # statistics
            running_loss += feat_loss.item() * inputs.size(0)
            # import ipdb
            # ipdb.set_trace()
            running_corrects_1 += lam * float(feats_10.eq(targets_a.data).cpu().sum()) + (1 - lam) * float(
                feats_10.eq(targets_b.data).cpu().sum())
            # running_corrects_1 += torch.sum(feats_54 == label_lists_54.data)  # .long())
        
        # print("feat preds_5:{}".format(feats_preds))
        print("labels_10 list:{}".format(labels_10))
        print("preds_10 list:{}".format(feats_10))
        epoch_loss = running_loss / len(dataloaders_['train'].dataset)
        # import ipdb
        # ipdb.set_trace()
        epoch_acc = running_corrects_1 / len(dataloaders_['train'].dataset)
        val_acc_history.append(epoch_acc)
        
        if Use_Two_outputs:
            pred_ = running_corrects_2.double() / len(dataloaders_['train'].dataset)
            print('{} Loss: {:.4f} pred_54_acc: {:.4f}, preds_5 acc:{}'.format('train', epoch_loss, epoch_acc, pred_))
        else:
            print('{} Loss: {:.4f} pred_54_acc: {:.4f}'.format('train', epoch_loss, epoch_acc))
        time_elapsed = time.time() - since
        print('Complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        
        lr_scheduler.step(epoch)
        
        # if epoch % (num_epochs_//3) == 0 and save_path_format_ != '':
        if epoch % 10 == 0 and save_path_format_ != '':
            save_path_ = save_path_format_.format(
                "input{}".format(input_size), use_different_lr, batch_size,
                learning_rate,
                epoch
            )
            torch.save(
                # {
                # 'epoch': num_epochs,
                # 'state_dict': model_to_train.state_dict(),
                # 'optimizer_state_dict': optimizer_ft.state_dict(),
                # },
                model_to_train.state_dict(),
                save_path_
            )
        # save best acc's model weights
        if epoch_acc > best_epoch_acc:
            best_epoch_acc = epoch_acc
            best_model_weights = copy.deepcopy(model_.state_dict())
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Directly use last train epoch_acc: {:4f}'.format(epoch_acc))
    print("epoch acc list:{}, \nbest epoch at:{}".format(val_acc_history, val_acc_history.index(max(val_acc_history))))
    
    return model_, best_model_weights


if __name__ == "__main__":
    train_data_base_path = '/home/kohou/cvgames/interest/contest/MARS/FuBuFu/data/data/train'
    # train_data_base_path = '/home/data/CVAR-B/study/interest/contest/HUAWEI/foods/data/merged_data/merged_baidu_google_all'
    batch_size = 8  # for finetuning resnext101_32x16d_swsl's fc only
    # batch_size = 8 # for finetuning whole resnext101_32x16d_swsl model
    # batch_size = 32  # for finetuning whole ghostNet model
    input_size = 224  # 224 for mixnet, resnext,rexnetv1
    # input_size = 416  # for resneSt269
    # input_size = 256  # for resneSt101
    # input_size = 269  # 269 for resneSt269
    # input_size = 600  # 600 for efficientbet_b7
    # input_size = 456  # 600 for efficientbet_b5
    # input_size = 300  # 600 for efficientbet_b3
    num_epochs = 30
    # num_epochs = 60 # for rexnetv1
    num_classes = 2
    learning_rate = 0.04  # originally 0.001
    learning_rate = 0.002  # originally 0.001
    weight_decay = 1e-4  # originally 1e-4
    mixup_alpha = 0.4  # originally 1.
    auto_augment = True
    use_cutout = True
    use_base_data_path = True
    load_pretrained = True
    use_different_lr = False
    finetune_fc_only = False
    net_type = 'mixnet_s'
    net_type = 'efficientnet-b3'
    cutmix_beta = 1.0
    cutmix_prob = 1.
    
    # model_to_train = EfficientNet.from_pretrained('efficientnet-b7', num_classes=num_classes)
    # model_to_train = EfficientNet.from_name('efficientnet-b3', override_params={'num_classes': num_classes})
    # model_to_train = EfficientNet.from_name('efficientnet-b5', override_params={'num_classes': num_classes})
    # model_to_train = MixNet(net_type=net_type, input_size=input_size, num_classes=5, feature_size=54)
    # model_to_train = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x16d_swsl')
    # model_to_train = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext50_32x4d_swsl')
    # model_to_train = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnext101_32x4d_swsl')
    model_to_train = resnext101_32x4d_swsl(num_classes=num_classes)
    # model_to_train = ResNetCBAM([3, 4, 23, 3], num_classes, 'ＣBAM')
    # model_to_train = dpns['dpn107'](num_classes=54)
    if load_pretrained:
        print("loading model...")
        loaded_model = torch.load(
            '/home/kohou/cvstudy/unicome/pretrained_models/resneXt/semi_weakly_supervised_resnext101_32x4-3f87e46b.pth',
            # map_location=device
        )
        # model_to_train.load_state_dict(loaded_model['state_dict'])  # mine pretrained model
        model_to_train.load_state_dict(loaded_model)  # torch's pretrained model
        del loaded_model
    # model_to_train.fc = nn.Linear(2048, num_classes)  # resnext101, resneSt, fcanet
    # model_to_train._fc = nn.Linear(1536, num_classes)  # efficientnetb3
    # # # # # # #
    # # #
    # # #
    # for m in model_to_train.modules():
    #     if isinstance(m, nn.Linear):
    #         init.normal_(m.weight, std=0.001)
    #         if m.bias is not None:
    #             init.constant_(m.bias, 0)
    # # #
    
    if finetune_fc_only:
        print('finetune fc layer only...')
        for name, param in model_to_train.named_parameters():
            param.requires_grad = False
        for name, param in model_to_train.fc.named_parameters():
            param.requires_grad = True
    
    # model_to_train.classifier = nn.Linear(2688, 54)  # for DPN107
    
    total_images = pd.read_csv('./datasets/train_label.csv')
    total_images = total_images.sample(frac=1., random_state=2021)
    print("total images:{}".format(len(total_images)))
    train_data_list = total_images
    # train_data_list, val_data_list = train_test_split(total_images, test_size=0.1, random_state=2019 + 1 + 1)
    
    train_gen = ReadImageDataset(total_images, train_data_base_path, mode="train",
                                 auto_augment=auto_augment, input_size=input_size, cutout=use_cutout,
                                 use_base_data_path=use_base_data_path
                                 )
    # val_gen = ReadImageDataset(val_data_list, train_data_base_path,
    #                            auto_augment=auto_augment, input_size=input_size,
    #                            mode="train", cutout=use_cutout,
    #                            )
    
    train_loader = DataLoader(train_gen, batch_size=batch_size, shuffle=True, pin_memory=True,
                              num_workers=4,
                              # drop_last=True
                              )
    # val_loader = DataLoader(val_gen, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=0)
    
    total_dataloader = {
        'train': train_loader,
        # 'val': val_loader,
        # 'test': test_loader,
    }
    
    model_to_train = model_to_train.to(device)
    # model_to_train = model_to_train.cuda()
    
    params_to_update = model_to_train.parameters()
    # Observe that all parameters are being optimized
    if not finetune_fc_only and not use_different_lr:
        print("train whole model...")
        optimizer_ft = optim.SGD(params_to_update, lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    elif not use_different_lr:
        optimizer_ft = optim.SGD(model_to_train.fc.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        # optimizer_ft = optim.SGD(model_to_train._fc.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)  # efficientnet
    else:
        print("use different lr.")
        backbone_params = []
        fc_params = list(map(id, model_to_train.fc.parameters()))  # resnext
        # fc_params = list(map(id, model_to_train._fc.parameters()))  # efficientnet
        backbone_params = filter(lambda x: id(x) not in fc_params, model_to_train.parameters())
        # for m in model_to_train.modules():
        #     if not isinstance(m, nn.Linear):
        #         backbone_params += m.parameters()
        
        # for name, param in model_to_train.named_parameters():
        #     if 'fc' not in name:
        #         backbone_params = backbone_params + param
        optimizer_ft = optim.SGD(
            params=[{'params': backbone_params, 'lr': learning_rate},
                    {'params': model_to_train.fc.parameters(), 'lr': 5 * learning_rate}],  # resnext101
            # {'params': model_to_train._fc.parameters(), 'lr': 5 * learning_rate}],  # efficientnet
            lr=learning_rate, momentum=0.9, weight_decay=weight_decay
        )
    
    cross_criterion = nn.CrossEntropyLoss()
    # KL_criterion = nn.KLDivLoss(reduction='batchmean')
    
    value_counts = train_data_list['label'].value_counts().to_dict()
    label_num = [value_counts[i] for i in range(len(value_counts))]
    ratio = [sum(label_num) / i for i in label_num]
    weight_ratio = [i / sum(ratio) for i in ratio]
    
    # print("train images:{}, valid images: {}".format(len(train_data_list), len(val_data_list)))
    print("train images:{}, valid images: {}".format(len(train_data_list), 0))
    print("train label:{}".format(train_data_list['label'].value_counts().to_dict()))
    # print("val label:{}".format(val_data_list['label'].value_counts().to_dict()))
    print("train weights:{}".format(weight_ratio))
    
    # weights = torch.tensor([1 - 0.62146, 0.62146])
    # feat_criterion = nn.CrossEntropyLoss(torch.tensor(weight_ratio).to(device))
    feat_criterion = LabelSmoothSoftmaxCEV1(weights=torch.tensor(weight_ratio).to(device))
    # feat_criterion = LabelSmoothSoftmaxCEV1()
    # feat_criterion = CircleLoss(m=0.25, gamma=2.56)
    # lr_scheduler = optim.lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
    # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer_ft, T_max=10)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer_ft, milestones=[15, 25], gamma=0.1)
    
    save_path_format = \
        '/home/kohou/cvgames/interest/contest/MARS/FuBuFu/trained_models/fcanet/FinetuneOnmixup_{}_DifferentLR__{}_batch{}_lr{}_epoch{}.pth'
    
    # # Train and evaluate
    # model_to_train, hist = train_model(model_to_train, total_dataloader, cross_criterion, KL_criterion, optimizer_ft,
    #                                    num_epochs_=num_epochs,
    #                                    tensorboard_writer=None
    #                                    )
    # # train without evaluate, use the last epoch directly. No bicycle!
    model_to_train, best_model_weights_trained = train_model_without_valid(model_to_train, total_dataloader, cross_criterion,
                                                                           feat_criterion, optimizer_ft,
                                                                           num_epochs_=num_epochs, save_path_format_=save_path_format
                                                                           )
    # # # MixUp,train without evaluate, use the last epoch directly. No bicycle!
    # model_to_train, best_model_weights_trained = train_model_withoutValid_withMixUp(model_to_train, total_dataloader, cross_criterion,
    #                                                                                 feat_criterion,
    #                                                                                 optimizer_ft,
    #                                                                                 num_epochs_=num_epochs,
    #                                                                                 save_path_format_=save_path_format
    #                                                                                 )
    
    save_path = '/home/kohou/cvgames/interest/contest/MARS/FuBuFu/trained_models/fcanet/FinetuneOnmixup_{}_DifferentLR__{}_batch{}_lr{}_epoch{}.pth'.format(
        "input{}".format(input_size), use_different_lr,
        batch_size, learning_rate,
        num_epochs)
    torch.save(
        #     {
        #     'epoch': num_epochs,
        #     'state_dict': model_to_train.state_dict(),
        #     'optimizer_state_dict': optimizer_ft.state_dict(),
        # },
        model_to_train.state_dict(),
        save_path
    )
    torch.save(best_model_weights_trained,
               os.path.join(os.path.dirname(save_path), "best_epoch_{}".format(os.path.basename(save_path))))
    print('model saved to {}.'.format(save_path))
    
    # cropped_test_image_path = '/home/data/CVAR-B/study/interest/contest/dataFountain/火眼金睛/data/cropped_images/stage3_datasets'
    # submission = '/home/data/CVAR-B/study/interest/contest/dataFountain/火眼金睛/submit/Mixnet+augment.csv'
    # test_image_base_path = '/home/data/CVAR-B/study/interest/contest/dataFountain/火眼金睛/data/stag3_dataset'
    # inference_results_classification_(model_to_train, test_image_base_path, cropped_test_image_path, submission)
