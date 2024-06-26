# -------------------------------------#
#       Train on the dataset
# -------------------------------------#
import os
import numpy as np
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from utils.config import Config
from torch.utils.data import DataLoader
from utils.dataloader import yolo_dataset_collate, YoloDataset
from nets.yolo_training import YOLOLoss, Generator
from nets.yolo3 import YoloBody
from tqdm import tqdm
from torchsummary import summary


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda):
    total_loss = 0
    val_loss = 0
    start_time = time.time()
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(3):
                loss_item = yolo_losses[i](outputs[i], targets)
                losses.append(loss_item[0])
            loss = sum(losses)
            loss.backward()
            optimizer.step()

            total_loss += loss
            waste_time = time.time() - start_time

            pbar.set_postfix(**{'total_loss': total_loss.item() / (iteration + 1),
                                'lr': get_lr(optimizer),
                                'step/s': waste_time})
            pbar.update(1)

            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            images_val, targets_val = batch[0], batch[1]

            with torch.no_grad():
                if cuda:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor)).cuda()
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                else:
                    images_val = Variable(torch.from_numpy(images_val).type(torch.FloatTensor))
                    targets_val = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets_val]
                optimizer.zero_grad()
                outputs = net(images_val)
                losses = []
                for i in range(3):
                    loss_item = yolo_losses[i](outputs[i], targets_val)
                    losses.append(loss_item[0])
                loss = sum(losses)
                val_loss += loss
            pbar.set_postfix(**{'total_loss': val_loss.item() / (iteration + 1)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), 'logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
        (epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))

    with open('train.log', 'a+') as f:
        f.write(
            '\n{},{:.5f},{:.5f}'.format((epoch + 1), total_loss / (epoch_size + 1), val_loss / (epoch_size_val + 1)))


if __name__ == "__main__":
    # Parameter initialization
    annotation_path = '2007_train.txt'
    model = YoloBody(Config)
    summary(model, (3, 416, 416), device='cpu')
    Cuda = True
    # -------------------------------#
    #   Use of Dataloder
    # -------------------------------#
    Use_Data_Loader = True

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load("model_data/yolo_weights.pth", map_location=device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    print('Finished!')

    net = model.train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    # Create loss function
    yolo_losses = []
    for i in range(3):
        yolo_losses.append(YOLOLoss(np.reshape(Config["yolo"]["anchors"], [-1, 2]),
                                    Config["yolo"]["classes"], (Config["img_w"], Config["img_h"]), Cuda))

    # 0.1 for validation and 0.9 for training
    val_split = 0.1
    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_val = int(len(lines) * val_split)
    num_train = len(lines) - num_val

    with open('train.log', 'w+') as f:
        f.write('epoch,loss,val_loss')
    # ------------------------------------------------------#
    #   Backbone feature extraction network features are universal, and frozen training can speed up training.
    #   It can also prevent the weights from being destroyed in the early stages of training.
    #   Init_Epoch is the starting epoch
    #   Freeze_Epoch is the epoch that freezes training
    #   Epoch is the total training epoch
    #   If it prompts OOM or insufficient video memory, please adjust the Batch_size smaller.
    # ------------------------------------------------------#
    if True:
        # Initially, using a learning rate of 1e-3 can converge faster.
        lr = 1e-3
        Batch_size = 16
        Init_Epoch = 0
        Freeze_Epoch = 50

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]))
            val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (Config["img_h"], Config["img_w"])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                                (Config["img_h"], Config["img_w"])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   Freeze certain portions of training
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = False

        for epoch in range(Init_Epoch, Freeze_Epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Freeze_Epoch, Cuda)
            lr_scheduler.step()

    if True:
        lr = 1e-4
        Batch_size = 8
        Freeze_Epoch = 50
        Unfreeze_Epoch = 100

        optimizer = optim.Adam(net.parameters(), lr)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)
        if Use_Data_Loader:
            train_dataset = YoloDataset(lines[:num_train], (Config["img_h"], Config["img_w"]))
            val_dataset = YoloDataset(lines[num_train:], (Config["img_h"], Config["img_w"]))
            gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                             drop_last=True, collate_fn=yolo_dataset_collate)
            gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=4, pin_memory=True,
                                 drop_last=True, collate_fn=yolo_dataset_collate)
        else:
            gen = Generator(Batch_size, lines[:num_train],
                            (Config["img_h"], Config["img_w"])).generate()
            gen_val = Generator(Batch_size, lines[num_train:],
                                (Config["img_h"], Config["img_w"])).generate()

        epoch_size = num_train // Batch_size
        epoch_size_val = num_val // Batch_size
        # ------------------------------------#
        #   Training after unfreezing
        # ------------------------------------#
        for param in model.backbone.parameters():
            param.requires_grad = True

        for epoch in range(Freeze_Epoch, Unfreeze_Epoch):
            fit_ont_epoch(net, yolo_losses, epoch, epoch_size, epoch_size_val, gen, gen_val, Unfreeze_Epoch, Cuda)
            lr_scheduler.step()
