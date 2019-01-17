from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os, sys, numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.metrics import average_precision_score

from utils import Logger

import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.transforms as transforms

import multiprocessing
CORES = 4#int(float(multiprocessing.cpu_count())*0.25)

from PascalLoader  import DataLoader
from PascalNetwork import resnet50

def compute_mAP(labels,outputs):
    y_true = labels
    y_pred = outputs.cpu().numpy()
    AP = []

    AP.append(average_precision_score(y_true,y_pred))
    return np.mean(AP)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

val_transform = transforms.Compose([
    # transforms.Scale(256),
    # transforms.CenterCrop(227),
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

val_data = DataLoader('VOC2007', 'test', transform=val_transform)
# val_loader = torch.utils.data.DataLoader(dataset=val_data,
#                                          batch_size=8,
#                                          shuffle=False,
#                                          num_workers=4)
net=resnet50(classes=21).cuda()
net.load_state_dict(torch.load('checkpoints/jps_155.pth'))
image=val_data[2][0]
labels=val_data[2][1]

images = image.view((-1,3,227,227))
images = Variable(images).cuda()

output=net(images).cpu().data
print(output)
print(labels)
# print(compute_mAP(labels,output))