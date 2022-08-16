import os
import sys
from torch.utils.data import Dataset, DataLoader
import numpy as np
#import cv2
from torchvision import transforms
import torch
import torchvision.datasets as datasets


#labels_t = []
#image_names = []
#with open('/DATA/disk1/zqq/IDARTS/data/tiny-imagenet-200/wnids.txt') as wnid:
#    for line in wnid:
#        labels_t.append(line.strip('\n'))
#for label in labels_t:
#    txt_path = '/DATA/disk1/zqq/IDARTS/data/tiny-imagenet-200/train/'+label+'/'+label+'_boxes.txt'
#    image_name = []
#    with open(txt_path) as txt:
#        for line in txt:
#            image_name.append(line.strip('\n').split('\t')[0])
#    image_names.append(image_name)
#labels = np.arange(200)
#
#
#
#val_labels_t = []
#val_labels = []
#val_names = []
#with open('/DATA/disk1/zqq/IDARTS/data/tiny-imagenet-200/val/val_annotations.txt') as txt:
#    for line in txt:
#        val_names.append(line.strip('\n').split('\t')[0])
#        val_labels_t.append(line.strip('\n').split('\t')[1])
#for i in range(len(val_labels_t)):
#    for i_t in range(len(labels_t)):
#        if val_labels_t[i] == labels_t[i_t]:
#            val_labels.append(i_t)
#val_labels = np.array(val_labels)
#
#
#class data(Dataset):
#    def __init__(self, type, transform):
#        self.type = type
#        if type == 'train':
#            i = 0
#            self.images = []
#            for label in labels_t:
#                image = []
#                for image_name in image_names[i]:
#                    image_path = os.path.join('/DATA/disk1/zqq/IDARTS/data/tiny-imagenet-200/train', label, 'images', image_name)
#                    image.append(cv2.imread(image_path))
#                self.images.append(image)
#                i = i + 1
#            self.images = np.array(self.images)
#            self.images = self.images.reshape(-1, 64, 64, 3)
#        elif type == 'val':
#            self.val_images = []
#            for val_image in val_names:
#                val_image_path = os.path.join('/DATA/disk1/zqq/IDARTS/data/tiny-imagenet-200/val/images', val_image)
#                self.val_images.append(cv2.imread(val_image_path))
#            self.val_images = np.array(self.val_images)
#        self.transform = transform
#        
#    def __getitem__(self, index):
#        label = []
#        image = []
#        if self.type == 'train':
#            label = index//500
#            image = self.images[index]
#        if self.type == 'val':
#            label = val_labels[index]
#            image = self.val_images[index]
#        return label, self.transform(image)
#        
#    def __len__(self):
#        len = 0
#        if self.type == 'train':
#            len = self.images.shape[0]
#        if self.type == 'val':
#            len = self.val_images.shape[0]
#        return len
#        
#        
#def get_tiny_datasets(config):
#    train_dataset = data('train', transform=transforms.Compose([transforms.ToTensor()]))
#    val_dataset = data('val', transform=transforms.Compose([transforms.ToTensor()]))
#    
#    train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size, shuffle=True)
#
#    val_dataloader = DataLoader(dataset=val_dataset, batch_size=config.batch_size, shuffle=False)
#    return train_dataloader, val_dataloader
    
def get_tiny_datasets(args):
    num_label = 200
    normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
    transform_train = transforms.Compose(
        [transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
         normalize, ])
    transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
    trainset = datasets.ImageFolder(root=os.path.join(args.train_dir, 'train'), transform=transform_train)
    testset = datasets.ImageFolder(root=os.path.join(args.val_dir, 'val'), transform=transform_test)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader