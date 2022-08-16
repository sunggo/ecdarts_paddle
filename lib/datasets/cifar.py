#import torch
import paddle
import numpy as np
#import torchvision.datasets as dset
from paddle.vision.datasets.cifar import Cifar100,Cifar10
from paddle.fluid.framework import _current_expected_place as _get_device
#import torchvision.transforms as transforms
import paddle.vision.transforms as transforms
from lib.datasets.data_utils import SubsetDistributedSampler
from lib.datasets.data_utils import CIFAR10Policy, Cutout
from paddle.io import Dataset, DistributedBatchSampler, DataLoader

def data_transforms_cifar(config, cutout=False):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    if config.use_aa:
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4, fill=128),
        transforms.RandomHorizontalFlip(), CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])
    else:
        train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])


    if cutout:
        train_transform.transforms.append(Cutout(config.cutout_length))

    valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform

def get_search_datasets(config):

    dataset = config.dataset.lower()
    if dataset == 'cifar10':
        dset_cls = Cifar10
        n_classes = 10
    elif dataset == 'cifar100':
        dset_cls = Cifar100
        n_classes = 100
    else:
        raise Exception("Not support dataset!")

    train_transform, valid_transform = data_transforms_cifar(config, cutout=False)
    train_data = dset_cls(data_file=config.data_dir, mode='train', download=True, transform=train_transform)
    test_data = dset_cls(data_file=config.data_dir, mode='test', download=True, transform=valid_transform)

    num_train = len(train_data)
    indices = list(range(num_train))
    split_mid = int(np.floor(0.5 * num_train))

    train_sampler = SubsetDistributedSampler(train_data, indices[:split_mid])
    valid_sampler = SubsetDistributedSampler(train_data, indices[split_mid:num_train])
    train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                places=_get_device(),
                num_workers=config.workers,
                batch_size=config.batch_size,
                return_list=True)
    # data_loader = run_manager.run_config
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=config.batch_size,
    #     sampler=train_sampler,
    #     pin_memory=True, num_workers=config.workers)
    valid_loader = DataLoader(
                train_data,
                batch_sampler=valid_sampler,
                places=_get_device(),
                num_workers=config.workers,
                batch_size=config.batch_size,
                return_list=True)
    # valid_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=config.batch_size,
    #     sampler=valid_sampler,
    #     pin_memory=True, num_workers=config.workers)

    return [train_loader, valid_loader], [train_sampler, valid_sampler]

def get_augment_datasets(config):

    dataset = config.dataset.lower()
    if dataset == 'cifar10':
        dset_cls = Cifar10
    elif dataset == 'cifar100':
        dset_cls = Cifar100
    else:
        raise Exception("Not support dataset!")

    train_transform, valid_transform = data_transforms_cifar(config, cutout=True)
    train_data = dset_cls(data_file=config.data_dir, mode='train', download=True, transform=train_transform)
    test_data = dset_cls(data_file=config.data_dir, mode='test', download=True, transform=valid_transform)

    train_sampler = DistributedBatchSampler(train_data,batch_size=config.batch_size)
    test_sampler = DistributedBatchSampler(test_data, batch_size=config.batch_size)
    #train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    #test_sampler = torch.utils.data.distributed.DistributedSampler(test_data)
    train_loader = DataLoader(
                train_data,
                batch_sampler=train_sampler,
                places=_get_device(),
                num_workers=config.workers,
                batch_size=config.batch_size,
                return_list=True)
    test_loader = DataLoader(
                test_data,
                batch_sampler=test_sampler,
                places=_get_device(),
                num_workers=config.workers,
                batch_size=config.batch_size,
                return_list=True)
    # train_loader = torch.utils.data.DataLoader(
    #     train_data, batch_size=config.batch_size,
    #     sampler=train_sampler,
    #     pin_memory=True, num_workers=config.workers)

    # test_loader = torch.utils.data.DataLoader(
    #     test_data, batch_size=config.batch_size,
    #     sampler=test_sampler,
    #     pin_memory=True, num_workers=config.workers)

    return [train_loader, test_loader], [train_sampler, test_sampler]

