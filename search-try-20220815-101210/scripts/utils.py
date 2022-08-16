""" Utilities """
import os
import logging
import shutil
import paddle
#import torch
from paddle.vision.datasets.cifar import Cifar100,Cifar10
#import torchvision.datasets as dset
import numpy as np
import preproc
from lib.datasets.data_utils import SubsetDistributedSampler
from paddle.fluid.framework import _current_expected_place as _get_device
import paddle.vision.transforms as transforms
#from torch.utils.data import DataLoader
from paddle.io import Dataset, DistributedBatchSampler, DataLoader,Subset,RandomSampler
#import torchvision.transforms as transforms
#from torch.autograd import Variable


def get_data(args, cutout_length, validation):
    """ Get torchvision dataset """
    dataset = args.dataset.lower()

    if dataset == 'cifar10':
        dset_cls = Cifar10
        n_classes = 10
    elif dataset == 'cifar100':
        dset_cls = Cifar100
        n_classes = 100
    elif dataset == 'tiny_imagenet':
        n_classes = 200
    elif dataset == 'imagenet':
        n_classes = 1000
    else:
        raise ValueError(dataset)

    trn_transform, val_transform = preproc.data_transforms(dataset, cutout_length)
    trn_data = dset_cls(mode='train', download=True, transform=trn_transform)

    # assuming shape is NHW or NHWC
    #shape = trn_data.data.shape
    shape =(50000,32,32,3)
    # shape = trn_data.train_data.shape
    input_channels = 3 if len(shape) == 4 else 1
    assert shape[1] == shape[2], "not expected shape = {}".format(shape)
    input_size = shape[1]

    if validation:
        n_train = len(trn_data)
        indices = list(range(n_train))
        train_loader = DataLoader(
                trn_data,
                #batch_sampler=train_sampler,
                places=_get_device(),
                num_workers=args.workers,
                batch_size=args.batch_size,
                return_list=True)
        
        ret = [input_size, input_channels, n_classes, train_loader]
    else:
        n_train = len(trn_data)
        split = n_train // 2
        indices = list(range(n_train))
        tra_data=Subset(trn_data,indices[:split])
        val_data=Subset(trn_data,indices[split:])
        # eval_sampler = DistributedBatchSampler(
        #         eval_data, batch_size=batch_size)
        # train_sampler=RandomSampler(tra_data,replacement=True, num_samples=args.batch_size)
        # valid_sampler=RandomSampler(val_data,replacement=True, num_samples=args.batch_size)
        train_sampler=DistributedBatchSampler(tra_data, batch_size=args.batch_size)
        valid_sampler=DistributedBatchSampler(val_data,batch_size=args.batch_size)
        #train_sampler = SubsetDistributedSampler(trn_data, indices[:split],batch_size=args.batch_size)
        #valid_sampler = SubsetDistributedSampler(trn_data,indices[split:],batch_size=args.batch_size)
        train_loader = DataLoader(trn_data,
                                                   #batch_size=args.batch_size,
                                                   batch_sampler=train_sampler,
                                                   num_workers=args.workers,
                                                   #shuffle=False,
                                                   #drop_last=False
                                                  )
        valid_loader = DataLoader(trn_data,
                                                   #batch_size=args.batch_size,
                                                   batch_sampler=valid_sampler,
                                                   num_workers=args.workers,
                                                   #shuffle=False,
                                                   #drop_last=False
                                                   )

        ret = [input_size, input_channels, n_classes, train_loader,valid_loader]
    if validation:  # append validation data
        ret = [input_size, input_channels, n_classes, train_loader]
        valid_loader = DataLoader(dset_cls(mode='test', download=True, transform=val_transform),
                                                   batch_size=args.batch_size,
                                                   shuffle=False,
                                                   num_workers=args.workers,
                                                   )
        ret.append(valid_loader)

    return ret

def drop_path(x, drop_prob):
  if drop_prob > 0.:
    keep_prob = 1.-drop_prob
    ft=paddle.empty(shape=[x.size(0), 1, 1, 1],dtype='float32')
    mask=paddle.bernoulli(paddle.full_like(ft,keep_prob))
    # torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob)
    mask = paddle.static.Variable(mask)
    x.div_(keep_prob)
    x.mul_(mask)
  return x

def param_size(model):
    """ Compute parameter size in MB """
    n_params=0
    for k, v in model.named_parameters():
        if not k.startswith('aux_head') and not k.startswith('criterion'):
            if v.stop_gradient==False:
                n_params+=np.prod(v.size)
    
    # n_params = sum(
    #     np.prod(v.size) for k, v in model.named_parameters() if not k.startswith('aux_head'))
    return n_params / 1024. / 1024.


class AverageMeter():
    """ Computes and stores the average and current value """
    def __init__(self):
        self.reset()

    def reset(self):
        """ Reset all statistics """
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        """ Update statistics """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """ Computes the precision@k for the specified values of k """
    maxk = max(topk)
    batch_size = target.size

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    # one-hot case
    if target.ndimension() > 1:
        target = target.max(1)[1]
    correct=pred==target.unsqueeze(axis=0).expand_as(pred)
    #correct = pred.eq(paddle.flatten(target,1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = paddle.to_tensor(correct.numpy()[:k].reshape(-1).sum(0))
        res.append(correct_k*(1.0 / batch_size))

    return res


def save_checkpoint(state, ckpt_dir, is_best=False):
    filename = os.path.join(ckpt_dir, 'checkpoint.pth.tar')
    params = state.state_dict()
    paddle.save(params, filename)
    if is_best:
        best_filename = os.path.join(ckpt_dir, 'best.pth.tar')
        shutil.copyfile(filename, best_filename)


def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.makedirs(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)
