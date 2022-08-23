""" Retraining searched model """
import os
import sys
import argparse
import time
import glob
import logging
import paddle
#import torch
#import torch.nn as nn
import paddle.nn as nn
import numpy as np
from collections import namedtuple
from tensorboardX import SummaryWriter
import utils
from models.augment_cnn import AugmentCNN
from lib.datasets.imagenet import get_augment_datasets
from lib.datasets.tiny_imagenet import get_tiny_datasets
#import torch.backends.cudnn as cudnn
import paddle.optimizer as optim
os.environ['CUDA_VISIBLE_DEVICES']='3'

parser = argparse.ArgumentParser("Retraining Config")
parser.add_argument('--dataset', default='cifar10', help='cifar10/100/tiny_imagenet/imagenet')
parser.add_argument('--train_dir', type=str, default='/data/zqq/datasets/', help='')
parser.add_argument('--val_dir', type=str, default='/data/zqq/datasets/', help='')
parser.add_argument('--use_aa', action='store_true', default=False, help='whether to use aa')
parser.add_argument('--batch_size', type=int, default=300, help='batch size')
parser.add_argument('--lr', type=float, default=0.025, help='lr for weights')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--grad_clip', type=float, default=5.,
                    help='gradient clipping for weights')
parser.add_argument('--print_freq', type=int, default=500, help='print frequency')
parser.add_argument('--gpus', default='all', help='gpu device ids separated by comma. '
                    '`all` indicates use all gpus.')
parser.add_argument('--epochs', type=int, default=100, help='# of training epochs')
parser.add_argument('--init_channels', type=int, default=34)
parser.add_argument('--n_layers', type=int, default=20, help='# of layers')
parser.add_argument('--arch', type=str, default="Genotype(normal=[[('sep_conv_3x3', 1), ('avg_pool_3x3', 0)], [('max_pool_3x3', 2), ('skip_connect', 1)], [('max_pool_3x3', 3), ('max_pool_3x3', 2)], [('avg_pool_3x3', 4), ('max_pool_3x3', 3)]], normal_concat=range(2, 6), reduce=[[('sep_conv_3x3', 0), ('dil_conv_3x3', 1)], [('max_pool_3x3', 2), ('skip_connect', 1)], [('max_pool_3x3', 3), ('skip_connect', 1)], [('max_pool_3x3', 3), ('skip_connect', 1)]], reduce_concat=range(2, 6))", help='which architecture to use')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--workers', type=int, default=16, help='# of workers')
parser.add_argument('--aux_weight', type=float, default=0.4, help='auxiliary loss weight')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path prob')
parser.add_argument('--save', type=str, default='checkpoints/retrain', help='experiment name')
parser.add_argument('--local_rank', type=int, default=0)
parser.add_argument('--world_size', type=int, default=8)
parser.add_argument('--note', type=str, default='try', help='note for this run')

args, unparsed = parser.parse_known_args()

if args.gpus == 'all':
    args.gpus = [3]#list(range(paddle.device.cuda.device_count()))
else:
    args.gpus = [int(s) for s in args.gpus.split(',')]

args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)
Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')
#device = paddle.device("cuda")

def main():
    # if not torch.cuda.is_available():
    #     logging.info('No GPU device available')
    #     sys.exit(1)
    # set default gpu device id
    paddle.set_device('gpu:{}'.format(args.gpus[0]))
    #paddle.cuda.set_device(args.gpus[0])
    g=paddle.seed(args.seed)
    #cudnn.benchmark = True
    g.manual_seed(args.seed)
    #cudnn.enabled=True
    #g=paddle.seed(args.seed)
    #g.manual_seed(args.seed)
    num_gpus = paddle.device.cuda.device_count()
    logging.info("args = %s", args)
    logging.info("unparsed_args = %s", unparsed)
    args.genotype = eval(args.arch)
    print('---------Genotype---------')
    logging.info(args.genotype)
    print('--------------------------')

    # get data with meta info
    if args.dataset == 'imagenet':
        train_loader, valid_loader = get_augment_datasets(args)
        input_size = 128
        input_channels = 3
        n_classes = 1000
    elif args.dataset == 'tiny_imagenet':
        train_loader, valid_loader = get_tiny_datasets(args)
        input_size = 64
        n_classes = 200
        input_channels = 3
    else:
        input_size, input_channels, n_classes, train_loader, valid_loader = utils.get_data(args, cutout_length=0, validation=True)

    criterion = nn.CrossEntropyLoss()#.to(device)
    use_aux = args.aux_weight > 0.
    model = AugmentCNN(args, input_size, input_channels, n_classes, use_aux)
    model = paddle.DataParallel(model)#.to(device)

    # model size
    logging.info("param size = %fMB", utils.param_size(model))

    # weights optimizer
    

    lr_scheduler = optim.lr.CosineAnnealingDecay(learning_rate=args.lr,T_max= int(args.epochs/2), eta_min=0,last_epoch=-1)
    optimizer=optim.Momentum(learning_rate=lr_scheduler,parameters= model.parameters(),weight_decay=args.weight_decay,
                            grad_clip=nn.ClipGradByNorm(args.grad_clip),momentum=args.momentum)
    #optimizer = optim.SGD(learning_rate=lr_scheduler,parameters= model.parameters(), 
    #                            weight_decay=args.weight_decay)
    best_top1 = 0.
    top1 =0
    # training loop
    for epoch in range(args.epochs):
        lr_scheduler.step()
        current_lr = lr_scheduler.get_lr()
        drop_prob = args.drop_path_prob * epoch / args.epochs
        #model.module.drop_path_prob(drop_prob)
        logging.info('Epoch: %d lr %e', epoch, current_lr)

        # training
        train_acc, train_obj = train(train_loader, model, optimizer, criterion, epoch)
        logging.info('Train_acc: %f', train_acc)

        # validation
        cur_step = (epoch+1) * len(train_loader)
        top1, valid_obj = validate(valid_loader, model, criterion, epoch, cur_step)
        logging.info('Valid_acc_top1: %f', top1)
        # save
        if best_top1 < top1:
            best_top1 = top1
            is_best = True
        else:
            is_best = False
        utils.save_checkpoint(model, args.save, is_best)




def train(train_loader, model, optimizer, criterion, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)

    model.train()

    for step, (X, y) in enumerate(train_loader):
        # X = data[0]["data"].cuda()
        # y = data[0]["label"].squeeze().long().cuda()
        #X, y = X.cuda(), y.cuda()
        N = X.shape[0]

        optimizer.clear_grad()
        logits, aux_logits = model(X)
        loss = criterion(logits, y)
        if args.aux_weight > 0.:
            loss += args.aux_weight * criterion(aux_logits, y)
        loss.backward()
        # gradient clipping
        #nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % args.print_freq == 0 or step == len(train_loader)-1:
           logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f',step, losses.avg, top1.avg, top5.avg)

        cur_step += 1
    return top1.avg, losses.avg


def validate(valid_loader, model, criterion, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with paddle.no_grad():
        for step, (X,y) in enumerate(valid_loader):
            #X, y = X.cuda(), y.cuda()
            N = X.size

            logits, _ = model(X)
            loss = criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % args.print_freq == 0 or step == len(valid_loader)-1:
                logging.info('VALID Step: %03d Objs: %e R1: %f R5: %f', step, losses.avg, top1.avg, top5.avg)

    return top1.avg, losses.avg


if __name__ == "__main__":
    main()
