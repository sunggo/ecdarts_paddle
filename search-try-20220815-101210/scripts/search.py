
""" Search cells """
import os
import sys
import argparse
import time
import glob
import paddle
#import torch
#import torch.nn as nn
import paddle.optimizer as optim
import paddle.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
import utils
import logging
from models.search_cnn import Network
from architect import Architect
from visualize import plot
from lib.datasets.imagenet import get_search_datasets
from lib.datasets.tiny_imagenet import get_tiny_datasets
from model import IST
#from lib.datasets.imagenet import get_imagenet_iter_dali
os.environ['CUDA_VISIBLE_DEVICES']='3'


parser = argparse.ArgumentParser("Search config")
parser.add_argument('--dataset', default='cifar10',help='cifar10/100/tiny_imagenet/imagenet')
parser.add_argument('--train_dir', type=str, default='data/zqq/datasets/cifar-10-batches-py/', help='')
parser.add_argument('--val_dir', type=str, default='data/zqq/datasets/cifar-10-batches-py/',
                    help='')
parser.add_argument('--batch_size', type=int, default=150, help='batch size')
parser.add_argument('--w_lr', type=float, default=0.025, help='lr for weights')
parser.add_argument('--w_lr_min', type=float, default=0.001, help='minimum lr for weights')
parser.add_argument('--w_momentum', type=float, default=0.9, help='momentum for weights')
parser.add_argument('--w_weight_decay', type=float, default=3e-4,
                    help='weight decay for weights')
parser.add_argument('--w_grad_clip', type=float, default=5.,
                    help='gradient clipping for weights')
parser.add_argument('--print_freq', type=int, default=50, help='print frequency')
parser.add_argument('--gpus', default='all', help='gpu device ids separated by comma. '
                                                  '`all` indicates use all gpus.')
parser.add_argument('--save_path', type=str, default='checkpoints/', help='experiment name')
parser.add_argument('--plot_path', type=str, default='plot/', help='experiment name')
parser.add_argument('--save', type=str, default='', help='experiment name')
parser.add_argument('--epochs', type=int, default=25, help='# of training epochs')
parser.add_argument('--init_channels', type=int, default=16)
parser.add_argument('--layers', type=int, default=8, help='# of layers')
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--workers', type=int, default=4, help='# of workers')
parser.add_argument('--alpha_lr', type=float, default=3e-4, help='lr for alpha')
parser.add_argument('--alpha_weight_decay', type=float, default=1e-3,
                    help='weight decay for alpha')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args = parser.parse_args()

if args.gpus == 'all':
    #paddle.device.cuda.device_count()
    args.gpus = [3]#list(range(1))
else:
    args.gpus = [int(s) for s in args.gpus.split(',')]
args.save = '{}search-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

#device = torch.device("cuda")
def clip_grad_value_(parameters, clip_value):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)

def main():
    if not paddle.device.is_compiled_with_cuda():
    # if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)
    logging.info("Logger is set - training start")

    # set default gpu device id
    paddle.set_device('gpu:{}'.format(3))

    # set seed
    g=paddle.seed(args.seed)
    np.random.seed(args.seed)
    g.manual_seed(args.seed)
    #g.manual_seed_all(args.seed)
    logging.info("args = %s", args)

    #torch.backends.cudnn.benchmark = True

    if args.dataset == 'imagenet':
        train_loader, valid_loader = get_search_datasets(args)
        args.input_channels = 3
        args.n_classes = 1000
    elif args.dataset == 'tiny-imagenet':
        train_loader, valid_loader = get_tiny_datasets(args)
        args.n_classes = 200
        args.input_channels = 3
    else:
        input_size, args.input_channels, args.n_classes, train_loader, valid_loader = utils.get_data(
        args, cutout_length=0, validation=False)


    net_crit = nn.CrossEntropyLoss()
    aux_net_crit = nn.CrossEntropyLoss()
    model = Network(args, net_crit, aux=False, device_ids=args.gpus)
    #model = model
    logging.info("param size = %fMB", utils.param_size(model))
    # weights optimizer
    
                              
    # alphas optimizer
    alpha_optim =optim.Adam(learning_rate=args.alpha_lr,parameters=model.alphas(), beta1=0.5, beta2=0.999,
                                   weight_decay=args.alpha_weight_decay)
    # weights optimizer
    lr_scheduler = optim.lr.CosineAnnealingDecay(learning_rate=args.w_lr,
        T_max= args.epochs, eta_min=args.w_lr_min)
    w_optim=optim.Momentum(learning_rate=lr_scheduler,parameters=model.weights(),
                              weight_decay=args.w_weight_decay,grad_clip=nn.ClipGradByNorm(args.w_grad_clip),momentum=args.w_momentum)
    #w_optim = optim.SGD(learning_rate=lr_scheduler,parameters=model.weights(), 
    #                          weight_decay=args.w_weight_decay,grad_clip=nn.ClipGradByNorm(args.w_grad_clip))

    lr_scheduler_aux = optim.lr.CosineAnnealingDecay(
        learning_rate=args.w_lr, T_max=args.epochs, eta_min=args.w_lr_min)
    #w_optim_aux = optim.SGD(learning_rate=lr_scheduler_aux ,parameters=model.weights(),
    #                          weight_decay=args.w_weight_decay,grad_clip=nn.ClipGradByNorm(args.w_grad_clip))
    w_optim_aux=optim.Momentum(learning_rate=lr_scheduler_aux,parameters=model.weights(),
                              weight_decay=args.w_weight_decay,grad_clip=nn.ClipGradByNorm(args.w_grad_clip),momentum=args.w_momentum)
    alpha_optim_aux = optim.Adam(learning_rate=args.alpha_lr,parameters=model.alphas(), beta1=0.5, beta2=0.999,
                                   weight_decay=args.alpha_weight_decay)

    
     
    architect = Architect(model, args.w_momentum, args.w_weight_decay,args.w_lr,args.w_grad_clip)

    # training loop
    best_top1 = 0.
    is_best = True
    for epoch in range(args.epochs):
        lr_scheduler.step()
        #w_optim.step()
        lr = lr_scheduler.get_lr()

        model.print_alphas(logging)

        # training
        train_acc, train_obj = train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch)
        logging.info('Train_acc %f', train_acc)

        # validation
        cur_step = (epoch+1) * len(train_loader)

        model, aux_model, train_aux_acc, train_aux_obj = IST(args, train_loader, valid_loader, model, architect, alpha_optim_aux, aux_net_crit, w_optim_aux, lr_scheduler_aux, epoch,logging)
        logging.info('Train_aux_acc %f', train_aux_acc)
        
        valid_acc, valid_obj = validate(valid_loader, aux_model, epoch, cur_step)
        logging.info('Valid_acc %f', valid_acc)
        logging.info('Epoch: %d lr: %e', epoch, lr)
        # log
        genotype = model.genotype()
        logging.info('genotype = %s', genotype)

        # genotype as a image
        plot_path = os.path.join(args.plot_path, "EP{:02d}".format(epoch+1))
        caption = "Epoch {}".format(epoch+1)
        plot(genotype.normal, plot_path + "-normal", caption)
        plot(genotype.reduce, plot_path + "-reduce", caption)

        # save
        if best_top1 < valid_acc:
            logging.info("niaoqianghuanpao")
            best_top1 = valid_acc
            best_genotype = genotype
            is_best = True
            save_path = args.save_path + 'aux_model_params.pt'
            paddle.save(aux_model.state_dict(), save_path)
            pretrained_dict = paddle.load(save_path)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'alpha' not in k}
            model_dict = model.state_dict()
            model_dict.update(pretrained_dict)
            model.set_state_dict(model_dict)
            
        else:
            is_best = False
        utils.save_checkpoint(model, args.save_path, is_best)
        print("")

            

    logging.info("Final best Prec@1 = {:.4%}".format(best_top1))
    logging.info("Best Genotype = {}".format(best_genotype))


def train(train_loader, valid_loader, model, architect, w_optim, alpha_optim, lr, epoch):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch*len(train_loader)

    model.train()
#     pdb.set_trace()
    for step, (data_train, data_val) in enumerate(zip(train_loader, valid_loader)):
        trn_X, trn_y = data_train[0], data_train[1]
        val_X, val_y = data_val[0], data_val[1]
        N = trn_X.size

        # phase 2. architect step (alpha)
        alpha_optim.clear_grad()
        architect.unrolled_backward(trn_X, trn_y, val_X, val_y, lr, w_optim)#计算alpha的梯度
        alpha_optim.step()

        w_optim.clear_grad()
        logits = model(trn_X)
        loss = model.criterion(logits, trn_y)
        loss.backward()
        # gradient clipping
        
        #clip_grad_value_(model.weights(), args.w_grad_clip)  #clip_grad防止梯度爆炸
        w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            logging.info('TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, losses.avg, top1.avg, top5.avg)

        cur_step += 1

    return top1.avg, losses.avg


def validate(valid_loader, model, epoch, cur_step):
    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    model.eval()

    with paddle.no_grad():
        for step, (X, y) in enumerate(valid_loader):
            
            N = X.size

            logits = model(X)
            loss = model.criterion(logits, y)

            prec1, prec5 = utils.accuracy(logits, y, topk=(1, 5))
            losses.update(loss.item(), N)
            top1.update(prec1.item(), N)
            top5.update(prec5.item(), N)

            if step % args.print_freq == 0:
                logging.info('valid %03d %e %f %f', step, losses.avg, top1.avg, top5.avg)


    return top1.avg, losses.avg



if __name__ == "__main__":
    main()
