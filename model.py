#import torch
import paddle
import paddle.nn as nn
#import torch.nn as nn
import os
from operations import *
#from torch.autograd import Variable
from utils import drop_path
from models.search_cnn import Network
from architect import Architect
import utils
def clip_grad_value_(parameters, clip_value):
    if isinstance(parameters, paddle.Tensor):
        parameters = [parameters]
    clip_value = float(clip_value)
    for p in filter(lambda p: p.grad is not None, parameters):
        paddle.clip(p.grad, min=-clip_value, max=clip_value)

def IST(args, train_loader, valid_loader, model, architect, alpha_optim, aux_net_crit, aux_w_optim, lr_scheduler_aux, epoch,
        logging):
    lr_scheduler_aux.step()
    lr = lr_scheduler_aux.get_lr()
    save_path = args.save_path + 'one-shot_params.pt'
    paddle.save(model.state_dict(), save_path)
    pretrained_dict = paddle.load(save_path)

    pretrained_dict = {k: v for k, v in pretrained_dict.items() if 'alpha' not in k}


    aux_model = Network(args, aux_net_crit, aux=True, alpha_normal=model.alpha_normal, alpha_reduce=model.alpha_reduce,
                            device_ids=args.gpus)
    aux_model_dict = aux_model.state_dict()
    aux_model_dict.update(pretrained_dict)
    aux_model.set_state_dict(aux_model_dict)

    top1 = utils.AverageMeter()
    top5 = utils.AverageMeter()
    losses = utils.AverageMeter()

    cur_step = epoch * len(train_loader)

    aux_model.train()
    for step, ((trn_X, trn_y), (val_X, val_y)) in enumerate(zip(train_loader, valid_loader)):
       
        N = trn_X.shape[0]

        aux_w_optim.clear_grad()
        logits = aux_model(trn_X)

        loss = aux_model.criterion(logits, trn_y)
        loss.backward()
        aux_w_optim.step()

        prec1, prec5 = utils.accuracy(logits, trn_y, topk=(1, 5))
        losses.update(loss.item(), N)
        top1.update(prec1.item(), N)
        top5.update(prec5.item(), N)

        if step % args.print_freq == 0 or step == len(train_loader) - 1:
            logging.info('Aux_TRAIN Step: %03d Objs: %e R1: %f R5: %f', step, losses.avg, top1.avg, top5.avg)
        cur_step += 1

    return model, aux_model, top1.avg, losses.avg


class Cell(nn.Layer):

    def __init__(self, genotype, C_prev_prev, C_prev, C, reduction, reduction_prev):
        super(Cell, self).__init__()
        print(C_prev_prev, C_prev, C)

        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)

        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction)

    def _compile(self, C, op_names, indices, concat, reduction):
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.LayerList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0, s1, drop_prob):
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]
        return paddle.concat([states[i] for i in self._concat], dim=1)


class AuxiliaryHeadCIFAR(nn.Layer):

    def __init__(self, C, num_classes):
        """assuming input size 8x8"""
        super(AuxiliaryHeadCIFAR, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2D(5, stride=3, padding=0, exclusive=False),  # image size = 2 x 2
            nn.Conv2D(C, 128, 1, bias=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 768, 2, bias_attr=False),
            nn.BatchNorm2D(768),
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class AuxiliaryHeadImageNet(nn.Layer):

    def __init__(self, C, num_classes):
        """assuming input size 14x14"""
        super(AuxiliaryHeadImageNet, self).__init__()
        self.features = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2D(5, stride=2, padding=0, exclusive=False),
            nn.Conv2D(C, 128, 1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 768, 2, bias_attr=False),
            # NOTE: This batchnorm was omitted in my earlier implementation due to a typo.
            # Commenting it out for consistency with the experiments in the paper.
            nn.BatchNorm2D(768),
            nn.ReLU()
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(paddle.flatten(x,0,-1))
        return x


class NetworkCIFAR(nn.Layer):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2D(3, C_curr, 3, padding=1, bias_attr=False),
            nn.BatchNorm2D(C_curr)
        )

        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.LayerList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadCIFAR(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AdaptiveAvgPool2D(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(paddle.flatten(out,0,-1))
        return logits, logits_aux


class NetworkImageNet(nn.Layer):

    def __init__(self, C, num_classes, layers, auxiliary, genotype):
        super(NetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2D(3, C // 2, kernel_size=3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2DD(C // 2),
            nn.ReLU(),
            nn.Conv2D(C // 2, C, 3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(C),
        )

        self.stem1 = nn.Sequential(
            nn.ReLU(),
            nn.Conv2D(C, C, 3, stride=2, padding=1, bias_attr=False),
            nn.BatchNorm2D(C),
        )

        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.LayerList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHeadImageNet(C_to_auxiliary, num_classes)
        self.global_pooling = nn.AvgPool2D(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input):
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(paddle.flatten(out,0,-1))
        return logits, logits_aux
