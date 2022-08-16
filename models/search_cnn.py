""" CNN for architecture search """
#import torch
import paddle
import paddle.nn as nn
#import torch.nn as nn
import paddle.nn.functional as F
#import torch.nn.functional as F
from models.search_cells import SearchCell
import genotypes as gt
from paddle.distributed import broadcast
#from torch.nn.parallel._functions import Broadcast
import logging
import numpy as np

def broadcast_list(l, device_ids):
    """ Broadcasting list """
    l_copies = broadcast(*l,device_ids)
    #l_copies = Broadcast.apply(device_ids, *l)
    l_copies = [l_copies[i:i+len(l)] for i in range(0, len(l_copies), len(l))]

    return l_copies


class SearchCNN(nn.Layer):
    """ Search CNN model """
    def __init__(self, C_in, C, n_classes, n_layers, n_nodes=4, stem_multiplier=3):
        """
        Args:
            C_in: # of input channels
            C: # of starting model channels
            n_classes: # of classes
            n_layers: # of layers
            n_nodes: # of intermediate nodes in Cell
            stem_multiplier
        """
        super().__init__()
        self.C_in = C_in
        self.C = C
        self.n_classes = n_classes
        self.n_layers = n_layers

        C_cur = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2D(C_in, C_cur, 3, 1, 1),
            nn.BatchNorm2D(C_cur,momentum=0.1,epsilon=1e-5)
        )

        # for the first cell, stem is used for both s0 and s1
        # [!] C_pp and C_p is output channel size, but C_cur is input channel size.
        C_pp, C_p, C_cur = C_cur, C_cur, C

        self.cells = nn.LayerList()
        reduction_p = False
        for i in range(n_layers):
            # Reduce featuremap size and double channels in 1/3 and 2/3 layer.
            if i in [n_layers//3, 2*n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = SearchCell(n_nodes, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * n_nodes
            C_pp, C_p = C_p, C_cur_out

        self.gap = nn.AdaptiveAvgPool2D(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x, weights_normal, weights_reduce):
        s0 = s1 = self.stem(x)
        for cell in self.cells:
            weights = weights_reduce if cell.reduction else weights_normal
            s0, s1 = s1, cell(s0, s1, weights)

        out = self.gap(s1)
        out = paddle.flatten(out,1, -1) # flatten
        logits = self.linear(out)
        return logits

class Network(nn.Layer):
    """ SearchCNN controller supporting multi-gpu """
    def __init__(self, args, criterion, aux, alpha_normal=None, alpha_reduce=None, n_nodes=4, stem_multiplier=3,
                 device_ids=None):
        super().__init__()
        C = args.init_channels
        C_in = args.input_channels
        n_layers = args.layers
        n_classes = args.n_classes
        self.n_nodes = n_nodes
        self.epoch=1
        self.aux=aux
        self.criterion = criterion
        if device_ids is None:
            device_ids = list(range(paddle.device.cuda.device_count()))
        self.device_ids = device_ids

        # initialize architect parameters: alphas
        n_ops = len(gt.PRIMITIVES)

        self.alpha_normal = nn.ParameterList()
        self.alpha_reduce = nn.ParameterList()
        if aux:
            for i in range(n_nodes):
                a_n=paddle.create_parameter([i+2,n_ops],dtype='float32')
                norm_value=paddle.zeros([i+2,n_ops],dtype='float32')
                a_n.set_value(norm_value)
                self.alpha_normal.append(a_n)
                a_r=paddle.create_parameter([i+2,n_ops],dtype='float32')
                reduce_value=paddle.zeros([i+2,n_ops],dtype='float32')
                a_r.set_value(reduce_value)
                self.alpha_reduce.append(a_r)
                # self.alpha_normal.append(nn.Parameter(paddle.zeros(i + 2, n_ops)))
                # self.alpha_reduce.append(nn.Parameter(paddle.zeros(i + 2, n_ops)))

                with paddle.no_grad():
                    # edges: Tensor(n_edges, n_ops)
                    edge_max, primitive_indices = paddle.topk(alpha_normal[i][:, :-1], 1)  # ignore 'none'每一行中的最大alpha值
                    topk_edge_values, topk_edge_indices = paddle.topk(paddle.flatten(edge_max,0,-1), 2)#这些alpha值的前两个值,每个节点只保留连接它的两条边
                    
                    an_tmp=self.alpha_normal[i].numpy()
                    for edge_idx in topk_edge_indices:
                        prim_idx = primitive_indices[edge_idx]
                        an_tmp[int(edge_idx)][int(prim_idx)]=1.0
                    self.alpha_normal[i].set_value(paddle.to_tensor(an_tmp))

                    ar_tmp=self.alpha_reduce[i].numpy()
                    edge_max, primitive_indices = paddle.topk(alpha_reduce[i][:, :-1], 1)  # ignore 'none'
                    topk_edge_values, topk_edge_indices = paddle.topk(paddle.flatten(edge_max,0,-1), 2)
                    for edge_idx in topk_edge_indices:
                        prim_idx = primitive_indices[edge_idx]
                        ar_tmp[int(edge_idx)][int(prim_idx)]=1.0
                    self.alpha_reduce[i].set_value(paddle.to_tensor(ar_tmp))
        else:
            for i in range(n_nodes):
                a_n=paddle.create_parameter([i+2,n_ops],dtype='float32')
                norm_value=paddle.randn([i+2,n_ops],dtype='float32')*1e-3
                a_n.set_value(norm_value)
                self.alpha_normal.append(a_n)
                a_r=paddle.create_parameter([i+2,n_ops],dtype='float32')
                reduce_value=paddle.randn([i+2,n_ops],dtype='float32')*1e-3
                a_r.set_value(reduce_value)
                self.alpha_reduce.append(a_r)

        
        # setup alphas list
        self._alphas = []
        for n, p in self.named_parameters():
            if 'alpha' in n:
                self._alphas.append((n, p))
        # C_in means the input data
        self.net = SearchCNN(C_in, C, n_classes, n_layers, n_nodes, stem_multiplier)
    # x means the input data
    def forward(self, x):
        # if self.aux:
        #     weights_normal = [alpha for alpha in self.alpha_normal]
        #     weights_reduce = [alpha for alpha in self.alpha_reduce]
        # else:
        weights_normal = [F.softmax(alpha, axis=0) for alpha in self.alpha_normal]   #CEN
        weights_reduce = [F.softmax(alpha, axis=0) for alpha in self.alpha_reduce]    #CEN

        #if len(self.device_ids) == 1:
        return self.net(x, weights_normal, weights_reduce)

        # scatter x
        # xs = paddle.distributed.scatter(x, self.device_ids)
        # # broadcast weights
        # wnormal_copies = broadcast_list(weights_normal, self.device_ids)
        # wreduce_copies = broadcast_list(weights_reduce, self.device_ids)

        # # replicate modules
        # replicas = nn.parallel.replicate(self.net, self.device_ids)
        # outputs = nn.parallel.parallel_apply(replicas,
        #                                      list(zip(xs, wnormal_copies, wreduce_copies)),
        #                                      devices=self.device_ids)
        # return nn.parallel.gather(outputs, self.device_ids[0])


    def loss(self, X, y):
        logits = self.forward(X)
        return self.criterion(logits, y)

    def print_alphas(self, logger):
        # remove formats
        org_formatters = []
        for handler in logger.getLogger('ecdarts').handlers:
            org_formatters.append(handler.formatter)
            handler.setFormatter(logging.Formatter("%(message)s"))

        logger.info("####### ALPHA #######")
        logger.info("# Alpha - normal")
        for alpha in self.alpha_normal:
            logger.info(F.softmax(alpha, axis=0))

        logger.info("\n# Alpha - reduce")
        for alpha in self.alpha_reduce:
            logger.info(F.softmax(alpha, axis=0))

        # restore formats
        for handler, formatter in zip(logger.getLogger('ecdarts').handlers, org_formatters):
            handler.setFormatter(formatter)

    def genotype(self):
        gene_normal = gt.parse(self.alpha_normal, k=2)
        gene_reduce = gt.parse(self.alpha_reduce, k=2)
        concat = range(2, 2+self.n_nodes) # concat all intermediate nodes

        return gt.Genotype(normal=gene_normal, normal_concat=concat,
                           reduce=gene_reduce, reduce_concat=concat)

    def weights(self):
        weight_list= []
        for w in self.net.parameters():
            if w.stop_gradient==False:
                weight_list.append(w) 
        return weight_list

    def named_weights(self):
        return self.net.named_parameters()

    def alphas(self):
        for n, p in self._alphas:  # n:alpha_normal.0,p:[2,8]
            yield p

    def named_alphas(self):
        for n, p in self._alphas:
            yield n, p
    def minmaxscaler(data):
        min = np.amin(data)
        max = np.amax(data)    
        return (data - min)/(max-min)

