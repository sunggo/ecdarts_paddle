""" Architect controls architecture of cell by computing gradients of alphas """
import copy
import paddle
#import torch
from numpy.linalg import eigvals
import numpy as np
#from paddle.autograd
#from torch.autograd import Variable
import pdb
from paddle.fluid.dygraph.base import to_variable
from paddle import fluid
#from search import args
import paddle.nn as nn
import  paddle.optimizer as optim
class Architect():
    """ Compute gradients of alphas """
    def __init__(self, net, w_momentum, w_weight_decay,w_lr,w_grad_clip):
        """
        Args:
            net
            w_momentum: weights momentum
        """
        self.net = net
        self.v_net = copy.deepcopy(net)
        for p,pv in zip(self.net.parameters(),self.v_net.parameters()):
            if p.stop_gradient==True:
                pv.stop_gradient=True
        self.w_momentum = w_momentum
        self.w_weight_decay = w_weight_decay
        self.hessian = None
        self.vw_optim=optim.Momentum(learning_rate=w_lr,parameters=self.v_net.weights(),
                              weight_decay=w_weight_decay,grad_clip=nn.ClipGradByNorm(w_grad_clip),momentum=w_momentum)

    def virtual_step(self, trn_X, trn_y, xi, w_optim):
        """
        Compute unrolled weight w' (virtual step)

        Step process:
        1) forward
        2) calc loss
        3) compute gradient (by backprop)
        4) update gradient

        Args:
            xi: learning rate for virtual gradient step (same as weights lr)
            w_optim: weights optimizer
        """
        # forward & calc loss
        for x, y in zip(self.v_net.parameters(),self.net.parameters()):
            fluid.layers.assign(y.detach(), x)
        
        loss = self.v_net.loss(trn_X, trn_y) # L_trn(w)
        loss.backward()
        lr=w_optim.get_lr()
        self.vw_optim.set_lr(lr)
        self.vw_optim.step()
        self.v_net.clear_gradients()
        
        
    def unrolled_backward(self, trn_X, trn_y, val_X, val_y, xi, w_optim):#计算alpha的梯度
        """ Compute unrolled loss and backward its gradients
        Args:
            xi: learning rate for virtual gradient step (same as net lr)
            w_optim: weights optimizer - for virtual step
        """
        self.virtual_step(trn_X, trn_y, xi, w_optim)
        loss = self.v_net.loss(val_X, val_y) # L_val(w`)
        loss.backward()
        dalpha = [
            to_variable(param[1]._grad_ivar().numpy())
            for param in self.v_net._alphas
        ]
        dw = [
            to_variable(param._grad_ivar().numpy())
            for param in self.v_net.weights()
        ]
        self.v_net.clear_gradients()
        hessian = self.compute_hessian(dw, trn_X, trn_y)
        with paddle.no_grad():
            for alpha, da, h in zip(self.net.alphas(), dalpha, hessian):
                alpha.grad.set_value( da - xi*h)
       
    def compute_hessian(self, dw, trn_X, trn_y,r=0.01):
        # """
        # dw = dw` { L_val(w`, alpha) }
        # w+ = w + eps * dw
        # w- = w - eps * dw
        # hessian = (dalpha { L_trn(w+, alpha) } - dalpha { L_trn(w-, alpha) }) / (2*eps)
        # eps = 0.01 / ||dw||
        # """
        # w_list=[]
        # for w in dw:
        #     if w is not None:
        #         w_list.append(paddle.flatten(w,0,-1))
        # norm=paddle.concat(w_list).norm()
        #norm = paddle.concat([paddle.flatten(w,0,-1) for w in dw]).norm()
        # eps = 0.01 / norm
        eps=r * fluid.layers.rsqrt(
            fluid.layers.sum([
                fluid.layers.reduce_sum(fluid.layers.square(v)) for v in dw
            ]))

        # w+ = w + eps*dw`
        with paddle.no_grad():
            for p, d in zip(self.net.weights(), dw):
                #if d is not None:
                p += eps * d
        loss = self.net.loss(trn_X, trn_y)
        loss.backward()
        dalpha_pos = [
            to_variable(param[1]._grad_ivar().numpy())
            for param in self.net._alphas
        ]
        
        with paddle.no_grad():
            for p, d in zip(self.net.weights(), dw):
                if d is not None:
                    p -= 2. * eps * d
        self.net.clear_gradients()
        loss = self.net.loss(trn_X, trn_y)
        loss.backward()

        #dalpha_neg = paddle.grad(loss, self.net.alphas()) # dalpha { L_trn(w-) }
        dalpha_neg = [
            to_variable(param[1]._grad_ivar().numpy())
            for param in self.net._alphas
        ]
       
        with paddle.no_grad():
            for p, d in zip(self.net.weights(), dw):
                if d is not None:
                    p += eps * d
        self.net.clear_gradients()
        hessian = [(p-n) /(2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        


        self.hessian = hessian
        
        
        return hessian
    
    def compute_eigenvalues(self):
        #hessian = self.compute_Hw(input, target)
        if self.hessian is None:
            raise ValueError
        mm = []
        i=0
        for t in self.hessian :
            if i == 0:
                mm=t.value.cpu().numpy()
                i=1
            else:
                mm = np.row_stack((mm, t.value.cpu().numpy()))
        print(mm.shape)
        return eigvals(mm)
