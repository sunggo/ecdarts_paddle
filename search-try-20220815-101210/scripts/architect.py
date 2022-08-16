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
        # gradients = [
        #     to_variable(param._grad_ivar().numpy())
        #     for param in self.net.weights()
        # ]
        # grad_list=[]
        # for w in self.net.weights():
        #     if w.grad is not None:
        #         grad_list.append(w.grad.clone())
        #     else:
        #         grad_list.append(w.grad)
        #     w.clear_grad()
        # for a in self.net._alphas:
                
        #         a[1].clear_grad()
        #gradients=grad_list
        # do virtual step (update gradient)
        # below operations do not need gradient tracking
        # with paddle.no_grad():
        #     # dict key is not the value, but the pointer. So original network weight have to
        #     # be iterated also.
        #     #for w, vw, g in zip(weights_list, v_weights_list, gradients):
        #     for w, vw, g in zip(self.net.weights(), self.v_net.weights(), gradients):
        #         m = w_optim.state[w].get('momentum_buffer', 0.) * self.w_momentum
        #         m=0
        #         # if g is not None:
        #         fluid.layers.assign((w - xi * (m + g + self.w_weight_decay*w)), vw)
        #         #vw=(w - xi * (m + g + self.w_weight_decay*w)).clone()

        #     # synchronize alphas
        #     for a, va in zip(self.net.alphas(), self.v_net.alphas()):
        #         fluid.layers.assign(a,va)
                #va=a.clone()
        
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
        # self.virtual_step(trn_X, trn_y, xi, w_optim)
        # loss = self.v_net.loss(val_X, val_y) # L_val(w`)
        # loss.backward()
        # dalpha = [
        #     to_variable(param[1]._grad_ivar().numpy())
        #     for param in self.v_net._alphas
        # ]
        # dw = [
        #     to_variable(param._grad_ivar().numpy())
        #     for param in self.v_net.weights()
        # ]
        # self.v_net.clear_gradients()

        # hessian = self.compute_hessian(dw, trn_X, trn_y)
        # for g, ig in zip(dalpha, hessian):
        #     new_g = g - (ig * self.vw_optim.get_lr())
        #     fluid.layers.assign(new_g.detach(), g)

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
        # dapos_list=[]
        # for w in self.net.weights():
        #     w.clear_grad()
        # for a in self.net._alphas:
        #         dapos_list.append(a[1].grad.clone())
        #         a[1].clear_grad()
        # #dalpha_pos = paddle.grad(loss, self.net.alphas()) # dalpha { L_trn(w+) }
        # dalpha_pos=dapos_list
        # w- = w - eps*dw`
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
        # da_list=[]
        # for a in self.net._alphas:
        #         da_list.append(a[1].grad)

        # dalpha_neg=da_list
        # recover w
        with paddle.no_grad():
            for p, d in zip(self.net.weights(), dw):
                if d is not None:
                    p += eps * d
        self.net.clear_gradients()
        hessian = [(p-n) /(2.*eps) for p, n in zip(dalpha_pos, dalpha_neg)]
        


        self.hessian = hessian
        
        
        return hessian
        # R = r * fluid.layers.rsqrt(
        #     fluid.layers.sum([
        #         fluid.layers.reduce_sum(fluid.layers.square(v)) for v in dw
        #     ]))

        # model_params = [
        #     p for p in self.net.parameters()
        #     if p.name not in [a.name for a in self.model.arch_parameters()] and
        #     p.trainable
        # ]
        # for param, grad in zip(model_params, dw):
        #     param_p = param + grad * R
        #     fluid.layers.assign(param_p.detach(), param)
        # loss = self.model._loss(input, trn_y)
        # if self.parallel:
        #     loss = self.parallel_model.scale_loss(loss)
        #     loss.backward()
        #     self.parallel_model.apply_collective_grads()
        # else:
        #     loss.backward()

        # grads_p = [
        #     to_variable(param._grad_ivar().numpy())
        #     for param in self.model.arch_parameters()
        # ]

        # for param, grad in zip(model_params, dw):
        #     param_n = param - grad * R * 2
        #     fluid.layers.assign(param_n.detach(), param)
        # self.model.clear_gradients()

        # loss = self.model._loss(input, trn_y)
        # if self.parallel:
        #     loss = self.parallel_model.scale_loss(loss)
        #     loss.backward()
        #     self.parallel_model.apply_collective_grads()
        # else:
        #     loss.backward()

        # grads_n = [
        #     to_variable(param._grad_ivar().numpy())
        #     for param in self.model.arch_parameters()
        # ]
        # for param, grad in zip(model_params, dw):
        #     param_o = param + grad * R
        #     fluid.layers.assign(param_o.detach(), param)
        # self.model.clear_gradients()
        # arch_grad = [(p - n) / (2 * R) for p, n in zip(grads_p, grads_n)]
        # return arch_grad
    
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
