""" CNN for network augmentation """
#import torch
import paddle
import paddle.nn as nn
#import torch.nn as nn
from models.augment_cells import AugmentCell
from models import ops
import pdb


class AuxiliaryHead(nn.Layer):
    """ Auxiliary head in 2/3 place of network to let the gradient flow well """
    def __init__(self, input_size, C, n_classes):
        """ assuming input size 7x7 or 8x8 """
#        pdb.set_trace()
        assert input_size in [7, 8,56]
        super().__init__()
        self.net = nn.Sequential(
            nn.ReLU(),
            nn.AvgPool2D(5, stride=input_size-5, padding=0, exclusive=False), # 2x2 out
            nn.Conv2D(C, 128, kernel_size=1, bias_attr=False),
            nn.BatchNorm2D(128),
            nn.ReLU(),
            nn.Conv2D(128, 768, kernel_size=2, bias_attr=False), # 1x1 out
            nn.BatchNorm2D(768),
            nn.ReLU()
        )
        self.linear = nn.Linear(768, n_classes)

    def forward(self, x):
        out = self.net(x)
        out = paddle.flatten(out,1, -1) # flatten
        logits = self.linear(out)
        return logits


class AugmentCNN(nn.Layer):
    """ Augmented CNN model """
    def __init__(self, args, input_size, C_in, n_classes, auxiliary, stem_multiplier=3):
        """
        Args:
            input_size: size of height and width (assuming height = width)
            C_in: # of input channels
            C: # of starting model channels
        """
        super().__init__()
        self.C_in = C_in
        self.C = args.init_channels
        self.n_classes = n_classes
        self.n_layers = args.n_layers
        self.genotype = args.genotype
        # aux head position
        self.aux_pos = 2*args.n_layers//3 if auxiliary else -1

        C_cur = stem_multiplier * self.C
        self.stem = nn.Sequential(
            nn.Conv2D(C_in, C_cur, 3, 1, 1, bias_attr=False),
            nn.BatchNorm2D(C_cur)
        )

        C_pp, C_p, C_cur = C_cur, C_cur, self.C

        self.cells = nn.LayerList()
        reduction_p = False
        for i in range(self.n_layers):
            if i in [self.n_layers//3, 2*self.n_layers//3]:
                C_cur *= 2
                reduction = True
            else:
                reduction = False

            cell = AugmentCell(self.genotype, C_pp, C_p, C_cur, reduction_p, reduction)
            reduction_p = reduction
            self.cells.append(cell)
            C_cur_out = C_cur * len(cell.concat)
            C_pp, C_p = C_p, C_cur_out

            if i == self.aux_pos:
                # [!] this auxiliary head is ignored in computing parameter size
                #     by the name 'aux_head'
                self.aux_head = AuxiliaryHead(input_size//4, C_p, n_classes)

        self.gap = nn.AdaptiveAvgPool2D(1)
        self.linear = nn.Linear(C_p, n_classes)

    def forward(self, x):
        s0 = s1 = self.stem(x)

        aux_logits = None
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1)
            if i == self.aux_pos and self.training:
                aux_logits = self.aux_head(s1)

        out = self.gap(s1)
        out = paddle.flatten(out,1,-1) # flatten
        logits = self.linear(out)
        return logits, aux_logits

    def drop_path_prob(self, p):
        """ Set drop path probability """
        for module in self.modules():
            if isinstance(module, ops.DropPath_):
                module.p = p
