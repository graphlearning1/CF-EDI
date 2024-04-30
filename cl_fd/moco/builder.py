# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from model.vit_decoder import Decoder


class MoCo(nn.Module):
    """
    Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0):
        """
        dim: feature dimension (default: 256)
        mlp_dim: hidden dimension in MLPs (default: 4096)
        T: softmax temperature (default: 1.0)
        """
        super(MoCo, self).__init__()

        self.T = T

        # build encoders
        self.base_encoder = base_encoder(num_classes=mlp_dim)
        self.momentum_encoder = base_encoder(num_classes=mlp_dim)
        self.base_decoder = Decoder()
        self.momentum_decoder = Decoder()

        self._build_projector_and_predictor_mlps(dim, mlp_dim)

        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()): # 初始化动量编码器参数
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

        for param_b, param_m in zip(self.base_decoder.parameters(), self.momentum_decoder.parameters()): # 初始化动量编码器参数
            param_m.data.copy_(param_b.data)  # initialize
            param_m.requires_grad = False  # not update by gradient

    def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
        mlp = []
        for l in range(num_layers):
            dim1 = input_dim if l == 0 else mlp_dim
            dim2 = output_dim if l == num_layers - 1 else mlp_dim

            mlp.append(nn.Linear(dim1, dim2, bias=False))

            if l < num_layers - 1:
                mlp.append(nn.BatchNorm1d(dim2))
                mlp.append(nn.ReLU(inplace=True))
            elif last_bn:
                # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
                # for simplicity, we further removed gamma in BN
                mlp.append(nn.BatchNorm1d(dim2, affine=False))

        return nn.Sequential(*mlp)

    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        pass

    @torch.no_grad()
    def _update_momentum_encoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def _update_momentum_decoder(self, m):
        """Momentum update of the momentum encoder"""
        for param_b, param_m in zip(self.base_decoder.parameters(), self.momentum_decoder.parameters()):
            param_m.data = param_m.data * m + param_b.data * (1. - m)

    def contrastive_loss(self, q, k):
        # normalize
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        # gather all targets
        k = concat_all_gather(k)
        # Einstein sum is more intuitive
        logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
        N = logits.shape[0]  # batch size per GPU
        labels = (torch.arange(N, dtype=torch.long) + N * torch.distributed.get_rank()).cuda()
        return nn.CrossEntropyLoss()(logits, labels) * (2 * self.T)

    def forward(self, x1, x2, m, epoch, epoch_mom):
        """
        Input:
            x1: first views of images
            x2: second views of images
            m: moco momentum
        Output:
            loss
        """

        # compute features
        z1, q_patches1, q_mask1, q_ids_restore1 = self.base_encoder(x1, epoch, mom=False)
        z2, q_patches2, q_mask2, q_ids_restore2 = self.base_encoder(x2, epoch, mom=False)
        q1 = self.predictor(z1)
        q2 = self.predictor(z2)

        with torch.no_grad():
            self._update_momentum_encoder(m)
            k1, k_patches1 = self.momentum_encoder(x1, epoch, mom=True)
            k2, k_patches2 = self.momentum_encoder(x2, epoch, mom=True)
        MAEq1, clsq1 = self.base_decoder.forward_st(x1, q_patches1, q_ids_restore1, q_mask1)
        MAEq2, clsq2 = self.base_decoder.forward_st(x2, q_patches2, q_ids_restore2, q_mask2)
        loss_contrastive = self.contrastive_loss(q1, k2) + self.contrastive_loss(q2, k1)
        if epoch < epoch_mom:
            with torch.no_grad():
                self._update_momentum_decoder(m)
                MAEk1, clsk1 = self.momentum_decoder.forward_st_100(x1, k_patches1)
                MAEk2, clsk2 = self.momentum_decoder.forward_st_100(x2, k_patches2)
        else:
            MAEk1, clsk1 = self.momentum_decoder.forward_st_100(x1, k_patches1)
            MAEk2, clsk2 = self.momentum_decoder.forward_st_100(x2, k_patches2)

        loss_MAE = (MAEq1 + MAEq2) / 2
        loss_contrastive_con = self.contrastive_loss(clsq1, clsk2) + self.contrastive_loss(clsq2, clsk1)
        loss = loss_contrastive + loss_MAE + 0.5 * loss_contrastive_con
        return loss, loss_contrastive, loss_MAE 


class MoCo_ResNet(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.fc.weight.shape[1]
        del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

        # projectors
        self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
        self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
    def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
        hidden_dim = self.base_encoder.head.weight.shape[1]  # 768
        del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

        # projectors
        self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)  # 3, 768, 4096, 256
        self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

        # predictor
        self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
