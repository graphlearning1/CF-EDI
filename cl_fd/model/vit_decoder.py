import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from model.pos_embed import get_2d_sincos_pos_embed
from timm.models.vision_transformer import PatchEmbed, Block

class Decoder(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=768, num_patches=196,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=partial(nn.LayerNorm, eps=1e-6), norm_pix_loss=False):

        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.decoder_cat_embed = nn.Linear(decoder_embed_dim * 2, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True)

        self.norm_pix_loss = norm_pix_loss
        self.initialize_weights()


    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=True)
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs


    def loss(self, imgs, pred, mask1, mask2):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)


        mask1 = torch.tensor(mask1, dtype=torch.int)
        mask2 = torch.tensor(mask2, dtype=torch.int)
        mask_h = mask1 | mask2
        mask_all = mask1 * mask2
        mask_only = abs(mask_h - mask_all)
        no_mask_all = 1 - mask_h

        loss_all = (loss * mask_all).sum() / mask_all.sum() if mask_all.sum() > 0 else 0   # mean loss on removed patches   掩码部分的平均损失，是个值
        loss_only = (loss * mask_only).sum() / mask_only.sum() if mask_only.sum() > 0 else 0 # mean loss on removed patches   掩码部分的平均损失，是个值
        loss_no_all = (loss * no_mask_all).sum() / no_mask_all.sum() if no_mask_all.sum() > 0 else 0
        loss = (loss_all + loss_only + loss_no_all) / 3

        # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss



    def forward(self, imgs, x_mask1, x_mask2, ids_restore1, ids_restore2, mask1, mask2):

        x_mask1 = self.decoder_embed(x_mask1)
        x_mask2 = self.decoder_embed(x_mask2)

        # append mask tokens to sequence
        mask_tokens1 = self.mask_token1.repeat(x_mask1.shape[0], ids_restore1.shape[1] + 1 - x_mask1.shape[1], 1)
        mask_tokens2 = self.mask_token2.repeat(x_mask2.shape[0], ids_restore2.shape[1] + 1 - x_mask2.shape[1], 1)
        x_1 = torch.cat([x_mask1[:, 1:, :], mask_tokens1], dim=1)
        x_2 = torch.cat([x_mask2[:, 1:, :], mask_tokens2], dim=1)
        x_1 = torch.gather(x_1, dim=1, index=ids_restore1.unsqueeze(-1).repeat(1, 1, x_mask1.shape[2]))
        x_2 = torch.gather(x_2, dim=1, index=ids_restore2.unsqueeze(-1).repeat(1, 1, x_mask2.shape[2]))
        x1 = torch.cat([x_mask1[:, :1, :], x_1], dim=1)
        x2 = torch.cat([x_mask2[:, :1, :], x_2], dim=1)

        # add pos embed
        # x = (x1 + x2) / 2
        x = torch.cat([x1, x2], dim=2)
        x = self.decoder_cat_embed(x)
        x = x + self.decoder_pos_embed
      # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        MAE = self.loss(imgs, x, mask1, mask2)

        return MAE

    def loss_100(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        b = pred.shape[0]
        l = pred.shape[1]
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = loss.sum()/(l * b)
        return loss

    def forward_100(self, imgs, x_mask1, x_mask2):
        x1 = self.decoder_embed(x_mask1)
        x2 = self.decoder_embed(x_mask2)
        x = torch.cat([x1, x2], dim=2)
        x = self.decoder_cat_embed(x)
        x = x + self.decoder_pos_embed
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        MAE = self.loss_100(imgs, x)
        return MAE


    def loss_st_100(self, imgs, pred):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        b = pred.shape[0]
        l = pred.shape[1]
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = loss.sum()/(l * b)
        return loss

    def forward_st_100(self, imgs, x_mask):
        x = self.decoder_embed(x_mask)
        x = x + self.decoder_pos_embed  #
        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        cls = x[:, 0, :]
        # remove cls token
        x = x[:, 1:, :]
        MAE = self.loss_st_100(imgs, x)
        return MAE, cls

    def loss_st(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)
        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss



    def forward_st(self, imgs, x_mask, ids_restore, mask):
        x_mask = self.decoder_embed(x_mask)  # 64*50*512

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x_mask.shape[0], ids_restore.shape[1] + 1 - x_mask.shape[1], 1)
        x_ = torch.cat([x_mask[:, 1:, :], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_mask.shape[2]))  # unshuffle
        x = torch.cat([x_mask[:, :1, :], x_], dim=1)  # append cls token 加token
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        # predictor projection
        x = self.decoder_pred(x)
        cls = x[:, 0, :]
        # remove cls token
        x = x[:, 1:, :]
        MAE = self.loss_st(imgs, x, mask)

        return MAE, cls




