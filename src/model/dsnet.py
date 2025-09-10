import os.path as osp
import pdb

import kornia as kn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.utils

import utils.util as util
from globalenv import *

from .arch.cross_nonlocal import CrossNonLocalBlock
from .arch.nonlocal_block_embedded_gaussian import NONLocalBlock2D
from .basic_loss import *
from .dconv import ColorDeformConv2d
from .single_net_basemodel import SingleNetBaseModel
from .HVI_transform import RGB_HVI


class tanh_L1Loss(nn.Module):
    def __init__(self):
        super(tanh_L1Loss, self).__init__()

    def forward(self, x, y):
        loss = torch.mean(torch.abs(torch.tanh(x) - torch.tanh(y)))
        return loss


class LitModel(SingleNetBaseModel):
    def __init__(self, opt):
        super().__init__(opt, DeepWBNet(opt[RUNTIME]), [TRAIN, VALID])

        self.pixel_loss = tanh_L1Loss()
        self.weighted_loss = WeightedL1Loss()
        self.tvloss = L_TV()
        self.ltv2 = LTVloss()
        self.cos = torch.nn.CosineSimilarity(1, 1e-8)
        self.histloss = HistogramLoss()
        self.vggloss = VGGLoss(model="vgg16", shift=2)
        self.vggloss.train()
        self.inter_histloss = IntermediateHistogramLoss()
        self.sparse_weight_loss = SparseWeightLoss()
        

    def training_step(self, batch, batch_idx):
        input_batch, gt_batch, output_batch, output_batch_hvi, gt_batch_hvi = super().training_step_forward(
            batch, batch_idx
        )
        loss_lambda_map = {
            # L1_LOSS: lambda: self.pixel_loss(output_batch, gt_batch),
            L1_LOSS_RGB: lambda: self.pixel_loss(output_batch, gt_batch),#
            L1_LOSS_HVI: lambda: self.pixel_loss(output_batch_hvi, gt_batch_hvi),#
            COS_LOSS: lambda: (1 - self.cos(output_batch, gt_batch).mean())
            * 0.5,
            COS_LOSS
            + "2": lambda: 1
            - F.sigmoid(self.cos(output_batch, gt_batch).mean()),
            LTV_LOSS: lambda: self.tvloss(output_batch),
            "tvloss1": lambda: self.tvloss(self.net.res[ILLU_MAP])
            + self.tvloss(self.net.res[BRIGHTEN_INPUT]),
            "tvloss2": lambda: self.tvloss(self.net.res[INVERSE_ILLU_MAP])
            + self.tvloss(self.net.res[DARKEN_INPUT]),
            "tvloss1_new": lambda: self.ltv2(
                input_batch, self.net.res[ILLU_MAP], 1
            ),
            "tvloss2_new": lambda: self.ltv2(
                1 - input_batch, self.net.res[INVERSE_ILLU_MAP], 1
            ),
            "illumap_loss": lambda: F.mse_loss(
                self.net.res[ILLU_MAP], 1 - self.net.res[INVERSE_ILLU_MAP]
            ),
            WEIGHTED_LOSS: lambda: self.weighted_loss(
                input_batch.detach(), output_batch, gt_batch
            ),
            SSIM_LOSS: lambda: kn.losses.ssim_loss(
                output_batch, gt_batch, window_size=11
            ),
            PSNR_LOSS: lambda: kn.losses.psnr_loss(
                output_batch, gt_batch, max_val=1.0
            ),
            HIST_LOSS: lambda: self.histloss(output_batch, gt_batch),
            INTER_HIST_LOSS: lambda: self.inter_histloss(
                input_batch,
                gt_batch,
                self.net.res[BRIGHTEN_INPUT],
                self.net.res[DARKEN_INPUT],
            ),
            VGG_LOSS: lambda: self.vggloss(input_batch, gt_batch),
        }
        if self.opt[RUNTIME][DEFORM]:
            loss_lambda_map.update(
                {
                    NORMAL_EX_LOSS: lambda: self.pixel_loss(
                        self.net.res[NORMAL], gt_batch
                    ),
                    NORMAL_EX_LOSS_HVI: lambda: self.pixel_loss
                    (self.net.res[NORMAL_HVI], gt_batch_hvi),
                }
            )
        loss = self.calc_and_log_losses(loss_lambda_map)

        # logging images:
        self.log_training_iogt_img(batch)
        return loss

    def validation_step(self, batch, batch_idx): ...

    def test_step(self, batch, batch_ix):
        super().test_step(batch, batch_ix)

        # save intermediate results
        for k, v in self.net.res.items():
            dirpath = Path(self.opt[IMG_DIRPATH]) / k
            fname = osp.basename(batch[INPUT_FPATH][0])
            if "illu" in k:
                util.mkdir(dirpath)
                torchvision.utils.save_image(v[0].unsqueeze(1), dirpath / fname)
            elif k == "guide_features":
                util.mkdir(dirpath)
                max_size = v[-1][-1].shape[-2:]
                final = []
                for level_guide in v:
                    gs = [F.interpolate(g, max_size) for g in level_guide]
                    final.extend(gs)
                region_num = final[0].shape[1]
                final = torch.stack(final).argmax(axis=2).float() / region_num
                torchvision.utils.save_image(final, dirpath / fname)
            else:
                self.save_img_batch(v, dirpath, fname)


class DeepWBNet(nn.Module):
    def build_illu_net(self):
        from .bilateralupsamplenet import BilateralUpsampleNet

        return BilateralUpsampleNet(self.opt[BUNET])
    
    def HVIT(self, x): #
        hvi = self.trans.HVIT(x)
        return hvi

    def backbone_forward(self, net, x):
        low_x = self.down_sampler(x)
        res = net(low_x, x)
        self.res.update({"guide_features": net.guide_features})
        return res

    def __init__(self, opt=None):
        super(DeepWBNet, self).__init__()
        self.opt = opt
        self.res = {}
        self.down_sampler = lambda x: F.interpolate(
            x, size=(256, 256), mode="bicubic", align_corners=False
        )
        # self.illu_net = self.build_illu_net()
        self.trans = RGB_HVI() #

        # Use non-local block for efficiency.
        nf = 32
        self.out_net = nn.Sequential(
            nn.Conv2d(9, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, nf, 3, 1, 1),
            nn.ReLU(inplace=True),
            NONLocalBlock2D(nf, sub_sample="bilinear", bn_layer=False),
            nn.Conv2d(nf, nf, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(nf, 3, 1),
            NONLocalBlock2D(3, sub_sample="bilinear", bn_layer=False),
        )
        # If you have enough computing resources, you can also use the following code to replace the above code.
        # Remember to modify the corresponding code in the forward function.
        # self.out_net = CrossNonLocalBlock(in_channels=3, inter_channels=16)

        if opt[DEFORM]:
            self.over_deform = ColorDeformConv2d(
                inc=3,
                outc=3,
                kernel_size=3,
                padding=1,
                stride=1,
                modulation=True,
                color_deform=True,
            )
            self.under_deform = ColorDeformConv2d(
                inc=3,
                outc=3,
                kernel_size=3,
                padding=1,
                stride=1,
                modulation=True,
                color_deform=True,
            )

    def decomp(self, x1, illu_map):
        condition = illu_map[:, 2:3, :, :] < x1[:, 2:3, :, :]  # [B,1,H,W]
        condition = condition.expand_as(illu_map)  # [B,C,H,W]
        denominator = torch.where(
            condition,
            x1,                    
            illu_map.float()       
        )
        return x1 / (denominator + 1e-7)

    def forward(self, x):
        x_hvi = self.HVIT(x)
        # x1 = x
        inverse_hvi = x_hvi.clone()
        reversed_channel = 1 - inverse_hvi[:, 2, :, :]
        inverse_hvi[:, 2, :, :] = reversed_channel

        brighten_x1_hvi = inverse_hvi
        darken_x1_hvi = x_hvi


        self.res.update(
            {
                BRIGHTEN_INPUT_HVI: brighten_x1_hvi,
                DARKEN_INPUT_HVI: darken_x1_hvi,
            }
        )

        weight_map_hvi = self.out_net(torch.cat([x_hvi, brighten_x1_hvi, darken_x1_hvi], dim=1))
        w1_hvi = weight_map_hvi[:, 0, ...].unsqueeze(1)
        w2_hvi = weight_map_hvi[:, 1, ...].unsqueeze(1)
        w3_hvi = weight_map_hvi[:, 2, ...].unsqueeze(1)

        out_hvi = x_hvi * w1_hvi + brighten_x1_hvi * w2_hvi + darken_x1_hvi * w3_hvi

        pseudo_normal = self.trans.PHVIT(out_hvi)
        pseudo_normal_hvi = out_hvi
        self.res.update({NORMAL: pseudo_normal,
                         NORMAL_HVI: pseudo_normal_hvi})

        # Deformation:
        # ──────────────────────────────────────────────────────────
        brighten_x2_hvi = self.over_deform(x=pseudo_normal_hvi, ref=brighten_x1_hvi)
        darken_x2_hvi = self.under_deform(x=pseudo_normal_hvi, ref=darken_x1_hvi)

        # Modulation:
        # ──────────────────────────────────────────────────────────

        weight_map_hvi = self.out_net(
            torch.cat([pseudo_normal_hvi, brighten_x2_hvi, darken_x2_hvi], dim=1)
        )
        w1_hvi = weight_map_hvi[:, 0, ...].unsqueeze(1)
        w2_hvi = weight_map_hvi[:, 1, ...].unsqueeze(1)
        w3_hvi = weight_map_hvi[:, 2, ...].unsqueeze(1)
        out_hvi = pseudo_normal_hvi * w1_hvi + brighten_x2_hvi * w2_hvi + darken_x2_hvi * w3_hvi
        out = self.trans.PHVIT(out_hvi)

        self.res.update(
            {
                BRIGHTEN_OFFSET: brighten_x2_hvi,
                DARKEN_OFFSET: darken_x2_hvi,
            }
        )

        assert out.shape == x.shape
        return out, out_hvi
