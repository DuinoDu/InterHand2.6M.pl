# -*- coding: utf-8 -*-
from argparse import ArgumentParser
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .nets.module import BackboneNet, PoseNet
from .nets.loss import JointHeatmapLoss, RelRootDepthLoss, HandTypeLoss
from .config import cfg


def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)


def render_gaussian_heatmap(joint_coord):
    """
    assert input is in cuda.
    """
    x = torch.arange(cfg.output_hm_shape[2])
    y = torch.arange(cfg.output_hm_shape[1])
    z = torch.arange(cfg.output_hm_shape[0])
    zz,yy,xx = torch.meshgrid(z,y,x)
    xx = xx[None,None,:,:,:].cuda().float()
    yy = yy[None,None,:,:,:].cuda().float()
    zz = zz[None,None,:,:,:].cuda().float()
    
    x = joint_coord[:,:,0,None,None,None]
    y = joint_coord[:,:,1,None,None,None]
    z = joint_coord[:,:,2,None,None,None];

    heatmap = torch.exp(-(((xx-x)/cfg.sigma)**2)/2 -(((yy-y)/cfg.sigma)**2)/2 - (((zz-z)/cfg.sigma)**2)/2)
    heatmap = heatmap * 255
    return heatmap


class Model(pl.LightningModule):
    def __init__(self, lr=1e-3, joint_num=21):
        super().__init__()
        self.save_hyperparameters()

        self.backbone_net = BackboneNet()
        self.pose_net = PoseNet(joint_num)

        self.backbone_net.init_weights()
        self.pose_net.apply(init_weights)

        # loss functions
        self.joint_heatmap_loss = JointHeatmapLoss()
        self.rel_root_depth_loss = RelRootDepthLoss()
        self.hand_type_loss = HandTypeLoss()

    def forward(self, x):
        img_feat = self.backbone_net(x)
        joint_heatmap_out, rel_root_depth_out, hand_type = self.pose_net(img_feat)
        return joint_heatmap_out, rel_root_depth_out, hand_type

    def training_step(self, batch, batch_idx):
        inputs, targets, meta_info = batch

        input_img = inputs['img']
        joint_heatmap_out, rel_root_depth_out, hand_type = self(input_img)
        batch_size = input_img.shape[0]

        target_joint_coord, target_rel_root_depth, target_hand_type = targets['joint_coord'], targets['rel_root_depth'], targets['hand_type']
        joint_valid, root_valid, hand_type_valid, inv_trans = meta_info['joint_valid'], meta_info['root_valid'], meta_info['hand_type_valid'], meta_info['inv_trans']
        target_joint_heatmap = render_gaussian_heatmap(target_joint_coord)

        joint_heatmap_loss = self.joint_heatmap_loss(
            joint_heatmap_out, target_joint_heatmap, joint_valid).mean()
        rel_root_depth_loss = self.rel_root_depth_loss(
            rel_root_depth_out, target_rel_root_depth, root_valid).mean()
        hand_type_loss = self.hand_type_loss(
            hand_type, target_hand_type, hand_type_valid).mean()
        loss = sum([joint_heatmap_loss, rel_root_depth_loss, hand_type_loss])
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.training_step(batch, batch_idx)
        # self.log('valid_loss', loss)

    def test_step(self, batch, batch_idx):
        inputs, targets, meta_info = batch

        input_img = inputs['img']
        target_joint_coord, target_rel_root_depth, target_hand_type = targets['joint_coord'], targets['rel_root_depth'], targets['hand_type']
        joint_valid, root_valid, hand_type_valid, inv_trans = meta_info['joint_valid'], meta_info['root_valid'], meta_info['hand_type_valid'], meta_info['inv_trans']
        
        batch_size = input_img.shape[0]
        joint_heatmap_out, rel_root_depth_out, hand_type = self(input_img)

        out = {}
        val_z, idx_z = torch.max(joint_heatmap_out,2)
        val_zy, idx_zy = torch.max(val_z,2)
        val_zyx, joint_x = torch.max(val_zy,2)
        joint_x = joint_x[:,:,None]
        joint_y = torch.gather(idx_zy, 2, joint_x)
        joint_z = torch.gather(idx_z, 2, joint_y[:,:,:,None].repeat(1,1,1,cfg.output_hm_shape[1]))[:,:,0,:]
        joint_z = torch.gather(joint_z, 2, joint_x)
        joint_coord_out = torch.cat((joint_x, joint_y, joint_z),2).float()
        out['joint_coord'] = joint_coord_out
        out['rel_root_depth'] = rel_root_depth_out
        out['hand_type'] = hand_type
        out['inv_trans'] = inv_trans
        out['target_joint'] = target_joint_coord
        out['joint_valid'] = joint_valid
        out['hand_type_valid'] = hand_type_valid
        return out

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--lr', type=float, default=0.0001)
        return parser
