import torch
from torch.autograd import Variable
import itertools
import math
import numpy as np
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks_contrastive_learning as networks
import SimpleITK as sitk
import torch.nn as nn
import torch.nn.functional as F
import copy
import utils.metrics as metrics
import os

# from utils.post_precessing import crf

start_point = 1
end_point = 15

csf = [4, 5, 14, 15, 24, 43, 44]
wm = [3, 8, 9, 10, 11, 12, 13, 17, 18, 19, 20, 26, 27, 28, 42, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 58, 59, 60]
gm = [2, 7, 16, 41, 46, 84]
good_labels_list = [csf, wm, gm]


class RegistrationModel(BaseModel):
    def name(self):
        return 'registration_model_contrastive_learning'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        # specify the training losses you want to print out. The program will call base_model.get_current_losses
        self.loss_names = ["total", "recon", "smooth", "contrastive_1", "contrastive_2", "contrastive_3"]
        self.metrics_names = ["mean_dcs"]

        # specify the images you want to save/display. The program will call base_model.get_current_visuals
        visual_names_A = [
            "input_moving", "input_fixed"
        ]
        visual_names_B = []
        self.visual_names = visual_names_A + visual_names_B

        # specify the models you want to save to the disk.
        self.model_names = ["Reg"]
        self.folder_names = opt.name

        # load/define networks
        net_mode = 'lpba40' if 'lpba40' in str(opt.dataset_mode) else NotImplementedError
        img_size = (128, 128, 128) if "small" in opt.dataroot else (128, 128, 128)
        self.netReg = networks.define_registration_model(gpu_ids=self.gpu_ids,
                                                         is_training=self.isTrain,
                                                         model_parallel=opt.model_parallel,
                                                         mode=net_mode,
                                                         img_size=img_size)

        if self.isTrain:
            # # CVPR 2018
            # self.criterionRecon = networks.recon_loss()
            # self.criterionSmooth = networks.smooth_loss()

            # MICCAI 2018
            self.criterionRecon = networks.lamda_mse_loss()
            self.criterionSmooth = networks.kl_loss()
            self.criterionContrastive = networks.contrastive_loss()

            # initialize optimizers
            self.optimizer_Reg = None

            params4T = [{"params": self.netReg.parameters()}]

            self.optimizer_Reg = torch.optim.Adam(params4T,
                                                  lr=opt.lr)

            self.optimizers = [self.optimizer_Reg]
            self.schedulers = []
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

            if opt.continue_train:
                self.load_networks(opt.which_epoch, opt.model_parallel)

        if not self.isTrain:
            self.load_networks(opt.which_epoch, opt.model_parallel)

        self.print_networks(opt.verbose)

    def set_input(self, input):
        moving = input['A']
        fixed = input['B']

        # images
        if len(self.gpu_ids) > 0:
            moving = moving.cuda(self.gpu_ids[0], non_blocking=True)
            fixed = fixed.cuda(self.gpu_ids[0], non_blocking=True)
        self.moving = moving
        self.fixed = fixed

        # masks, only for test!!!
        moving_atlas = input['A_atlas']
        fixed_atlas = input['B_atlas']

        if len(self.gpu_ids) > 0:
            moving_atlas = moving_atlas.cuda(self.gpu_ids[0], non_blocking=True)
            fixed_atlas = fixed_atlas.cuda(self.gpu_ids[0], non_blocking=True)
        self.moving_atlas = moving_atlas
        self.fixed_atlas = fixed_atlas

        # paths
        self.moving_paths = input['A_paths']
        self.fixed_paths = input['B_paths']

    def forward(self):
        self.input_moving = Variable(self.moving)
        self.input_fixed = Variable(self.fixed)
        self.input_moving_atlas = Variable(self.moving_atlas).cuda()
        self.input_fixed_atlas = Variable(self.fixed_atlas).cuda()

    def backward_Reg(self):
        wrapped_image, _, flow, flow_mean, flow_log_sigma, \
        contrastive_features_x_three, contrastive_features_x_five, contrastive_features_x_seven, contrastive_features_y_three, contrastive_features_y_five, contrastive_features_y_seven = self.netReg(self.input_moving, self.input_fixed, self.input_moving_atlas)
        contrastive_features_x_three = F.normalize(contrastive_features_x_three, dim=1)
        contrastive_features_x_five = F.normalize(contrastive_features_x_five, dim=1)
        contrastive_features_x_seven = F.normalize(contrastive_features_x_seven, dim=1)
        contrastive_features_y_three = F.normalize(contrastive_features_y_three, dim=1)
        contrastive_features_y_five = F.normalize(contrastive_features_y_five, dim=1)
        contrastive_features_y_seven = F.normalize(contrastive_features_y_seven, dim=1)

        # # CVPR 2018
        # self.loss_recon = self.criterionRecon(wrapped_image, self.fixed)
        # self.loss_smooth = self.criterionSmooth(flow)

        # MICCAI 2018
        self.loss_recon = self.criterionRecon(wrapped_image, self.fixed)
        self.loss_smooth = self.criterionSmooth(flow_mean, flow_log_sigma)
        self.loss_contrastive_1 = 0.0001 * self.criterionContrastive(contrastive_features_x_three, contrastive_features_y_three)
        self.loss_contrastive_2 = 0.0001 * self.criterionContrastive(contrastive_features_x_five, contrastive_features_y_five)
        self.loss_contrastive_3 = 0.0001 * self.criterionContrastive(contrastive_features_x_seven, contrastive_features_y_seven)
        # backward
        self.loss_total = self.loss_recon + self.loss_smooth + self.loss_contrastive_1 + self.loss_contrastive_2 + self.loss_contrastive_3
        self.loss_total.backward()

        return self.loss_total

    def optimize_parameters(self):
        self.forward()

        self.optimizer_Reg.zero_grad()
        self.backward_Reg()
        self.optimizer_Reg.step()


    def test(self):
        # ~~~~~~~~~~~~~~~~~ Test for segmentation instead of registration ~~~~~~~~~~~~~~~~~
        self.netReg.eval()

        with torch.no_grad():
            self.input_moving = self.moving.cuda()
            self.input_fixed = self.fixed.cuda()
            self.input_moving_atlas =self.moving_atlas.cuda()
            self.input_fixed_atlas = self.fixed_atlas.cuda()

            warpeded, segmentation_result, flow, _, _, _, _, _, _, _, _ = self.netReg(
                self.input_moving, self.input_fixed, self.input_moving_atlas)

            # ~~~~~~~~~~~~~~~~~ Evaluation. ~~~~~~~~~~~~~~~~~
            self.metric_mean_dcs = 0

            # Calculate the Dice coefficient
            dsc, volume = metrics.dice(segmentation_result.data.int().cpu().numpy(),
                                       self.input_fixed_atlas.int().cpu().numpy())
            print('Total_Mask_Dice={:0.4f}. '.format(np.nanmean(dsc)))

            if not os.path.exists(str('./checkpoints/{}/output_{}'.format(
                    self.folder_names, self.folder_names))):
                os.makedirs(str('./checkpoints/{}/output_{}'.format(
                    self.folder_names, self.folder_names)))
                
            sitk.WriteImage(sitk.GetImageFromArray(self.input_fixed_atlas.data.int().cpu().numpy().squeeze()),
                            str('./checkpoints/{}/output_{}/fixed_{}'.format(
                                self.folder_names, self.folder_names, self.moving_paths[0].split('/')[-1])))
            
            sitk.WriteImage(sitk.GetImageFromArray(warpeded.data.int().cpu().numpy().squeeze()),
                            str('./checkpoints/{}/output_{}/warpeded_{}'.format(
                                self.folder_names, self.folder_names, self.moving_paths[0].split('/')[-1])))

            sitk.WriteImage(sitk.GetImageFromArray(segmentation_result.data.int().cpu().numpy().squeeze()),
                            str('./checkpoints/{}/output_{}/seg_of_moving_{}'.format(
                                self.folder_names, self.folder_names, self.moving_paths[0].split('/')[-1])))

            sitk.WriteImage(sitk.GetImageFromArray(np.transpose(flow.data.float().squeeze().cpu().numpy(), (1, 2, 3, 0))),
                            str('./checkpoints/{}/output_{}/deformation_field_{}'.format(
                                self.folder_names, self.folder_names, self.moving_paths[0].split('/')[-1])))


            self.metric_mean_dcs = np.nanmean(dsc)

            return self.metric_mean_dcs, dsc

    def reset(self):
        self.netReg.reset()