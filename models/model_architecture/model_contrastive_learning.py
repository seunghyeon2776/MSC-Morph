import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
from torchvision.models import vgg16, resnet50
from torch.distributions import normal
from torch.nn import init
import SimpleITK as sitk
from torch.distributions.normal import Normal


class UNet(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1_three = self.contracting_block_three(in_channels=in_channel // 2, out_channels=16)
        self.conv_encode1_five = self.contracting_block_five(in_channels=in_channel // 2, out_channels=16)
        self.conv_encode1_seven = self.contracting_block_seven(in_channels=in_channel // 2, out_channels=16)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2_three = self.contracting_block_three(16, 32)
        self.conv_encode2_five = self.contracting_block_five(16, 32)
        self.conv_encode2_seven = self.contracting_block_seven(16, 32)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3_three = self.contracting_block_three(32, 64)
        self.conv_encode3_five = self.contracting_block_five(32, 64)
        self.conv_encode3_seven = self.contracting_block_seven(32, 64)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_x = torch.nn.Linear(64, 64)
        self.linear_y = torch.nn.Linear(64, 64)

        # Bottleneck
        mid_channel = 128
        '''
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
            torch.nn.BatchNorm3d(mid_channel * 2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        '''
        
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding='same'),
            torch.nn.BatchNorm3d(mid_channel * 2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding='same'),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )
        
        self.skip_block_4 = self.skip_block(384, 128)
        self.skip_block_3 = self.skip_block(384, 128)
        self.skip_block_2 = self.skip_block(192, 64)
        self.skip_block_1 = self.skip_block(96, 32)
        
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, 3)

        self._init_weight()
    
    def contracting_block_three(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    
    '''
    def contracting_block_three(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    '''
    
    def contracting_block_five(self, in_channels, out_channels, kernel_size=5):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    
    '''
    def contracting_block_five(self, in_channels, out_channels, kernel_size=5):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    '''
    
    def contracting_block_seven(self, in_channels, out_channels, kernel_size=7):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    
    '''
    def contracting_block_seven(self, in_channels, out_channels, kernel_size=7):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    '''
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    
    '''
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding='same'),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block
    '''
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            # For Probablisitic Model - Softmax is Better
            torch.nn.Softmax() 
            # # For Non-Probablisitic Model - Relu is Better
            # torch.nn.ReLU()
        )
        return block
    
    def skip_block(self, in_channels, out_channels, kernel_size=3):
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU() 
        )
        return block
    
    '''
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding='same'),
            torch.nn.BatchNorm3d(out_channels),
            # For Probablisitic Model - Softmax is Better
            torch.nn.Softmax() 
            # # For Non-Probablisitic Model - Relu is Better
            # torch.nn.ReLU()
        )
        return block
    '''
    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose3d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.uniform_(0.0, 1.0)
                m.bias.data.fill_(0)

    def forward(self, x, y):
        # Encode 1
        encode_block1_three_x = self.conv_encode1_three(x)
        encode_block1_five_x = self.conv_encode1_five(x)
        encode_block1_seven_x = self.conv_encode1_seven(x)
        encode_pool1_three_x = self.conv_maxpool1(encode_block1_three_x)
        encode_pool1_five_x = self.conv_maxpool1(encode_block1_five_x)
        encode_pool1_seven_x = self.conv_maxpool1(encode_block1_seven_x)
        encode_block2_three_x = self.conv_encode2_three(encode_pool1_three_x)
        encode_block2_five_x = self.conv_encode2_five(encode_pool1_five_x)
        encode_block2_seven_x = self.conv_encode2_seven(encode_pool1_seven_x)
        encode_pool2_three_x = self.conv_maxpool2(encode_block2_three_x)
        encode_pool2_five_x = self.conv_maxpool2(encode_block2_five_x)
        encode_pool2_seven_x = self.conv_maxpool2(encode_block2_seven_x)
        encode_block3_three_x = self.conv_encode3_three(encode_pool2_three_x)
        encode_block3_five_x = self.conv_encode3_five(encode_pool2_five_x)
        encode_block3_seven_x = self.conv_encode3_seven(encode_pool2_seven_x)
        encode_pool3_three_x = self.conv_maxpool3(encode_block3_three_x)
        encode_pool3_five_x = self.conv_maxpool3(encode_block3_five_x)
        encode_pool3_seven_x = self.conv_maxpool3(encode_block3_seven_x)
        
        f_three_x = self.avgpool(encode_pool3_three_x)
        f_five_x = self.avgpool(encode_pool3_five_x)
        f_seven_x = self.avgpool(encode_pool3_seven_x)
        f_three_x = f_three_x.squeeze()
        f_five_x = f_five_x.squeeze()
        f_seven_x = f_seven_x.squeeze()
        f_three_x = self.linear_x(f_three_x)
        f_five_x = self.linear_x(f_five_x)
        f_seven_x = self.linear_x(f_seven_x)
        f_three_x = f_three_x / f_three_x.norm(dim=-1, keepdim=True)
        f_five_x = f_five_x / f_five_x.norm(dim=-1, keepdim=True)
        f_seven_x = f_seven_x / f_seven_x.norm(dim=-1, keepdim=True)

        
        # Encode 2
        encode_block1_three_y = self.conv_encode1_three(y)
        encode_block1_five_y = self.conv_encode1_five(y)
        encode_block1_seven_y = self.conv_encode1_seven(y)
        encode_pool1_three_y = self.conv_maxpool1(encode_block1_three_y)
        encode_pool1_five_y = self.conv_maxpool1(encode_block1_five_y)
        encode_pool1_seven_y = self.conv_maxpool1(encode_block1_seven_y)
        encode_block2_three_y = self.conv_encode2_three(encode_pool1_three_y)
        encode_block2_five_y = self.conv_encode2_five(encode_pool1_five_y)
        encode_block2_seven_y = self.conv_encode2_seven(encode_pool1_seven_y)
        encode_pool2_three_y = self.conv_maxpool2(encode_block2_three_y)
        encode_pool2_five_y = self.conv_maxpool2(encode_block2_five_y)
        encode_pool2_seven_y = self.conv_maxpool2(encode_block2_seven_y)
        encode_block3_three_y = self.conv_encode3_three(encode_pool2_three_y)
        encode_block3_five_y = self.conv_encode3_five(encode_pool2_five_y)
        encode_block3_seven_y = self.conv_encode3_seven(encode_pool2_seven_y)
        encode_pool3_three_y = self.conv_maxpool3(encode_block3_three_y)
        encode_pool3_five_y = self.conv_maxpool3(encode_block3_five_y)
        encode_pool3_seven_y = self.conv_maxpool3(encode_block3_seven_y)
        
        f_three_y = self.avgpool(encode_pool3_three_y)
        f_five_y = self.avgpool(encode_pool3_five_y)
        f_seven_y = self.avgpool(encode_pool3_seven_y)
        f_three_y = f_three_y.squeeze()
        f_five_y = f_five_y.squeeze()
        f_seven_y = f_seven_y.squeeze()
        f_three_y = self.linear_y(f_three_y)
        f_five_y = self.linear_y(f_five_y)
        f_seven_y = self.linear_y(f_seven_y)
        f_three_y = f_three_y / f_three_y.norm(dim=-1, keepdim=True)
        f_five_y = f_five_y / f_five_y.norm(dim=-1, keepdim=True)
        f_seven_y = f_seven_y / f_seven_y.norm(dim=-1, keepdim=True)

        
        # Bottleneck
        
        cat_4 = torch.cat((encode_pool3_three_x, encode_pool3_five_x, encode_pool3_seven_x, encode_pool3_three_y, encode_pool3_five_y, encode_pool3_seven_y), 1)
        skip_4 = self.skip_block_4(cat_4)
        bottleneck1 = self.bottleneck(skip_4)

        # Decode
        
        cat_3 = torch.cat((encode_block3_three_x, encode_block3_five_x, encode_block3_seven_x, encode_block3_three_y, encode_block3_five_y, encode_block3_seven_y), 1)
        skip_3 = self.skip_block_3(cat_3)
        decode_block3 = self.crop_and_concat(bottleneck1, skip_3)
        cat_layer2 = self.conv_decode3(decode_block3)
        
        cat_2 = torch.cat((encode_block2_three_x, encode_block2_five_x, encode_block2_seven_x, encode_block2_three_y, encode_block2_five_y, encode_block2_seven_y), 1)
        skip_2 = self.skip_block_2(cat_2)
        decode_block2 = self.crop_and_concat(cat_layer2, skip_2)
        cat_layer1 = self.conv_decode2(decode_block2)
        
        
        cat_1 = torch.cat((encode_block1_three_x, encode_block1_five_x, encode_block1_seven_x, encode_block1_three_y, encode_block1_five_y, encode_block1_seven_y), 1)
        skip_1 = self.skip_block_1(cat_1)
        decode_block1 = self.crop_and_concat(cat_layer1, skip_1)
        final_layer = self.final_layer(decode_block1)

        return final_layer, f_three_x, f_five_x, f_seven_x, f_three_y, f_five_y, f_seven_y


class ProbabilisticModel(nn.Module):
    def __init__(self, is_training=True):
        super(ProbabilisticModel, self).__init__()

        self.mean = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.log_sigma = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        # Manual Initialization
        self.mean.weight.data.normal_(0, 1e-5)
        self.log_sigma.weight.data.normal_(0, 1e-10)
        self.log_sigma.bias.data.fill_(-10.)

        self.is_training=is_training

    def forward(self, final_layer):
        flow_mean = self.mean(final_layer)
        flow_log_sigma = self.log_sigma(final_layer)
        noise = torch.randn_like(flow_mean).cuda()

        if self.is_training:
            flow = flow_mean + flow_log_sigma * noise 
        else:
            flow = flow_mean + flow_log_sigma # No noise at testing time

        return flow, flow_mean, flow_log_sigma


class VoxelMorph3d(nn.Module):
    def __init__(self, in_channels=2, use_gpu=False, is_training=True, img_size=(128, 128, 128)):
        super(VoxelMorph3d, self).__init__()
        self.unet = UNet(in_channels, 3)
        self.probabilistic_model = ProbabilisticModel(is_training=is_training)
        self.spatial_transform = SpatialTransformer(img_size)

        if use_gpu:
            self.unet = self.unet.cuda()
            self.probabilistic_model = self.probabilistic_model.cuda()
            self.spatial_transform = self.spatial_transform.cuda()

    def forward(self, moving_image, fixed_image, moving_atlas):
        flow, f_three_x, f_five_x, f_seven_x, f_three_y, f_five_y, f_seven_y = self.unet(moving_image, fixed_image)

        deformation_matrix, flow_mean, flow_log_sigma = self.probabilistic_model(flow)
        warped_image = self.spatial_transform(moving_image, deformation_matrix)
        warped_image_atlas = self.spatial_transform(moving_atlas, deformation_matrix, mode="nearest")

        return warped_image, warped_image_atlas, deformation_matrix, flow_mean, flow_log_sigma, f_three_x, f_five_x, f_seven_x, f_three_y, f_five_y, f_seven_y


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode='bilinear'): #nearest도 한 번 해보면 좋을듯?
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode)