from .dec_net import DecNet
from . import resnet
import torch.nn as nn
import numpy as np
from timm import create_model

class SpineNet(nn.Module):
    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(SpineNet, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        channels = [3, 64, 128, 256, 512, 1024]
        self.l1 = int(np.log2(down_ratio))
        # 使用 Swin Transformer 作为主干网络
        pretrained_cfg_overlay = {'file': r"C:\Users\susu\.cache\torch\hub\checkpoints\pytorch_model.bin"}
        self.base_network = create_model('hrnet_w32.ms_in1k', pretrained=pretrained,
                                         pretrained_cfg_overlay=pretrained_cfg_overlay, features_only=True)
        self.dec_net = DecNet(heads, final_kernel, head_conv, channels[self.l1])

    def forward(self, x):
        base_features = self.base_network(x)

        # 将原始输入图像作为第一个特征，手动添加到输出列表中
        # base_features = [x] + base_features  # 将输入图像添加到列表开头

        # 根据 HRNet 的结构，最后几层的分辨率递减，需要根据实际需求调整选择
        dec_dict = self.dec_net(base_features)
        return dec_dict

