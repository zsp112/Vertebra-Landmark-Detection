import torch.nn as nn
import torch
from .model_parts import CombinationModule

class DecNet(nn.Module):
    def __init__(self, heads, final_kernel, head_conv, channel):
        super(DecNet, self).__init__()
        # Feature 4 和 Feature 3 融合
        self.dec_c4 = CombinationModule(1024, 512, batch_norm=True)  # 输入 1024，输出 512
        # Feature 3 和 Feature 2 融合
        self.dec_c3 = CombinationModule(512, 256, batch_norm=True)  # 输入 512，输出 256
        # Feature 2 和 Feature 1 融合
        self.dec_c2 = CombinationModule(256, 128, batch_norm=True)  # 输入 256，输出 128
        # self.dec_c1 = CombinationModule(128, 64, batch_norm=True)  # 输入 256，输出 128
        self.heads = heads
        for head in self.heads:
            classes = self.heads[head]
            if head == 'wh':
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=7, padding=7//2, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=7, padding=7 // 2, bias=True))
            else:
                fc = nn.Sequential(nn.Conv2d(channel, head_conv, kernel_size=3, padding=1, bias=True),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(head_conv, classes, kernel_size=final_kernel, stride=1,
                                             padding=final_kernel // 2, bias=True))
            if 'hm' in head:
                fc[-1].bias.data.fill_(-2.19)
            else:
                self.fill_fc_weights(fc)

            self.__setattr__(head, fc)


    def fill_fc_weights(self, layers):
        for m in layers.modules():
            if isinstance(m, nn.Conv2d):
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x 是 HRNet 的特征输出列表，从底层到高层逐步融合
        c4_combine = self.dec_c4(x[-1], x[-2])  # (batch_size, 512, 64, 32)
        c3_combine = self.dec_c3(c4_combine, x[-3])  # (batch_size, 256, 128, 64)
        c2_combine = self.dec_c2(c3_combine, x[-4])  # (batch_size, 128, 256, 128)
        # c1_combine = self.dec_c1(c2_combine, x[-5])  # (batch_size, 64, 512, 256)

        # 生成最终的输出
        dec_dict = {}
        for head in self.heads:
            dec_dict[head] = self.__getattr__(head)(c2_combine)  # 使用 c2_combine 作为输入
            if 'hm' in head:
                dec_dict[head] = torch.sigmoid(dec_dict[head])  # 热图使用 sigmoid
        return dec_dict