import torch
import torch.nn as nn
import torch.nn.functional as F

# 自动填充函数，用于根据卷积核大小自动计算填充量
def autopad(k, p=None, d=1):
    """根据卷积核大小自动填充，使得输出与输入尺寸一致。"""
    if d > 1:  # 考虑卷积核的扩张
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:  # 自动计算填充量
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p

# 标准卷积模块
class Conv(nn.Module):
    """标准卷积模块，包含卷积、批归一化和激活函数。"""
    default_act = nn.SiLU()  # 默认激活函数

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """初始化卷积层，包含输入通道、输出通道、卷积核大小、步长等参数。"""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)  # 批归一化层
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """前向传播，依次应用卷积、批归一化和激活函数。"""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """在推理时使用卷积操作，省略批归一化。"""
        return self.act(self.conv(x))

# ODConv的注意力机制类，用于计算不同维度上的注意力
class Attention(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, groups=1, reduction=0.0625, kernel_num=4, min_channel=16):
        """初始化注意力机制层，包含输入通道、输出通道、卷积核大小、组数等参数。"""
        super(Attention, self).__init__()
        attention_channel = max(int(in_planes * reduction), min_channel)
        self.kernel_size = kernel_size
        self.kernel_num = kernel_num
        self.temperature = 1.0

        self.avgpool = nn.AdaptiveAvgPool2d(1)  # 自适应全局平均池化
        self.fc = nn.Conv2d(in_planes, attention_channel, 1, bias=False)  # 全连接层，用于通道缩减
        self.bn = nn.BatchNorm2d(attention_channel)  # 批归一化
        self.relu = nn.ReLU(inplace=True)  # ReLU激活函数

        self.channel_fc = nn.Conv2d(attention_channel, in_planes, 1, bias=True)  # 通道注意力层
        self.func_channel = self.get_channel_attention  # 获取通道注意力

        if in_planes == groups and in_planes == out_planes:  # 深度卷积
            self.func_filter = self.skip  # 如果输入通道数和组数相等，则跳过过滤器注意力
        else:
            self.filter_fc = nn.Conv2d(attention_channel, out_planes, 1, bias=True)  # 过滤器注意力层
            self.func_filter = self.get_filter_attention  # 获取过滤器注意力

        if kernel_size == 1:  # 如果是点卷积，跳过空间注意力
            self.func_spatial = self.skip
        else:
            self.spatial_fc = nn.Conv2d(attention_channel, kernel_size * kernel_size, 1, bias=True)  # 空间注意力层
            self.func_spatial = self.get_spatial_attention  # 获取空间注意力

        if kernel_num == 1:  # 如果只有一个卷积核，跳过核注意力
            self.func_kernel = self.skip
        else:
            self.kernel_fc = nn.Conv2d(attention_channel, kernel_num, 1, bias=True)  # 卷积核注意力层
            self.func_kernel = self.get_kernel_attention  # 获取卷积核注意力

        self._initialize_weights()

    def _initialize_weights(self):
        """初始化权重。"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias

, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def update_temperature(self, temperature):
        """更新温度参数，用于控制注意力机制的敏感性。"""
        self.temperature = temperature

    @staticmethod
    def skip(_):
        """跳过某个注意力机制。"""
        return 1.0

    def get_channel_attention(self, x):
        """计算通道注意力。"""
        channel_attention = torch.sigmoid(self.channel_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return channel_attention

    def get_filter_attention(self, x):
        """计算过滤器注意力。"""
        filter_attention = torch.sigmoid(self.filter_fc(x).view(x.size(0), -1, 1, 1) / self.temperature)
        return filter_attention

    def get_spatial_attention(self, x):
        """计算空间注意力。"""
        spatial_attention = self.spatial_fc(x).view(x.size(0), 1, 1, 1, self.kernel_size, self.kernel_size)
        spatial_attention = torch.sigmoid(spatial_attention / self.temperature)
        return spatial_attention

    def get_kernel_attention(self, x):
        """计算卷积核注意力。"""
        kernel_attention = self.kernel_fc(x).view(x.size(0), -1, 1, 1, 1, 1)
        kernel_attention = F.softmax(kernel_attention / self.temperature, dim=1)
        return kernel_attention

    def forward(self, x):
        """前向传播，依次计算通道、过滤器、空间和卷积核的注意力。"""
        x = self.avgpool(x)
        x = self.fc(x)
        x = self.relu(x)
        return self.func_channel(x), self.func_filter(x), self.func_spatial(x), self.func_kernel(x)

# ODConv2d类实现，集成注意力机制的卷积层
class ODConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=1, dilation=1, groups=1, reduction=0.0625, kernel_num=4):
        """初始化ODConv卷积层，包含输入通道、输出通道、卷积核大小、步长、填充等参数。"""
        super(ODConv2d, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.kernel_num = kernel_num
        self.attention = Attention(in_planes, out_planes, kernel_size, groups=groups, reduction=reduction, kernel_num=kernel_num)
        self.weight = nn.Parameter(torch.randn(kernel_num, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        self._initialize_weights()

        if self.kernel_size == 1 and self.kernel_num == 1:
            self._forward_impl = self._forward_impl_pw1x
        else:
            self._forward_impl = self._forward_impl_common

    def _initialize_weights(self):
        """初始化权重。"""
        for i in range(self.kernel_num):
            nn.init.kaiming_normal_(self.weight[i], mode='fan_out', nonlinearity='relu')

    def update_temperature(self, temperature):
        """更新温度参数。"""
        self.attention.update_temperature(temperature)

    def _forward_impl_common(self, x):
        """常规前向传播实现，包含多个卷积核的卷积操作。"""
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x * channel_attention
        x = x.reshape(1, -1, height, width)
        aggregate_weight = spatial_attention * kernel_attention * self.weight.unsqueeze(dim=0)
        aggregate_weight = torch.sum(aggregate_weight, dim=1).view([-1, self.in_planes // self.groups, self.kernel_size, self.kernel_size])
        output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups * batch_size)
        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        output = output * filter_attention
        return output

    def _forward_impl_pw1x(self, x):
        """当卷积核大小为1x1时的前向传播实现。"""
        channel_attention, filter_attention, spatial_attention, kernel_attention = self.attention(x)
        x = x * channel_attention
        output = F.conv2d(x, weight=self.weight.squeeze(dim=0), bias=None, stride=self.stride, padding=self.padding, dilation=self.dilation, groups=self.groups)
        output = output * filter_attention
        return output

    def forward(self, x):
        """调用前向传播方法。"""
        return self._forward_impl(x)
