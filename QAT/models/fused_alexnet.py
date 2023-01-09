import torch
import torch.nn as nn
from typing import Any
from .layers.conv2d import *
from .layers.linear import *
from .quant_noise import _quant_noise
from .quantization_utils import *


class FusedAlexNet(nn.Module):
    def __init__(self, arg_dict: dict, num_classes: int = 1000) -> None:
        super(FusedAlexNet, self).__init__()
        arg_bit, self.smooth, self.runtime_helper = itemgetter('bit', 'smooth', 'runtime_helper')(arg_dict)
        self.bit = torch.nn.Parameter(torch.tensor(arg_bit, dtype=torch.int8), requires_grad=False)
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)

        self.apply_ema = False

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.conv1 = FusedConv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = FusedConv2d(64, 192, kernel_size=5, stride=1, padding=2, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv3 = FusedConv2d(192, 384, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv4 = FusedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv5 = FusedConv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.fc1 = FusedLinear(256 * 6 * 6, 4096, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc2 = FusedLinear(4096, 4096, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc3 = FusedLinear(4096, num_classes, arg_dict=arg_dict)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            if self.apply_ema:
                self.in_range[0], self.in_range[1] = ema(x, self.in_range, self.smooth)
                if self.runtime_helper.apply_fake_quantization:
                    s, z = calc_qparams(self.in_range[0], self.in_range[1], self.bit)
                    x = fake_quantize(x, s, z, self.bit)
            else:
                self.in_range[0], self.in_range[1] = get_range(x)
                self.apply_ema.data = torch.tensor(True, dtype=torch.bool)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x

    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.bit)
        prev_s, prev_z = self.conv1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv4.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv5.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        self.fc3.set_qparams(prev_s, prev_z)


class FusedAlexNetSmall(nn.Module):
    def __init__(self, arg_dict: dict, num_classes: int = 10) -> None:
        super(FusedAlexNetSmall, self).__init__()
        target_bit, bit_first, bit_classifier, self.smooth, self.runtime_helper, self.mixed_precision \
            = itemgetter('bit', 'bit_first', 'bit_classifier', 'smooth', 'runtime_helper', 'mixed_precision')(arg_dict)
        self.target_bit = torch.nn.Parameter(torch.tensor(target_bit, dtype=torch.int8), requires_grad=False)
        self.in_bit = torch.nn.Parameter(torch.tensor(bit_first, dtype=torch.int8), requires_grad=False)

        # if self.mixed_precision:
        #     self.mixed_range = nn.Parameter(torch.zeros(2, 3), requires_grad=False)
        self.in_range = nn.Parameter(torch.zeros(2), requires_grad=False)
        self.apply_ema = nn.Parameter(torch.zeros(1), requires_grad=False)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = FusedConv2d(3, 96, kernel_size=5, stride=1, padding=2, bias=True,
                                 w_bit=bit_first, activation=nn.ReLU, arg_dict=arg_dict)
        self.conv2 = FusedConv2d(96, 256, kernel_size=5, stride=1, padding=2, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv3 = FusedConv2d(256, 384, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv4 = FusedConv2d(384, 384, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.conv5 = FusedConv2d(384, 256, kernel_size=3, stride=1, padding=1, bias=True,
                                 activation=nn.ReLU, arg_dict=arg_dict)
        self.fc1 = FusedLinear(256, 4096, bias=True, activation=nn.ReLU, arg_dict=arg_dict)
        self.fc2 = FusedLinear(4096, 4096, bias=True, activation=nn.ReLU, a_bit=bit_classifier, arg_dict=arg_dict)
        self.fc3 = FusedLinear(4096, num_classes, bias=True, is_classifier=True,
                               w_bit=bit_classifier, a_bit=bit_classifier, arg_dict=arg_dict)

        if self.mixed_precision:
            for module in self.modules():
                if isinstance(module, FusedConv2d):
                    module.input_range = nn.Parameter(torch.zeros((2, module.in_channels)), requires_grad=False)
                    module.mixed_ema = nn.Parameter(torch.tensor(0, dtype=torch.bool), requires_grad=False)
            self.percentile = None
            self.total_ch_sum = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            self._update_input_ranges(x)
            if self.runtime_helper.apply_fake_quantization:
                x = self._fake_quantize_input(x)

        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

    @torch.no_grad()
    def _update_input_ranges(self, x):
        _min = x.min().item()
        _max = x.max().item()
        if self.apply_ema:
            self.in_range[0] = self.in_range[0] * self.smooth + _min * (1 - self.smooth)
            self.in_range[1] = self.in_range[1] * self.smooth + _max * (1 - self.smooth)
        else:
            self.in_range[0], self.in_range[1] = _min, _max
            self.apply_ema.data = torch.tensor(True, dtype=torch.bool)


    def _fake_quantize_input(self, x):
        s, z = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)
        if self.mixed_precision:
            return fake_quantize_per_input_channel(x, self.conv1.low_bit, self.conv1.low_group, self.conv1.high_group,
                                                   scale=s, zero_point=z)
        else:
            return fake_quantize(x, s, z, self.in_bit)


    def set_quantization_params(self):
        self.scale, self.zero_point = calc_qparams(self.in_range[0], self.in_range[1], self.in_bit)

        prev_s, prev_z = self.conv1.set_qparams(self.scale, self.zero_point)
        prev_s, prev_z = self.conv2.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv3.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv4.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.conv5.set_qparams(prev_s, prev_z)

        prev_s, prev_z = self.fc1.set_qparams(prev_s, prev_z)
        prev_s, prev_z = self.fc2.set_qparams(prev_s, prev_z)
        _, _ = self.fc3.set_qparams(prev_s, prev_z)


    def set_mixed_bits(self, x: torch.Tensor) -> torch.Tensor:
        def record_input_range(x, module):
            data = x.transpose(1, 0).reshape(x.size(1), -1)
            _max = data.max(dim=1).values
            _min = data.min(dim=1).values

            if self.runtime_helper.input_ema_method == 'max':
                if module.mixed_ema:
                    updated_min = module.val_input_range[0] * self.smooth + _min * (1 - self.smooth)
                    updated_max = module.val_input_range[1] * self.smooth + _max * (1 - self.smooth)

                    module.val_input_range[0], module.val_input_range[1] = updated_min, updated_max
                else:
                    module.val_input_range[0], module.val_input_range[1] = _min, _max
                    module.mixed_ema.data = torch.tensor(True, dtype=torch.bool)

            else:
                module.val_input_range[0] += _min
                module.val_input_range[1] += _max

            return module(x)

        x = record_input_range(x, self.conv1)
        x = self.maxpool(x)
        x = record_input_range(x, self.conv2)
        x = self.maxpool(x)
        x = record_input_range(x, self.conv3)
        x = record_input_range(x, self.conv4)
        x = record_input_range(x, self.conv5)
        x = self.maxpool(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x


def fused_alexnet(arg_dict: dict, **kwargs: Any) -> FusedAlexNet:
    return FusedAlexNet(arg_dict, **kwargs)


def fused_alexnet_small(arg_dict: dict, num_classes=10, **kwargs: Any) -> FusedAlexNetSmall:
    return FusedAlexNetSmall(arg_dict, num_classes=num_classes, **kwargs)


def set_fused_alexnet(fused, pre):
    """
        Copy pre model's params & set fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    fused.conv1 = copy_from_pretrained(fused.conv1, pre.features[0])
    fused.conv2 = copy_from_pretrained(fused.conv2, pre.features[3])
    fused.conv3 = copy_from_pretrained(fused.conv3, pre.features[6])
    fused.conv4 = copy_from_pretrained(fused.conv4, pre.features[8])
    fused.conv5 = copy_from_pretrained(fused.conv5, pre.features[10])
    fused.fc1 = copy_from_pretrained(fused.fc1, pre.classifier[1])
    fused.fc2 = copy_from_pretrained(fused.fc2, pre.classifier[4])
    fused.fc3 = copy_from_pretrained(fused.fc3, pre.classifier[6])
    return fused

def modify_fused_alexnet_qn_pre_hook(model):
    model.conv1.conv = _quant_noise(model.conv1.conv, model.runtime_helper.qn_prob, 1, bit=model.bit)
    model.conv2.conv = _quant_noise(model.conv2.conv, model.runtime_helper.qn_prob, 1, bit=model.bit)
    model.conv3.conv = _quant_noise(model.conv3.conv, model.runtime_helper.qn_prob, 1, bit=model.bit)
    model.conv4.conv = _quant_noise(model.conv4.conv, model.runtime_helper.qn_prob, 1, bit=model.bit)
    model.conv5.conv = _quant_noise(model.conv5.conv, model.runtime_helper.qn_prob, 1, bit=model.bit)
    model.fc1.fc = _quant_noise(model.fc1.fc, model.runtime_helper.qn_prob, 1, bit=model.bit)
    model.fc2.fc = _quant_noise(model.fc2.fc, model.runtime_helper.qn_prob, 1, bit=model.bit)
    model.fc3.fc = _quant_noise(model.fc3.fc, model.runtime_helper.qn_prob, 1, bit=model.bit)
