import torch
import torch.nn as nn
import numpy as np
import os


class SkipBN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


def calc_qprams(_min, _max, q_max):
    assert q_max == 15 or q_max == 255, print("Not Supported int type!\nPlz use uint4 or int8")
    # if q_max == 15:
    #     s = (_max - _min) / q_max
    #     z = - round(_min / s)
    #     return s, np.clip(z, 0, q_max)
    # elif q_max == 255:
    #     s = (_max - _min) / q_max
    #     z = -128 - round(_min / s)
    #     return s, np.clip(z, -128, q_max - 128)
    if q_max == 15:
        s = _max.sub(_min).div(q_max)
        z = - torch.round(_min.div(s))
        return nn.Parameter(s, requires_grad=False), nn.Parameter(torch.clamp(z, 0, q_max), requires_grad=False)
    elif q_max == 255:
        s = _max.sub(_min).div(q_max)
        z = -128 - torch.round(_min.div(s))
        return nn.Parameter(s, requires_grad=False), nn.Parameter(torch.clamp(z, -128, 127), requires_grad=False)



def set_fused_resnet18_params(fused, pre):
    """
        Copy from pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    # First layer
    fused.first_conv.copy_from_pretrained(pre.conv1, pre.bn1)

    # Block 1
    block = fused.layer1
    for i in range(len(block)):
        block[i].conv1.copy_from_pretrained(pre.layer1[i].conv1, pre.layer1[i].bn1)
        block[i].conv2.copy_from_pretrained(pre.layer1[i].conv2, pre.layer1[i].bn2)

    # Block 2
    block = fused.layer2
    block[0].downsample.copy_from_pretrained(pre.layer2[0].downsample[0], pre.layer2[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1.copy_from_pretrained(pre.layer2[i].conv1, pre.layer2[i].bn1)
        block[i].conv2.copy_from_pretrained(pre.layer2[i].conv2, pre.layer2[i].bn2)

    # Block 3
    block = fused.layer3
    block[0].downsample.copy_from_pretrained(pre.layer3[0].downsample[0], pre.layer3[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1.copy_from_pretrained(pre.layer3[i].conv1, pre.layer3[i].bn1)
        block[i].conv2.copy_from_pretrained(pre.layer3[i].conv2, pre.layer3[i].bn2)

    # Block 4
    block = fused.layer4
    block[0].downsample.copy_from_pretrained(pre.layer4[0].downsample[0], pre.layer4[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1.copy_from_pretrained(pre.layer4[i].conv1, pre.layer4[i].bn1)
        block[i].conv2.copy_from_pretrained(pre.layer4[i].conv2, pre.layer4[i].bn2)

    # Classifier
    fused.fc.fc.weight = torch.nn.Parameter(pre.fc.weight)
    fused.fc.fc.bias = torch.nn.Parameter(pre.fc.bias)
    return fused


def set_fused_resnet20_params(fused, pre):
    """
        Copy pre model's params to fused layers.
        Use fused architecture, but not really fused (use CONV & BN seperately)
    """
    # First layer
    fused.first_conv.copy_from_pretrained(pre.conv1, pre.bn1)

    # Block 1
    block = fused.layer1
    for i in range(len(block)):
        block[i].conv1.copy_from_pretrained(pre.layer1[i].conv1, pre.layer1[i].bn1)
        block[i].conv2.copy_from_pretrained(pre.layer1[i].conv2, pre.layer1[i].bn2)

    # Block 2
    block = fused.layer2
    block[0].downsample.copy_from_pretrained(pre.layer2[0].downsample[0], pre.layer2[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1.copy_from_pretrained(pre.layer2[i].conv1, pre.layer2[i].bn1)
        block[i].conv2.copy_from_pretrained(pre.layer2[i].conv2, pre.layer2[i].bn2)

    # Block 3
    block = fused.layer3
    block[0].downsample.copy_from_pretrained(pre.layer3[0].downsample[0], pre.layer3[0].downsample[1])
    for i in range(len(block)):
        block[i].conv1.copy_from_pretrained(pre.layer3[i].conv1, pre.layer3[i].bn1)
        block[i].conv2.copy_from_pretrained(pre.layer3[i].conv2, pre.layer3[i].bn2)

    # Classifier
    fused.fc.fc.weight = torch.nn.Parameter(pre.fc.weight)
    fused.fc.fc.bias = torch.nn.Parameter(pre.fc.bias)
    return fused


def fuse_resnet(model, arch):
    # First layer
    model.first_conv.fuse_conv_and_bn()

    # Block 1
    block = model.layer1
    for i in range(len(block)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()

    # Block 2
    block = model.layer2
    block[0].downsample.fuse_conv_and_bn()
    for i in range(len(block)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()

    # Block 3
    block = model.layer3
    block[0].downsample.fuse_conv_and_bn()
    for i in range(len(block)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()

    # Block 4
    if arch in ['resnet18']:
        block = model.layer4
        block[0].downsample.fuse_conv_and_bn()
        for i in range(len(block)):
            block[i].conv1.fuse_conv_and_bn()
            block[i].conv2.fuse_conv_and_bn()
    return model


def transform(param):
    return param.flatten().numpy().astype('float32')


def save_qparams(m, prev_s, prev_z, f):
    assert m.layer_type in ['FusedConv2d', 'FusedLinear'],\
        "Can't parse Q-params from {}".format(type(m))
    if m.layer_type == 'FusedConv2d':
        s2, z2 = m.conv.scale, m.conv.zero_point
    else:
        s2, z2 = m.fc.scale, m.fc.zero_point
    s3, z3 = m.scale, m.zero_point

    print("S1: {} | Z1: {}".format(prev_s, prev_z))
    print("S2: {} | Z2: {}".format(s2, z2))
    print("S3: {} | Z3: {}".format(s3, z3))
    prev_s.numpy().astype('float32').tofile(f)
    s2.numpy().astype('float32').tofile(f)
    s3.numpy().astype('float32').tofile(f)
    prev_z.numpy().astype('int32').tofile(f)
    z2.numpy().astype('int32').tofile(f)
    z3.numpy().astype('int32').tofile(f)
    return s3, z3


def save_block_qparams(block, bypass_in_s, bypass_in_z, f):
    in_s, in_z = bypass_in_s, bypass_in_z
    # Downsampling after bypass-connection
    if block.downsample:
        in_s, in_z = save_qparams(block.downsample, bypass_in_s, bypass_in_z, f)

    # CONV after bypass-connection
    prev_s, prev_z = save_qparams(block.conv1, bypass_in_s, bypass_in_z, f)

    # 2nd CONV in a block
    prev_s, prev_z = save_qparams(block.conv2, prev_s, prev_z, f)

    # SHORTCUT layer in Darknet
    in_s.numpy().astype('float32').tofile(f)
    prev_s.numpy().astype('float32').tofile(f)
    block.scale.numpy().astype('float32').tofile(f)
    in_z.numpy().astype('int32').tofile(f)
    prev_z.numpy().astype('int32').tofile(f)
    block.zero_point.numpy().astype('int32').tofile(f)
    return block.scale, block.zero_point


def save_params(model, path):
    with open(path, 'w') as f:
        weight = None
        for name, param in model.named_parameters():
            if 'weight' in name:
                print(">> Layer: {}\tshape={}".format(name, str(param.data.shape).replace('torch.Size', '')))
                weight = transform(param.data)
            elif 'bias' in name:
                print(">> Layer: {}\tshape={}".format(name, str(param.data.shape).replace('torch.Size', '')))
                transform(param.data).tofile(f)
                weight.tofile(f)


def save_fused_alexnet_qparams(model, path):
    with open(path, 'w') as f:
        prev_s3, prev_z3 = model.get_input_qparams()
        for name, m in model.named_children():
            if 'features' in name:
                prev_s3, prev_z3 = save_qparams(m[0], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_qparams(m[2], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_qparams(m[4], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_qparams(m[5], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_qparams(m[6], prev_s3, prev_z3, f)

            elif 'classifier' in name:
                prev_s3, prev_z3 = save_qparams(m[0], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_qparams(m[1], prev_s3, prev_z3, f)
                _, _ = save_qparams(m[2], prev_s3, prev_z3, f)


def save_fused_resnet_qparams(model, path):
    with open(path, 'wb') as f:
        prev_s, prev_z = model.scale, model.zero_point
        for name, m in model.named_children():
            if "layer" in name:
                for i in range(len(m)):
                    prev_s, prev_z = save_block_qparams(m[i], prev_s, prev_z, f)
            elif name in ["first_conv", "fc"]:
                prev_s, prev_z = save_qparams(m, prev_s, prev_z, f)


def save_fused_network_in_darknet_form(model, arch):
    path = './result/darknet'
    if not os.path.exists(path):
        os.makedirs(path)
    path = os.path.join(path, '{}.fused.torch.'.format(arch))

    model.cpu()
    save_params(model, path + 'weights')
    if "resnet" in arch:
        save_fused_resnet_qparams(model, path + 'qparams')
    elif arch == "alexnet":
        save_fused_alexnet_qparams(model, path + 'qparams')
