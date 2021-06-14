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
    if q_max == 15:
        s = (_max - _min) / q_max
        z = - round(_min / s)
        return s, np.clip(z, 0, q_max)
    elif q_max == 255:
        # Darknet uses INT8, not UINT8
        s = (_max - _min) / q_max
        z = -128 - round(_min / s)
        return s, np.clip(z, -128, q_max - 128)


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


def fuse_resnet18(model):
    # First layer
    model.first_conv.fuse_conv_and_bn()

    # Block 1
    block = model.layer1
    for i in range(1, len(model.layer1)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()

    # Block 2
    block = model.layer2
    block.downsample.fuse_conv_and_bn()
    for i in range(1, len(model.layer2)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()

    # Block 3
    block = model.layer3
    block.downsample.fuse_conv_and_bn()
    for i in range(1, len(model.layer3)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()

    # Block 4
    block = model.layer4
    block.downsample.fuse_conv_and_bn()
    for i in range(1, len(model.layer4)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()
    return model


def fuse_resnet20(model):
    # First layer
    model.first_conv.fuse_conv_and_bn()

    # Block 1
    block = model.layer1
    for i in range(1, len(model.layer1)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()

    # Block 2
    block = model.layer2
    block.downsample.fuse_conv_and_bn()
    for i in range(1, len(model.layer2)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()

    # Block 3
    block = model.layer3
    block.downsample.fuse_conv_and_bn()
    for i in range(1, len(model.layer3)):
        block[i].conv1.fuse_conv_and_bn()
        block[i].conv2.fuse_conv_and_bn()
    return model


def transform(param):
    return param.flatten().numpy().astype('float32')


def save_fc_qparams(m, prev_s, prev_z, f):
    s2, z2 = m.fc.get_weight_qparams()
    s3, z3 = m.get_activation_qparams()
    print("S1: {} | Z1: {}".format(prev_s, prev_z))
    print("S2: {} | Z2: {}".format(s2, z2))
    print("S3: {} | Z3: {}".format(s3, z3))
    np.array(prev_s).astype('float32').tofile(f)
    np.array(s2).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(prev_z).astype('int32').tofile(f)
    np.array(z2).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)
    return s3, z3


def save_conv_qparams(m, prev_s, prev_z, f):
    s2, z2 = m.conv.get_weight_qparams()
    s3, z3 = m.get_activation_qparams()
    print("S1: {} | Z1: {}".format(prev_s, prev_z))
    print("S2: {} | Z2: {}".format(s2, z2))
    print("S3: {} | Z3: {}".format(s3, z3))
    np.array(prev_s).astype('float32').tofile(f)
    np.array(s2).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(prev_z).astype('int32').tofile(f)
    np.array(z2).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)
    return s3, z3


def save_block_qparams(m, bypass_in_s, bypass_in_z, f):
    # CONV after bypass-connection
    s2, z2 = m.conv1.conv.get_weight_qparams()
    s3, z3 = m.conv1.get_activation_qparams()
    print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
    print("S2: {} | Z2: {}".format(s2, z2))
    print("S3: {} | Z3: {}".format(s3, z3))
    np.array(bypass_in_s).astype('float32').tofile(f)
    np.array(s2).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(bypass_in_z).astype('int32').tofile(f)
    np.array(z2).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)
    prev_s3, prev_z3 = s3, z3

    # 2nd CONV in a block
    s2, z2 = m.conv2.conv.get_weight_qparams()
    s3, z3 = m.conv2.get_activation_qparams()
    print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
    print("S2: {} | Z2: {}".format(s2, z2))
    print("S3: {} | Z3: {}".format(s3, z3))
    np.array(prev_s3).astype('float32').tofile(f)
    np.array(s2).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(prev_z3).astype('int32').tofile(f)
    np.array(z2).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)

    # Darknet's Shortcut layer
    bypass_out_s, bypass_out_z = m.get_activation_qparams()
    print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
    print("S2: {} | Z2: {}".format(s3, z3))
    print("S3: {} | Z3: {}".format(bypass_out_s, bypass_out_z))
    np.array(bypass_in_s).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(bypass_out_s).astype('float32').tofile(f)
    np.array(bypass_in_z).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)
    np.array(bypass_out_z).astype('int32').tofile(f)
    return bypass_out_s, bypass_out_z


def save_block_qparams_with_downsample(m, bypass_in_s, bypass_in_z, f):
    # Downsampling after bypass-connection
    s2, z2 = m.downsample.conv.get_weight_qparams()
    s3, z3 = m.downsample.get_activation_qparams()
    print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
    print("S2: {} | Z2: {}".format(s2, z2))
    print("S3: {} | Z3: {}".format(s3, z3))
    np.array(bypass_in_s).astype('float32').tofile(f)
    np.array(s2).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(bypass_in_z).astype('int32').tofile(f)
    np.array(z2).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)
    prev_s3, prev_z3 = s3, z3  # Will be saved as bypass_in after using current bypass_in at the next layer

    # CONV after bypass-connection
    s2, z2 = m.conv1.conv.get_weight_qparams()
    s3, z3 = m.conv1.get_activation_qparams()
    print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
    print("S2: {} | Z2: {}".format(s2, z2))
    print("S3: {} | Z3: {}".format(s3, z3))
    np.array(bypass_in_s).astype('float32').tofile(f)
    np.array(s2).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(bypass_in_z).astype('int32').tofile(f)
    np.array(z2).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)
    bypass_in_s, bypass_in_z = prev_s3, prev_z3  # Save downsampling layer's s3, z3 for next shortcut
    prev_s3, prev_z3 = s3, z3

    # 2nd CONV in a block
    s2, z2 = m.conv2.conv.get_weight_qparams()
    s3, z3 = m.conv2.get_activation_qparams()
    print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
    print("S2: {} | Z2: {}".format(s2, z2))
    print("S3: {} | Z3: {}".format(s3, z3))
    np.array(prev_s3).astype('float32').tofile(f)
    np.array(s2).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(prev_z3).astype('int32').tofile(f)
    np.array(z2).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)

    # Darknet's Shortcut layer
    bypass_out_s, bypass_out_z = m.get_activation_qparams()
    print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
    print("S2: {} | Z2: {}".format(s3, z3))
    print("S3: {} | Z3: {}".format(bypass_out_s, bypass_out_z))
    np.array(bypass_in_s).astype('float32').tofile(f)
    np.array(s3).astype('float32').tofile(f)
    np.array(bypass_out_s).astype('float32').tofile(f)
    np.array(bypass_in_z).astype('int32').tofile(f)
    np.array(z3).astype('int32').tofile(f)
    np.array(bypass_out_z).astype('int32').tofile(f)
    return bypass_out_s, bypass_out_z


def save_params(model, path):
    with open(path, 'w') as f:
        weight = None
        for name, param in model.named_parameters():
            print(">> Layer: {}\tshape={}".format(name, str(param.data.shape).replace('torch.Size', '')))
            if "bias" in name:
                transform(param.data).tofile(f)
                weight.tofile(f)
            else:
                weight = transform(param.data)


def save_fused_alexnet_qparams(model, path):
    with open(path, 'w') as f:
        # All conv layers before bypass connections use the same S,Z from the last block's activation
        prev_s3, prev_z3 = None, None
        print("input's range: {} , {}".format(model.in_range[0], model.in_range[1]))
        for name, m in model.named_children():
            if 'features' in name:
                input_s, input_z = model.get_input_qparams()
                prev_s3, prev_z3 = save_conv_qparams(m[0], input_s, input_z, f)
                prev_s3, prev_z3 = save_conv_qparams(m[2], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_conv_qparams(m[4], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_conv_qparams(m[5], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_conv_qparams(m[6], prev_s3, prev_z3, f)

            elif 'classifier' in name:
                prev_s3, prev_z3 = save_fc_qparams(m[0], prev_s3, prev_z3, f)
                prev_s3, prev_z3 = save_fc_qparams(m[1], prev_s3, prev_z3, f)
                _, _ = save_fc_qparams(m[2], prev_s3, prev_z3, f)


def save_fused_resnet_qparams(model, path):
    with open(path, 'w') as f:
        bypass_in_s, bypass_in_z = None, None
        print("input's range: {} , {}".format(model.in_range[0], model.in_range[1]))
        for name, m in model.named_children():
            if 'first_conv' in name:
                input_s, input_z = model.get_input_qparams()
                bypass_in_s, bypass_in_z = save_conv_qparams(m, input_s, input_z, f)

            elif "layer" in name:
                if "layer1" in name:
                    bypass_in_s, bypass_in_z = save_block_qparams(m[0], bypass_in_s, bypass_in_z, f)
                else:
                    bypass_in_s, bypass_in_z = save_block_qparams_with_downsample(m[0], bypass_in_s, bypass_in_z, f)
                for i in range(1, len(m)):
                    bypass_in_s, bypass_in_z = save_block_qparams(m[i], bypass_in_s, bypass_in_z, f)

            elif 'fc' in name:
                _, _ = save_fc_qparams(m, bypass_in_s, bypass_in_z, f)


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
