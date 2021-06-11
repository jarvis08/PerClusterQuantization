import torch.nn as nn
import numpy as np


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


def transform(param):
    return param.flatten().numpy().astype('float32')


def save_network(model, path):
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
                s1, z1 = model.get_input_qparams()
                s2, z2 = m[0].conv.get_weight_qparams()
                s3, z3 = m[0].get_activation_qparams()
                print("S1: {} | Z1: {}".format(s1, z1))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(s3, z3))
                np.array(s1).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(z1).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                prev_s3, prev_z3 = s3, z3

                s2, z2 = m[2].conv.get_weight_qparams()
                s3, z3 = m[2].get_activation_qparams()
                print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(s3, z3))
                np.array(prev_s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(prev_z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                prev_s3, prev_z3 = s3, z3

                s2, z2 = m[4].conv.get_weight_qparams()
                s3, z3 = m[4].get_activation_qparams()
                print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(s3, z3))
                np.array(prev_s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(prev_z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                prev_s3, prev_z3 = s3, z3

                s2, z2 = m[5].conv.get_weight_qparams()
                s3, z3 = m[5].get_activation_qparams()
                print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(s3, z3))
                np.array(prev_s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(prev_z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                prev_s3, prev_z3 = s3, z3

                s2, z2 = m[6].conv.get_weight_qparams()
                s3, z3 = m[6].get_activation_qparams()
                print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(s3, z3))
                np.array(prev_s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(prev_z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                prev_s3, prev_z3 = s3, z3

            elif 'classifier' in name:
                s2, z2 = m[0].fc.get_weight_qparams()
                s3, z3 = m[0].get_activation_qparams()
                print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S2: {} | Z2: {}".format(s3, z3))
                np.array(prev_s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(prev_z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                prev_s3, prev_z3 = s3, z3

                s2, z2 = m[1].fc.get_weight_qparams()
                s3, z3 = m[1].get_activation_qparams()
                print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S2: {} | Z2: {}".format(s3, z3))
                np.array(prev_s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(prev_z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                prev_s3, prev_z3 = s3, z3

                s2, z2 = m[2].fc.get_weight_qparams()
                s3, z3 = m[2].get_activation_qparams()
                print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S2: {} | Z2: {}".format(s3, z3))
                np.array(prev_s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(prev_z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)


def save_fused_resnet18_qparams(model, path):
    with open(path, 'w') as f:
        # All conv layers before bypass connections use the same S,Z from the last block's activation
        bypass_in_s, bypass_in_z = None, None
        # prev_s3, prev_z3 = None, None
        # bypass_s, bypass_z = None, None
        # bypass_s, bypass_z = model.layer1[0].get_activation_qparams()  # used in integrated S, Z
        print("input's range: {} , {}".format(model.in_range[0], model.in_range[1]))
        for name, m in model.named_children():
            if 'first_conv' in name:
                s1, z1 = model.get_input_qparams()
                s2, z2 = m.conv.get_weight_qparams()
                s3, z3 = m.get_activation_qparams()
                print("S1: {} | Z1: {}".format(s1, z1))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(s3, z3))
                np.array(s1).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(z1).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                bypass_in_s, bypass_in_z = s3, z3

            elif "layer1" in name:
                s2, z2 = m[0].conv1.conv.get_weight_qparams()
                s3, z3 = m[0].conv1.get_activation_qparams()
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

                s2, z2 = m[0].conv2.conv.get_weight_qparams()
                s3, z3 = m[0].conv2.get_activation_qparams()
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
                bypass_out_s, bypass_out_z = m[0].get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
                print("S2: {} | Z2: {}".format(s3, z3))
                print("S3: {} | Z3: {}".format(bypass_out_s, bypass_out_z))
                np.array(bypass_in_s).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_out_s).astype('float32').tofile(f)
                np.array(bypass_in_z).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                np.array(bypass_out_z).astype('int32').tofile(f)
                bypass_in_s, bypass_in_z = bypass_out_s, bypass_out_z

                # CONV after bypass-connection
                s2, z2 = m[1].conv1.conv.get_weight_qparams()
                s3, z3 = m[1].conv1.get_activation_qparams()
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

                s2, z2 = m[1].conv2.conv.get_weight_qparams()
                s3, z3 = m[1].conv2.get_activation_qparams()
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
                bypass_out_s, bypass_out_z = m[1].get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
                print("S2: {} | Z2: {}".format(s3, z3))
                print("S3: {} | Z3: {}".format(bypass_out_s, bypass_out_z))
                np.array(bypass_in_s).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_out_s).astype('float32').tofile(f)
                np.array(bypass_in_z).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                np.array(bypass_out_z).astype('int32').tofile(f)
                bypass_in_s, bypass_in_z = bypass_out_s, bypass_out_z

            elif "layer" in name:
                # Downsampling after bypass-connection
                s2, z2 = m[0].downsample.conv.get_weight_qparams()
                s3, z3 = m[0].downsample.get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(s3, z3))
                np.array(bypass_in_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_in_z).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                prev_s3, prev_z3 = s3, z3                # Will be saved as bypass_in after using current bypass_in at the next layer

                # CONV after bypass-connection
                s2, z2 = m[0].conv1.conv.get_weight_qparams()
                s3, z3 = m[0].conv1.get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(s3, z3))
                np.array(bypass_in_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_in_z).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                bypass_in_s, bypass_in_z = prev_s3, prev_z3   # Save downsampling layer's s3, z3 for next shortcut
                prev_s3, prev_z3 = s3, z3

                s2, z2 = m[0].conv2.conv.get_weight_qparams()
                s3, z3 = m[0].conv2.get_activation_qparams()
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
                bypass_out_s, bypass_out_z = m[0].get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
                print("S2: {} | Z2: {}".format(s3, z3))
                print("S3: {} | Z3: {}".format(bypass_out_s, bypass_out_z))
                np.array(bypass_in_s).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_out_s).astype('float32').tofile(f)
                np.array(bypass_in_z).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                np.array(bypass_out_z).astype('int32').tofile(f)
                bypass_in_s, bypass_in_z = bypass_out_s, bypass_out_z

                # CONV after bypass-connection
                s2, z2 = m[1].conv1.conv.get_weight_qparams()
                s3, z3 = m[1].conv1.get_activation_qparams()
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

                s2, z2 = m[1].conv2.conv.get_weight_qparams()
                s3, z3 = m[1].conv2.get_activation_qparams()
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
                bypass_out_s, bypass_out_z = m[1].get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
                print("S2: {} | Z2: {}".format(s3, z3))
                print("S3: {} | Z3: {}".format(bypass_out_s, bypass_out_z))
                np.array(bypass_in_s).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_out_s).astype('float32').tofile(f)
                np.array(bypass_in_z).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                np.array(bypass_out_z).astype('int32').tofile(f)
                bypass_in_s, bypass_in_z = bypass_out_s, bypass_out_z

            #### Before using paper's formula in bypass
            # if 'first_conv' in name:
            #     s1, z1 = model.get_input_qparams()
            #     s2, z2 = m.conv.get_weight_qparams()
            #     s3, z3 = m.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(s1, z1))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(s1).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(z1).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     prev_s3, prev_z3 = s3, z3
            #
            # elif "layer1" in name:
            #     s2, z2 = m[0].conv1.conv.get_weight_qparams()
            #     s3, z3 = m[0].conv1.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S2: {} | Z2: {}".format(s3, z3))
            #     np.array(prev_s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(prev_z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     prev_s3, prev_z3 = s3, z3
            #
            #     s2, z2 = m[0].conv2.conv.get_weight_qparams()
            #     s3, z3 = m[0].conv2.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(prev_s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(prev_z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #
            #     # CONV after bypass-connection
            #     bypass_s, bypass_z = m[0].get_activation_qparams()
            #     s2, z2 = m[1].conv1.conv.get_weight_qparams()
            #     s3, z3 = m[1].conv1.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     prev_s3, prev_z3 = s3, z3
            #
            #     s2, z2 = m[1].conv2.conv.get_weight_qparams()
            #     s3, z3 = m[1].conv2.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(prev_s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(prev_z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     bypass_s, bypass_z = m[1].get_activation_qparams()
            #
            # elif "layer" in name:
            #     # CONV after bypass-connection
            #     s2, z2 = m[0].downsample.conv.get_weight_qparams()
            #     s3, z3 = m[0].downsample.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #
            #     # CONV after bypass-connection
            #     s2, z2 = m[0].conv1.conv.get_weight_qparams()
            #     s3, z3 = m[0].conv1.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     prev_s3, prev_z3 = s3, z3
            #
            #     s2, z2 = m[0].conv2.conv.get_weight_qparams()
            #     s3, z3 = m[0].conv2.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(prev_s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(prev_z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #
            #     # CONV after bypass-connection
            #     bypass_s, bypass_z = m[0].get_activation_qparams()
            #     s2, z2 = m[1].conv1.conv.get_weight_qparams()
            #     s3, z3 = m[1].conv1.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     prev_s3, prev_z3 = s3, z3
            #
            #     s2, z2 = m[1].conv2.conv.get_weight_qparams()
            #     s3, z3 = m[1].conv2.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(prev_s3, prev_z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(s3, z3))
            #     np.array(prev_s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(prev_z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     bypass_s, bypass_z = m[1].get_activation_qparams()


            #### Integrated S, Z in bypass
            # if 'first_conv' in name:
            #     s1, z1 = model.get_input_qparams()
            #     s2, z2 = m.conv.get_weight_qparams()
            #     print("S1: {} | Z1: {}".format(s1, z1))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
            #     np.array(s1).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(z1).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #
            # elif "layer1" in name:
            #     s2, z2 = m[0].conv1.conv.get_weight_qparams()
            #     s3, z3 = m[0].conv1.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S2: {} | Z2: {}".format(s3, z3))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #
            #     s2, z2 = m[0].conv2.conv.get_weight_qparams()
            #     print("S1: {} | Z1: {}".format(s3, z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #
            #     s2, z2 = m[1].conv1.conv.get_weight_qparams()
            #     s3, z3 = m[1].conv1.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S2: {} | Z2: {}".format(s3, z3))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #
            #     s2, z2 = m[1].conv2.conv.get_weight_qparams()
            #     print("S1: {} | Z1: {}".format(s3, z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #
            # elif "layer" in name:
            #     s2, z2 = m[0].downsample.conv.get_weight_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #
            #     s2, z2 = m[0].conv1.conv.get_weight_qparams()
            #     s3, z3 = m[0].conv1.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S2: {} | Z2: {}".format(s3, z3))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #
            #     s2, z2 = m[0].conv2.conv.get_weight_qparams()
            #     print("S1: {} | Z1: {}".format(s3, z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #
            #     s2, z2 = m[1].conv1.conv.get_weight_qparams()
            #     s3, z3 = m[1].conv1.get_activation_qparams()
            #     print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S2: {} | Z2: {}".format(s3, z3))
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #
            #     s2, z2 = m[1].conv2.conv.get_weight_qparams()
            #     print("S1: {} | Z1: {}".format(s3, z3))
            #     print("S2: {} | Z2: {}".format(s2, z2))
            #     print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
            #     np.array(s3).astype('float32').tofile(f)
            #     np.array(s2).astype('float32').tofile(f)
            #     np.array(bypass_s).astype('float32').tofile(f)
            #     np.array(z3).astype('int32').tofile(f)
            #     np.array(z2).astype('int32').tofile(f)
            #     np.array(bypass_z).astype('int32').tofile(f)

            elif 'fc' in name:
                # FC after bypass-connection
                s2, z2 = m.fc.get_weight_qparams()
                s3, z3 = m.get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_in_s, bypass_in_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z2: {}".format(s3, z3))
                np.array(bypass_in_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_in_z).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)


def save_fused_resnet18_weight(model, path):
    """
        Darknet calculates downsampling before block start, not after like torch.
        Therefore, by using queue, move downsampling layer to starting point of block.
    """
    with open(path, 'w') as f:
        weight = None
        for name, param in model.named_parameters():
            print(">> Layer: {}\tshape={}".format(name, str(param.data.shape).replace('torch.Size', '')))
            if "bias" in name:
                transform(param.data).tofile(f)
                weight.tofile(f)
            else:
                weight = transform(param.data)


def save_fused_network_in_darknet_form(model, arch):
    """
        1. PyTroch and Darknet both use Row-major format as default, but use Col-major for FC weights.
        2. Darknet save/load params in bias/weight sequence, so save in the pattern
    """
    model.cpu()
    path = "./result/darknet/{}.fused.torch.".format(arch)
    if arch == "resnet18":
        save_fused_resnet18_weight(model, path + 'weights')
        save_fused_resnet18_qparams(model, path + 'qparams')
    elif arch == "alexnet":
        save_network(model, path + 'weights')
        save_fused_alexnet_qparams(model, path + 'qparams')
