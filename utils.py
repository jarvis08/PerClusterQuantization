import torch
import numpy as np


def transform(param):
    return param.flatten().numpy().astype('float32')


def save_fused_network(model, path):
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


def save_fused_resnet18_qparams(model, path):
    with open(path, 'w') as f:
        # All conv layers before bypass connections use the same S,Z from the last block's activation
        bypass_s, bypass_z = model.layer4[1].get_activation_qparams()
        print("input's range: {} , {}".format(model.in_range[0], model.in_range[1]))
        for name, m in model.named_children():
            if 'first_conv' in name:
                s1, z1 = model.get_input_qparams()
                s2, z2 = m.conv.get_weight_qparams()
                print("S1: {} | Z1: {}".format(s1, z1))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
                np.array(s1).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(z1).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)

            elif "layer1" in name:
                s2, z2 = m[0].conv1.conv.get_weight_qparams()
                s3, z3 = m[0].conv1.get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S2: {} | Z2: {}".format(s3, z3))
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)

                s2, z2 = m[0].conv2.conv.get_weight_qparams()
                print("S1: {} | Z1: {}".format(s3, z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
                np.array(s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)

                s2, z2 = m[1].conv1.conv.get_weight_qparams()
                s3, z3 = m[1].conv1.get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S2: {} | Z2: {}".format(s3, z3))
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)

                s2, z2 = m[1].conv2.conv.get_weight_qparams()
                print("S1: {} | Z1: {}".format(s3, z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
                np.array(s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)

            elif "layer" in name:
                s2, z2 = m[0].downsample.conv.get_weight_qparams()
                print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)

                s2, z2 = m[0].conv1.conv.get_weight_qparams()
                s3, z3 = m[0].conv1.get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S2: {} | Z2: {}".format(s3, z3))
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)

                s2, z2 = m[0].conv2.conv.get_weight_qparams()
                print("S1: {} | Z1: {}".format(s3, z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
                np.array(s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)

                s2, z2 = m[1].conv1.conv.get_weight_qparams()
                s3, z3 = m[1].conv1.get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S2: {} | Z2: {}".format(s3, z3))
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(z3).astype('int32').tofile(f)

                s2, z2 = m[1].conv2.conv.get_weight_qparams()
                print("S1: {} | Z1: {}".format(s3, z3))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S3: {} | Z3: {}".format(bypass_s, bypass_z))
                np.array(s3).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(z3).astype('int32').tofile(f)
                np.array(z2).astype('int32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)

            elif 'fc' in name:
                s2, z2 = m.get_weight_qparams()
                s3, z3 = m.get_activation_qparams()
                print("S1: {} | Z1: {}".format(bypass_s, bypass_z))
                print("S2: {} | Z2: {}".format(s2, z2))
                print("S2: {} | Z2: {}".format(s3, z3))
                np.array(bypass_s).astype('float32').tofile(f)
                np.array(s2).astype('float32').tofile(f)
                np.array(s3).astype('float32').tofile(f)
                np.array(bypass_z).astype('int32').tofile(f)
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
    path = "./result/darknet/test-{}.fused.torch.".format(arch)
    if arch == "resnet18":
        save_fused_resnet18_weight(model, path + 'weights')
        save_fused_resnet18_qparams(model, path + 'qparams')
    # elif arch == "densenet121":
    #     save_fused_densnet(model, path)
    else:
        save_fused_network(model, path + 'weights')
        # save_qparams(model, path + 'weights')


def load_preprocessed_cifar10_from_darknet():
    input = torch.tensor(np.fromfile("result/darknet/cifar_test_dataset.bin", dtype='float32').reshape((10000, 1, 3, 32, 32)))
    target = torch.tensor(np.fromfile("result/darknet/cifar_test_target.bin", dtype='int32').reshape((10000, 1)), dtype=torch.long)
    return input, target
