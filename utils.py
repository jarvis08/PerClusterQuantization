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


def save_fused_resnet_qparams(model, path):
    #with open(path, 'w') as f:
    #    rst = []
    #    rst.append([get_qprams(model.conv1)])
    #    s = np.zeros((3, 3), dtype=np.float32)
    #    z = np.zeros((3, 3), dtype=np.float32)
    #    s[0] = model.conv1.a_scale
    #    z[0] = model.conv1.a_zero
    #    model.layer1[0].conv1, model.layer1[0].bn1 = _fuse(model.layer1[0].conv1, model.layer1[0].bn1)
    #    model.layer1[0].conv2, model.layer1[0].bn2 = _fuse(model.layer1[0].conv2, model.layer1[0].bn2)
    #    model.layer1[1].conv1, model.layer1[1].bn1 = _fuse(model.layer1[1].conv1, model.layer1[1].bn1)
    #    model.layer1[1].conv2, model.layer1[1].bn2 = _fuse(model.layer1[1].conv2, model.layer1[1].bn2)
    #    model.layer2[0].conv1, model.layer2[0].bn1 = _fuse(model.layer2[0].conv1, model.layer2[0].bn1)
    #    model.layer2[0].conv2, model.layer2[0].bn2 = _fuse(model.layer2[0].conv2, model.layer2[0].bn2)
    #    model.layer2[0].downsample[0], model.layer2[0].downsample[1] = _fuse(model.layer2[0].downsample[0],
    #                                                                         model.layer2[0].downsample[1])
    #    model.layer2[1].conv1, model.layer2[1].bn1 = _fuse(model.layer2[1].conv1, model.layer2[1].bn1)
    #    model.layer2[1].conv2, model.layer2[1].bn2 = _fuse(model.layer2[1].conv2, model.layer2[1].bn2)
    #    model.layer3[0].conv1, model.layer3[0].bn1 = _fuse(model.layer3[0].conv1, model.layer3[0].bn1)
    #    model.layer3[0].conv2, model.layer3[0].bn2 = _fuse(model.layer3[0].conv2, model.layer3[0].bn2)
    #    model.layer3[0].downsample[0], model.layer3[0].downsample[1] = _fuse(model.layer3[0].downsample[0],
    #                                                                         model.layer3[0].downsample[1])
    #    model.layer3[1].conv1, model.layer3[1].bn1 = _fuse(model.layer3[1].conv1, model.layer3[1].bn1)
    #    model.layer3[1].conv2, model.layer3[1].bn2 = _fuse(model.layer3[1].conv2, model.layer3[1].bn2)
    #    model.layer4[0].conv1, model.layer4[0].bn1 = _fuse(model.layer4[0].conv1, model.layer4[0].bn1)
    #    model.layer4[0].conv2, model.layer4[0].bn2 = _fuse(model.layer4[0].conv2, model.layer4[0].bn2)
    #    model.layer4[0].downsample[0], model.layer4[0].downsample[1] = _fuse(model.layer4[0].downsample[0],
    #                                                                         model.layer4[0].downsample[1])
    #    model.layer4[1].conv1, model.layer4[1].bn1 = _fuse(model.layer4[1].conv1, model.layer4[1].bn1)
    #    model.layer4[1].conv2, model.layer4[1].bn2 = _fuse(model.layer4[1].conv2, model.layer4[1].bn2)
    pass


def save_fused_resnet_weight(model, path):
    """
        Darknet calculates downsampling before block start, not after like torch.
        Therefore, by using queue, move downsampling layer to starting point of block.
    """
    with open(path, 'w') as f:
        weight = None
        block = []
        for name, param in model.named_parameters():
            print(">> Layer: {}\tshape={}".format(name, str(param.data.shape).replace('torch.Size', '')))

            if "layer1" in name:
                if "bias" in name:
                    transform(param.data).tofile(f)
                    weight.tofile(f)
                else:
                    weight = transform(param.data)

            # Layers with downsampling, modify sequence of saving params file for the reason of docstring[2]
            elif "layer" in name and '.0.' in name:
                if "downsample" in name:
                    if "bias" in name:
                        transform(param.data).tofile(f)
                        weight.tofile(f)

                        # Save rest of the stacked block & initialize stack for next the block
                        for b in block:
                            b.tofile(f)
                        block = []
                    else:
                        weight = transform(param.data)
                else:
                    if "bias" in name:
                        block.append(transform(param.data))
                        block.append(weight)
                    else:
                        weight = transform(param.data)

            else:
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
        save_fused_resnet_weight(model, path + 'weights')
        save_fused_resnet_qparams(model, path + 'weights')
    # elif arch == "densenet121":
    #     save_fused_densnet(model, path)
    else:
        save_fused_network(model, path + 'weights')
        # save_qparams(model, path + 'weights')


def load_preprocessed_cifar10_from_darknet():
    input = torch.tensor(np.fromfile("result/darknet/cifar_test_dataset.bin", dtype='float32').reshape((10000, 1, 3, 32, 32)))
    target = torch.tensor(np.fromfile("result/darknet/cifar_test_target.bin", dtype='int32').reshape((10000, 1)), dtype=torch.long)
    return input, target
