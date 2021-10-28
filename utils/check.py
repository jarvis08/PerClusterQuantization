def pcq_alexnet_trained_activation_ranges(model):
    ranges = [[] for _ in range(4)]
    for c in range(4):
        ranges[c].append([model.conv1.act_range[c][0].item(), model.conv1.act_range[c][1].item()])
        ranges[c].append([model.conv2.act_range[c][0].item(), model.conv2.act_range[c][1].item()])
        ranges[c].append([model.conv3.act_range[c][0].item(), model.conv3.act_range[c][1].item()])
        ranges[c].append([model.conv4.act_range[c][0].item(), model.conv4.act_range[c][1].item()])
        ranges[c].append([model.conv5.act_range[c][0].item(), model.conv5.act_range[c][1].item()])
        ranges[c].append([model.fc1.act_range[c][0].item(), model.fc1.act_range[c][1].item()])
        ranges[c].append([model.fc2.act_range[c][0].item(), model.fc2.act_range[c][1].item()])
        ranges[c].append([model.fc3.act_range[c][0].item(), model.fc3.act_range[c][1].item()])
    names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']

    import csv
    for c in range(4):
        with open('pcq_alexnet_trained_activation_ranges_c{}.csv'.format(c + 1), 'w', newline='') as csvfile:
            fieldnames = ['name', 'min', 'max']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for n, r in zip(names, ranges[c]):
                writer.writerow({'name': n, 'min': r[0], 'max': r[1]})


def qat_alexnet_trained_activation_ranges(model):
    ranges = []
    ranges.append([model.conv1.act_range[0].item(), model.conv1.act_range[1].item()])
    ranges.append([model.conv2.act_range[0].item(), model.conv2.act_range[1].item()])
    ranges.append([model.conv3.act_range[0].item(), model.conv3.act_range[1].item()])
    ranges.append([model.conv4.act_range[0].item(), model.conv4.act_range[1].item()])
    ranges.append([model.conv5.act_range[0].item(), model.conv5.act_range[1].item()])
    ranges.append([model.fc1.act_range[0].item(), model.fc1.act_range[1].item()])
    ranges.append([model.fc2.act_range[0].item(), model.fc2.act_range[1].item()])
    ranges.append([model.fc3.act_range[0].item(), model.fc3.act_range[1].item()])
    names = ['input', 'conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'fc3']
    for n, r in zip(names, ranges):
        print("{}\t{}".format(n, r))

    import csv
    with open('qat_alexnet_trained_activation_ranges.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'min', 'max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for n, r in zip(names, ranges):
            writer.writerow({'name': n, 'min': r[0], 'max': r[1]})


def qat_resnet20_trained_activation_ranges(model):
    ranges = []
    ranges.append([model.bn1.act_range[0].item(), model.bn1.act_range[1].item()])
    for i in range(len(model.layer1)):
        ranges.append([model.layer1[i].bn1.act_range[0].item(), model.layer1[i].bn1.act_range[1].item()])
    for i in range(len(model.layer2)):
        ranges.append([model.layer2[i].bn1.act_range[0].item(), model.layer2[i].bn1.act_range[1].item()])
    for i in range(len(model.layer3)):
        ranges.append([model.layer3[i].bn1.act_range[0].item(), model.layer3[i].bn1.act_range[1].item()])
    names = ['first_bn', 'layer1_bn1', 'layer1_bn2', 'layer2_bn1', 'layer2_bn2', 'layer3_bn1', 'layer3_bn2']
    for n, r in zip(names, ranges):
        print("{}\t{}".format(n, r))

    import csv
    with open('qat_resnet_trained_activation_ranges.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'min', 'max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for n, r in zip(names, ranges):
            writer.writerow({'name': n, 'min': r[0], 'max': r[1]})


def qat_resnet50_trained_activation_ranges(model):
    ranges = []
    ranges.append([model.bn1.act_range[0].item(), model.bn1.act_range[1].item()])
    for i in range(len(model.layer1)):
        ranges.append([model.layer1[i].bn1.act_range[0].item(), model.layer1[i].bn1.act_range[1].item()])
        ranges.append([model.layer1[i].bn2.act_range[0].item(), model.layer1[i].bn2.act_range[1].item()])
    for i in range(len(model.layer2)):
        ranges.append([model.layer2[i].bn1.act_range[0].item(), model.layer2[i].bn1.act_range[1].item()])
        ranges.append([model.layer2[i].bn2.act_range[0].item(), model.layer2[i].bn2.act_range[1].item()])
    for i in range(len(model.layer3)):
        ranges.append([model.layer3[i].bn1.act_range[0].item(), model.layer3[i].bn1.act_range[1].item()])
        ranges.append([model.layer3[i].bn2.act_range[0].item(), model.layer3[i].bn2.act_range[1].item()])
    for i in range(len(model.layer4)):
        ranges.append([model.layer3[i].bn1.act_range[0].item(), model.layer3[i].bn1.act_range[1].item()])
        ranges.append([model.layer3[i].bn2.act_range[0].item(), model.layer3[i].bn2.act_range[1].item()])

    names = ['First_BN']
    num_blocks = 3 + 4 + 6 + 3
    for block in range(1, num_blocks + 1):
        names.append('Block{}_BN1'.format(block))
        names.append('Block{}_BN2'.format(block))

    for n, r in zip(names, ranges):
        print("{}\t{}".format(n, r))

    import csv
    with open('qat_resnet50_trained_activation_ranges.csv', 'w', newline='') as csvfile:
        fieldnames = ['name', 'min', 'max']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for n, r in zip(names, ranges):
            writer.writerow({'name': n, 'min': r[0], 'max': r[1]})


def pcq_resnet50_trained_activation_ranges(model):
    ranges = [[] for _ in range(4)]
    for c in range(4):
        ranges[c].append([model.bn1.act_range[c][0].item(), model.bn1.act_range[c][1].item()])
        for i in range(len(model.layer1)):
            ranges[c].append([model.layer1[i].bn1.act_range[c][0].item(), model.layer1[i].bn1.act_range[c][1].item()])
            ranges[c].append([model.layer1[i].bn2.act_range[c][0].item(), model.layer1[i].bn2.act_range[c][1].item()])
        for i in range(len(model.layer2)):
            ranges[c].append([model.layer2[i].bn1.act_range[c][0].item(), model.layer2[i].bn1.act_range[c][1].item()])
            ranges[c].append([model.layer2[i].bn2.act_range[c][0].item(), model.layer2[i].bn2.act_range[c][1].item()])
        for i in range(len(model.layer3)):
            ranges[c].append([model.layer3[i].bn1.act_range[c][0].item(), model.layer3[i].bn1.act_range[c][1].item()])
            ranges[c].append([model.layer3[i].bn2.act_range[c][0].item(), model.layer3[i].bn2.act_range[c][1].item()])
        for i in range(len(model.layer4)):
            ranges[c].append([model.layer3[i].bn1.act_range[c][0].item(), model.layer3[i].bn1.act_range[c][1].item()])
            ranges[c].append([model.layer3[i].bn2.act_range[c][0].item(), model.layer3[i].bn2.act_range[c][1].item()])

    names = ['First_BN']
    num_blocks = 3 + 4 + 6 + 3
    for block in range(1, num_blocks + 1):
        names.append('Block{}_BN1'.format(block))
        names.append('Block{}_BN2'.format(block))

    import csv
    for c in range(4):
        with open('pcq_resnet50_trained_activation_ranges_c{}.csv'.format(c + 1), 'w', newline='') as csvfile:
            fieldnames = ['name', 'min', 'max']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for n, r in zip(names, ranges[c]):
                writer.writerow({'name': n, 'min': r[0], 'max': r[1]})
