import os
import sys

import torch, math
import pickle
import numpy as np
from sklearn.utils import shuffle

import threading
from torch.multiprocessing import Event
from torch._six import queue

try:
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator
    from nvidia.dali.pipeline import Pipeline
    import nvidia.dali.ops as ops
    import nvidia.dali.types as types
except ImportError:
    raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")


class HybridTrainPipe(Pipeline):
    """
    DALI Train Pipeline
    Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
    In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
    This dataloader implements ImageNet style training preprocessing, namely:
    -random resized crop
    -random horizontal flip
    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
    containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    local_rank (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
    min_crop_size (float, optional, default = 0.08) - Minimum random crop size
    """

    def __init__(self, batch_size, num_threads, device_id, data_dir, crop,
                 mean, std, local_rank=0, world_size=1, shuffle=True, min_crop_size=0.08):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(HybridTrainPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)

        # Enabling read_ahead slowed down processing ~40%
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size,
                                    random_shuffle=shuffle)

        # Let user decide which pipeline works best with the chosen model
        decode_device = "mixed"
        self.dali_device = "gpu"

        output_dtype = types.FLOAT

        self.cmn = ops.CropMirrorNormalize(device="gpu",
                                           output_dtype=output_dtype,
                                           output_layout=types.NCHW,
                                           crop=(crop, crop),
                                           image_type=types.RGB,
                                           mean=mean,
                                           std=std,)

        # To be able to handle all images from full-sized ImageNet, this padding sets the size of the internal
        # nvJPEG buffers without additional reallocations
        device_memory_padding = 211025920 if decode_device == 'mixed' else 0
        host_memory_padding = 140544512 if decode_device == 'mixed' else 0
        self.decode = ops.ImageDecoderRandomCrop(device=decode_device, output_type=types.RGB,
                                                 device_memory_padding=device_memory_padding,
                                                 host_memory_padding=host_memory_padding,
                                                 # random_aspect_ratio=[0.8, 1.25],
                                                 random_aspect_ratio=[0.8, 1.0],
                                                 random_area=[min_crop_size, 1.0],
                                                 num_attempts=100)

        # Resize as desired.  To match torchvision data loader, use triangular interpolation.
        self.res = ops.Resize(device=self.dali_device, resize_x=crop, resize_y=crop,
                              interp_type=types.INTERP_TRIANGULAR)

        self.coin = ops.CoinFlip(probability=0.5)
        print('DALI "{0}" variant'.format(self.dali_device))

    def define_graph(self):
        rng = self.coin()
        self.jpegs, self.labels = self.input(name="Reader")

        # Combined decode & random crop
        images = self.decode(self.jpegs)

        # Resize as desired
        images = self.res(images)

        output = self.cmn(images, mirror=rng)

        self.labels = self.labels.gpu()
        return [output, self.labels]


class HybridValPipe(Pipeline):
    """
    DALI Validation Pipeline
    Based on the official example: https://github.com/NVIDIA/DALI/blob/master/docs/examples/pytorch/resnet50/main.py
    In comparison to the example, the CPU backend does more computation on CPU, reducing GPU load & memory use.
    This dataloader implements ImageNet style validation preprocessing, namely:
    -resize to specified size
    -center crop to desired size
    batch_size (int): how many samples per batch to load
    num_threads (int): how many DALI workers to use for data loading.
    device_id (int): GPU device ID
    data_dir (str): Directory to dataset.  Format should be the same as torchvision dataloader,
        containing train & val subdirectories, with image class subfolders
    crop (int): Image output size (typically 224 for ImageNet)
    size (int): Resize size (typically 256 for ImageNet)
    mean (tuple): Image mean value for each channel
    std (tuple): Image standard deviation value for each channel
    local_rank (int, optional, default = 0) – Id of the part to read
    world_size (int, optional, default = 1) - Partition the data into this many parts (used for multiGPU training)
    dali_cpu (bool, optional, default = False) - Use DALI CPU mode instead of GPU
    shuffle (bool, optional, default = True) - Shuffle the dataset each epoch
    fp16 (bool, optional, default = False) - Output the data in fp16 instead of fp32 (GPU mode only)
    """

    def __init__(self, batch_size, num_threads, device_id, data_dir, crop, size,
                 mean, std, local_rank=0, world_size=1, shuffle=False):

        # As we're recreating the Pipeline at every epoch, the seed must be -1 (random seed)
        super(HybridValPipe, self).__init__(batch_size, num_threads, device_id, seed=-1)

        # Enabling read_ahead slowed down processing ~40%
        # Note: initial_fill is for the shuffle buffer.  As we only want to see every example once, this is set to 1
        self.input = ops.FileReader(file_root=data_dir, shard_id=local_rank, num_shards=world_size, random_shuffle=shuffle, initial_fill=1)
        decode_device = "mixed"
        self.dali_device = "gpu"

        output_dtype = types.FLOAT

        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=output_dtype,
                                            output_layout=types.NCHW,
                                            crop=(crop, crop),
                                            image_type=types.RGB,
                                            mean=mean,
                                            std=std)

        self.decode = ops.ImageDecoder(device=decode_device, output_type=types.RGB)

        # Resize to desired size.  To match torchvision dataloader, use triangular interpolation
        self.res = ops.Resize(device=self.dali_device, resize_shorter=size, interp_type=types.INTERP_TRIANGULAR)

    def define_graph(self):
        self.jpegs, self.labels = self.input(name="Reader")
        images = self.decode(self.jpegs)
        images = self.res(images)
        output = self.cmnp(images)

        self.labels = self.labels.gpu()
        return [output, self.labels]


class DaliIterator():
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __init__(self, pipelines, size, **kwargs):
        self._dali_iterator = DALIClassificationIterator(pipelines=pipelines, size=size)

    def __iter__(self):
        return self

    def __len__(self):
        return int(math.ceil(self._dali_iterator._size / self._dali_iterator.batch_size))


class DaliIteratorGPU(DaliIterator):
    """
    Wrapper class to decode the DALI iterator output & provide iterator that functions the same as torchvision
    pipelines (Pipeline): DALI pipelines
    size (int): Number of examples in set
    Note: allow extra inputs to keep compatibility with CPU iterator
    """

    def __next__(self):
        try:
            data = next(self._dali_iterator)
        except StopIteration:
            print('Resetting DALI loader')
            self._dali_iterator.reset()
            raise StopIteration

        # Decode the data output
        input = data[0]['data']
        target = data[0]['label'].squeeze().long()

        return input, target


def _preproc_worker(dali_iterator, cuda_stream, fp16, mean, std, output_queue, proc_next_input, done_event, pin_memory):
    """
    Worker function to parse DALI output & apply final pre-processing steps
    """

    while not done_event.is_set():
        # Wait until main thread signals to proc_next_input -- normally once it has taken the last processed input
        proc_next_input.wait()
        proc_next_input.clear()

        if done_event.is_set():
            print('Shutting down preproc thread')
            break

        try:
            data = next(dali_iterator)

            # Decode the data output
            input_orig = data[0]['data']
            target = data[0]['label'].squeeze().long()  # DALI should already output target on device

            # Copy to GPU and apply final processing in separate CUDA stream
            with torch.cuda.stream(cuda_stream):
                input = input_orig
                if pin_memory:
                    input = input.pin_memory()
                    del input_orig  # Save memory
                input = input.cuda(non_blocking=True)

                input = input.permute(0, 3, 1, 2)

                # Input tensor is kept as 8-bit integer for transfer to GPU, to save bandwidth
                if fp16:
                    input = input.half()
                else:
                    input = input.float()

                input = input.sub_(mean).div_(std)

            # Put the result on the queue
            output_queue.put((input, target))

        except StopIteration:
            print('Resetting DALI loader')
            dali_iterator.reset()
            output_queue.put(None)


class HybridTrainPipeCifar10(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data_dir, crop=32, local_rank=0, world_size=1, cutout=0):
        super(HybridTrainPipeCifar10, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.iterator = iter(Cifar10InputIter(batch_size, 'train', root=data_dir))
        dali_device = "gpu"
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.pad = ops.Paste(device=dali_device, ratio=1.25, fill_value=0)
        self.uniform = ops.Uniform(range=(0., 1.))
        self.crop = ops.Crop(device=dali_device, crop_h=crop, crop_w=crop)
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
                                            )
        self.coin = ops.CoinFlip(probability=0.5)

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")
        self.feed_input(self.labels, labels)

    def define_graph(self):
        rng = self.coin()
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.pad(output.gpu())
        output = self.crop(output, crop_pos_x=self.uniform(), crop_pos_y=self.uniform())
        output = self.cmnp(output, mirror=rng)
        return [output, self.labels]


class HybridTestPipeCifar10(Pipeline):
    def __init__(self, mode, batch_size, num_threads, device_id, data_dir, local_rank=0, world_size=1, phase2_idx=None):
        super(HybridTestPipeCifar10, self).__init__(batch_size, num_threads, device_id, seed=12 + device_id)
        self.iterator = iter(Cifar10InputIter(batch_size, mode, root=data_dir, phase2_idx=phase2_idx))
        self.input = ops.ExternalSource()
        self.input_label = ops.ExternalSource()
        self.cmnp = ops.CropMirrorNormalize(device="gpu",
                                            output_dtype=types.FLOAT,
                                            output_layout=types.NCHW,
                                            image_type=types.RGB,
                                            mean=[0.49139968 * 255., 0.48215827 * 255., 0.44653124 * 255.],
                                            std=[0.24703233 * 255., 0.24348505 * 255., 0.26158768 * 255.]
                                            )

    def iter_setup(self):
        (images, labels) = self.iterator.next()
        self.feed_input(self.jpegs, images, layout="HWC")  # can only in HWC order
        self.feed_input(self.labels, labels)

    def define_graph(self):
        self.jpegs = self.input()
        self.labels = self.input_label()
        output = self.jpegs
        output = self.cmnp(output.gpu())
        return [output, self.labels]


class Cifar10InputIter():
    base_folder = 'cifar-10-batches-py'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]

    def __init__(self, batch_size, type='train', root='/userhome/memory_data/cifar10', phase2_idx=None):
        self.root = root
        self.batch_size = batch_size
        self.mode = type
        if self.mode == 'test':
            downloaded_list = self.test_list
        else:
            downloaded_list = self.train_list

        self.data = []
        self.targets = []
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                if sys.version_info[0] == 2:
                    entry = pickle.load(f)
                else:
                    entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])

        data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        targets = np.vstack(self.targets)
        if self.mode == 'train':
            self.data = data[:-5000]
            self.targets = targets[:-5000]
            self.n = 45000
            np.save("cifar_train.npy", self.data)
            self.data = np.load('cifar_train.npy')  # to serialize, increase locality
        elif self.mode == 'phase2':
            self.data = np.take(data, phase2_idx, axis=0)
            self.targets = np.take(targets, indices, axis=0)
            self.n = len(self.data)
            np.save("cifar_phase2.npy", self.data)
            self.data = np.load('cifar_phase2.npy')  # to serialize, increase locality
        elif self.mode == 'val':
            self.data = data[-5000:]
            self.targets = targets[-5000:]
            self.n = 5000
            np.save("cifar_val.npy", self.data)
            self.data = np.load('cifar_val.npy')  # to serialize, increase locality
        else:
            self.data = data
            self.targets = targets
            self.n = 10000
            np.save("cifar_test.npy", self.data)
            self.data = np.load('cifar_test.npy')  # to serialize, increase locality
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        batch = []
        labels = []
        for _ in range(self.batch_size):
            if self.mode == 'train' and self.i % self.n == 0:
                self.data, self.targets = shuffle(self.data, self.targets, random_state=0)
            img, label = self.data[self.i], self.targets[self.i]
            batch.append(img)
            labels.append(label)
            self.i = (self.i + 1) % self.n
        return (batch, labels)

    next = __next__
