import cv2
import glob
import os, sys, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchvision.transforms import Normalize
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

image_size = 224
batch_size = 64

crops_dir = "/raid/scratch/tf_train/dset/dfdc_train_part_10/?/*.png"
val_crops_dir = "/raid/scratch/tf_train/dset/feat_val/dfdc_train_part_0/?/*.png"

img_path = random.choice(glob.glob(crops_dir))
print(img_path)
# plt.imshow(cv2.imread(img_path)[..., ::-1])
print(cv2.imread(img_path)[..., ::-1])

class Unnormalize:
    """Converts an image tensor that was previously Normalize'd
    back to an image with pixels in the range [0, 1]."""
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        mean = torch.as_tensor(self.mean, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        std = torch.as_tensor(self.std, dtype=tensor.dtype, device=tensor.device).view(3, 1, 1)
        return torch.clamp(tensor*std + mean, 0., 1.)


mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize_transform = Normalize(mean, std)
unnormalize_transform = Unnormalize(mean, std)


def random_hflip(img, p=0.5):
    """Random horizontal flip."""
    if random.random() < p:
        return cv2.flip(img, 1)
    else:
        return img


def load_image(filename, augment):
    """Loads an image into a tensor. Also returns its label."""
    # dir = os.path.dirname(filename)
    # path = os.path.normpath(dir)
    # label = int(path.split(os.sep)[-1])

    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if augment: 
        img = random_hflip(img)

    # img = cv2.resize(img, (image_size, image_size))

    img = torch.tensor(img).permute((2, 0, 1)).float().div(255)
    img = normalize_transform(img)

    return img


img = load_image("/raid/scratch/tf_train/dset/dfdc_train_part_10/1/aajtsolpjq.mp4_0.png", augment=True)
print(img.shape)

# plt.imshow(unnormalize_transform(img).permute((1, 2, 0)))
# plt.show()


class VideoDataset(Dataset):
    """Face crops dataset.

    Arguments:
        crops_dir: base folder for face crops
        df: Pandas DataFrame with metadata
        split: if "train", applies data augmentation
        image_size: resizes the image to a square of this size
        sample_size: evenly samples this many videos from the REAL
            and FAKE subfolders (None = use all videos)
        seed: optional random seed for sampling
    """
    def __init__(self, crops_dir, split, image_size, sample_size=None, seed=None):
        self.crops_dir = crops_dir
        self.split = split
        self.image_size = image_size
        self.crops_files = []
        
        real_crops_files = []
        fake_crops_files = []
        
        for fpath in glob.glob(crops_dir):
            dir = os.path.dirname(fpath)
            # path = os.path.normpath(dir)
            label = int(dir.split(os.sep)[-1])
            if label == 0:
                real_crops_files.append((fpath, 0))
            elif label == 1:
                fake_crops_files.append((fpath, 1))
            else:
                raise('Invalid label: %d' % label)

        if sample_size is not None:
            sample_size = np.min(np.array([sample_size, len(real_crops_files), len(fake_crops_files)]))
            print("%s: sampling %d from %d real crops" % (split, sample_size, len(real_crops_files)))
            print("%s: sampling %d from %d fake crops" % (split, sample_size, len(fake_crops_files)))
            real_crops_files = random.sample(real_crops_files, sample_size)
            fake_crops_files = random.sample(fake_crops_files, sample_size)
            for i in range(sample_size):
                self.crops_files.append(real_crops_files[i])
                self.crops_files.append(fake_crops_files[i])
        else:
            self.crops_files = real_crops_files + fake_crops_files
            # random.shuffle(self.crops_files)

        num_real = len(real_crops_files)
        num_fake = len(fake_crops_files)
        print("%s dataset has %d real videos, %d fake videos" % (split, num_real, num_fake))

    def __getitem__(self, index):
        filename = self.crops_files[index][0]
        label = self.crops_files[index][1]
        img = load_image(filename, self.split == "train")
        return img, label
        
    def __len__(self):
        return len(self.crops_files)


dataset = VideoDataset(crops_dir, "val", image_size, sample_size=1000, seed=1234)
# plt.imshow(unnormalize_transform(dataset[0][0]).permute(1, 2, 0))
print(unnormalize_transform(dataset[0][0]).permute(1, 2, 0))

del dataset


def create_data_loaders(train_crops_dir, eval_crops_dir, image_size, batch_size, num_workers):

    train_dataset = VideoDataset(train_crops_dir, "train", image_size, sample_size=10000)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)

    val_dataset = VideoDataset(eval_crops_dir, "val", image_size, sample_size=500, seed=1234)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader

train_loader, val_loader = create_data_loaders(crops_dir, val_crops_dir, image_size, 
                                               batch_size, num_workers=2)

X, y = next(iter(train_loader))
# plt.imshow(unnormalize_transform(X[0]).permute(1, 2, 0))
print(unnormalize_transform(X[0]).permute(1, 2, 0))
print(y[0])

X, y = next(iter(val_loader))
# plt.imshow(unnormalize_transform(X[0]).permute(1, 2, 0))
print(unnormalize_transform(X[0]).permute(1, 2, 0))
print(y[0])


def evaluate(net, data_loader, device, silent=False):
    net.train(False)

    bce_loss = 0
    total_examples = 0

    with tqdm(total=len(data_loader), desc="Evaluation", leave=False, disable=silent) as pbar:
        for batch_idx, data in enumerate(data_loader):
            with torch.no_grad():
                batch_size = data[0].shape[0]
                x = data[0].to(device)
                y_true = data[1].to(device).float()

                y_pred = net(x)
                y_pred = y_pred.squeeze()

                bce_loss += F.binary_cross_entropy_with_logits(y_pred, y_true).item() * batch_size

            total_examples += batch_size
            pbar.update()

    bce_loss /= total_examples

    if silent:
        return bce_loss
    else:
        print("BCE: %.4f" % (bce_loss))


def fit(epochs):
    global history, iteration, epochs_done, lr

    with tqdm(total=len(train_loader), leave=False) as pbar:
        for epoch in range(epochs):
            pbar.reset()
            pbar.set_description("Epoch %d" % (epochs_done + 1))
            
            bce_loss = 0
            total_examples = 0

            net.train(True)

            for batch_idx, data in enumerate(train_loader):
                batch_size = data[0].shape[0]
                x = data[0].to(gpu)
                y_true = data[1].to(gpu).float()
                
                optimizer.zero_grad()

                y_pred = net(x)
                y_pred = y_pred.squeeze()
                
                loss = F.binary_cross_entropy_with_logits(y_pred, y_true)
                loss.backward()
                optimizer.step()
                
                batch_bce = loss.item()
                bce_loss += batch_bce * batch_size
                history["train_bce"].append(batch_bce)

                total_examples += batch_size
                iteration += 1
                pbar.update()

            bce_loss /= total_examples
            epochs_done += 1

            print("Epoch: %3d, train BCE: %.4f" % (epochs_done, bce_loss))

            val_bce_loss = evaluate(net, val_loader, device=gpu, silent=True)
            history["val_bce"].append(val_bce_loss)
            
            print("              val BCE: %.4f" % (val_bce_loss))

            # TODO: can do LR annealing here
            # TODO: can save checkpoint here

            print("")


checkpoint = torch.load("pretrained/resnext50_32x4d-7cdf4587.pth")


class MyResNeXt(models.resnet.ResNet):
    def __init__(self, training=True):
        super(MyResNeXt, self).__init__(block=models.resnet.Bottleneck,
                                        layers=[3, 4, 6, 3], 
                                        groups=32, 
                                        width_per_group=4)

        self.load_state_dict(checkpoint)

        # Override the existing FC layer with a new one.
        self.fc = nn.Linear(2048, 1)

net = MyResNeXt().to(gpu)
# if torch.cuda.device_count() > 1: 
#     net = nn.DataParallel(net) #enabling data parallelism
del checkpoint

out = net(torch.zeros((10, 3, image_size, image_size)).to(gpu))
print(out.shape)


def freeze_until(net, param_name):
    found_name = False
    for name, params in net.named_parameters():
        if name == param_name:
            found_name = True
        params.requires_grad = found_name


freeze_until(net, "layer4.0.conv1.weight")

print([k for k,v in net.named_parameters() if v.requires_grad])

evaluate(net, val_loader, device=gpu)

lr = 0.01
wd = 0.

history = { "train_bce": [], "val_bce": [] }
iteration = 0
epochs_done = 0

optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=wd)

# checkpoint = torch.load("model-checkpoint.pth")
# net.load_state_dict(checkpoint)

# checkpoint = torch.load("optimizer-checkpoint.pth")
# optimizer.load_state_dict(checkpoint)

fit(5)

# def set_lr(optimizer, lr):
#     for param_group in optimizer.param_groups:
#         param_group["lr"] = lr

# lr /= 10
# set_lr(optimizer, lr)

# fit(5)

torch.save(net.state_dict(), "checkpoint.pth")

# plt.plot(history["val_bce"])
print(history["val_bce"])