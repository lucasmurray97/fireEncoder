import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
import numpy as np
import pickle
from matplotlib import pyplot as plt

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, root, tform=None):
    super(MyDataset, self).__init__()
    self.root = root
    self.tform = tform
    file = open(root, 'rb')
    self.data = pickle.load(file)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    return np.expand_dims(self.data[i][0].astype('float32'), axis=0), np.array(10*self.data[i][1], dtype="float32")


def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images, model, n_images ,n_rows):

    with torch.no_grad():

        images = images
        images = model(images)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torch.round(torchvision.utils.make_grid(images[0:n_images], n_rows)).numpy()
        plt.imshow(np.transpose(np_imagegrid, (1, 2, 0)))
        plt.show()