import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import torchvision.datasets as D
import torch.nn.functional as F
from torchvision.transforms import Normalize
import numpy as np
import pickle
from matplotlib import pyplot as plt
import codecs

class MyDataset(torch.utils.data.Dataset):
  def __init__(self, root, tform=None, normalize = False):
    super(MyDataset, self).__init__()
    self.root = root
    self.tform = tform
    file = open(root+"Sub20x20_full_grid.pkl", 'rb')
    self.data = pickle.load(file)
    self.normalize = normalize
    self.rewards = []
    for i in range(len(self.data)):
       self.rewards.append(self.data[i][1])

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    if not self.normalize:
      return np.expand_dims((self.data[i][0] < 0).astype('float32'), axis=0), np.array(self.rewards[i], dtype="float32")
    else:
      return np.expand_dims((self.data[i][0] < 0).astype('float32'), axis=0), np.array(self.scale(self.rewards[i]), dtype="float32")
  
  
class MyDatasetV2(torch.utils.data.Dataset):
  def __init__(self, root, tform=None, normalize = False):
    super(MyDatasetV2, self).__init__()
    """
    Function that creates a pytorch dataset.
    """
    # Loading file with all solutions
    self.root = root
    self.tform = tform
    file = open(root+"Sub20x20_full_grid.pkl", 'rb')
    self.data = pickle.load(file)
    # storing rewards seperately
    self.rewards = []
    for i in range(len(self.data)):
       self.rewards.append(self.data[i][1])
    
    # Directory with landscape information
    self.landscape_dir = f"{self.root}/Sub20x20"

    # Loads elevation .asc into a numpy array
    with codecs.open(f'{self.landscape_dir}/elevation.asc', encoding='utf-8-sig', ) as f:
      line = "_"
      elevation = []
      while line:
          line = f.readline()
          line_list = line.split()
          if len(line_list) > 2:
              elevation.append([float(i) for i in line_list])
    elevation = np.array(elevation)

    # Loads slope .asc into a numpy array
    with codecs.open(f'{self.landscape_dir}/slope.asc', encoding='utf-8-sig', ) as f:
      line = "_"
      slope = []
      while line:
          line = f.readline()
          line_list = line.split()
          if len(line_list) > 2:
              slope.append([float(i) for i in line_list])
    slope = np.array(slope)

    # Loads elevation .saz into a numpy array
    with codecs.open(f'{self.landscape_dir}/saz.asc', encoding='utf-8-sig', ) as f:
      line = "_"
      saz = []
      while line:
          line = f.readline()
          line_list = line.split()
          if len(line_list) > 2:
              saz.append([float(i) for i in line_list])
    saz = np.array(saz)

    # Stacks array into a tensor, generating a landscape tensor
    self.landscape = torch.from_numpy(np.stack([elevation, slope, saz]))
    # We compute means + std per channel to normalize
    means = torch.mean(self.landscape, dim=(1,2))
    stds = torch.std(self.landscape, dim=(1,2))
    norm = Normalize(means, stds)
    # Normalizes landscape
    self.landscape = norm(self.landscape)

  def __len__(self):
    return len(self.data)

  def __getitem__(self, i):
    """
    Returns the ith element of the dataset, each composed of:
    data: Tensor of (4, 20, 20)
    valuation: float Tensor
    """
    # We load solution
    firebreaks = torch.from_numpy(np.expand_dims((self.data[i][0] > 0).astype('float32'), axis=0))
    # We generate a tensor of solution + landscape
    data = torch.cat([firebreaks, self.landscape], dim=0).float()
    return data, torch.Tensor([self.rewards[i]])
  

def to_img(x):
    x = x.clamp(0, 1)
    return x

def show_image(img):
    img = to_img(img)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def visualise_output(images,r, model, n_images ,n_rows):

    with torch.no_grad():

        images = images
        images = model(images, r)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torchvision.utils.make_grid(images[0:n_images], n_rows).numpy()
        plt.imshow(np.transpose(np.where(np_imagegrid >= 0.5, 1.0, 0.0), (1, 2, 0)))

def visualise_output_reward(images,r, model, n_images ,n_rows):

    with torch.no_grad():

        images = images
        images, r = model(images, r)
        images = images.cpu()
        images = to_img(images)
        np_imagegrid = torch.round(torchvision.utils.make_grid(images[0:n_images], n_rows)).numpy()
        plt.imshow(np.transpose((np_imagegrid, (1, 2, 0))))
        plt.show()
