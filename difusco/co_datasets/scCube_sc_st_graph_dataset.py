"""SC_ST Graph Dataset"""

import numpy as np
import os
import torch
import torch.nn.functional as F
from sklearn.neighbors import KDTree
# from torch_geometric.data import Data as GraphData
from torch_geometric.loader import DataLoader as GraphDataLoader

from tqdm import *
from torch_geometric.data import Batch,Data
from torch_geometric import utils

import torch
from torch.utils import data


class scCube_data(torch.utils.data.Dataset):
  def __init__(self, data_file, list_IDs, gene_shuff=True):
    self.data_file = data_file
    self.list_IDs = list_IDs
    self.gene_shuff = gene_shuff

    # self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.list_IDs)} lines')

  def __len__(self):
    return len(self.list_IDs)

  def get_example(self, idx):
    
    
    count_sc = torch.tensor(np.loadtxt(self.data_file+f'sc_count/sc_count{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    count_st = torch.tensor(np.loadtxt(self.data_file+f'st_count/st_count{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    assignment_matrix = torch.tensor(np.loadtxt(self.data_file+f'assign/assignment_matrix{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1)).type(torch.float)
    cell_type_sc = torch.tensor(np.loadtxt(self.data_file+f'cell_type_sc/cell_type_sc{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    de_conv_post_norm = torch.tensor(np.loadtxt(self.data_file+f'cell_type_st_post/cell_type_st_post{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    pos = torch.tensor(np.loadtxt(self.data_file+f'position/position{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
  
    if self.gene_shuff:
      gene_index = torch.randperm(2000)
      count_sc = count_sc[:,gene_index]
      count_st = count_st[:,gene_index]
      # print(idx, gene_index)
      # 5534 tensor([1728,  843, 1775,  ...,  105,  301,  367])
      # 63 tensor([1728,  843, 1775,  ...,  105,  301,  367])
      # 1471 tensor([1728,  843, 1775,  ...,  105,  301,  367])
      # 2188 tensor([1728,  843, 1775,  ...,  105,  301,  367])
      # 6886 tensor([ 104, 1873,   24,  ..., 1946,  944,  902])
      # 283 tensor([ 104, 1873,   24,  ..., 1946,  944,  902])
      # 6051 tensor([ 104, 1873,   24,  ..., 1946,  944,  902])
      # 603 tensor([ 104, 1873,   24,  ..., 1946,  944,  902])

    return count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, self.list_IDs[idx]

  def __getitem__(self, idx):
    count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, self.list_IDs[idx] = self.get_example(idx)
    
      
    return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, self.list_IDs[idx])
    

class real_data(torch.utils.data.Dataset):
  def __init__(self, data_file, args, split):
    self.data_file = data_file
    self.args = args
    self.split = split
    self.list_IDs = list(range(0,len(os.listdir(self.data_file + 'sc_count' ))))
    if self.args.test_data=='merfish':
      if self.split == 'test':
        self.list_IDs=[]
        # self.list_IDs = list(range(100))
        for i in range(100):
          self.list_IDs.append(self.args.idx_tune)
    if self.args.test_data=='PDAC':
      if self.split == 'train' or self.split == 'val':
        self.list_IDs=[]
        for i in range(100):
          self.list_IDs.append(0)


    # self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" {len(self.list_IDs)} lines')

  def __len__(self):
    return len(self.list_IDs)

  def get_example(self, idx):
    if self.args.test_data=='merfish':
      if self.args.fine_tune:
        if self.split == 'test':
          # ind = self.list_IDs[idx]
          ind = self.args.idx_tune
        else:
          ind = self.list_IDs[self.args.idx_tune]
      else:
        ind = self.list_IDs[idx]
      
      animal_list = np.loadtxt('/data/jjingliu/real_data/merfish/animal_list.csv', delimiter=",", skiprows=0)
      slice_list = np.loadtxt('/data/jjingliu/real_data/merfish/slice_list.csv', delimiter=",", skiprows=0)
      id_animal = int(animal_list[ind])
      id_slice = int(slice_list[ind])

      count_sc = torch.tensor(np.loadtxt(self.data_file+f'sc_count/sc_count{id_animal}_{id_slice}.csv', delimiter=",", skiprows=1))
      count_st = torch.tensor(np.loadtxt(self.data_file+f'{self.args.resolution}/st_count/st_count_{id_animal}_{id_slice}.csv', delimiter=",", skiprows=1))
      num_sc, num_gene = count_sc.shape[-2], count_sc.shape[-1]
      num_st = count_st.shape[-2]
      assignment_matrix = torch.tensor(np.loadtxt(self.data_file+f'{self.args.resolution}/assignment_matrix/assignment_matrix_{id_animal}_{id_slice}.csv', delimiter=",", skiprows=1)).type(torch.float)
      cell_type_sc = torch.tensor(np.loadtxt(self.data_file+f'{self.args.resolution}/cell_type_sc/cell_type_sc_{id_animal}_{id_slice}.csv', delimiter=",", skiprows=1))
      de_conv_post_norm = torch.tensor(np.loadtxt(self.data_file+f'{self.args.resolution}/cell_type_st_post/cell_type_st_post_{id_animal}_{id_slice}.csv', delimiter=",", skiprows=1))
      pos = torch.tensor(np.loadtxt(self.data_file+f'{self.args.resolution}/position/position_{id_animal}_{id_slice}.csv', delimiter=",", skiprows=1))
      if self.args.fine_tune:
        if self.split=='train':
          seg = self.args.seg
          sc_inds = torch.randperm(num_sc)[0:num_sc//seg]
          st_inds = torch.randperm(num_st)[0:num_st//seg]
          gene_inds = torch.randperm(self.args.gen_num)
          count_sc = count_sc[sc_inds,:][:,gene_inds]
          count_st = count_st[st_inds,:][:,gene_inds]

          assignment_matrix = assignment_matrix[sc_inds,:][:,st_inds]
          cell_type_sc = cell_type_sc[sc_inds,:]
          de_conv_post_norm = de_conv_post_norm[st_inds,:]
          pos = torch.tensor(0)
        elif self.split=='val':
          seg = self.args.seg
          sc_inds = list(range(num_sc//seg))
          st_inds = list(range(num_st//seg))
          gene_inds = torch.randperm(self.args.gen_num)
          # count_sc = count_sc[sc_inds,:][:,gene_inds]
          count_sc = count_sc[:,gene_inds]
          count_st = count_st[st_inds,:][:,gene_inds]

          # assignment_matrix = assignment_matrix[sc_inds,:][:,st_inds]
          # cell_type_sc = cell_type_sc[sc_inds,:]
          assignment_matrix = assignment_matrix[:,st_inds]
          # cell_type_sc = cell_type_sc[sc_inds,:]
          de_conv_post_norm = de_conv_post_norm[st_inds,:]
          pos = torch.tensor(0)
      # if self.args.zero_padding:
      #   padding = torch.zeros((1, 2000 - num_gene))
      #   count_sc = torch.cat((count_sc, padding.repeat(num_sc,1)),-1)
      #   count_st = torch.cat((count_st, padding.repeat(num_st,1)),-1)
      # else:
      #   count_sc = count_sc.repeat(1, 2000//num_gene+1)[:,:2000]
      #   count_st = count_st.repeat(1, 2000//num_gene+1)[:,:2000]
    elif self.args.test_data=='seqfish+':
      ind = self.list_IDs[idx]
      if self.args.fine_tune:
        idx=self.args.idx_tune
      count_sc = torch.tensor(np.loadtxt(self.data_file+f'sc_count/sc_count{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
      count_st = torch.tensor(np.loadtxt(self.data_file+f'st_count/st_count{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
      assignment_matrix = torch.tensor(np.loadtxt(self.data_file+f'assign/assignment_matrix{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1)).type(torch.float)
      cell_type_sc = torch.tensor(np.loadtxt(self.data_file+f'cell_type_sc/cell_type_sc{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
      de_conv_post_norm = torch.tensor(np.loadtxt(self.data_file+f'cell_type_st_post/cell_type_st_post{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
      pos = torch.tensor(np.loadtxt(self.data_file+f'position/position{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))

    elif self.args.test_data=='PDAC':
      # idx = 0
      ind = self.list_IDs[idx]
      count_sc = torch.tensor(np.loadtxt(self.data_file+f'sc_count/sc_count_{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
      count_st = torch.tensor(np.loadtxt(self.data_file+f'st_count/st_count_{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
      num_sc, num_st, num_gene = count_sc.shape[0], count_st.shape[0], count_sc.shape[1]
      # assignment_matrix = torch.tensor(np.loadtxt('/data/jjingliu/temp/A.txt', delimiter=",", skiprows=0))
      assignment_matrix =  torch.zeros(num_sc, num_st).type(torch.float)
      cell_type_sc = torch.tensor(np.loadtxt(self.data_file+f'cell_type_sc/cell_type_sc_{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
      num_ct = cell_type_sc.shape[1]
      de_conv_post_norm = torch.ones(num_sc, num_ct).type(torch.float)
      pos = torch.tensor(np.loadtxt(self.data_file+f'position/position_{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))

      st_inds = list(range(len(count_st)))
      if self.args.fine_tune:
        if self.split=='train':
          seg = self.args.seg
          st_inds = torch.randperm(num_st)[0:num_st//seg]
          # st_inds = torch.randperm(num_st)[0:num_st//seg]
          gene_inds = torch.randperm(self.args.gen_num)
          count_sc = count_sc[:,gene_inds]
          count_st = count_st[st_inds,:][:,gene_inds]

          de_conv_post_norm = de_conv_post_norm[st_inds,:]
          pos = torch.tensor(0)

    return count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, ind, st_inds

  def __getitem__(self, idx):
    # if self.args.fine_tune:
    #   idx=self.args.idx_tune

    count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, ind, st_inds = self.get_example(idx)

    return (
      torch.LongTensor(np.array([idx], dtype=np.int64)),
      count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, ind, st_inds)
    
    
class shuff_data(torch.utils.data.Dataset):
  def __init__(self, data_file, list_IDs):
    self.data_file = data_file
    self.list_IDs = list_IDs

    # self.file_lines = open(data_file).read().splitlines()
    print(f'Loaded "{data_file}" with {len(self.list_IDs)} lines')

  def __len__(self):
    return len(self.list_IDs)

  def get_example(self, idx):
    gene_index = torch.randperm(2000)
    count_sc = torch.tensor(np.loadtxt(self.data_file+f'sc_count/sc_count{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    count_st = torch.tensor(np.loadtxt(self.data_file+f'st_count/st_count{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    assignment_matrix = torch.tensor(np.loadtxt(self.data_file+f'assign/assignment_matrix{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1)).type(torch.float)
    cell_type_sc = torch.tensor(np.loadtxt(self.data_file+f'cell_type_sc/cell_type_sc{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    de_conv_post_norm = torch.tensor(np.loadtxt(self.data_file+f'cell_type_st_post/cell_type_st_post{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    pos = torch.tensor(np.loadtxt(self.data_file+f'position/position{self.list_IDs[idx]}.csv', delimiter=",", skiprows=1))
    count_sc = count_sc[:,gene_index]
    count_st = count_st[:,gene_index]

    return count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, self.list_IDs[idx]

  def __getitem__(self, idx):
    count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, self.list_IDs[idx] = self.get_example(idx)
    
      
    return (
          torch.LongTensor(np.array([idx], dtype=np.int64)),
          count_sc, count_st, assignment_matrix, cell_type_sc, de_conv_post_norm, pos, self.list_IDs[idx]) 

if __name__ == '__main__':
  

  bsz = 1

  path_file = '/media/imin/DATA/jjingliu/STDD/tensor_simulation_by_scCube/'
  train_inds = list(range(0,7000))
  val_inds = list(range(7000,9000))
  test_inds = list(range(9000,10000))

  train_dataset = scCube_data(path_file, train_inds)
  train_dataloader = GraphDataLoader(
        train_dataset, batch_size=bsz, shuffle=True,
        num_workers=0, pin_memory=True,
        persistent_workers=False, drop_last=True)

  for [i_batch, batch] in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
  # loop = tqdm(enumerate(train_dataloader_dense), total=len(train_dataloader_dense))
  # for step, batch in loop:
    idx, count_sc, count_st, assignment_matrix, cell_type_sc, cell_type_st, pos = batch

    # print(cell_type_sc, cell_type_sc.shape)

    # print(pos, pos.shape)

    sc_adj = cell_type_sc @ (cell_type_sc.permute(0,2,1)) - torch.eye(cell_type_sc.shape[1])
    # print(sc_adj.shape, sc_adj)

    def knn_matrix(points, k):
      from scipy.spatial.distance import cdist
      points = np.array(points)
      n = len(points)
      dist_matrix = cdist(points, points)
      eye_matrix = np.eye(n) * 1.0e100
      dist_matrix += eye_matrix
      knn_values = np.take_along_axis(dist_matrix, np.argsort(dist_matrix, axis=1)[:, :k], axis=1)
      matrix = np.where(dist_matrix <= knn_values[:, -1][:, None], 1, 0)
      # 返回结果矩阵
      return torch.tensor(matrix)

    st_adj =  knn_matrix(pos.squeeze(), 5)
    print(st_adj.shape, st_adj)



  