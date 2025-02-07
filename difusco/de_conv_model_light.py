"""Lightning module for training the DIFUSCO TSP model."""

import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from pytorch_lightning.utilities import rank_zero_info

# from co_datasets.sc_st_graph_dataset import SC2ST_GraphDataset
# from co_datasets.seqFISH_sc_st_graph_dataset import SC2ST_GraphDataset_light as real_dataset
from co_datasets.scCube_sc_st_graph_dataset import scCube_data, real_data, shuff_data
from pl_meta_model import COMetaModel
from utils.diffusion_schedulers import InferenceSchedule
from utils.sc2st_utils import file_name, test_file, norm_sc, norm_st, loss_t, Reconstruction_loss

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


class SC2STModel_light(COMetaModel):
  def __init__(self,
               param_args=None):
    super(SC2STModel_light, self).__init__(param_args=param_args, node_feature_only=False)

    # self.train_dataset = SC2ST_GraphDataset_light(
    #     data_file=file_name(self.args, split=self.args.training_split),
    #     sparse_factor=self.args.sparse_factor,
    # )


    # self.validation_dataset = SC2ST_GraphDataset_light(
    #     data_file=file_name(self.args, self.args.validation_split),
    #     sparse_factor=self.args.sparse_factor,
    # )

    # if self.args.do_train:
    #   self.test_dataset = SC2ST_GraphDataset_light(
    #       data_file=file_name(self.args, self.args.test_split),
    #       sparse_factor=self.args.sparse_factor,
    #   )
    # else: 
    #   self.test_dataset = SC2ST_GraphDataset_light(
    #       data_file=test_file(self.args),
    #       sparse_factor=self.args.sparse_factor,
    #   )

    # if self.args.test_data == 'seqFISH':
    #   self.count_sc = torch.load(self.args.data_path + 'seqFISH_count_SC.pt')
    # elif self.args.test_data == 'merfish_part_20':
    #   self.count_sc = torch.load(self.args.data_path + 'merFISH_1_count_SC.pt')


    # path_file = '/media/imin/DATA/jjingliu/STDD/post_simulation_by_scCube/'
    path_file = self.args.data_path
    train_inds = list(range(0,9000))
    val_inds = list(range(9100, 10000))
    test_inds = list(range(9000,9100))
    if self.args.fine_tune:
      print('fine_tune on test data:')
      self.train_dataset = real_data(path_file, self.args, 'train')
      self.validation_dataset = real_data(path_file, self.args, 'val')
    else:
      print('training:')
      self.train_dataset = scCube_data(path_file, train_inds, gene_shuff=self.args.gene_shuff_train)
      self.validation_dataset = scCube_data(path_file, val_inds, gene_shuff=self.args.gene_shuff_val)

    


    if self.args.do_train:
      self.test_dataset = scCube_data(path_file, test_inds, gene_shuff=self.args.gene_shuff_test)
      if self.args.fine_tune:
        self.test_dataset = real_data(path_file, self.args, 'test')
    else:
      if self.args.test_data=='simulation': 
        self.test_dataset = scCube_data(path_file, test_inds, gene_shuff=self.args.gene_shuff_test)
      else:
        self.test_dataset = real_data(path_file, self.args, 'test')
      

  def forward(self, x, y, bi_adj, sc_adj, st_adj, t, edge_index):
    if self.args.self_graph:
      return self.model(x, y, bi_adj, sc_adj, st_adj, t, edge_index)
    else:
      return self.model(x, y, bi_adj, t, edge_index)

  def categorical_training_step(self, batch, batch_idx):
    edge_index = None


    real_batch_idx, count_sc, count_st, bi_graph, cell_type_sc, de_conv_st, pos, sample_i, st_inds = batch
    
    
    # print(count_sc.shape, count_st.shape, bi_graph.shape)
    # print(count_sc[0,:,:].shape, count_st[0,:,:].shape, bi_graph[0,:,:].shape)
    # print(count_sc[1,:,:].shape, count_st[1,:,:].shape, bi_graph[1,:,:].shape)
    # st_graph = knn_matrix(pos[0], self.args.st_k)

    # C_INDEX = torch.arange(0, self.args.cell_type, 1)
    # sc_graph = F.one_hot(C_INDEX.repeat(self.args.cells_per_type, 1).T.reshape(-1), num_classes= self.args.cell_type)

    num_sampling = self.args.train_sampling
    t = np.random.randint(1, self.diffusion.T + 1, num_sampling).astype(int)

    # count_sc = self.count_sc.type_as(count_st) !!!!!!!!!!!!!
    if self.args.sc_norm:
      log_sc = norm_sc(count_sc)
      log_st = norm_st(count_st)
    else:
      log_sc = count_sc
      log_st = count_st


    # Sample from diffusion 
    bi_graph_onehot = F.one_hot(bi_graph.long(), num_classes=2).float()
    if self.args.fine_tune:
      bi_graph = torch.tensor(np.loadtxt(self.args.tune_A_file, delimiter=",", skiprows=0)).type(torch.float).unsqueeze(0).repeat(count_st.shape[0],1,1).to(count_st.device)[:,:,st_inds]
      A00_hat, A01_hat = (1. - bi_graph.float()), bi_graph.float()
      bi_graph_onehot = torch.cat((A00_hat, A01_hat), 2).permute(0,1,3,2)
      bi_graph = bi_graph.squeeze(2)

    
    bi_graph_t = self.diffusion.sample(bi_graph_onehot, t)
    A0 = bi_graph


    t = torch.from_numpy(t).float().view(num_sampling)

    device = batch[-1].device
    cell_type_sc, de_conv_st = cell_type_sc.type(torch.float).to(device), de_conv_st.to(device)
    # sc_adj = cell_type_sc @ (cell_type_sc.permute(0,2,1)) - torch.eye(cell_type_sc.shape[1]).to(device)
    # st_adj = knn_matrix(pos.cpu().squeeze(0), self.args.st_k).unsqueeze(0).to(device)
    sc_adj = torch.tensor(0)
    st_adj = torch.tensor(0)


    # Denoise

    bi_graph_0_pred, sc_hat, st_hat = self.forward(
        log_sc.to(bi_graph.device),
        log_st.to(bi_graph.device),
        bi_graph_t.float().to(bi_graph.device),
        sc_adj.float().to(bi_graph.device),
        st_adj.float().to(bi_graph.device),
        t.float().to(bi_graph.device),
        edge_index,
    )

    bi_graph_0_pred_prob = bi_graph_0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)[..., 1]
    
    if self.args.save_numpy_heatmap:
      # print(bi_graph_0_pred_prob.shape, bi_graph_t.shape, A0.shape, real_batch_idx, t[0])
      self.run_save_numpy_heatmap(bi_graph_0_pred_prob, bi_graph_t, A0[0], real_batch_idx, t[0], 'train', 'A0_prob')

    # Compute loss
    
    # print(bi_graph_0_pred.shape, A0.shape)#([3, 2, 1224, 255])([1, 1224, 255])
    # print(bi_graph_0_pred_prob.shape, A0.shape, count_sc.shape, count_st.shape, cell_type_sc.shape)
    bi_graph_0_pred_prob = bi_graph_0_pred_prob.log().softmax(dim=-1)
    loss_CE, RMSE_single, RMSE_global, MSE_hard, MSE_soft, count, term, regular_01, KL, cos_dis = loss_t(bi_graph_0_pred_prob, A0, count_sc, count_st, cell_type_sc, self.args)
    

    l1_norm = 0
    for bt in range(self.args.train_sampling):
      l1_norm = l1_norm + torch.linalg.norm(bi_graph_0_pred_prob[bt], ord=1, dim=(0, 1))
    l1_norm = l1_norm/self.args.train_sampling

    # loss_func = nn.CrossEntropyLoss()
    if self.args.decode:
      Reconstruction_loss_sc, Reconstruction_loss_st = Reconstruction_loss(count_sc, count_st, sc_hat, st_hat, self.args)
      loss = loss_CE + self.args.alpha_1 * RMSE_global + self.args.alpha_2 * term + self.args.alpha_3 * l1_norm + self.args.alpha_4 * (Reconstruction_loss_sc + Reconstruction_loss_st) + self.args.alpha_5 * regular_01 + self.args.alpha_6 * KL
      split = 'train'
      metrics = {
          f"{split}/loss": loss,
          f"{split}/MSE_hard": MSE_hard,
          f"{split}/MSE_soft": MSE_soft,
          f"{split}/RMSE":  RMSE_global,
          f"{split}/loss_CE":  loss_CE,
          f"{split}/count":  count,
          f"{split}/matrix_term":  term,
          f"{split}/l1_norm":  l1_norm,
          f"{split}/regular_01":  regular_01,
          f"{split}/kl":  KL,
          f"{split}/Reconstruction_loss_sc":  Reconstruction_loss_sc,
          f"{split}/Reconstruction_loss_st":  Reconstruction_loss_st,
      }
    else:
      loss = self.args.alpha_0 * loss_CE + self.args.alpha_1 * RMSE_global + self.args.alpha_2 * term + self.args.alpha_3 * l1_norm + self.args.alpha_5 * regular_01 + self.args.alpha_6 * KL  + self.args.alpha_7 * cos_dis  + self.args.alpha_8 * MSE_soft
      split = 'train'
      metrics = {
          f"{split}/loss": loss,
          f"{split}/MSE_hard": MSE_hard,
          f"{split}/MSE_soft": MSE_soft,
          f"{split}/RMSE":  RMSE_global,
          f"{split}/loss_CE":  loss_CE,
          f"{split}/count":  count,
          f"{split}/matrix_term":  term,
          f"{split}/l1_norm":  l1_norm,
          f"{split}/regular_01":  regular_01,
          f"{split}/kl":  KL,
          f"{split}/cos_dis":  cos_dis,
      }

    
    for k, v in metrics.items():
      self.log(k, v, on_epoch=True, sync_dist=True)
      
    # return loss
    return loss.requires_grad_(True)

  def gaussian_training_step(self, batch, batch_idx):
    if self.sparse:
      # TODO: Implement Gaussian diffusion with sparse graphs
      raise ValueError("DIFUSCO with sparse graphs are not supported for Gaussian diffusion")
    _, points, adj_matrix, _ = batch

    adj_matrix = adj_matrix * 2 - 1
    adj_matrix = adj_matrix * (1.0 + 0.05 * torch.rand_like(adj_matrix))
    # Sample from diffusion
    t = np.random.randint(1, self.diffusion.T + 1, adj_matrix.shape[0]).astype(int)
    xt, epsilon = self.diffusion.sample(adj_matrix, t)

    t = torch.from_numpy(t).float().view(adj_matrix.shape[0])
    # Denoise
    epsilon_pred = self.forward(
        points.float().to(adj_matrix.device),
        xt.float().to(adj_matrix.device),
        t.float().to(adj_matrix.device),
        None,
    )
    epsilon_pred = epsilon_pred.squeeze(1)

    # Compute loss
    loss = F.mse_loss(epsilon_pred, epsilon.float())
    self.log("train/loss", loss)
    return loss

  def training_step(self, batch, batch_idx):
    if self.diffusion_type == 'gaussian':
      return self.gaussian_training_step(batch, batch_idx)
    elif self.diffusion_type == 'categorical':
      return self.categorical_training_step(batch, batch_idx)

  def categorical_denoise_step(self, log_x, log_y, At, sc_adj, st_adj, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)

      A0_pred,_,_ = self.forward(
          log_x.to(device),#(N, G)
          log_y.to(device),#(bsz, S, G)
          At.float().to(device),#(bsz, N, S)
          sc_adj.float().to(device), 
          st_adj.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )

      A0_pred_prob = A0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)[..., 1]
        


      # 保证每行只有一个1
      # print(A0_pred_prob.shape)
      # A0_pred_prob = A0_pred_prob[..., 1]
      # A0_pred_prob = (A0_pred_prob == A0_pred_prob.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float).unsqueeze(3)
      # A0_pred_prob = torch.cat((1-A0_pred_prob, A0_pred_prob),-1)
      # print(A0_pred_prob.shape)
      A0_pred_prob = A0_pred_prob.log().softmax(dim=-1)
      A0_pred_prob0_hat, A0_pred_prob1_hat = (1. - A0_pred_prob).unsqueeze(3), A0_pred_prob.unsqueeze(3)
      A0_pred_prob = torch.cat((A0_pred_prob0_hat, A0_pred_prob1_hat),3)

      At, At_pro = self.categorical_posterior(target_t, t, A0_pred_prob, At)
      return At, At_pro, A0_pred_prob[:,:,:,1]

  def gaussian_denoise_step(self, points, xt, t, device, edge_index=None, target_t=None):
    with torch.no_grad():
      t = torch.from_numpy(t).view(1)
      pred = self.forward(
          points.float().to(device),
          xt.float().to(device),
          t.float().to(device),
          edge_index.long().to(device) if edge_index is not None else None,
      )
      pred = pred.squeeze(1)
      xt = self.gaussian_posterior(target_t, t, pred, xt)
      return xt

  def test_step(self, batch, batch_idx, split='test'):
    with torch.no_grad():
      edge_index = None
      np_edge_index = None
      device = batch[-2].device
      if not self.sparse:
        # real_batch_idx, adj_matrix, count_st = batch
        # if self.args.test_data=='seqFISH':
        #   count_sc = self.count_sc.type_as(count_st)
        # if self.args.test_data=='merfish_part_20':
        #   count_sc = self.count_sc[batch_idx].type_as(count_st)

        # real_batch_idx, count_sc, count_st, adj_matrix, cell_type_sc, _, _ = batch
        real_batch_idx, count_sc, count_st, bi_graph, cell_type_sc, de_conv_st, pos, sample_i,_ = batch
        
        if self.args.sc_norm:
          log_sc = norm_sc(count_sc)
          log_st = norm_st(count_st)
        else:
          log_sc = count_sc
          log_st = count_st

        A0 = bi_graph
        # A0 = bi_graph.squeeze(0)

        cell_type_sc, de_conv_st = cell_type_sc.type(torch.float).to(device), de_conv_st.to(device)
        sc_adj = torch.tensor(0)
        st_adj = torch.tensor(0)
        # sc_adj = cell_type_sc @ (cell_type_sc.permute(0,2,1)) - torch.eye(cell_type_sc.shape[1]).to(device)
        # st_adj = knn_matrix(pos.cpu().squeeze(0), self.args.st_k).unsqueeze(0).to(device)

      if self.args.parallel_sampling > 1:
        if not self.sparse:
          log_st = log_st.repeat(self.args.parallel_sampling, 1, 1)
          # points = points.repeat(self.args.parallel_sampling, 1, 1)

      for _ in range(self.args.sequential_sampling):
        # At = torch.randn_like(bi_graph.float())
        At = torch.rand_like(bi_graph.float())
        if self.args.parallel_sampling > 1:
          if not self.sparse:
            At = At.repeat(self.args.parallel_sampling, 1, 1)
            A0 = A0.repeat(self.args.parallel_sampling, 1, 1)
          else:
            xt = xt.repeat(self.args.parallel_sampling, 1)
          # At = torch.randn_like(At)
          At = torch.rand_like(At)

        if self.diffusion_type == 'gaussian':
          xt.requires_grad = True
        else:
          # At = (At > 0).long()
          At = torch.round(At)

        steps = self.args.inference_diffusion_steps
        time_schedule = InferenceSchedule(inference_schedule=self.args.inference_schedule,
                                          T=self.diffusion.T, inference_T=steps)

        # Diffusion iterations
        for i in tqdm(range(steps), total=steps, desc=f'Processing: sample {real_batch_idx.cpu().numpy().reshape(-1)[0]}'):
          t1, t2 = time_schedule(i)
          t1 = np.array([t1]).astype(int)
          t2 = np.array([t2]).astype(int)
          if t1==self.args.diffusion_steps:
              AT = At
              
          At, At_prob, A0_prob = self.categorical_denoise_step(
              log_sc, log_st, At, sc_adj, st_adj, t1, device, edge_index, target_t=t2)
            
          if self.args.save_numpy_heatmap:
            if t2%(steps/10) == 0:
              self.run_save_numpy_heatmap(At, AT, A0, real_batch_idx, t2, split, 'sample')
              self.run_save_numpy_heatmap(At_prob, AT, A0, real_batch_idx, t2, split, 'At_prob')
              self.run_save_numpy_heatmap(A0_prob, AT, A0, real_batch_idx, t2, split, 'A0_prob')

            # loss_CE, RMSE_single, RMSE_global, MSE_hard, MSE_soft, count= loss_t(At, adj_matrix, count_sc, count_st, self.args)
            # loss_CEt_prob, RMSEt_prob_single, RMSEt_prob_global, MSEt_hard_prob, MSEt_soft_prob, countt_prob = loss_t(At_prob, adj_matrix, count_sc, count_st, self.args)
            # loss_CE0_prob, RMSE0_prob_single, RMSE0_prob_global, MSE0_hard_prob, MSE0_soft_prob, count0_prob = loss_t(A0_prob, adj_matrix, count_sc, count_st, self.args)    

          if self.args.save_plt_loss:
            At = At[0,:,:].unsqueeze(0)
            At_prob = At_prob[0,:,:].unsqueeze(0)
            A0_prob = A0_prob[0,:,:].unsqueeze(0)
            adj_matrix = bi_graph
            loss_CE, RMSE_single, RMSE_global, MSE_hard, MSE_soft, count, matrix_term,_ ,_,_ = loss_t(At, adj_matrix, count_sc, count_st, cell_type_sc, self.args)
            loss_CEt_prob, RMSEt_prob_single, RMSEt_prob_global, MSEt_hard_prob, MSEt_soft_prob, countt_prob, matrix_term_t_pro,_ ,_,_   = loss_t(At_prob, adj_matrix, count_sc, count_st, cell_type_sc, self.args)
            loss_CE0_prob, RMSE0_prob_single, RMSE0_prob_global, MSE0_hard_prob, MSE0_soft_prob, count0_prob, matrix_term_0_pro,_ ,_ ,_  = loss_t(A0_prob, adj_matrix, count_sc, count_st, cell_type_sc, self.args)    
            
            
            if t1==self.args.diffusion_steps:      
              CE = loss_CE.unsqueeze(0)
              RMSE = RMSE_global.unsqueeze(0)
              mse_hard = MSE_hard.unsqueeze(0)
              mse_soft = MSE_soft.unsqueeze(0)
              Count = count.unsqueeze(0)
              Matrix_term = matrix_term.unsqueeze(0)

              CEt_prob = loss_CEt_prob.unsqueeze(0)
              RMSEt_prob =  RMSEt_prob_global.unsqueeze(0)
              mset_hard_prob = MSEt_hard_prob.unsqueeze(0)
              mset_soft_prob = MSEt_soft_prob.unsqueeze(0)
              Countt_prob = countt_prob.unsqueeze(0)

              CE0_prob = loss_CE0_prob.unsqueeze(0)
              RMSE0_prob =  RMSE0_prob_global.unsqueeze(0)
              mse0_hard_prob = MSE0_hard_prob.unsqueeze(0)
              mse0_soft_prob = MSE0_soft_prob.unsqueeze(0)
              Count0_prob = count0_prob.unsqueeze(0)

              RMSE_Single = RMSE_single.unsqueeze(0)

            else:
              CE = torch.cat((CE, loss_CE.unsqueeze(0)), 0)
              RMSE = torch.cat((RMSE, RMSE_global.unsqueeze(0)), 0)
              mse_hard = torch.cat((mse_hard, MSE_hard.unsqueeze(0)), 0)
              mse_soft = torch.cat((mse_soft, MSE_soft.unsqueeze(0)), 0)
              Count = torch.cat((Count, count.unsqueeze(0)), 0)   
              Matrix_term = torch.cat((Matrix_term, matrix_term.unsqueeze(0)), 0)   

              CEt_prob = torch.cat((CEt_prob, loss_CEt_prob.unsqueeze(0)), 0)
              RMSEt_prob = torch.cat((RMSEt_prob, RMSEt_prob_global.unsqueeze(0)), 0)
              mset_hard_prob = torch.cat((mset_hard_prob, MSEt_hard_prob.unsqueeze(0)), 0)
              mset_soft_prob = torch.cat((mset_soft_prob, MSEt_soft_prob.unsqueeze(0)), 0)
              Countt_prob = torch.cat((Countt_prob, countt_prob.unsqueeze(0)), 0)   

              CE0_prob = torch.cat((CE0_prob, loss_CE0_prob.unsqueeze(0)), 0)
              RMSE0_prob = torch.cat((RMSE0_prob, RMSE0_prob_global.unsqueeze(0)), 0)
              mse0_hard_prob = torch.cat((mse0_hard_prob, MSE0_hard_prob.unsqueeze(0)), 0)
              mse0_soft_prob = torch.cat((mse0_soft_prob, MSE0_soft_prob.unsqueeze(0)), 0)
              Count0_prob = torch.cat((Count0_prob, count0_prob.unsqueeze(0)), 0)      
                    
              RMSE_Single = torch.cat((RMSE_Single, RMSE_single.unsqueeze(0)), 0)
            # x0_truth = adj_matrix.repeat(self.args.parallel_sampling, 1)

            At = At.repeat(self.args.parallel_sampling,1,1)
            At_prob = At_prob.repeat(self.args.parallel_sampling,1,1)
            A0_prob = A0_prob.repeat(self.args.parallel_sampling,1,1)

        if self.diffusion_type == 'gaussian':
          adj_mat = xt.cpu().detach().numpy() * 0.5 + 0.5
        # else:
        #   adj_mat = At.float().cpu().detach().numpy() + 1e-6

        # print('test', adj_mat.shape, xt.shape)

      if self.args.save_plt_loss:
        self.run_save_loss_t(CE, real_batch_idx, steps, split, loss_type='CE')
        self.run_save_loss_t(RMSE, real_batch_idx, steps, split, loss_type='RMSE')
        self.run_save_loss_t(mse_hard, real_batch_idx, steps, split, loss_type='MSE_hard')
        self.run_save_loss_t(mse_soft, real_batch_idx, steps, split, loss_type='MSE_soft')
        self.run_save_loss_t(Count, real_batch_idx, steps, split, loss_type='Count')

        self.run_save_loss_t(CEt_prob, real_batch_idx, steps, split, loss_type='CEt_prob')
        self.run_save_loss_t(RMSEt_prob, real_batch_idx, steps, split, loss_type='RMSEt_prob')
        self.run_save_loss_t(mset_hard_prob, real_batch_idx, steps, split, loss_type='MSEt_hard_prob')
        self.run_save_loss_t(mset_soft_prob, real_batch_idx, steps, split, loss_type='MSEt_soft_prob')
        self.run_save_loss_t(Countt_prob, real_batch_idx, steps, split, loss_type='Countt_prob') 

        self.run_save_loss_t(CE0_prob, real_batch_idx, steps, split, loss_type='CE0_prob')
        self.run_save_loss_t(RMSE0_prob, real_batch_idx, steps, split, loss_type='RMSE0_prob')
        self.run_save_loss_t(mse0_hard_prob, real_batch_idx, steps, split, loss_type='MSE0_hard_prob')
        self.run_save_loss_t(mse0_soft_prob, real_batch_idx, steps, split, loss_type='MSE0_soft_prob')
        self.run_save_loss_t(Count0_prob, real_batch_idx, steps, split, loss_type='Count0_prob') 

        # self.run_save_RMSE_single_t(RMSE, RMSE_Single, real_batch_idx, steps, split, loss_type='RMSE_Single')
      
        
      A00_hat, A01_hat = (1. - At.float()).unsqueeze(1), At.float().unsqueeze(1)
      A0_hat = torch.cat((A00_hat, A01_hat),1)
      # x0_hat = F.one_hot(xt.long(), num_classes=2).permute((0, 3, 1, 2)).float()
      CrossEntropy = nn.CrossEntropyLoss()
      # print(A0_hat.shape, adj_matrix.shape)
      bi_graph = bi_graph.repeat(A0_hat.shape[0], 1, 1)
      loss_CE = CrossEntropy(A0_hat, bi_graph.long().to(A0_hat.device))
      # loss_CE = 0

      # A0_pred_prob = A0_hat.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)[..., 1]
      A0_pred_prob = A0_prob
      # A0_pred_hard = (A0_pred_prob == A0_pred_prob.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float)
      A0_pred_hard = A0_pred_prob.round()
      
      Y_hat_hard = A0_pred_hard.permute(0, 2, 1).float() @ count_sc.to(A0_hat.device).float()
      Y_hat_soft = A0_pred_prob.permute(0, 2, 1) @ count_sc.to(A0_hat.device).float()
      
      
      Y = count_st.to(A0_hat.device).float()

      if self.args.save_A0_only:
        log_path = self.args.path_A0 + self.args.project_name + '/'
        if not os.path.exists(log_path):
          os.makedirs(log_path)

        #  sample_i = str(sample_i.cpu().detach().numpy())
        np.savetxt(log_path + f'M_{sample_i.cpu().detach().numpy()[0]}_{real_batch_idx.cpu().detach().numpy()[0]}.txt', A0_pred_prob[0].cpu().detach().numpy(), delimiter=",")
        np.savetxt(log_path + f'At_{sample_i.cpu().detach().numpy()[0]}_{real_batch_idx.cpu().detach().numpy()[0]}.txt', At[0].cpu().detach().numpy(), delimiter=",")
        np.savetxt(log_path + f'A0_pre_{sample_i.cpu().detach().numpy()[0]}_{real_batch_idx.cpu().detach().numpy()[0]}.txt', A0_pred_hard[0].cpu().detach().numpy(), delimiter=",")
        np.savetxt(log_path + f'At_prob_{sample_i.cpu().detach().numpy()[0]}_{real_batch_idx.cpu().detach().numpy()[0]}.txt', At_prob[0].cpu().detach().numpy(), delimiter=",")


      MSELoss = nn.MSELoss(reduction='mean')
      # print(Y_hat_hard.shape, Y.shape)
      MSE_hard = MSELoss(Y_hat_hard, Y)
      MSE_soft = MSELoss(Y_hat_soft, Y)

      metrics = {
          f"{split}/loss": loss_CE,
          f"{split}/MSE_hard": MSE_hard,
          f"{split}/MSE_soft": MSE_soft,
      }
      for k, v in metrics.items():
        self.log(k, v, on_epoch=True, sync_dist=True)
      self.log(f"{split}/solved_cost", loss_CE, prog_bar=True, on_epoch=True, sync_dist=True)
      return metrics

  def run_save_numpy_heatmap(self, At, AT, A0, real_batch_idx, t, split, type):
    
    # if self.args.parallel_sampling > 1 or self.args.sequential_sampling > 1:
    #   raise NotImplementedError("Save numpy heatmap only support single sampling")
    real_batch_index = real_batch_idx.cpu().numpy().reshape(-1)[0]
    if split == 'train':
      t = t.int()
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    heatmap_path = os.path.join(exp_save_dir, 'numpy_heatmap', f"epoch_{self.current_epoch}", f"{split}-{type}-heatmap", f"{real_batch_index}")
    # rank_zero_info(f"Saving heatmap to {heatmap_path}")
    os.makedirs(os.path.join(heatmap_path, "txt"), exist_ok=True)
    os.makedirs(os.path.join(heatmap_path, "fig"), exist_ok=True)

    adj_mat = At[0].cpu().detach().numpy()
    AT, A0= AT[0].cpu().detach().numpy(), A0.cpu().detach().numpy()
    if A0.ndim == 3:
      A0 = A0.squeeze(0)

    if split != 'train' and self.current_epoch == 0:
      np.savetxt(os.path.join(heatmap_path, f"txt/epoch_0_heatmap-t{t}.txt"), adj_mat, delimiter=",")
      fig_path = os.path.join(heatmap_path, f"fig/epoch_0_heatmap-t{t}.png")
    else:
      np.savetxt(os.path.join(heatmap_path, f"txt/heatmap-t{t}.txt"), adj_mat, delimiter=",")
      fig_path = os.path.join(heatmap_path, f"fig/heatmap-t{t}.png")

    plt.figure(figsize=(10, 10))
    plt.imshow(adj_mat, cmap='Blues')
    plt.xlabel('Spots')
    plt.ylabel('Cells')
    plt.title('Assignment Matrix')
    plt.colorbar()
    plt.savefig(fig_path, dpi=500, bbox_inches='tight')
    plt.close()

    if t==0 or split=='train':
      fig_truth = os.path.join(heatmap_path, "fig/heatmap-truth.png")
      fig_AT = os.path.join(heatmap_path, f"fig/heatmap-t{self.args.diffusion_steps}.png")
      plt.figure(figsize=(10, 10))
      plt.imshow(A0, cmap='Blues')
      plt.xlabel('Spots')
      plt.ylabel('Cells')
      plt.title('Assignment Matrix')
      plt.colorbar()
      plt.savefig(fig_truth, dpi=500, bbox_inches='tight')
      plt.close()
      np.savetxt(os.path.join(heatmap_path, "txt/heatmap-truth.txt"), A0, delimiter=",")

      plt.figure(figsize=(10, 10))
      plt.imshow(AT, cmap='Blues')
      plt.xlabel('Spots')
      plt.ylabel('Cells')
      plt.title('Assignment Matrix')
      plt.colorbar()
      plt.savefig(fig_AT, dpi=500, bbox_inches='tight')
      plt.close()
      np.savetxt(os.path.join(heatmap_path, f"txt/heatmap-t{self.args.diffusion_steps}.txt"), AT, delimiter=",")
    
  def run_save_loss_t(self, loss_fcn, real_batch_idx, steps, split, loss_type='MSE_hard'):
    loss_fcn = loss_fcn[0:steps]

    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    loss_path = os.path.join(exp_save_dir, 'LOSS&COUNT', f'epoch_{self.current_epoch}', loss_type)
    T = [n for n in range(0,steps)]

    if not os.path.exists(loss_path):
      os.makedirs(loss_path)

    # loss分界线*******************************************************
    plt.plot(T, loss_fcn.cpu().detach().numpy(), 'b-', alpha=0.5, linewidth=1, label=f"mean_norm={'%.2f' % (loss_fcn.sum().item()/steps)}, {loss_type}_T={'%.2f' % loss_fcn[-1].item()}")#'bo-'表示蓝色实线，数据点实心原点标注

    plt.legend()  #显示上面的label
    plt.xlabel('time') #x_label
    plt.ylabel(loss_type)#y_label
    
    #plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.savefig(loss_path + f'/{split}-{loss_type}-{real_batch_idx}.png', dpi=500, bbox_inches='tight')
    plt.close()

  def run_save_RMSE_single_t(self, RMSE, loss_fcn, real_batch_idx, steps, split, loss_type='RMSE_single'):
    
    color=['b','g','r','c','m','y','k','w']
   
    loss_fcn = loss_fcn[0:steps]

    real_batch_idx = real_batch_idx.cpu().numpy().reshape(-1)[0]
    exp_save_dir = os.path.join(self.logger.save_dir, self.logger.name, self.logger.version)
    loss_path = os.path.join(exp_save_dir, 'LOSS&COUNT', f'epoch_{self.current_epoch}', loss_type)
    T = [n for n in range(0,steps)]

    if not os.path.exists(loss_path):
      os.makedirs(loss_path)

    # loss分界线*******************************************************
    num_types = loss_fcn.shape[-1]
    for type in range(num_types):
      plt.plot(T,loss_fcn.cpu().detach().numpy()[:, type], color[type], alpha=0.5, linewidth=1)

    # plt.legend()  #显示上面的label
    plt.xlabel('time') #x_label
    plt.ylabel(loss_type)#y_label
    
    #plt.ylim(-1,1)#仅设置y轴坐标范围
    plt.savefig(loss_path + f'/{split}-{loss_type}-{real_batch_idx}.png', dpi=500, bbox_inches='tight')
    plt.close()

    np.savetxt(loss_path + f'/{split}-{loss_type}-{real_batch_idx}.txt', loss_fcn.cpu().detach().numpy(), delimiter=",")
    np.savetxt(loss_path + f"/{split}-{'RMSE_global'}-{real_batch_idx}.txt", RMSE.cpu().detach().numpy(), delimiter=",")

  def validation_step(self, batch, batch_idx):
    with torch.no_grad():
      # return self.test_step(batch, batch_idx, split='val')
      real_batch_idx, count_sc, count_st, bi_graph, cell_type_sc, de_conv_st, pos, sample_i, _ = batch
      edge_index = None
      
      # num_sampling = self.args.train_sampling
      # t = np.random.randint(1, self.diffusion.T + 1, num_sampling).astype(int)
      t = np.array([1000, 800, 600, 400, 200, 100, 50, 25])

      # A0 = bi_graph[real_batch_idx]
      A0 = bi_graph
      # count_sc = self.count_sc.type_as(count_st) !!!!!!!!!!!!!
      if self.args.sc_norm:
        log_sc = norm_sc(count_sc)
        log_st = norm_st(count_st)
      else:
        log_sc = count_sc
        log_st = count_st


      # Sample from diffusion
      bi_graph_onehot = F.one_hot(bi_graph.long(), num_classes=2).float()

      bi_graph_t = self.diffusion.sample(bi_graph_onehot, t)
      if self.args.test_data=='PDAC':
        # bi_graph_t = torch.rand_like(bi_graph_t.float())
        # print(bi_graph_t.shape, bi_graph.shape)
        # print(count_sc.shape)
        bi_graph_t = torch.rand(8, count_sc.shape[1], count_st.shape[1])
        t = np.array([1000, 1000, 1000, 1000, 1000, 1000, 1000, 1000])

      t = torch.from_numpy(t).float().view(8)

      device = batch[-2].device
      cell_type_sc, de_conv_st = cell_type_sc.type(torch.float).to(device), de_conv_st.to(device)
      # sc_adj = cell_type_sc @ (cell_type_sc.permute(0,2,1)) - torch.eye(cell_type_sc.shape[1]).to(device)
      # st_adj = knn_matrix(pos.cpu().squeeze(0), self.args.st_k).unsqueeze(0).to(device)
      sc_adj = torch.tensor(0)
      st_adj = torch.tensor(0)
      # Denoise

      bi_graph_0_pred, sc_hat, st_hat = self.forward(
          log_sc.to(bi_graph.device),
          log_st.to(bi_graph.device),
          bi_graph_t.float().to(bi_graph.device),
          sc_adj.float().to(bi_graph.device),
          st_adj.float().to(bi_graph.device),
          t.float().to(bi_graph.device),
          edge_index,
      )

      bi_graph_0_pred_prob = bi_graph_0_pred.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)[..., 1]
      bi_graph_0_pred_prob = bi_graph_0_pred_prob.log().softmax(dim=-1)
      
      if self.args.save_numpy_heatmap:
        # print(bi_graph_0_pred_prob.shape, bi_graph_t.shape, A0.shape, real_batch_idx, t[0])
        self.run_save_numpy_heatmap(bi_graph_0_pred_prob[0].unsqueeze(0), bi_graph_t[0].unsqueeze(0), A0, real_batch_idx, t[0], 'val', 'A0_prob')
        self.run_save_numpy_heatmap(bi_graph_0_pred_prob[1].unsqueeze(0), bi_graph_t[1].unsqueeze(0), A0, real_batch_idx, t[1], 'val', 'A0_prob')
        self.run_save_numpy_heatmap(bi_graph_0_pred_prob[2].unsqueeze(0), bi_graph_t[2].unsqueeze(0), A0, real_batch_idx, t[2], 'val', 'A0_prob')
        self.run_save_numpy_heatmap(bi_graph_0_pred_prob[3].unsqueeze(0), bi_graph_t[3].unsqueeze(0), A0, real_batch_idx, t[3], 'val', 'A0_prob')
        self.run_save_numpy_heatmap(bi_graph_0_pred_prob[4].unsqueeze(0), bi_graph_t[4].unsqueeze(0), A0, real_batch_idx, t[4], 'val', 'A0_prob')
        self.run_save_numpy_heatmap(bi_graph_0_pred_prob[5].unsqueeze(0), bi_graph_t[5].unsqueeze(0), A0, real_batch_idx, t[5], 'val', 'A0_prob')
        self.run_save_numpy_heatmap(bi_graph_0_pred_prob[6].unsqueeze(0), bi_graph_t[6].unsqueeze(0), A0, real_batch_idx, t[6], 'val', 'A0_prob')
        self.run_save_numpy_heatmap(bi_graph_0_pred_prob[7].unsqueeze(0), bi_graph_t[7].unsqueeze(0), A0, real_batch_idx, t[7], 'val', 'A0_prob')

      # Compute loss
      
      # print(bi_graph_0_pred.shape, A0.shape)#([3, 2, 1224, 255])([1, 1224, 255])
      # print(bi_graph_0_pred_prob.permute(0, 2, 1) @ count_sc.float())
      # print(bi_graph_0_pred_prob.shape, A0.shape)
      loss_CE, RMSE_single, RMSE_global, MSE_hard, MSE_soft, count, term, regular_01, KL, cos_dis = loss_t(bi_graph_0_pred_prob, A0, count_sc, count_st, cell_type_sc, self.args)
      loss_CE0, _, RMSE_global0, _, _, count0, term0, regular_010, KL0, _ = loss_t(bi_graph_0_pred_prob[0].unsqueeze(0), A0, count_sc, count_st, cell_type_sc, self.args)
      loss_CE1, _, RMSE_global1, _, _, count1, term1, regular_011, KL1, _ = loss_t(bi_graph_0_pred_prob[1].unsqueeze(0), A0, count_sc, count_st, cell_type_sc, self.args)
      loss_CE2, _, RMSE_global2, _, _, count2, term2, regular_012, KL2, _ = loss_t(bi_graph_0_pred_prob[2].unsqueeze(0), A0, count_sc, count_st, cell_type_sc, self.args)
      loss_CE3, _, RMSE_global3, _, _, count3, term3, regular_013, KL3, _ = loss_t(bi_graph_0_pred_prob[3].unsqueeze(0), A0, count_sc, count_st, cell_type_sc, self.args)
      loss_CE4, _, RMSE_global4, _, _, count4, term4, regular_014, KL4, _ = loss_t(bi_graph_0_pred_prob[4].unsqueeze(0), A0, count_sc, count_st, cell_type_sc, self.args)
      loss_CE5, _, RMSE_global5, _, _, count5, term5, regular_015, KL5, _ = loss_t(bi_graph_0_pred_prob[5].unsqueeze(0), A0, count_sc, count_st, cell_type_sc, self.args)
      loss_CE6, _, RMSE_global6, _, _, count6, term6, regular_016, KL6, _ = loss_t(bi_graph_0_pred_prob[6].unsqueeze(0), A0, count_sc, count_st, cell_type_sc, self.args)
      loss_CE7, _, RMSE_global7, _, _, count7, term7, regular_017, KL7, _ = loss_t(bi_graph_0_pred_prob[7].unsqueeze(0), A0, count_sc, count_st, cell_type_sc, self.args)
    if self.args.fine_tune:
      np.savetxt(self.args.tune_A_file, bi_graph_0_pred_prob[0].cpu().detach().numpy(), delimiter=",")
      

      l1_norm = 0
      for bt in range(self.args.train_sampling):
        l1_norm = l1_norm + torch.linalg.norm(bi_graph_0_pred_prob[bt], ord=1, dim=(0, 1))
      l1_norm = l1_norm/self.args.train_sampling

      # loss_func = nn.CrossEntropyLoss()
      if self.args.decode:
        Reconstruction_loss_sc, Reconstruction_loss_st = Reconstruction_loss(count_sc, count_st, sc_hat, st_hat, self.args)
        loss = loss_CE + self.args.alpha_1 * RMSE_global + self.args.alpha_2 * term + self.args.alpha_3 * l1_norm + self.args.alpha_4 * (Reconstruction_loss_sc + Reconstruction_loss_st) + self.args.alpha_5 * regular_01 + self.args.alpha_6 * KL
        split = 'val'
        metrics = {
            f"{split}/loss": loss,
            f"{split}/MSE_hard": MSE_hard,
            f"{split}/MSE_soft": MSE_soft,
            f"{split}/RMSE":  RMSE_global,
            f"{split}/loss_CE":  loss_CE,
            f"{split}/count":  count,
            f"{split}/matrix_term":  term,
            f"{split}/l1_norm":  l1_norm,
            f"{split}/regular_01":  regular_01,
            f"{split}/kl":  KL,
            f"{split}/Reconstruction_loss_sc":  Reconstruction_loss_sc,
            f"{split}/Reconstruction_loss_st":  Reconstruction_loss_st,
        }
      else:
        loss = self.args.alpha_0 * loss_CE + self.args.alpha_1 * RMSE_global + self.args.alpha_2 * term + self.args.alpha_3 * l1_norm + self.args.alpha_5 * regular_01 + self.args.alpha_6 * KL  + self.args.alpha_7 * cos_dis + self.args.alpha_8 * MSE_soft
        split = 'val'
        metrics = {
            f"{split}/loss": loss,
            f"{split}/MSE_hard": MSE_hard,
            f"{split}/MSE_soft": MSE_soft,
            f"{split}/RMSE":  RMSE_global,
            f"{split}/loss_CE":  loss_CE,
            f"{split}/matrix_term":  term,
            f"{split}/l1_norm":  l1_norm,
            f"{split}/regular_01":  regular_01,
            f"{split}/loss_CE0": loss_CE0,
            f"{split}/RMSE0":  RMSE_global0,
            f"{split}/matrix_term0":  term0,
            f"{split}/regular_010":  regular_010,
            f"{split}/KL0":  KL0,
            f"{split}/loss_CE1": loss_CE1,
            f"{split}/RMSE1":  RMSE_global1,
            f"{split}/matrix_term1":  term1,
            f"{split}/regular_011":  regular_011,
            f"{split}/KL1":  KL1,
            f"{split}/loss_CE2": loss_CE2,
            f"{split}/RMSE2":  RMSE_global2,
            f"{split}/loss_CE3": loss_CE3,
            f"{split}/RMSE3":  RMSE_global3,
            f"{split}/loss_CE4": loss_CE4,
            f"{split}/RMSE4":  RMSE_global4,
            f"{split}/loss_CE5": loss_CE5,
            f"{split}/RMSE5":  RMSE_global5,
            f"{split}/loss_CE6": loss_CE6,
            f"{split}/RMSE6":  RMSE_global6,
            f"{split}/loss_CE7": loss_CE7,
            f"{split}/RMSE7":  RMSE_global7,
            f"{split}/cos_dis":  cos_dis,

        }


      for k, v in metrics.items():
        self.log(k, v, on_epoch=True, sync_dist=True)
      self.log(f"{split}/solved_cost", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        
      # return loss
      return metrics
