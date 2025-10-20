import os
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.nn.functional as F

def cos_sim(Y_hat, Y):

  Y_hat = Y_hat / torch.norm(Y_hat, dim=-1, keepdim=True) 
  Y = Y / torch.norm(Y, dim=-1, keepdim=True) 
  similarity_matrix = Y_hat@Y.permute(0,2,1)
  sim = torch.diagonal(similarity_matrix)

  return sim


def file_name(FLAGS, split='train.pt'):
  num_sc, num_st, num_gens, num_celltype = FLAGS.N, FLAGS.S, FLAGS.gen_num, FLAGS.cell_type
  data_path = FLAGS.data_path
  if FLAGS.source == 'real':
    log_name = f"{FLAGS.source}_{FLAGS.data}_{FLAGS.num_train}_"
  elif FLAGS.source == 'simulation':
    log_name = f"{FLAGS.source}_count_ST_N{FLAGS.N}_S{FLAGS.S}_"
  name =  data_path + log_name + split
  
  return os.path.join(data_path, name)

def test_file(FLAGS):

  data_path = FLAGS.data_path
  test_name = FLAGS.test_file

  name =  data_path + test_name
  
  return os.path.join(data_path, name)

def norm_sc(x):
  bsz, S, G = x.shape[0], x.shape[1], x.shape[2]
  x = x.reshape(-1, G)
  log_x = sc.AnnData(x.cpu().detach().numpy().copy())
  sc.pp.normalize_total(log_x, target_sum=1e1)
  sc.pp.log1p(log_x)
  log_X = torch.tensor(log_x.X).reshape(bsz, S, G)
  return log_X

def norm_st(y):
  bsz, S, G = y.shape[0], y.shape[1], y.shape[2]
  y = y.reshape(-1, G)
  log_y = sc.AnnData(y.cpu().detach().numpy().copy())
  sc.pp.normalize_total(log_y, target_sum=1e1)
  sc.pp.log1p(log_y)
  log_Y = torch.tensor(log_y.X).reshape(bsz, S, G)
  return log_Y

def loss_t(At, adj_matrix, count_sc, count_st, cell_type_sc, FLAGS):
  num_sc, num_st = At.shape[1], At.shape[2]
  adj_matrix = adj_matrix.repeat(At.shape[0],1,1)
  count_sc = count_sc.squeeze(0)

  device = At.device
  d = (torch.ones(num_st)/num_st).unsqueeze(0).repeat(At.shape[0],1).to(device)
  At_m = At.sum(1)/num_sc
  adj_matrix, count_sc, count_st, cell_type_sc = adj_matrix.to(device), count_sc.to(device), count_st.to(device), cell_type_sc.to(device)
  At0_hat, At1_hat = (1. - At.float()).unsqueeze(1), At.float().unsqueeze(1)
  At_hat = torch.cat((At0_hat, At1_hat),1)
  weight = torch.tensor([1.0, 1000.0]).to(device)
  CrossEntropy = nn.CrossEntropyLoss(weight=weight)
  # print(At_hat.shape, adj_matrix.shape) #num_smaple,2,1224,252

  loss_CE = CrossEntropy(At_hat, adj_matrix.long())

  # At_pred_prob = At_hat.permute((0, 2, 3, 1)).contiguous().softmax(dim=-1)[..., 1]
  At_pred_prob = At
  
  # At_hard = (At_pred_prob == At_pred_prob.max(dim=-1, keepdim=True)[0]).to(dtype=torch.float)
  At_hard = At.round().long()

  RMSE_single, RMSE_global = RMSE_deconv(deconv(At, FLAGS, cell_type_sc).squeeze(),deconv(adj_matrix, FLAGS, cell_type_sc).squeeze())
  
  Y_hat_hard = At_pred_prob.round().long().permute(0, 2, 1).float() @ count_sc.float()
  Y_hat_soft = At_pred_prob.permute(0, 2, 1) @ count_sc.float()


  Y = count_st.float().repeat(At.shape[0],1,1)

  MSELoss = nn.MSELoss(reduction='mean')
  # print(Y_hat_hard.shape, Y.shape)
  MSE_hard = MSELoss(Y_hat_hard, Y)
  MSE_soft = MSELoss(Y_hat_soft, Y)

  count = torch.sum(At[0], dim=1)
  count = torch.mean(count)

  num_st = At.shape[-1]
  num_sc = At.shape[-2]
  # print(At.shape, torch.ones(num_st).shape)
  # print((At @ torch.ones(num_st).to(device)).shape, torch.ones(num_st).repeat(At.shape[0],1).shape)
  matrix_term = (torch.norm(At @ torch.ones(num_st).to(device) - torch.ones(num_sc).repeat(At.shape[0],1).to(device)))/num_st

  regular_01 = torch.norm(At - At * At)/num_sc
  if FLAGS.train_sampling>1:
    At_m, d = At_m.reshape(-1), d.reshape(-1)
  KL = F.kl_div(At_m.log(), d, reduction='batchmean')

  sim = cos_sim(Y_hat_soft, Y)
  # print(sim.T)
  cos_dis = (1 - sim).mean()
  
  # print(KL)
  # print(At_m)
  # print(d)

  # print(loss_CE.shape, RMSE_single.shape, RMSE_global.shape, MSE_hard.shape, MSE_soft.shape, count.shape)
  # print(loss_CE, RMSE_single, RMSE_global, MSE_hard, MSE_soft, count)

  return loss_CE, RMSE_single, RMSE_global, MSE_hard, MSE_soft, count, matrix_term, regular_01, KL, cos_dis

def Reconstruction_loss(sc, st, sc_hat, st_hat, FLAGS):
  device = sc_hat.device
  sc, st = sc.to(device), st.to(device)
  L_sc, L_st = 0, 0
  for bt in range(FLAGS.train_sampling):
    L_sc = L_sc + torch.linalg.norm(sc.repeat(2,1,1)[bt]-sc_hat[bt], ord=2, dim=(0, 1))
    L_st = L_st + torch.linalg.norm(st.repeat(2,1,1)[bt]-st_hat[bt], ord=2, dim=(0, 1))
  L_sc, L_st =  L_sc / FLAGS.train_sampling, L_st / FLAGS.train_sampling
 
  return L_sc, L_st

  # return CE, mse_hard, mse_soft, Count

def deconv(A, args, C):
  if C is False:
    if args.test_data == 'seqFISH':
      path_cell_type = '/data/jjingliu/seqFISH/cell_types.csv'
      id_cell_type = np.loadtxt(open(path_cell_type,"rb"),delimiter=",",skiprows=0,usecols=(1),dtype=str)
    elif args.test_data == 'merfish_part_20':
      path_cell_type = '/data/jjingliu/MERFISH_part_20/cell_type/cell_type_0.csv'
      id_cell_type = np.loadtxt(open(path_cell_type,"rb"),delimiter=",",skiprows=0,usecols=(0),dtype=str)
  

    le = LabelEncoder()
    id_cell_type = le.fit_transform(id_cell_type)
    num_types = id_cell_type.max() + 1

    cell_type = F.one_hot(torch.tensor(id_cell_type), num_classes=num_types).type(torch.cuda.FloatTensor)

  else:
    cell_type = C.type(torch.cuda.FloatTensor)
    num_types = cell_type.shape[-1]

  count_type = A.type(torch.cuda.FloatTensor).permute(0, 2, 1) @ cell_type #(bsz, S, N)*(N, num_type)->(bsz, S, num_type)
  count_raw = count_type.sum(-1).unsqueeze(2).repeat(1, 1, num_types) #每一spot里面有多少细胞(bsz, S, num_type)
  de_conv = (count_type / (count_raw + 1.0e-8)) #spot里的细胞成分
  # print(cell_type.shape, count_type.shape, count_raw.shape, de_conv.shape, de_conv.sum(-1)) #assert == 1

  return de_conv

def prop(sc_ct):
  # sc_ct_sum = np.expand_dims(sc_ct.sum(-1), axis = 1)
  sc_ct_sum = sc_ct.sum(-1, keepdims=True)
  num_types = sc_ct.shape[-1]
  sc_ct_raw = np.repeat(sc_ct_sum, num_types, axis = 1)
  p = (sc_ct / (sc_ct_raw  + 1.0e-4))

  return p


def RMSE_deconv(output, target_data):

  # print(type(output), type(output)=="<class 'numpy.ndarray'>", type(output)=='numpy.ndarray', type(output)==np.ndarray)
  if type(output)==np.ndarray:
    output, target_data = torch.tensor(output), torch.tensor(target_data)
  else:
    output, target_data = output.clone().detach().requires_grad_(True), target_data.clone().detach().requires_grad_(True)
  criterion_none = nn.MSELoss(reduction='none')
  loss_single = torch.sqrt(criterion_none(output, target_data).mean(0))
  
  criterion = nn.MSELoss(reduction='mean')
  loss_global = torch.sqrt(criterion(output, target_data))

  return loss_single, loss_global


