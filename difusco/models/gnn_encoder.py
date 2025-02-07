import functools
import math

import torch
import torch.nn.functional as F
from torch import nn
from models.nn import (
    linear,
    zero_module,
    normalization,
    timestep_embedding,
)
from torch_sparse import SparseTensor
from torch_sparse import sum as sparse_sum
from torch_sparse import mean as sparse_mean
from torch_sparse import max as sparse_max
import torch.utils.checkpoint as activation_checkpoint

# from utils.sc2st_utils import Aggregate
from models.decoder import Decoder


class GNNLayer(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(GNNLayer, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"

    self.U1 = nn.Linear(hidden_dim, hidden_dim, bias=True) #SC loop
    self.U2 = nn.Linear(hidden_dim, hidden_dim, bias=True) #ST loop

    self.V1 = nn.Linear(hidden_dim, hidden_dim, bias=True) #SC2ST
    self.V2 = nn.Linear(hidden_dim, hidden_dim, bias=True) #ST2SC

    self.W1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.W2 = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.bi_A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.bi_B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.bi_C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.sc_A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.sc_B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.sc_C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.st_A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.st_B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.st_C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h_sc, h_st, bi_e, bi_graph, sc_e, sc_graph, st_e, st_graph, mode="residual", edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          h: Input node features (B x V x H)
          e: Input edge features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          mode: str
        In Sparse version:
          h: Input node features (V x H)
          e: Input edge features (E x H)
          graph: torch_sparse.SparseTensor
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Updated node and edge features
    """
    if not sparse:
      batch_size, num_sc, hidden_dim = h_sc.shape
      num_st = h_st.shape[1]
    else:
      batch_size = None
      num_sc, hidden_dim = h_sc.shape
      num_st = h_st.shape[0]

    h_sc_in, h_st_in = h_sc, h_st
    bi_e_in, sc_e_in, st_e_in = bi_e, sc_e, st_e

    # Linear transformations for node update
    Uh_sc, Uh_st = self.U1(h_sc), self.U2(h_st)  # B x V x H

    if not sparse: 
      # Vh_sc = self.V1(h_sc).unsqueeze(2).expand(-1, -1, num_st, -1)    # B x num_sc x num_st x H
      # Vh_st = self.V2(h_st).unsqueeze(1).expand(-1, num_sc, -1, -1)    # B x num_sc x num_st x H

      # Wh_sc = self.W1(h_sc).unsqueeze(2).expand(-1, -1, num_sc, -1)    # B x num_sc x num_sc x H
      # Wh_st = self.W2(h_st).unsqueeze(1).expand(-1, num_st, -1, -1)    # B x num_st x num_st x H

      Vh_sc = self.V1(h_sc)    # B x num_sc x H
      Vh_st = self.V2(h_st)    # B x num_st x H

      Wh_sc = self.W1(h_sc)    # B x num_sc x H
      Wh_st = self.W2(h_st)    # B x num_st x H

    else:
      Vh = self.V(h[edge_index[1]])  # E x H

    # Linear transformations for edge update and gating
    bi_Ah_sc = self.bi_A(h_sc)  # B x V1 x H, sc
    bi_Bh_st = self.bi_B(h_st)  # B x V2 x H, st
    bi_Ce = self.bi_C(bi_e)  # B x V1 x V2 x H 

    sc_Ah_sc = self.sc_A(h_sc)  # B x V1 x H, sc
    sc_Bh_sc = self.sc_B(h_sc)  # B x V1 x H, sc
    sc_Ce = self.sc_C(sc_e)  # B x V1 x V1 x H

    st_Ah_st = self.st_A(h_st)  # B x V2 x H, st
    st_Bh_st = self.st_B(h_st)  # B x V2 x H, st
    st_Ce = self.st_C(st_e)  # B x V2 x V2 x H 

    # Update edge features and compute edge gates
    if not sparse:
      # print('********', h1.shape, h2.shape, e.shape)
      # print(Ah.shape, Bh.shape,Ce.shape)
      # print(Ah.unsqueeze(1).shape, Bh.unsqueeze(2).shape)
      # print(Ah.repeat(Bh.shape[1], 1, 1, 1).permute(1,2,0,3).shape, Bh.repeat(Ah.shape[1], 1, 1, 1).permute(1,0,2,3).shape)
      # e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H

      # print('bi_e  shape:', bi_Ah_sc.shape, bi_Bh_st.shape, bi_Ce.shape)
      # 'bi_e'  shape: torch.Size([1, 1224, 128]) torch.Size([1, 253, 128]) torch.Size([1, 1224, 253, 128])
      bi_e = bi_Ah_sc.repeat(bi_Bh_st.shape[1], 1, 1, 1).permute(1,2,0,3) + bi_Bh_st.repeat(bi_Ah_sc.shape[1], 1, 1, 1).permute(1,0,2,3) + bi_Ce
      sc_e = sc_Ah_sc.repeat(sc_Bh_sc.shape[1], 1, 1, 1).permute(1,2,0,3) + sc_Bh_sc.repeat(sc_Ah_sc.shape[1], 1, 1, 1).permute(1,0,2,3) + sc_Ce
      st_e = st_Ah_st.repeat(st_Bh_st.shape[1], 1, 1, 1).permute(1,2,0,3) + st_Bh_st.repeat(st_Ah_st.shape[1], 1, 1, 1).permute(1,0,2,3) + st_Ce
      # torch.Size([1, 1224, 255, 128]) torch.Size([1, 1224, 1224, 128]) torch.Size([1, 255, 255, 128])  

    else:
      bi_e = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x H

    bi_gates = torch.sigmoid(bi_e)  # B x V1 x V2 x H 
    sc_gates = torch.sigmoid(sc_e)  # B x V1 x V1 x H 
    st_gates = torch.sigmoid(st_e)  # B x V2 x V2 x H 

    # Update node features
    # print(Uh_sc.shape, self.aggregate(Vh_st, bi_graph, bi_gates, edge_index=edge_index, sparse=sparse, target='st2sc').shape, self.aggregate(Wh_sc, sc_graph, sc_gates, edge_index=edge_index, sparse=sparse, target='sc2sc').shape,'1111111111111')
    
    '''print('test:')
    print('shape for graph:', bi_graph.shape, sc_graph.shape, st_graph.shape)

    h_st2sc = self.aggregate(Vh_st, bi_graph, bi_gates, edge_index=edge_index, sparse=sparse, target='st2sc')
    print('done for st2sc, shape:', h_st2sc.shape)
    h_sc2sc = self.aggregate(Wh_sc, sc_graph, sc_gates, edge_index=edge_index, sparse=sparse, target='sc2sc')
    print('done for sc2sc, shape:', h_sc2sc.shape)

    h_sc2st = self.aggregate(Vh_sc, bi_graph, bi_gates, edge_index=edge_index, sparse=sparse, target='sc2st')
    print('done for sc2st, shape:', h_sc2st.shape)
    h_st2st = self.aggregate(Wh_st, st_graph, st_gates, edge_index=edge_index, sparse=sparse, target='st2st')
    print('done for st2st, shape:', h_st2st.shape)'''

    h_sc = Uh_sc + self.aggregate(Vh_st, bi_graph, bi_gates, edge_index=edge_index, sparse=sparse, target='st2sc') + self.aggregate(Wh_sc, sc_graph, sc_gates, edge_index=edge_index, sparse=sparse, target='sc2sc') # B x V x H
    # print(h_sc.shape, 'h_sc Done')

    h_st = Uh_st + self.aggregate(Vh_sc, bi_graph, bi_gates, edge_index=edge_index, sparse=sparse, target='sc2st') + self.aggregate(Wh_st, st_graph, st_gates, edge_index=edge_index, sparse=sparse, target='st2st') # B x V x H
    
    # Normalize node features
    if not sparse:
      h_sc = self.norm_h(
          h_sc.view(-1, hidden_dim)
      ).view(batch_size, -1, hidden_dim) if self.norm_h else h_sc

      h_st = self.norm_h(
          h_st.view(-1, hidden_dim)
      ).view(batch_size, -1, hidden_dim) if self.norm_h else h_st
    else:
      h = self.norm_h(h) if self.norm_h else h

    # Normalize edge features
    if not sparse:
      bi_e = self.norm_e(
          bi_e.reshape(batch_size * num_sc * num_st, hidden_dim)
      ).reshape(batch_size, num_sc, num_st, hidden_dim) if self.norm_e else bi_e

      sc_e = self.norm_e(
          sc_e.reshape(batch_size * num_sc * num_sc, hidden_dim)
      ).reshape(batch_size, num_sc, num_sc, hidden_dim) if self.norm_e else sc_e

      st_e = self.norm_e(
          st_e.reshape(batch_size * num_st * num_st, hidden_dim)
      ).reshape(batch_size, num_st, num_st, hidden_dim) if self.norm_e else st_e

    else:
      e = self.norm_e(e) if self.norm_e else e

    # Apply non-linearity
    h_sc = F.relu(h_sc)
    h_st = F.relu(h_st)
    bi_e = F.relu(bi_e)
    sc_e = F.relu(sc_e)
    st_e = F.relu(st_e)

    # Make residual connection
    if mode == "residual":
      h_sc = h_sc_in + h_sc
      h_st = h_st_in + h_st
      bi_e = bi_e_in + bi_e
      sc_e = sc_e_in + sc_e
      st_e = st_e_in + st_e

    return h_sc, h_st, bi_e, sc_e, st_e

  def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False, target='st2sc'):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
        In Sparse version:
          Vh: Neighborhood features (E x H)
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    # print(gates.shape,Vh.shape,target,'44444444444444444444444444')
    
    # Vh = gates * Vh  # B x V1 x V2 x H
    # print(Vh.shape)
    # import sys
    # sys.stdout.flush()
    # input('[Press Any Key to start another run]')
    # graph = 
    # edge_index = graph.nonzero(as_tuple=False).t()

    # A = Aggregate()

    # Aggregate neighborhood features
    bsz, n_v1, n_v2, _ = gates.shape

    if target == 'sc2st':
      dim_aggr = 1
      Vh = gates * (Vh.unsqueeze(2).expand(-1, -1, n_v2, -1))
    elif target == 'st2sc':
      dim_aggr = 2
      Vh = gates * (Vh.unsqueeze(1).expand(-1, n_v1, -1, -1))
    elif target == 'sc2sc':
      dim_aggr = 2
      Vh = gates * (Vh.unsqueeze(1).expand(-1, n_v1, -1, -1))
    elif target == 'st2st':
      dim_aggr = 1
      Vh = gates * (Vh.unsqueeze(2).expand(-1, -1, n_v2, -1))
    else:
      return print('wrong!')

    if not sparse:
      if (mode or self.aggregation) == "mean":
        error = 1.0e-8
        Vh = torch.div(graph @ Vh, (graph.sum(dim=dim_aggr) + (error*torch.ones_like(graph.sum(dim=dim_aggr)))).unsqueeze(-1))
        return Vh
      elif (mode or self.aggregation) == "max":
        return torch.max(Vh, dim=dim_aggr)[0] #?
      else:
        # print(graph.shape, Vh.shape, target) 
        #(torch.Size([1, 1224, 255]) torch.Size([1, 1224, 255, 128])
        #(torch.Size([1, 1224, 1224]) torch.Size([1, 1224, 1224, 128]))
        Vh = graph.unsqueeze(3).expand(-1, -1, -1, self.hidden_dim) * Vh
        return torch.sum(Vh, dim=dim_aggr)
    else:
      sparseVh = SparseTensor(
          row=edge_index[0],
          col=edge_index[1],
          value=Vh,
          sparse_sizes=(graph.size(0), graph.size(1))
      )

      if (mode or self.aggregation) == "mean":
        return sparse_mean(sparseVh, dim=1)

      elif (mode or self.aggregation) == "max":
        return sparse_max(sparseVh, dim=1)

      else:
        return sparse_sum(sparseVh, dim=1)


class PositionEmbeddingSine(nn.Module):
  """
  This is a more standard version of the position embedding, very similar to the one
  used by the Attention is all you need paper, generalized to work on images.
  """

  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    y_embed = x[:, :, 0]
    x_embed = x[:, :, 1]
    if self.normalize:
      # eps = 1e-6
      y_embed = y_embed * self.scale
      x_embed = x_embed * self.scale

    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2.0 * (torch.div(dim_t, 2, rounding_mode='trunc')) / self.num_pos_feats)

    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos = torch.cat((pos_y, pos_x), dim=2).contiguous()
    return pos


class ScalarEmbeddingSine(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, :, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
    # print(pos_x.shape, pos_x[:, :, :, 0::2].shape) torch.Size([1, 1224, 255, 128]) torch.Size([1, 1224, 255, 64])
    return pos_x


class ScalarEmbeddingSine1D(nn.Module):
  def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
    super().__init__()
    self.num_pos_feats = num_pos_feats
    self.temperature = temperature
    self.normalize = normalize
    if scale is not None and normalize is False:
      raise ValueError("normalize should be True if scale is passed")
    if scale is None:
      scale = 2 * math.pi
    self.scale = scale

  def forward(self, x):
    x_embed = x
    dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)
    return pos_x


def run_sparse_layer(layer, time_layer, out_layer, adj_matrix, edge_index, add_time_on_edge=True):
  def custom_forward(*inputs):
    x_in = inputs[0]
    e_in = inputs[1]
    time_emb = inputs[2]
    x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
    if add_time_on_edge:
      e = e + time_layer(time_emb)
    else:
      x = x + time_layer(time_emb)
    x = x_in + x
    e = e_in + out_layer(e)
    return x, e
  return custom_forward


class GNNEncoder(nn.Module):
  """Configurable GNN Encoder
  """

  def __init__(self, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               *args, **kwargs):
    super(GNNEncoder, self).__init__()
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    self.node_embed = nn.Linear(hidden_dim, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    if not node_feature_only:
      self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
      self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    else:
      self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    self.bi_e_out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.sc_e_out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.st_e_out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.layers = nn.ModuleList([
        GNNLayer(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_bi_e_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_sc_e_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_st_e_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint

  def dense_forward(self, x_sc, x_st, bi_graph, sc_graph, st_graph, timesteps, edge_index=None):
    """
    Args:
        x: Input node coordinates (B x V x 2)
        graph: Graph adjacency matrices (B x V x V)
        timesteps: Input node timesteps (B)
        edge_index: Edge indices (2 x E)
    Returns:
        Updated edge features (B x V x V)
    """
    # Embed edge features
    del edge_index
    x_sc, x_st = x_sc.repeat(x_st.shape[0], 1, 1).to(torch.float32), x_st.to(torch.float32)
    x_sc = self.node_embed(self.pos_embed(x_sc))
    x_st = self.node_embed(self.pos_embed(x_st))

    sc_graph = sc_graph.repeat(x_sc.shape[0], 1, 1)
    st_graph = st_graph.repeat(x_st.shape[0], 1, 1)

    bi_e = self.edge_embed(self.edge_pos_embed(bi_graph))
    sc_e = self.edge_embed(self.edge_pos_embed(sc_graph))
    st_e = self.edge_embed(self.edge_pos_embed(st_graph))

    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))


    for layer, time_layer, bi_e_out_layer, sc_e_out_layer, st_e_out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_bi_e_out, self.per_layer_sc_e_out, self.per_layer_st_e_out):
      x_sc_in, x_st_in, bi_e_in, sc_e_in, st_e_in = x_sc, x_st, bi_e, sc_e, st_e

      if self.use_activation_checkpoint:
        raise NotImplementedError

      x_sc, x_st, bi_e, sc_e, st_e = layer(x_sc, x_st, bi_e, bi_graph, sc_e, sc_graph, st_e, st_graph, mode="direct")
      if not self.node_feature_only:
        bi_e = bi_e + time_layer(time_emb)[:, None, None, :]
      else:
        x = x + time_layer(time_emb)[:, None, :]
        y = y + time_layer(time_emb)[:, None, :]
      x_sc = x_sc_in + x_sc
      x_st = x_st_in + x_st
      bi_e = bi_e_in + bi_e_out_layer(bi_e)
      sc_e_in = sc_e_in + sc_e_out_layer(sc_e)
      st_e_in = st_e_in + st_e_out_layer(st_e)

    bi_e = self.bi_e_out(bi_e.permute((0, 3, 1, 2)))
    return bi_e

  def sparse_forward(self, x, graph, timesteps, edge_index):
    """
    Args:
        x: Input node coordinates (V x 2)
        graph: Graph edge features (E)
        timesteps: Input edge timestep features (E)
        edge_index: Adjacency matrix for the graph (2 x E)
    Returns:
        Updated edge features (E x H)
    """
    # Embed edge features
    x = self.node_embed(self.pos_embed(x.unsqueeze(0)).squeeze(0))
    e = self.edge_embed(self.edge_pos_embed(graph.expand(1, 1, -1)).squeeze())
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    e = e.reshape((1, x.shape[0], -1, e.shape[-1])).permute((0, 3, 1, 2))
    e = self.out(e).reshape(-1, edge_index.shape[1]).permute((1, 0))
    return e

  def sparse_forward_node_feature_only(self, x, timesteps, edge_index):
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  def sparse_encoding(self, x, e, edge_index, time_emb):
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones_like(edge_index[0].float()),
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    adj_matrix = adj_matrix.to(x.device)

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        single_time_emb = time_emb[:1]

        run_sparse_layer_fn = functools.partial(
            run_sparse_layer,
            add_time_on_edge=not self.node_feature_only
        )

        out = activation_checkpoint.checkpoint(
            run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
            x_in, e_in, single_time_emb
        )
        x = out[0]
        e = out[1]
      else:
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
        if not self.node_feature_only:
          e = e + time_layer(time_emb)
        else:
          x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
    return x, e

  def forward(self,  x_sc, x_st, bi_graph, sc_graph, st_graph, timesteps, edge_index=None):
    if self.node_feature_only:
      if self.sparse:
        return self.sparse_forward_node_feature_only(x, timesteps, edge_index)
      else:
        raise NotImplementedError
    else:
      if self.sparse:
        return self.sparse_forward(x_sc, x_st, bi_graph, sc_graph, st_graph, timesteps, edge_index)
      else:
        return self.dense_forward(x_sc, x_st, bi_graph, sc_graph, st_graph, timesteps, edge_index)


class GNNLayer_light(nn.Module):
  """Configurable GNN Layer
  Implements the Gated Graph ConvNet layer:
      h_i = ReLU ( U*h_i + Aggr.( sigma_ij, V*h_j) ),
      sigma_ij = sigmoid( A*h_i + B*h_j + C*e_ij ),
      e_ij = ReLU ( A*h_i + B*h_j + C*e_ij ),
      where Aggr. is an aggregation function: sum/mean/max.
  References:
      - X. Bresson and T. Laurent. An experimental study of neural networks for variable graphs. In International Conference on Learning Representations, 2018.
      - V. P. Dwivedi, C. K. Joshi, T. Laurent, Y. Bengio, and X. Bresson. Benchmarking graph neural networks. arXiv preprint arXiv:2003.00982, 2020.
  """

  def __init__(self, hidden_dim, aggregation="sum", norm="batch", learn_norm=True, track_norm=False, gated=True):
    """
    Args:
        hidden_dim: Hidden dimension size (int)
        aggregation: Neighborhood aggregation scheme ("sum"/"mean"/"max")
        norm: Feature normalization scheme ("layer"/"batch"/None)
        learn_norm: Whether the normalizer has learnable affine parameters (True/False)
        track_norm: Whether batch statistics are used to compute normalization mean/std (True/False)
        gated: Whether to use edge gating (True/False)
    """
    super(GNNLayer_light, self).__init__()
    self.hidden_dim = hidden_dim
    self.aggregation = aggregation
    self.norm = norm
    self.learn_norm = learn_norm
    self.track_norm = track_norm
    self.gated = gated
    assert self.gated, "Use gating with GCN, pass the `--gated` flag"

    self.U1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V1 = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.U2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.V2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.A = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.B = nn.Linear(hidden_dim, hidden_dim, bias=True)
    self.C = nn.Linear(hidden_dim, hidden_dim, bias=True)

    self.norm_h = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

    self.norm_e = {
        "layer": nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
        "batch": nn.BatchNorm1d(hidden_dim, affine=learn_norm, track_running_stats=track_norm)
    }.get(self.norm, None)

  def forward(self, h1, h2, e, graph, mode="residual", edge_index=None, sparse=False):
    """
    Args:
        In Dense version:
          h: Input node features (B x V x H)
          e: Input edge features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          mode: str
        In Sparse version:
          h: Input node features (V x H)
          e: Input edge features (E x H)
          graph: torch_sparse.SparseTensor
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Updated node and edge features
    """
    # print(h1.shape, h2.shape, e.shape, graph.shape)
    if not sparse:
      _, num_sc, hidden_dim = h1.shape
      num_st = h2.shape[1]
      batch_size = e.shape[0]
    else:
      batch_size = None
      num_sc, hidden_dim = h1.shape
      num_st = h2.shape[0]

    h1_in, h2_in = h1, h2
    e_in = e

    # Linear transformations for node update
    Uh1, Uh2 = self.U1(h1), self.U2(h2)  # B x V x H

    if not sparse: 
      Vh1 = self.V1(h1).unsqueeze(2).expand(-1, -1, num_st, -1)    # B x V x V x H
      Vh2 = self.V2(h2).unsqueeze(1).expand(-1, num_sc, -1, -1)
    else:
      Vh = self.V(h[edge_index[1]])  # E x H

    # Linear transformations for edge update and gating
    Ah = self.A(h1)  # B x V1 x H, source
    Bh = self.B(h2)  # B x V2 x H, target
    Ce = self.C(e)  # B x V x V x H / E x H

    # Update edge features and compute edge gates
    if not sparse:
      # print('********', h1.shape, h2.shape, e.shape)
      # print(Ah.shape, Bh.shape,Ce.shape)
      # print(Ah.unsqueeze(1).shape, Bh.unsqueeze(2).shape)
      # print(Ah.repeat(Bh.shape[1], 1, 1, 1).permute(1,2,0,3).shape, Bh.repeat(Ah.shape[1], 1, 1, 1).permute(1,0,2,3).shape)
      # e = Ah.unsqueeze(1) + Bh.unsqueeze(2) + Ce  # B x V x V x H
      # print(Ah.shape, Bh.shape)
      e = Ah.repeat(Bh.shape[1], 1, 1, 1).permute(1,2,0,3) + Bh.repeat(Ah.shape[1], 1, 1, 1).permute(1,0,2,3) + Ce
    else:
      e = Ah[edge_index[1]] + Bh[edge_index[0]] + Ce  # E x H

    gates = torch.sigmoid(e)  # B x V x V x H / E x H
    # Update node features
    h1 = Uh1 + self.aggregate(Vh2, graph, gates, edge_index=edge_index, sparse=sparse, target='sc')  # B x V x H
    h2 = Uh2 + self.aggregate(Vh1, graph, gates, edge_index=edge_index, sparse=sparse, target='st')
    
    # Normalize node features
    h1 = self.norm_h(
        h1.view(-1, hidden_dim)
    ).view(batch_size, -1, hidden_dim) if self.norm_h else h

    h2 = self.norm_h(
        h2.view(-1, hidden_dim)
    ).view(batch_size, -1, hidden_dim) if self.norm_h else h

    # Normalize edge features
    if not sparse:
      e = self.norm_e(
          e.reshape(batch_size * num_sc * num_st, hidden_dim)
      ).reshape(batch_size, num_sc, num_st, hidden_dim) if self.norm_e else e
    else:
      e = self.norm_e(e) if self.norm_e else e

    # Apply non-linearity
    h1 = F.relu(h1)
    h2 = F.relu(h2)
    e = F.relu(e)

    # Make residual connection
    if mode == "residual":
      h1 = h1_in + h1
      h2 = h2_in + h2
      e = e_in + e

    return h1, h2, e

  def aggregate(self, Vh, graph, gates, mode=None, edge_index=None, sparse=False, target='st'):
    """
    Args:
        In Dense version:
          Vh: Neighborhood features (B x V x V x H)
          graph: Graph adjacency matrices (B x V x V)
          gates: Edge gates (B x V x V x H)
          mode: str
        In Sparse version:
          Vh: Neighborhood features (E x H)
          graph: torch_sparse.SparseTensor (E edges for V x V adjacency matrix)
          gates: Edge gates (E x H)
          mode: str
          edge_index: Edge indices (2 x E)
        sparse: Whether to use sparse tensors (True/False)
    Returns:
        Aggregated neighborhood features (B x V x H)
    """
    # Perform feature-wise gating mechanism
    # Vh = gates * Vh  # B x V x V x H
    graph = torch.repeat_interleave(graph.unsqueeze(3), repeats=self.hidden_dim, dim=-1)
    if self.aggregation == 'graph_sum':
      Vh = gates * Vh * graph
    else:
      Vh = gates * Vh
    # Enforce graph structure through masking
    # Vh[graph.unsqueeze(-1).expand_as(Vh)] = 0

    # Aggregate neighborhood features
    if target == 'st':
      dim_aggr = 1
    elif target == 'sc':
      dim_aggr = 2
    else:
      return print('wrong!')

    if not sparse:
      if (mode or self.aggregation) == "mean":
        return torch.sum(Vh, dim=dim_aggr) / (torch.sum(graph, dim=2).unsqueeze(-1).type_as(Vh))
      elif (mode or self.aggregation) == "max":
        return torch.max(Vh, dim=dim_aggr)[0]
      else:
        # print(graph.shape, graph.sum(dim_aggr), torch.sum(Vh, dim=dim_aggr), torch.sum(Vh, dim=dim_aggr).shape)
        return torch.sum(Vh, dim=dim_aggr)

      
class GNNEncoder_light(nn.Module):
  """Configurable GNN Encoder
  """

  def __init__(self, param_args, n_layers, hidden_dim, out_channels=1, aggregation="sum", norm="layer",
               learn_norm=True, track_norm=False, gated=True,
               sparse=False, use_activation_checkpoint=False, node_feature_only=False,
               *args, **kwargs):
    super(GNNEncoder_light, self).__init__()
    self.args = param_args
    self.sparse = sparse
    self.node_feature_only = node_feature_only
    self.hidden_dim = hidden_dim
    time_embed_dim = hidden_dim // 2
    self.node_embed = nn.Linear(2000, hidden_dim)
    self.edge_embed = nn.Linear(hidden_dim, hidden_dim)

    if not node_feature_only:
      self.pos_embed = PositionEmbeddingSine(hidden_dim // 2, normalize=True)
      self.edge_pos_embed = ScalarEmbeddingSine(hidden_dim, normalize=False)
    else:
      self.pos_embed = ScalarEmbeddingSine1D(hidden_dim, normalize=False)
    self.time_embed = nn.Sequential(
        linear(hidden_dim, time_embed_dim),
        nn.ReLU(),
        linear(time_embed_dim, time_embed_dim),
    )
    self.out = nn.Sequential(
        normalization(hidden_dim),
        nn.ReLU(),
        # zero_module(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=True)
        # ),
    )

    self.layers = nn.ModuleList([
        GNNLayer_light(hidden_dim, aggregation, norm, learn_norm, track_norm, gated)
        for _ in range(n_layers)
    ])

    self.time_embed_layers = nn.ModuleList([
        nn.Sequential(
            nn.ReLU(),
            linear(
                time_embed_dim,
                hidden_dim,
            ),
        ) for _ in range(n_layers)
    ])

    self.per_layer_out = nn.ModuleList([
        nn.Sequential(
          nn.LayerNorm(hidden_dim, elementwise_affine=learn_norm),
          nn.SiLU(),
          zero_module(
              nn.Linear(hidden_dim, hidden_dim)
          ),
        ) for _ in range(n_layers)
    ])
    self.use_activation_checkpoint = use_activation_checkpoint
    self.Decoder = Decoder(self.hidden_dim)

  def dense_forward(self, x, y, graph, timesteps, edge_index=None):
    """
    Args:
        x: Input node coordinates (B x V x 2)
        graph: Graph adjacency matrices (B x V1 x V2)
        timesteps: Input node timesteps (B)
        edge_index: Edge indices (2 x E)
    Returns:
        Updated edge features (B x V x V)
    """
    # Embed edge features
    del edge_index
    # x = x.repeat(y.shape[0], 1, 1)
    x, y = x.to(torch.float32), y.to(torch.float32)

    if x.shape[-1]<2000:
      w = self.node_embed.weight.data[:,0:x.shape[-1]].T
      # print(w, self.node_embed.weight.shape)
      if self.node_embed.weight.data.shape[-1]<2000:
        x = self.node_embed(x)
        y = self.node_embed(y)
      else:
        x = x @ w
        y = y @ w
    else:
      x = self.node_embed(x)
      y = self.node_embed(y)
    e = self.edge_embed(self.edge_pos_embed(graph))

    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    # graph = torch.ones_like(graph).long()

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, y_in, e_in = x, y, e

      if self.use_activation_checkpoint:
        raise NotImplementedError

      
      e = e + time_layer(time_emb)[:, None, None, :]
      x, y, e = layer(x, y, e, graph, mode="no_residual")
      

      
      # x = x + time_layer(time_emb)[:, None, :]
      # y = y + time_layer(time_emb)[:, None, :]

      x = x_in + out_layer(x)
      y = y_in + out_layer(y)
      e = e_in + out_layer(e)
    e = self.out(e.permute((0, 3, 1, 2)))

    if self.args.decode:
      x, y = self.Decoder(x, y)
    return e, x, y

  def sparse_forward_node_feature_only(self, x, timesteps, edge_index):
    x = self.node_embed(self.pos_embed(x))
    x_shape = x.shape
    e = torch.zeros(edge_index.size(1), self.hidden_dim, device=x.device)
    time_emb = self.time_embed(timestep_embedding(timesteps, self.hidden_dim))
    edge_index = edge_index.long()

    x, e = self.sparse_encoding(x, e, edge_index, time_emb)
    x = x.reshape((1, x_shape[0], -1, x.shape[-1])).permute((0, 3, 1, 2))
    x = self.out(x).reshape(-1, x_shape[0]).permute((1, 0))
    return x

  def sparse_encoding(self, x, e, edge_index, time_emb):
    adj_matrix = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=torch.ones_like(edge_index[0].float()),
        sparse_sizes=(x.shape[0], x.shape[0]),
    )
    adj_matrix = adj_matrix.to(x.device)

    for layer, time_layer, out_layer in zip(self.layers, self.time_embed_layers, self.per_layer_out):
      x_in, e_in = x, e

      if self.use_activation_checkpoint:
        single_time_emb = time_emb[:1]

        run_sparse_layer_fn = functools.partial(
            run_sparse_layer,
            add_time_on_edge=not self.node_feature_only
        )

        out = activation_checkpoint.checkpoint(
            run_sparse_layer_fn(layer, time_layer, out_layer, adj_matrix, edge_index),
            x_in, e_in, single_time_emb
        )
        x = out[0]
        e = out[1]
      else:
        x, e = layer(x_in, e_in, adj_matrix, mode="direct", edge_index=edge_index, sparse=True)
        if not self.node_feature_only:
          e = e + time_layer(time_emb)
        else:
          x = x + time_layer(time_emb)
        x = x_in + x
        e = e_in + out_layer(e)
    return x, e

  def forward(self, x, y, graph, timesteps, edge_index=None):
    if self.node_feature_only:
      if self.sparse:
        return self.sparse_forward_node_feature_only(x, timesteps, edge_index)
      else:
        raise NotImplementedError
    else:
      if self.sparse:
        return self.sparse_forward(x, graph, timesteps, edge_index)
      else:
        return self.dense_forward(x, y, graph, timesteps, edge_index)