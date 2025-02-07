import torch
import torch.nn.functional as F
from torch import nn



class Decoder(nn.Module):

  def __init__(self, hidden_dim):

    super(Decoder, self).__init__()
    self.hidden_dim = hidden_dim

    self.decoder = nn.Sequential(
        # nn.BatchNorm1d(hidden_dim, affine=True, track_running_stats=True),
        # nn.ReLU(),
        nn.Linear(hidden_dim, 2*hidden_dim, bias=True),
        # nn.BatchNorm1d(2*hidden_dim, affine=True, track_running_stats=True),
        nn.ReLU(),
        nn.Linear(2*hidden_dim, 2*hidden_dim, bias=True),
        nn.ReLU(),
        nn.ReLU(),
        nn.Linear(2*hidden_dim, 2000),
    )

   
  def forward(self, h_sc, h_st):
    bsz, sc_num, dim = h_sc.shape
    # h_sc, h_st = h_sc.reshape(-1, dim), h_st.reshape(-1, dim)
    h_sc, h_st = self.decoder(h_sc), self.decoder(h_st)
    # h_sc, h_st = h_sc.reshape(bsz, -1, 2000), h_st.reshape(bsz, -1, 2000)

    return h_sc, h_st
