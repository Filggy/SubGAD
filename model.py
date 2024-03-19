import torch
import torch.nn as nn
from SoftClustering import SoftClustering
from transformer import *
from einops import repeat

class outlier_model(nn.Module):
    def __init__(self, num_gcn_layer, in_dim, hid1_dim, hid_dim, dim_feedforward, cluster_dim, depth, heads,
                emb_dp, transformer_dp):
        super(outlier_model, self).__init__()

        self.softgcn_feat = SoftClustering(num_gcn_layer, in_dim, hid1_dim, hid_dim, cluster_dim)
        self.softgcn_str = SoftClustering(num_gcn_layer, 18, hid1_dim, hid_dim, cluster_dim)
        
        self.z_g = nn.Parameter(torch.randn(1, 1, hid_dim))
        
        self.dropout = nn.Dropout(emb_dp)
        self.to_z_g = nn.Identity()
        self.encoder_layers = TransformerEncoderLayer(hid_dim, heads, dim_feedforward, transformer_dp)

        self.encoder_feat = TransformerEncoder(self.encoder_layers, depth)
        self.encoder_str = TransformerEncoder(self.encoder_layers, depth)

        self.proj_head = nn.Sequential(nn.Linear(hid_dim, hid_dim), nn.ReLU(inplace=True), nn.Linear(hid_dim, hid_dim))

        self.linear = nn.Linear(2*hid_dim, hid_dim)

    def forward(self, x, x_s, adj):
        b, n, _ = x.shape 

        g_f, substr_tensor_f = self.softgcn_feat(x, adj)
        x = substr_tensor_f

        g_s, substr_tensor_s = self.softgcn_str(x_s, adj)
        x_s = substr_tensor_s
        
        x_fusion = self.linear(torch.cat((x,x_s),dim=2))
        z_g = repeat(self.z_g, '() n d -> b n d', b=b)

        x_final = torch.cat((z_g, x_fusion), dim=1)
        x_final = self.dropout(x_final)
        x_final = self.encoder_feat(x_final)
        x_final = self.to_z_g(x_final[:, 0]) 
        z = self.proj_head(x_final)
        
        return z, substr_tensor_f, substr_tensor_s







    