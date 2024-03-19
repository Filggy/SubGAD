import torch
import torch.nn as nn
from GCNlayer import GraphConv
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim, num_gc_layers=2):
        super().__init__()
        self.num_gc_layers = num_gc_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()

        if num_gc_layers == 1:
            self.convs.append(GraphConv(in_dim, out_dim, True, False))
            bn = nn.BatchNorm1d(out_dim)
            self.bns.append(bn)
        else:
            for i in range(num_gc_layers - 1):
                if i:
                    self.convs.append(GraphConv(hid_dim, hid_dim, True, False))
                else:
                    self.convs.append(GraphConv(in_dim, hid_dim, True, False))
                bn = nn.BatchNorm1d(hid_dim)
                self.bns.append(bn)

            self.convs.append(GraphConv(hid_dim, out_dim, True, False))
            bn = nn.BatchNorm1d(out_dim)
            self.bns.append(bn)

    def forward(self, x, adj):
        for i in range(self.num_gc_layers):
            x = F.relu(self.convs[i](x, adj))
            x = x.permute(0, 2, 1)
            x = self.bns[i](x)
            x = x.permute(0, 2, 1)   
        return x

class SoftClustering(nn.Module):
    def __init__(self, num_gcn_layer, in_dim, hid1_dim, hid2_dim, cluster_dim):
        super(SoftClustering, self).__init__()
        self.emb_block = GCN(in_dim, hid1_dim, hid2_dim, num_gcn_layer)
        self.assign_block = GCN(in_dim, hid2_dim, cluster_dim, num_gcn_layer)

    def forward(self, x, adj):
        embedding_tensor = self.emb_block(x, adj)
        embedding_h = self.assign_block(x, adj)
        embedding_h = nn.Softmax(dim=-1)(embedding_h)
        substr_tensor = torch.matmul(torch.transpose(embedding_h, 1, 2), embedding_tensor)
        
        return embedding_h, substr_tensor
