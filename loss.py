import torch
import numpy as np
from torch.autograd import Variable

def cal_losscl(x,x_aug):
    batch =x.shape[0]
    x= x.reshape(-1,x.shape[-1])
    x_aug = x_aug.reshape(-1,x.shape[-1])
    T = 0.2
    batch_size, _ = x.size()
    x_abs = x.norm(dim=1)
    x_aug_abs = x_aug.norm(dim=1)
    sim_matrix = torch.einsum('ik,jk->ij', x, x_aug) / torch.einsum('i,j->ij', x_abs, x_aug_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]

    loss_0 = pos_sim / (sim_matrix.sum(dim=0) - pos_sim + 1e-12)
    loss_1 = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + 1e-12)

    loss_0 = - torch.log(loss_0)
    loss_1 = - torch.log(loss_1)
    loss = ((loss_0 + loss_1) / 2.0)
    loss = loss.reshape(batch,-1)
    loss = torch.mean(loss,dim=1)

    return loss

def loss_dualAE(args, label_center, h, sub_f, sub_s,alpha):
    # dist, scores = anomaly_score(label_center,h)
    dist = torch.sum((h - label_center) ** 2, dim=1)
    loss_svdd = torch.mean(dist)
    loss_g = torch.mean(cal_losscl(sub_f, sub_s))
    loss = alpha * loss_svdd + (1-alpha) * loss_g
    scores = alpha * torch.sqrt(dist) + (1-alpha) * cal_losscl(sub_f, sub_s)
    return loss, scores

def anomaly_score(data_center, outputs):
    dist = torch.sum((outputs - data_center) ** 2, dim=1) #把维度加起来了
    scores = torch.sqrt(dist)
    return dist, scores

def init_center_dual(args, data, model, eps=0.001):
    """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
    outputs_A = []
    c_A = torch.zeros(args.hidden_dim).to(args.device)
    # model.eval()
    with torch.no_grad():
        for index, graph in enumerate(data):
            h0 = Variable(graph['feats'].float(), requires_grad=False).cuda(args.device) 
            h0_struct = graph['str_feats'].float().cuda(args.device)  
            adj = Variable(graph['adj'].float(), requires_grad=False).cuda(args.device)
            cls, sub_f, sub_s = model(h0, h0_struct, adj)
            outputs_A.append(torch.mean(cls, dim=0))
        if len(outputs_A) == 1:
            outputs_A = torch.unsqueeze(outputs_A[0], 0)
        else:
            # outputs_S=torch.tensor(outputs_S)
            outputs_A = torch.stack(outputs_A, 0)

        # get the inputs of the batch
        n_samples_A = outputs_A.shape[0]
        c_A = torch.sum(outputs_A, dim=0)
    c_A /= n_samples_A

    # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
    c_A[(abs(c_A) < eps) & (c_A < 0)] = -eps
    c_A[(abs(c_A) < eps) & (c_A > 0)] = eps
    return c_A


def get_radius(dist: torch.Tensor, nu: float):
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    radius = np.quantile(np.sqrt(dist.clone().data.cpu().numpy()), 1 - nu)
    # if radius<0.1:
    #     radius=0.1
    return radius
