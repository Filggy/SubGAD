# -*- coding: utf-8 -*-
import sys
import torch
import time
import copy
import random
import argparse
import numpy as np
from numpy.random import seed
from torch.autograd import Variable
# -------my packages-----------------
from graph_sampler import GraphSampler
from model import outlier_model
from load_data import read_graphfile
from loss import loss_dualAE, init_center_dual
from torch.nn.utils import clip_grad_norm_
import util
from util import AverageMeter
# ------------------------------------
from sklearn.metrics import auc, roc_curve
np.set_printoptions(threshold=sys.maxsize) 
np.set_printoptions(linewidth=400)  
import csv
from collections import Counter

def arg_parse():
    parser = argparse.ArgumentParser(description='GASR Arguments.')
    #dataset dependent args
    parser.add_argument('--grad_clip', dest='grad_clip', default=1.0,type=float, help='Gradient clipping.') 
    parser.add_argument('--datadir', dest='datadir', default='/home/lab/GLAD_new/GASR/dataset', help='Directory where benchmark is located')
    parser.add_argument('--dataset', dest='dataset', default='Tox21_MMP', help='dataset name')
    parser.add_argument('--batch_size', type=int, default=2000, help='batch size')
    parser.add_argument('--epochs', type=int, default=30, help='num-of-epoch')
    parser.add_argument('--lr', dest='lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--result_file', default='contrastive_results.csv', help='result csv file saving dir')
    parser.add_argument('--seed', type=list, dest='seed',
                        default=[1, 42, 666, 777, 10086], help='seed')
    parser.add_argument('--coef', type=float, default=1.0)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--step_size', dest='step_size', type=int,default=30, help='step_size') 
    parser.add_argument('--gamma', dest='gamma', type=int,default=0.1, help='gamma')
    parser.add_argument('--scheduler', dest='scheduler', action='store_const',const=False,default=True, help='scheduler')  # 是否需要学习率衰减
    parser.add_argument('--k', type=float,dest='k', default=0.9)
    parser.add_argument("--outlier_label", dest='outlier_label', type=int, default=1,help="outlier_label") 
    parser.add_argument('--normalize', dest='normalize', action='store_const',const=True,default=True, help='Whether adj normalization is used')  # 邻接矩阵是否需要标准化
    parser.add_argument('--alpha', type=float, default=0.5, help='loss balance') 

    
    #graph setting
    parser.add_argument('--feat', type=str, default='node-label', choices=['node-label', 'node-feat'],help='node feature type')
    parser.add_argument('--feature', dest='feature_type', default="default",help='Feature used for encoder. Can be: id, deg')
    
    #SoftClustering
    parser.add_argument('--pool_ratio', type=float, default=0.5)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--hidden_dim2', type=int, default=64)
    parser.add_argument('--num_gcn_layer', type=int, default=3)
    parser.add_argument('--bias', dest='bias', action='store_const',const=True, default=True, help='switch for bias')
    parser.add_argument('--emb_dp', type=float, default=0.)

    #Multi-head Attention
    parser.add_argument('--depth', type=int, default=2, help="the depth of transformer encoder layer")
    parser.add_argument('--heads', type=int, default=2, help="the number of attention heads")
    parser.add_argument('--dim_feedforward', type=int, default=128, help='feedforward dimension of transformer encoder')
    parser.add_argument('--cat', type=bool, default=False)
    parser.add_argument('--transformer_dp', type=float, default=0.)
    
    #device
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train(data_train_loader,data_valid_loader,model,dataname,lr,args,alpha):
    print('------Training--------')
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30,50], gamma=args.gamma)  # 50
    iter = 0
    
    label_center = init_center_dual(args, data_train_loader, model)
    min_auc_val = 0
    best_epoch = 0
    begin_time = time.time()

    for epoch in range(args.epochs):
        min_epoch = -1 
        total_loss_train = 0.0
        model.train()
        grad_norm = AverageMeter()
        loss_avg = AverageMeter()
        for batch_idx, data in enumerate(data_train_loader): 
            optimizer.zero_grad()  
            h0_tr = data['feats'].float().cuda(args.device) 
            h0_tr_struct = data['str_feats'].float().cuda(args.device) 
            adj_tr =data['adj'].float().cuda(args.device)
            cls_tr, sub_f_tr, sub_s_tr = model(h0_tr, h0_tr_struct, adj_tr)
            loss_tr,  score_tr = loss_dualAE(args,label_center,cls_tr, sub_f_tr, sub_s_tr,alpha)
            loss_tr.backward()
            g_norm = clip_grad_norm_(model.parameters(), args.grad_clip)
            grad_norm.update(g_norm)
            iter += 1
            loss_avg.update(loss_tr.item())
            optimizer.step()
            total_loss_train += loss_tr

        if epoch >= 0:
            total_loss_valid = 0.0
            score_valid = []
            y_ = []
            emb_h_v = []

            model.eval() 
            for batch_idx, data in enumerate(data_valid_loader):
                h0_v = Variable(data['feats'].float(), requires_grad=False).cuda(args.device) 
                h0_v_struct = data['str_feats'].float().cuda(args.device)  
                adj_v = Variable(data['adj'].float(), requires_grad=False).cuda(args.device)
                cls_v, sub_f_v, sub_s_v = model(h0_v, h0_v_struct, adj_v)
                loss_svdd_v,  score_v = loss_dualAE(args,label_center,cls_v, sub_f_v, sub_s_v, alpha)
                total_loss_valid += loss_svdd_v
                
                score_v_ = np.array(score_v.cpu().detach())
                score_valid.append(score_v_)
                if data['graph_label'] == args.outlier_label:
                    y_.append(1)
                else:
                    y_.append(0)
                emb_h_v.append(cls_v.cpu().detach().numpy())
            label_valid = np.array(score_valid)
            fpr_ab, tpr_ab, _ = roc_curve(y_, label_valid)
            valid_roc_ab = auc(fpr_ab, tpr_ab)
            t_ = total_loss_valid / (len(label_valid))
            print("epoch:", epoch, "total_loss_train: %.10f " % total_loss_train,"total_loss_valid: %.10f" % t_ ," valid auc", valid_roc_ab)

        if epoch > min_epoch and valid_roc_ab >= min_auc_val:  # 将验证集最小的结果视为最优模型
            min_auc_val = valid_roc_ab
            best_model = copy.deepcopy(model)
            print("valid_roc_ab", valid_roc_ab, "total_loss_valid",
                    total_loss_valid, "min_auc_val", min_auc_val)
            best_epoch = epoch
        scheduler.step()
            
    total_time = time.time() - begin_time
    best_model_path = "/home/lab/GLAD_new/GASR/Model/"+dataname+".pt"
    torch.save(best_model.state_dict(), best_model_path)
    print("best_epoch", best_epoch, "min_auc_val", min_auc_val)
    return best_epoch, best_model_path, label_center,total_time

def test(data_test_loader, model, best_model_path, label_center,dataname, args, alpha):
    print('------Testing---------')
    model.load_state_dict(torch.load(best_model_path))
    loss_test = []
    emb_h_t = []
    y = []
    begin_time = time.time()
    model.eval()
    with torch.no_grad():
        for batch_idx, data in enumerate(data_test_loader):
            h0_t = Variable(data['feats'].float(), requires_grad=False).cuda(args.device)  
            h0_t_struct = data['str_feats'].float().cuda(args.device) 
            adj_t = Variable(data['adj'].float(), requires_grad=False).cuda(args.device)
            cls_t, sub_f_t, sub_s_t = model(h0_t, h0_t_struct, adj_t)
            loss_dist,  score_v = loss_dualAE(args,label_center,cls_t, sub_f_t, sub_s_t, alpha)
            loss_ = np.array(score_v.cpu().detach())
            loss_test.append(loss_)
            if data['graph_label'] == args.outlier_label:
                y.append(1)
            else:
                y.append(0)
            emb_h_t.append(cls_t.cpu().detach().numpy())

        label_test = np.array(loss_test)
        fpr_ab, tpr_ab, _ = roc_curve(y, label_test)
        test_roc_ab = auc(fpr_ab, tpr_ab)
        total_time = time.time() - begin_time
        print('abnormal detection: test auc: {}'.format(test_roc_ab))
    return test_roc_ab,total_time

if __name__ == '__main__':
    
    args = arg_parse()
    seed_list = args.seed

    result_auc = []
    result_raw=[]
    filepath="/home/lab/GLAD_new/GASR/results/result_Tox.csv"
    with open(filepath,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["Model","Dataset","AUC","alpha","OutlierClass","Seed","emb_dp","Batch_Size","Epoch_Num","Optimizer","LR","Hidden_Dim","dim_feedforward","result","Train_Time","Test_Time","TimeStamp"])
    

    result_aucs=[]
    train_total_times=[]
    test_total_times=[]

    for seed in seed_list:
        setup_seed(seed)
        print("dataset:", args.dataset)
        train_graphs,num_graph_labels, max_node_train, = read_graphfile(
            args.datadir, args.dataset+'_training', isTox=True)
        valid_graphs,num_graph_labels, max_node_valid = read_graphfile(
            args.datadir, args.dataset+'_evaluation', isTox=True)
        test_graphs, num_graph_labels,max_node_test = read_graphfile(
            args.datadir, args.dataset+'_testing', isTox=True)

        datanum = len(train_graphs) + \
            len(valid_graphs)+len(test_graphs)
        print("num_graphs:", datanum)
        graphs_train_label = [G.graph['label'] for G in train_graphs]
        label_dist=Counter(graphs_train_label)

        graphs_train = []
        for graph in train_graphs:
            if graph.graph['label'] != args.outlier_label:
                graphs_train.append(graph)

        num_graph_labels = 2
        
        max_nodes_num_train = max([G.number_of_nodes() for G in train_graphs])
        max_nodes_num_valid = max([G.number_of_nodes() for G in valid_graphs])
        max_nodes_num_test = max([G.number_of_nodes() for G in test_graphs])
        max_nodes_num = max(
            [max_nodes_num_train, max_nodes_num_test, max_nodes_num_valid])

        mean_nodes_num_train = np.mean([G.number_of_nodes() for G in train_graphs])
        mean_nodes_num_valid =np.mean([G.number_of_nodes() for G in valid_graphs])
        mean_nodes_num_test = np.mean([G.number_of_nodes() for G in test_graphs])
        avg_num_nodes =np.mean(
            [mean_nodes_num_train, mean_nodes_num_valid, mean_nodes_num_test])
        
        print("max_nodes_num:", max_nodes_num)

        example_node = train_graphs[0].nodes[0]
        if args.feat == 'node-feat' and 'feat_dim' in train_graphs[0].graph:
            print('Using node features')
            input_dim = train_graphs[0].graph['feat_dim']
        elif args.feat == 'node-label' and 'label' in example_node:
            print('Using node labels')
            for G in train_graphs:
                for u in G.nodes():
                    util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
            for G in valid_graphs:
                for u in G.nodes():
                    util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])
            for G in test_graphs:
                for u in G.nodes():
                    util.node_dict(G)[u]['feat'] = np.array(util.node_dict(G)[u]['label'])        
        else:
            args.feature_type = 'deg'

        input_dim = len(example_node['feat'])
        print("input feat_dim: ", input_dim)

        num_train = len(graphs_train)
        num_valid = len(valid_graphs)
        num_test = len(test_graphs)
        print("train, vaild, test graph nums: ", num_train, num_valid, num_test)

        graphs_train_label = [G.graph['label'] for G in graphs_train]
        graphs_valid_label = [G.graph['label'] for G in valid_graphs]
        graphs_test_label = [G.graph['label'] for G in test_graphs]

        dataset_sampler_test = GraphSampler(test_graphs, normalize=args.normalize, max_num_nodes=max_nodes_num)
        data_test_loader = torch.utils.data.DataLoader(dataset_sampler_test,shuffle=False,batch_size=1)

        dataset_sampler_train = GraphSampler(graphs_train, normalize=args.normalize, max_num_nodes=max_nodes_num)
        data_train_loader = torch.utils.data.DataLoader(dataset_sampler_train,shuffle=True,batch_size=args.batch_size)
        
        dataset_sampler_valid = GraphSampler(valid_graphs, normalize=args.normalize, max_num_nodes=max_nodes_num)
        data_valid_loader = torch.utils.data.DataLoader(dataset_sampler_valid,shuffle=False,batch_size=1)

        my_model = outlier_model(args.num_gcn_layer, input_dim, args.hidden_dim2, args.hidden_dim, args.dim_feedforward, int(avg_num_nodes * args.pool_ratio), args.depth, args.heads, args.emb_dp, args.transformer_dp).to(args.device)

        best_epoch, best_model_path, label_center,train_total_time = train(data_train_loader, data_valid_loader, my_model,args.dataset,args.lr,args,args.alpha)
        
        result,test_total_time = test(data_test_loader, my_model,  best_model_path, label_center,args.dataset,args,args.alpha)
        
        result_aucs.append(result)
        train_total_times.append(train_total_time)
        test_total_times.append(test_total_time)
        

    train_total_time_seeds=np.array(train_total_times)
    train_total_time_seeds = np.mean(train_total_times)
        
    test_total_time_seeds=np.array(test_total_times)
    test_total_time_seeds = np.mean(test_total_times)
    
    result_auc = np.array(result_aucs)    
    auc_avg = np.mean(result_auc)
    auc_std = np.std(result_auc)
    
    print('auroc{}, average: {}, std: {}'.format(result_auc, auc_avg, auc_std))

    result_raw=['GASR', args.dataset, str(format(auc_avg, '.4f'))+"±"+str(format(auc_std, '.4f')),args.alpha,args.outlier_label,seed,args.emb_dp,args.batch_size,args.epochs,'AdamW',args.lr,args.hidden_dim,args.dim_feedforward,result_auc,format(train_total_time_seeds, '.4f'),format(test_total_time_seeds, '.4f'),time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())]
    with open(filepath,"a") as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(result_raw)