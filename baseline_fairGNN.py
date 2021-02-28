import dgl
import time
import tqdm
import ipdb
import argparse
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.utils import dropout_adj, convert

import networkx as nx
from fairgnn_utils import *
from models import *
from aif360.sklearn.metrics import consistency_score as cs


def train(model, x, edge_index, labels, idx_train, sens, idx_sens_train):
    model.train()

    train_g_loss = 0
    train_a_loss = 0

    ### update E, G
    model.adv.requires_grad_(False)
    optimizer_G.zero_grad()

    s = model.estimator(x, edge_index)
    h = model.GNN(x, edge_index)
    y = model.classifier(h)

    s_g = model.adv(h)
    s_score_sigmoid = torch.sigmoid(s.detach())
    s_score = s.detach()
    s_score[idx_train]=sens[idx_train].unsqueeze(1).float()
    y_score = torch.sigmoid(y)
    cov =  torch.abs(torch.mean((s_score_sigmoid[idx_train] - torch.mean(s_score_sigmoid[idx_train])) * (y_score[idx_train] - torch.mean(y_score[idx_train]))))
    
    cls_loss = criterion(y[idx_train], labels[idx_train].unsqueeze(1).float())
    adv_loss = criterion(s_g[idx_train], s_score[idx_train])
    G_loss = cls_loss  + args.alpha * cov - args.beta * adv_loss
    G_loss.backward()
    optimizer_G.step()

    ## update Adv
    model.adv.requires_grad_(True)
    optimizer_A.zero_grad()
    s_g = model.adv(h.detach())
    A_loss = criterion(s_g[idx_train], s_score[idx_train])
    A_loss.backward()
    optimizer_A.step()
    return G_loss.detach(), A_loss.detach()


def fair_metric(pred, labels, sens):
    idx_s0 = sens==0
    idx_s1 = sens==1
    idx_s0_y1 = np.bitwise_and(idx_s0, labels==1)
    idx_s1_y1 = np.bitwise_and(idx_s1, labels==1)
    parity = abs(sum(pred[idx_s0])/sum(idx_s0)-sum(pred[idx_s1])/sum(idx_s1))
    equality = abs(sum(pred[idx_s0_y1])/sum(idx_s0_y1)-sum(pred[idx_s1_y1])/sum(idx_s1_y1))
    return parity.item(), equality.item()


if __name__ == '__main__':

    # Training settings
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='Disables CUDA training.')
    parser.add_argument('--fastmode', action='store_true', default=False,
                        help='Validate during training pass.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--epochs', type=int, default=2000,
                        help='Number of epochs to train.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=32,
                        help='Number of hidden units of the sensitive attribute estimator')
    parser.add_argument('--dropout', type=float, default=.5,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--alpha', type=float, default=4,
                        help='The hyperparameter of alpha')
    parser.add_argument('--beta', type=float, default=0.01,
                        help='The hyperparameter of beta')
    parser.add_argument('--model', type=str, default="GAT",
                        help='the type of model GCN/GAT')
    parser.add_argument('--dataset', type=str, default='pokec_n',
                        choices=['pokec_z','pokec_n','nba', 'german', 'bail', 'credit'])
    parser.add_argument('--num-hidden', type=int, default=32,
                        help='Number of hidden units of classifier.')
    parser.add_argument("--num-heads", type=int, default=1,
                            help="number of hidden attention heads")
    parser.add_argument("--num-out-heads", type=int, default=1,
                        help="number of output attention heads")
    parser.add_argument("--num-layers", type=int, default=1,
                        help="number of hidden layers")
    parser.add_argument("--residual", action="store_true", default=False,
                        help="use residual connection")
    parser.add_argument("--attn-drop", type=float, default=.0,
                        help="attention dropout")
    parser.add_argument('--negative-slope', type=float, default=0.2,
                        help="the negative slope of leaky relu")
    parser.add_argument('--acc', type=float, default=0.688,
                        help='the selected FairGNN accuracy on val would be at least this high')
    parser.add_argument('--roc', type=float, default=0.745,
                        help='the selected FairGNN ROC score on val would be at least this high')
    parser.add_argument('--sens_number', type=int, default=200,
                        help="the number of sensitive attributes")
    parser.add_argument('--label_number', type=int, default=500,
                        help="the number of labels")
    parser.add_argument('--run', type=int, default=0,
                        help="kth run of the model")
    parser.add_argument('--pretrained', type=bool, default=False,
                        help="load a pretrained model")

    args = parser.parse_known_args()[0]
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(args)

    #%%
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.run)

    # Load data
    # print(args.dataset)

    if args.dataset == 'german':
        dataset = 'german'
        sens_attr = "Gender"
        predict_attr = "GoodCustomer"
        label_number = 100
        sens_number = 100
        path = "./dataset/german"
        test_idx = True        
        adj, features, labels, idx_train, idx_val, idx_test,sens, idx_sens_train = load_german(dataset,
                                                                                    sens_attr,
                                                                                    predict_attr,
                                                                                    path=path,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number)
    # Load credit_scoring dataset
    elif args.dataset == 'credit':
        sens_attr = "Age"  # column number after feature process is 1
        sens_idx = 1
        predict_attr = 'NoDefaultNextMonth'
        label_number = 6000
        sens_number = 6000
        path_credit = "./dataset/credit"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_credit(args.dataset, sens_attr, 
                                                                                     predict_attr, path=path_credit, 
                                                                                     label_number=label_number, 
                                                                                     sens_number=sens_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    # Load bail dataset
    elif args.dataset == 'bail':
        sens_attr = "WHITE"  # column number after feature process is 0
        sens_idx = 0
        predict_attr = "RECID"
        label_number = 100
        sens_number = 100
        path_bail = "./dataset/bail"
        adj, features, labels, idx_train, idx_val, idx_test, sens, idx_sens_train = load_bail(args.dataset, sens_attr, 
                                                                                    predict_attr, path=path_bail,
                                                                                    label_number=label_number,
                                                                                    sens_number=sens_number)
        norm_features = feature_norm(features)
        norm_features[:, sens_idx] = features[:, sens_idx]
        features = norm_features

    edge_index = convert.from_scipy_sparse_matrix(adj)[0]
     
    # Model and optimizer
    model = FairGNN(nfeat = features.shape[1], args = args).to(device)
    if args.cuda:
        model.cuda()
        features = features.cuda()
        edge_index = edge_index.cuda()
        labels = labels.cuda()
        sens = sens.cuda()
        idx_sens_train = idx_sens_train.cuda()

    from sklearn.metrics import accuracy_score,roc_auc_score,recall_score,f1_score

    # Train model
    t_total = time.time()
    best_result = {}
    best_fair = 100
    G_params = list(model.GNN.parameters()) + list(model.classifier.parameters()) + list(model.estimator.parameters())
    optimizer_G = torch.optim.Adam(G_params, lr = args.lr, weight_decay = args.weight_decay)
    optimizer_A = torch.optim.Adam(model.adv.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    best_roc_val = 0
    for epoch in range(args.epochs):
        t = time.time()

        # model.train()
        loss = train(model, features, edge_index, labels, idx_train, sens, idx_sens_train)
        model.eval()
        output, ss, z = model(features, edge_index)
        output_preds = (output.squeeze()>0).type_as(labels)
        ss_preds = (ss.squeeze()>0).type_as(labels)

        # Store accuracy
        acc_val = accuracy(output_preds[idx_val], labels[idx_val]).item()
        roc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
        acc_sens = accuracy(ss_preds[idx_test], sens[idx_test]).item()
        parity_val, equality_val = fair_metric(output_preds[idx_val].cpu().numpy(), labels[idx_val].cpu().numpy(), sens[idx_val].cpu().numpy())

        acc_test = accuracy(output_preds[idx_test], labels[idx_test]).item()
        roc_test = roc_auc_score(labels[idx_test].cpu().numpy(), output[idx_test].detach().cpu().numpy())
        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].cpu().numpy())

        # if epoch % 100 == 0:
        #     print('Epoch: {:04d}'.format(epoch+1), 'acc_val: {:.4f}'.format(acc_val), "roc_val: {:.4f}".format(roc_val), "parity_val: {:.4f}".format(parity_val), "equality: {:.4f}".format(equality_val))  

        if roc_val > best_roc_val:
            best_roc_val = roc_val
            best_result['acc'] = acc_test
            best_result['roc'] = roc_test
            best_result['parity'] = parity
            best_result['equality'] = equality    

            # SaVE models
            torch.save(model.state_dict(), f'./fairgnn_model_{(args.run+1):02d}.pth')
            out_preds = output.squeeze()
            out_preds = (out_preds>0).type_as(labels)

            # print("=================================")
            # print('Epoch: {:04d}'.format(epoch+1),
            #     'cov: {:.4f}'.format(cov.item()),
            #     'cls: {:.4f}'.format(cls_loss.item()),
            #     'adv: {:.4f}'.format(adv_loss.item()),
            #     'acc_val: {:.4f}'.format(acc_val.item()),
            #     "roc_val: {:.4f}".format(roc_val),
            #     "parity_val: {:.4f}".format(parity_val),
            #     "equality: {:.4f}".format(equality_val))
            # print("Test:",
            #         "accuracy: {:.4f}".format(acc_test.item()),
            #         "roc: {:.4f}".format(roc_test),
            #         "acc_sens: {:.4f}".format(acc_sens),
            #         "parity: {:.4f}".format(parity),
            #         "equality: {:.4f}".format(equality))

    # print("Optimization Finished!")
    # print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # print('============performace on test set=============')
    if len(best_result) > 0:
        # Load best weights
        model.load_state_dict(torch.load(f'./fairgnn_model_{(args.run+1):02d}.pth'))
        model.eval()
        output, _, _ = model(features, edge_index)
        output_preds = (output.squeeze()>0).type_as(labels)
        ss_preds = (ss.squeeze()>0).type_as(labels)
        noisy_features = features.clone() + torch.ones(features.shape).normal_(0, 1).to(device)
        noisy_output, _, _ = model(noisy_features.to(device), edge_index.to(device))
        noisy_output_preds = (noisy_output.squeeze()>0).type_as(labels)
        auc_roc_test = roc_auc_score(labels.cpu().numpy()[idx_test], output.detach().cpu().numpy()[idx_test])
        robustness_score = 1 - (output_preds.eq(noisy_output_preds)[idx_test].sum().item()/idx_test.shape[0])
        parity, equality = fair_metric(output_preds[idx_test].cpu().numpy(), labels[idx_test].cpu().numpy(), sens[idx_test].cpu().numpy())
        f1_s = f1_score(labels[idx_test].cpu().numpy(), output_preds[idx_test].cpu().numpy())
        print("The AUCROC of estimator: {:.4f}".format(auc_roc_test))
        print(f'Parity: {parity} | Equality: {equality}')
        print(f'F1-score: {f1_s}')
        print(f'CounterFactual Fairness: N/A')
        print(f'Robustness Score: {robustness_score}')
    else:
        print("Please set smaller acc/roc thresholds")
