'''
    Robust Graph Convolutional Networks Against Adversarial Attacks. KDD 2019.
        http://pengcui.thumedialab.com/papers/RGCN.pdf
    Author's Tensorflow implemention:
        https://github.com/thumanlab/nrlweb/tree/master/static/assets/download
'''

import torch.nn.functional as F
import math
import torch
import numpy as np
import scipy.sparse as sp
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.distributions.multivariate_normal import MultivariateNormal
# from deeprobust.graph import utils
import torch.optim as optim
from copy import deepcopy
from sklearn.metrics import f1_score, roc_auc_score

# TODO sparse implementation

class GGCL_F(Module):
    """GGCL: the input is feature"""

    def __init__(self, in_features, out_features, dropout=0.6):
        super(GGCL_F, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, features, adj_norm1, adj_norm2, gamma=1):
        features = F.dropout(features, self.dropout, training=self.training)
        self.miu = F.elu(torch.mm(features, self.weight_miu))
        self.sigma = F.relu(torch.mm(features, self.weight_sigma))
        # torch.mm(previous_sigma, self.weight_sigma)
        Att = torch.exp(-gamma * self.sigma)
        miu_out = torch.matmul(adj_norm1.float(), (self.miu * Att).float())  # adj_norm1.float() @ (self.miu * Att).float()  # double()
        sigma_out = torch.matmul(adj_norm2.float(), (self.sigma * Att * Att).float())  # double()
        return miu_out, sigma_out

class GGCL_D(Module):

    """GGCL_D: the input is distribution"""
    def __init__(self, in_features, out_features, dropout):
        super(GGCL_D, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        # self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, miu, sigma, adj_norm1, adj_norm2, gamma=1):
        miu = F.dropout(miu, self.dropout, training=self.training)
        sigma = F.dropout(sigma, self.dropout, training=self.training)
        miu = F.elu(torch.matmul(miu, self.weight_miu.float()))    # F.elu(miu @ self.weight_miu.float())  # double())
        sigma = F.relu(torch.matmul(sigma, self.weight_sigma.float())) # F.relu(sigma @ self.weight_sigma.float())  # double())

        Att = torch.exp(-gamma * sigma)
        mean_out = torch.matmul(adj_norm1.float(), (miu * Att).float())  # adj_norm1.float() @ (miu * Att).float()  # double()
        sigma_out = torch.matmul(adj_norm2.float(), (sigma * Att * Att).float())  # adj_norm2.float() @ (sigma * Att * Att).float()  # double()
        return mean_out, sigma


class GaussianConvolution(Module):

    def __init__(self, in_features, out_features):
        super(GaussianConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_miu = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight_sigma = Parameter(torch.FloatTensor(in_features, out_features))
        # self.sigma = Parameter(torch.FloatTensor(out_features))
        # self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # TODO
        torch.nn.init.xavier_uniform_(self.weight_miu)
        torch.nn.init.xavier_uniform_(self.weight_sigma)

    def forward(self, previous_miu, previous_sigma, adj_norm1=None, adj_norm2=None, gamma=1):

        if adj_norm1 is None and adj_norm2 is None:
            return torch.mm(previous_miu, self.weight_miu), \
                    torch.mm(previous_miu, self.weight_miu)
                    # torch.mm(previous_sigma, self.weight_sigma)

        Att = torch.exp(-gamma * previous_sigma)
        M = torch.matmul(adj_norm1, torch.matmul((previous_miu * Att), self.weight_miu))   # adj_norm1 @ (previous_miu * Att) @ self.weight_miu
        Sigma = torch.matmul(adj_norm2, torch.matmul((previous_sigma * Att * Att), self.weight_sigma))  # adj_norm2 @ (previous_sigma * Att * Att) @ self.weight_sigma
        return M, Sigma

        # M = torch.mm(torch.mm(adj, previous_miu * A), self.weight_miu)
        # Sigma = torch.mm(torch.mm(adj, previous_sigma * A * A), self.weight_sigma)

        # TODO sparse implemention
        # support = torch.mm(input, self.weight)
        # output = torch.spmm(adj, support)
        # return output + self.bias

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class RobustGCN(Module):

    def __init__(self, nnodes, nfeat, nhid, nclass, gamma=1.0, beta1=1e-7, beta2=5e-4, lr=0.001, dropout=0.6, device='cpu', seed=1):
        super(RobustGCN, self).__init__()

        self.device = device
        self.seed = seed
        # adj_norm = normalize(adj)
        # first turn original features to distribution
        self.lr = lr
        self.gamma = gamma
        self.beta1 = beta1
        self.beta2 = beta2
        self.nclass = nclass
        self.nhid = nhid // 2
        # self.gc1 = GaussianConvolution(nfeat, nhid, dropout=dropout)
        # self.gc2 = GaussianConvolution(nhid, nclass, dropout)
        self.gc1 = GGCL_F(nfeat, self.nhid, dropout=dropout).to(self.device)
        self.gc2 = GGCL_D(self.nhid, nclass, dropout=dropout).to(self.device)

        self.dropout = dropout
        # self.gaussian = MultivariateNormal(torch.zeros(self.nclass), torch.eye(self.nclass))
        self.gaussian = MultivariateNormal(torch.zeros(nnodes, self.nclass),
                torch.diag_embed(torch.ones(nnodes, self.nclass)))
        self.adj_norm1, self.adj_norm2 = None, None
        self.features, self.labels = None, None


    def forward(self):
        features = self.features
        # ipdb.set_trace()
        miu, sigma = self.gc1(features.to(self.device), self.adj_norm1.to(self.device), self.adj_norm2.to(self.device), self.gamma)
        miu, sigma = self.gc2(miu, sigma, self.adj_norm1.to(self.device), self.adj_norm2.to(self.device), self.gamma)
        output = miu + self.gaussian.sample().to(self.device) * torch.sqrt(sigma + 1e-8)
        return output  # F.log_softmax(output, dim=1)


    def fit(self, features, adj, labels, idx_train, idx_val=None, idx_test=None, train_iters=1000, verbose=True, attention=None):

        # adj, features, labels = utils.to_tensor(adj.todense(), features.todense(), labels, device=self.device)

        self.features, self.labels = features, labels

        self.adj_norm1 = self._normalize_adj(adj, power=-1/2).float().cpu()
        self.adj_norm2 = self._normalize_adj(adj, power=-1).float().cpu()
        # print('=== training rgcn model ===')
        self._initialize()
        if idx_val is None:
            self._train_without_val(labels, idx_train, train_iters, verbose)
        else:
            self._train_with_val(labels, idx_train, idx_val, train_iters, verbose)

    def _train_without_val(self, labels, idx_train, train_iters, verbose=True):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.train()
        for i in range(train_iters):
            optimizer.zero_grad()
            output = self.forward()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            if verbose and i % 100 == 0:
                print('Epoch {}, training loss: {}'.format(i, loss_train.item()))

        self.eval()
        output = self.forward()
        self.output = output

    def _train_with_val(self, labels, idx_train, idx_val, train_iters, verbose):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)

        best_loss_val = 100
        best_acc_val = 0
        import tqdm
        for i in tqdm.tqdm(range(train_iters)):
            torch.cuda.empty_cache()
            self.train()
            optimizer.zero_grad()
            output = self.forward().to('cpu')
            # ipdb.set_trace()
            loss_train = self._loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()

            self.eval()
            output = self.forward().to('cpu')
            loss_val = F.binary_cross_entropy_with_logits(output[idx_val][:, 0], labels[idx_val].float()).detach().item()  # double())  # F.nll_loss(output[idx_val], labels[idx_val])

            # preds = (output.squeeze()>0).type_as(labels)
            # ipdb.set_trace()
            # acc_val = utils.accuracy(preds[idx_val], labels[idx_val])
            acc_val = roc_auc_score(labels[idx_val].cpu().numpy(), output[idx_val].detach().cpu().numpy())
            # if verbose and i % 100 == 0:
            #     print('Epoch {}, training loss: {:.4f} || val loss: {:.4f}'.format(i, loss_train.detach().item(), loss_val))

            if best_loss_val > loss_val:
                best_loss_val = loss_val
                # self.output = output
                torch.save(self.state_dict(), f'weights_rogcn_{self.seed}.pt')

            if acc_val > best_acc_val:
                best_acc_val = acc_val
                # self.output = output
                torch.save(self.state_dict(), f'weights_rogcn_{self.seed}.pt')

        # print('=== picking the best model according to the performance on validation ===')


    def test(self, idx_test):
        # output = self.forward()
        output = self.output
        # loss_test = F.nll_loss(output[idx_test], self.labels[idx_test])
        loss_test = F.binary_cross_entropy_with_logits(output[idx_test][:, 0], self.labels[idx_test].float())
#        ipdb.set_trace()
        preds = (output.squeeze()>0).type_as(self.labels)
#        acc_test = utils.accuracy(output[idx_test], self.labels[idx_test])
        acc_test = roc_auc_score(self.labels[idx_test].cpu().numpy(), output.detach()[idx_test].cpu().numpy())
        print("Test set results:",
              "loss= {:.4f}".format(loss_test.item()),
              "accuracy= {:.4f}".format(acc_test.item()))
        return acc_test, output

    def predict(self, features):
        # ipdb.set_trace()
        miu, sigma = self.gc1(features.to('cpu'), self.adj_norm1.to('cpu'), self.adj_norm2.to('cpu'), self.gamma)
        miu, sigma = self.gc2(miu.to('cpu'), sigma.to('cpu'), self.adj_norm1.to('cpu'), self.adj_norm2.to('cpu'), self.gamma)
        output = miu.to('cpu') + self.gaussian.sample().to('cpu') * torch.sqrt(sigma + 1e-8)
        return output

    def _loss(self, input, labels):
        loss = F.binary_cross_entropy_with_logits(input[:, 0], labels.float())  # double())  # F.binary_cross_entropy_with_logits(input, labels)
        miu1 = self.gc1.miu
        sigma1 = self.gc1.sigma
        kl_loss = 0.5 * (miu1.pow(2) + sigma1 - torch.log(1e-8 + sigma1)).mean(1)
        kl_loss = kl_loss.sum()
        norm2 = torch.norm(self.gc1.weight_miu, 2).pow(2) + \
                torch.norm(self.gc1.weight_sigma, 2).pow(2)

        # print(f'gcn_loss: {loss.item()}, kl_loss: {self.beta1 * kl_loss.item()}, norm2: {self.beta2 * norm2.item()}')
        return loss  + self.beta1 * kl_loss + self.beta2 * norm2

    def _initialize(self):
        self.gc1.reset_parameters()
        self.gc2.reset_parameters()


    def _normalize_adj(self, adj, power=-1/2):

        """Row-normalize sparse matrix"""
	# TO DO: change it to our implementation
        # rowsum = np.array(mx.sum(1))
        # r_inv = np.power(rowsum, -1).flatten()
        # r_inv[np.isinf(r_inv)] = 0.
        # r_mat_inv = sp.diags(r_inv)
        # mx = r_mat_inv.dot(mx)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        A = torch.tensor(adj.todense()).float().to(device)  # + torch.eye(adj.shape[0]).to(self.device)
        D_power = (A.sum(1)).pow(power)
        D_power[torch.isinf(D_power)] = 0.
        D_power = torch.diag(D_power)
        # ipdb.set_trace()
        # return torch.spmm(D_power, torch.spmm(A, D_power))
        return torch.bmm(D_power.unsqueeze(0), torch.bmm(A.unsqueeze(0), D_power.unsqueeze(0)))[0, :]
#        return torch.matmul(D_power, torch.matmul(A, D_power))  # D_power @ A @ D_power
