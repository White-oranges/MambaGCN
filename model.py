import torch
import numpy as np
from torch import nn
from torch_geometric.nn import GCNConv
from mamba_ssm import Mamba


class MambaGNN(nn.Module):
    def __init__(self):
        super(MambaGNN, self).__init__()

        self.shift = 8

        self.device = torch.device("cuda")

        self.A0 = np.array([[1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                            [0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 1],
                            [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])

        self.mamba1 = Mamba(d_model=self.shift, d_state=4, d_conv=2, expand=1)

        self.fc1 = nn.Linear(self.shift * 18, 18)

        self.fl = nn.Flatten()

        self.b1 = nn.LayerNorm([18, self.shift])

        self.gcn1 = GCNConv(self.shift, self.shift)

    def forward(self, x):
        x = self.layer(x)
        x = self.fl(x)
        x = self.fc1(x)
        return x

    def layer(self, X):
        X = torch.concatenate([X, torch.ones(len(X), self.shift, 1).to(self.device)], dim=2)
        X = torch.permute(X, (0, 2, 1))
        X2 = torch.flatten(X, start_dim=0, end_dim=1)
        X1 = X / torch.norm(X, dim=2, keepdim=True)
        A = torch.bmm(X1, X1.permute(0, 2, 1))
        A = A*self.A0
        A = self.make_sparse(len(X), A)
        A_sparse = A.to_sparse_coo()
        edge_index = A_sparse.indices()
        weight = A_sparse.values()
        X2 = self.gcn1(X2, edge_index=edge_index, edge_weight=weight)
        X3 = X2.reshape(-1, 19, self.shift)
        X = X3[:, :18, :]
        X4 = X
        X = self.b1(X)
        X = self.mamba1(X)
        X = X4+X
        X = torch.permute(X, (0, 2, 1))
        return X

    def make_sparse(self, n, X):
        X = X
        A = X[0]
        for i in range(n - 1):
            A = torch.block_diag(A, X[i + 1])
        return A
