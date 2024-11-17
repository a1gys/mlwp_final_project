import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import degree, add_self_loops as add_self_loops_fn
from torch_geometric.data import InMemoryDataset
from torch_scatter import scatter_add
from torch_sparse import SparseTensor


class PARMA(nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 period: int,
                 timestamps: int,
                 act: nn.Module = nn.ReLU(),
                 dropout: float = 0.,
                 bias: bool = True):
        super(PARMA, self).__init__()

        self.name = 'PARMA'
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = period
        self.T = timestamps
        self.act = act
        self.dropout = dropout

        self.weight = nn.ModuleDict({
            str(k): nn.Linear(out_dim, out_dim, dtype=torch.float32) for k in range(self.K)
        })
        self.root_weight = nn.ModuleDict({
            str(k): nn.Linear(in_dim, out_dim, dtype=torch.float32) for k in range(self.K)
        })
        self.init_weight = nn.Linear(in_dim, out_dim, dtype=torch.float32)
        if bias:
            self.bias = nn.Parameter(torch.empty(self.T, self.K, out_dim, dtype=torch.float32))
            # self.bias = nn.Parameter(torch.empty(self.T, 1, out_dim, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        for k in range(self.K):
            glorot(self.weight[str(k)].weight)
            glorot(self.root_weight[str(k)].weight)
        glorot(self.init_weight.weight)
        if self.bias is not None:
            # zeros(self.bias)
            nn.init.uniform_(self.bias, a=-0.01, b=0.01)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,
                shift_op: Tensor) -> Tensor:
        out = self.init_weight(x) # (N, F_in) @ (F_in, F_out) => (N, F_out)

        for t in range(self.T):
            for k in range(self.K):
                root = F.dropout(x, p=self.dropout, training=self.training)
                root = self.root_weight[str(k)](root) # (N, F_in) @ (F_in, F_out) => (N, F_out)
                # root = F.dropout(root, p=self.dropout, training=self.training)
                out = self.weight[str(k)](out)
                out = shift_op @ out + root

            if self.bias is not None:
                out = out + self.bias[t, k]
                # out = out + self.bias[t]
            if self.act is not None:
                out = self.act(out)

        return F.log_softmax(out, dim=1)


class PARMAImproved(nn.Module):

    def __init__(self,
                 in_dim: int,
                 out_dim: int,
                 period: int,
                 timestamps: int,
                 act: nn.Module = nn.ReLU(),
                 dropout: float = 0.,
                 bias: bool = True):
        super(PARMAImproved, self).__init__()

        self.name = 'PARMA'
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.K = period
        self.T = timestamps
        self.act = act
        self.dropout = dropout
        self.out_dim = out_dim

        self.weight = nn.Parameter(torch.empty(self.K, out_dim, out_dim))
        self.root_weight = nn.Parameter(torch.empty(self.K, in_dim, out_dim))
        self.init_weight = nn.Linear(in_dim, out_dim, dtype=torch.float32)

        if bias:
            self.bias = nn.Parameter(torch.empty(self.T, self.K, out_dim, dtype=torch.float32))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight.data)
        glorot(self.root_weight.data)
        glorot(self.init_weight.weight)
        if self.bias is not None:
            zeros(self.bias)

    def forward(self,
                x: Tensor,
                edge_index: Tensor,) -> Tensor:

        num_nodes = x.size(0)
        edge_index, edge_weight = gcn_norm(edge_index=edge_index, add_self_loops=True, num_nodes=num_nodes,
                                           flow="source_to_target", )
        out = self.init_weight(x)  # (B, N, F_in) @ (F_in, F_out) => (B, N, F_out)

        for t in range(self.T):
            for k in range(self.K):
                root = F.dropout(x, p=self.dropout, training=self.training)
                root = torch.matmul(root, self.root_weight[k])  # (B, N, F_in) @ (F_in, F_out) => (B, N, F_out)
                # root = F.dropout(root, p=self.dropout, training=self.training)

                out = scatter_add(edge_weight.unsqueeze(-1) * out[edge_index[0]],
                                  edge_index[1], dim=0, dim_size=num_nodes)

                out = torch.matmul(out, self.weight[k]) + root

            if self.bias is not None:
                out = out + self.bias[t, k]

            if self.act is not None:
                out = self.act(out)

        return F.log_softmax(out, dim=-1)


def graph_laplacian(data: InMemoryDataset,
                    add_self_loops: bool = True,):
    num_nodes = data.x.size(0)
    if add_self_loops:
        edge_index, _ = add_self_loops_fn(data.edge_index, num_nodes=num_nodes)
    else:
        edge_index = data.edge_index
    row, col = edge_index
    deg = degree(row, num_nodes=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
    norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    laplacian = SparseTensor(row=row, col=col, value=norm, sparse_sizes=(num_nodes, num_nodes))
    return laplacian
