import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, global_max_pool, global_mean_pool
from torch_geometric.data import Data


def get_conv_layer(in_dim: int,
                   out_dim: int,
                   conv_layer: str,
                   dropout: float = 0.2,
                   heads: int = 1):
    if conv_layer == 'gcn':
        return GCNConv(in_dim, out_dim)
    elif conv_layer == 'sage':
        return SAGEConv(in_dim, out_dim)
    elif conv_layer == 'gat':
        return GATConv(in_dim, out_dim, heads=heads, dropout=dropout)
    else:
        raise NotImplementedError(f'Unknown conv layer: {conv_layer}')


class UPFDNet(nn.Module):

    def __init__(self,
                 conv_layer: str,
                 in_dim: int,
                 hidden_dim: int,
                 out_dim: int,
                 concat: bool,
                 dropout: float):
        super(UPFDNet, self).__init__()

        self.concat = concat
        self.dropout = dropout
        self.conv = get_conv_layer(in_dim, hidden_dim, conv_layer)

        if self.concat:
            self.lin0 = nn.Linear(in_dim, hidden_dim)
            self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = F.relu(self.conv(x, edge_index))
        out = global_max_pool(out, batch)

        if self.concat:
            news = torch.stack([x[(batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
            news = F.relu(self.lin0(news))
            out = torch.cat([out, news], dim=1)
            out = F.relu(self.lin1(out))

        out = self.lin2(out)
        out = F.log_softmax(out, dim=-1)

        return out


class UPFDSingle(nn.Module):

    def __init__(self,
                 conv_layer: str,
                 feature_type: str,
                 hidden_dim: int,
                 out_dim: int,
                 dropout: float):
        super(UPFDSingle, self).__init__()

        self.dropout = dropout

        if feature_type == 'content':
            in_dim = 310
        elif feature_type == 'bert':
            in_dim = 768
        elif feature_type == 'profile':
            in_dim = 10
        else:
            in_dim = 300

        self.conv1 = get_conv_layer(in_dim, hidden_dim, conv_layer)
        self.conv2 = get_conv_layer(hidden_dim, hidden_dim, conv_layer)

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        out = self.conv1(x, edge_index)
        out = F.relu(out)

        out = self.conv2(out, edge_index)
        out = F.relu(out)

        out = global_mean_pool(out, batch)

        out = F.relu(self.lin1(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)
        out = F.log_softmax(out, dim=-1)

        return out


class MultiFeatureNet(nn.Module):

    def __init__(self,
                 features: list[str],
                 conv_layer: str,
                 hidden_dim: int,
                 out_dim: int,
                 dropout: float):
        super(MultiFeatureNet, self).__init__()

        self.features = features
        self.dropout = dropout

        if 'content' in features:
            self.content_dim = 310
            self.content_lin = nn.Linear(self.content_dim, hidden_dim)
        if 'bert' in features:
            self.bert_dim = 768
            self.bert_lin = nn.Linear(self.bert_dim, hidden_dim)
        if 'profile' in features:
            self.profile_dim = 10
            self.profile_lin = nn.Linear(self.profile_dim, hidden_dim)
        if 'spacy' in features:
            self.spacy_dim = 300
            self.spacy_lin = nn.Linear(self.spacy_dim, hidden_dim)

        combined_dim = hidden_dim * len(features)

        self.conv1 = get_conv_layer(combined_dim, hidden_dim, conv_layer)
        self.conv2 = get_conv_layer(hidden_dim, hidden_dim, conv_layer)

        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def forward(self,
                data: list[Data]) -> Tensor:
        embeddings = []
        i = 0
        if 'content' in self.features:
            content_x, edge_index, batch = data[i].x, data[i].edge_index, data[i].batch
            content_h = F.relu(self.content_lin(content_x))
            embeddings.append(content_h)
            i += 1
        if 'bert' in self.features:
            bert_x, edge_index, batch = data[i].x, data[i].edge_index, data[i].batch
            bert_h = F.relu(self.bert_lin(bert_x))
            embeddings.append(bert_h)
            i += 1
        if 'profile' in self.features:
            profile_x, edge_index, batch = data[i].x, data[i].edge_index, data[i].batch
            profile_h = F.relu(self.profile_lin(profile_x))
            embeddings.append(profile_h)
            i += 1
        if 'spacy' in self.features:
            spacy_x, edge_index, batch = data[i].x, data[i].edge_index, data[i].batch
            spacy_h = F.relu(self.spacy_lin(spacy_x))
            embeddings.append(spacy_h)

        out = torch.cat(embeddings, dim=1)

        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.conv1(out, edge_index))

        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.conv2(out, edge_index))

        out = global_mean_pool(out, batch)

        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.lin1(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)

        out = F.log_softmax(out, dim=-1)

        return out


class ParallelFeatureNet(nn.Module):

    def __init__(self,
                 features: list[str],
                 conv_layer: str,
                 hidden_dim: int,
                 out_dim: int,
                 dropout: float):
        super(ParallelFeatureNet, self).__init__()

        self.dropout = dropout
        self.features = features
        if 'content' in features:
            self.content_conv1 = get_conv_layer(310, hidden_dim, conv_layer)
            self.content_conv2 = get_conv_layer(hidden_dim, hidden_dim, conv_layer)
        if 'bert' in features:
            self.bert_conv1 = get_conv_layer(768, hidden_dim, conv_layer)
            self.bert_conv2 = get_conv_layer(hidden_dim, hidden_dim, conv_layer)
        if 'profile' in features:
            self.profile_conv1 = get_conv_layer(10, hidden_dim, conv_layer)
            self.profile_conv2 = get_conv_layer(hidden_dim, hidden_dim, conv_layer)
        if 'spacy' in features:
            self.spacy_conv1 = get_conv_layer(300, hidden_dim, conv_layer)
            self.spacy_conv2 = get_conv_layer(hidden_dim, hidden_dim, conv_layer)

        self.lin1 = nn.Linear(hidden_dim * len(self.features), hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, out_dim)

    def process_branch(self, x, edge_index, conv1, conv2):
        out = F.dropout(x, p=self.dropout, training=self.training)
        out = conv1(out, edge_index)
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = conv2(out, edge_index)
        return out

    def forward(self,
                data: dict[Data]) -> Tensor:
        embeddings = []
        i = 0
        if 'content' in self.features:
            content_x, edge_index, batch = data[i].x, data[i].edge_index, data[i].batch
            content_h = self.process_branch(content_x, edge_index, self.content_conv1, self.content_conv2)
            embeddings.append(content_h)
            i += 1
        if 'bert' in self.features:
            bert_x, edge_index, batch = data[i].x, data[i].edge_index, data[i].batch
            bert_h = self.process_branch(bert_x, edge_index, self.bert_conv1, self.bert_conv2)
            embeddings.append(bert_h)
            i += 1
        if 'profile' in self.features:
            profile_x, edge_index, batch = data[i].x, data[i].edge_index, data[i].batch
            profile_h = self.process_branch(profile_x, edge_index, self.profile_conv1, self.profile_conv2)
            embeddings.append(profile_h)
            i += 1
        if 'spacy' in self.features:
            spacy_x, edge_index, batch = data[i].x, data[i].edge_index, data[i].batch
            spacy_h = self.process_branch(spacy_x, edge_index, self.spacy_conv1, self.spacy_conv2)
            embeddings.append(spacy_h)

        out = torch.cat(embeddings, dim=1)

        out = global_mean_pool(out, batch)

        out = F.dropout(out, p=self.dropout, training=self.training)
        out = F.relu(self.lin1(out))
        out = F.dropout(out, p=self.dropout, training=self.training)
        out = self.lin2(out)
        out = F.log_softmax(out, dim=-1)

        return out
