import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv


class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(4, 64)
        self.conv15 = GCNConv(64, 64)
        self.conv2 = GCNConv(64, 1)

    #        self.lin1 = Linear(data.num_features, hidden_channels)
    #        self.lin2 = Linear(hidden_channels, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        final = torch.cat([x], dim=1)
        return torch.sigmoid(final)


class GAT(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.hidden = 8
        self.in_head = 8
        self.out_head = 1
        self.num_features = 4
        self.num_classes = 1
        self.conv1 = GATConv(
            self.num_features, self.hidden, heads=self.in_head, dropout=0.6
        )

        self.conv2 = GATConv(
            self.hidden * self.in_head,
            self.num_classes,
            concat=False,
            heads=self.out_head,
            dropout=0.6,
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        final = torch.cat([x], dim=1)
        return torch.sigmoid(final)
