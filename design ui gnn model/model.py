import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class LigandGNN(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=128):
        super(LigandGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)


class ProteinGNN(nn.Module):
    def __init__(self, in_channels=23, hidden_dim=128):
        super(ProteinGNN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)


class ProteinLigandGNN(nn.Module):
    def __init__(self, hidden_dim=128):
        super(ProteinLigandGNN, self).__init__()

        self.ligand_gnn = LigandGNN(in_channels=4, hidden_dim=hidden_dim)
        self.protein_gnn = ProteinGNN(in_channels=23, hidden_dim=hidden_dim)

        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.fc_pkd = nn.Linear(hidden_dim, 1)
        self.fc_pki = nn.Linear(hidden_dim, 1)
        self.fc_ba  = nn.Linear(hidden_dim, 1)


    def forward(self, protein_data, ligand_data):
        ligand_emb = self.ligand_gnn(ligand_data)
        protein_emb = self.protein_gnn(protein_data)

        combined = torch.cat([protein_emb, ligand_emb], dim=1)
        combined = F.relu(self.fc1(combined))

        pkd = self.fc_pkd(combined)
        pki = self.fc_pki(combined)
        ba  = self.fc_ba(combined)

        return pkd, pki, ba
