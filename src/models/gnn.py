"""
Graph Neural Network for DILI prediction.

Operates on molecular graphs (atoms as nodes, bonds as edges),
learning structure-toxicity relationships directly from the
molecular topology -- a core approach at Axiom Bio for
understanding how structural features drive toxicity.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.metrics import roc_auc_score

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.featurization import mol_to_graph


class MolGCN(nn.Module):
    """
    Graph Convolutional Network for molecular property prediction.
    Uses multiple GCN layers with residual connections and a readout
    that combines mean and max pooling.
    """

    def __init__(self, node_feat_dim: int, hidden: int = 128, num_layers: int = 3, dropout: float = 0.2):
        super().__init__()

        self.input_proj = nn.Linear(node_feat_dim, hidden)

        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden, hidden))
            self.bns.append(nn.BatchNorm1d(hidden))

        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.input_proj(x)
        x = F.relu(x)

        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            x = x + h  # residual

        pool_mean = global_mean_pool(x, batch)
        pool_max = global_max_pool(x, batch)
        graph_emb = torch.cat([pool_mean, pool_max], dim=-1)

        return self.classifier(graph_emb).squeeze(-1)


def smiles_to_pyg_data(smiles: str, label: float) -> Data:
    """Convert SMILES to PyG Data object."""
    graph = mol_to_graph(smiles)
    if graph is None:
        return None

    return Data(
        x=torch.tensor(graph["node_feats"], dtype=torch.float32),
        edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
        y=torch.tensor([label], dtype=torch.float32),
    )


def build_graph_dataset(smiles_list: list[str], labels: np.ndarray) -> list[Data]:
    dataset = []
    for smi, lab in zip(smiles_list, labels):
        data = smiles_to_pyg_data(smi, lab)
        if data is not None:
            dataset.append(data)
    return dataset


def collate_fn(batch: list[Data]) -> Batch:
    return Batch.from_data_list(batch)


def train_gnn(
    smiles_train: list[str],
    y_train: np.ndarray,
    smiles_test: list[str],
    y_test: np.ndarray,
    epochs: int = 80,
    batch_size: int = 32,
    lr: float = 1e-3,
    device: str = "auto",
) -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print("    Building molecular graphs...")
    train_data = build_graph_dataset(smiles_train, y_train)
    test_data = build_graph_dataset(smiles_test, y_test)
    print(f"    Graphs: {len(train_data)} train, {len(test_data)} test")

    if len(train_data) == 0:
        raise ValueError("No valid molecular graphs in training data")

    node_feat_dim = train_data[0].x.shape[1]

    train_loader = TorchDataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = TorchDataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    model = MolGCN(node_feat_dim=node_feat_dim).to(device)

    pos_count = sum(1 for d in train_data if d.y.item() > 0.5)
    neg_count = len(train_data) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                logits = model(batch)
                probs = torch.sigmoid(logits).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch.y.squeeze().cpu().numpy())

        auc = roc_auc_score(all_labels, all_probs)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            avg_loss = total_loss / len(train_data)
            print(f"    Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  test_AUC={auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            logits = model(batch)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch.y.squeeze().cpu().numpy())

    test_probs = np.array(all_probs)
    test_preds = (test_probs >= 0.5).astype(int)

    return {
        "model": model,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "test_labels": np.array(all_labels),
        "best_auc": best_auc,
        "device": device,
    }
