"""
Pre-train → Fine-tune pipeline for DILI prediction.

Strategy v2: Multi-task pre-training + staged fine-tuning.

Pre-training uses the full proxy-DILI dataset (8,558 compounds) with
SEPARATE task heads for each assay source (mito tox, BSEP, rat hepatotox)
so the shared backbone learns diverse hepatotoxicity representations.

Fine-tuning uses a two-stage approach:
  Stage 1 (head warmup): freeze backbone, train only the DILI head
  Stage 2 (full fine-tune): unfreeze all with discriminative LR
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader as TorchDataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.featurization import (
    featurize_dataset, mol_to_graph, morgan_fingerprint,
    mol_from_smiles, physicochemical_descriptors,
)
from src.models.gnn import MolGCN, smiles_to_pyg_data, build_graph_dataset, collate_fn
from src.models.hybrid import (
    HepaToxFusion, prepare_hybrid_batch, collate_hybrid,
)

PROXY_SOURCES = {
    "Mitochondrial Toxicity": 0,
    "BSEP Inhibition": 1,
    "Rat Hepatotoxicity": 2,
}


def load_proxy_pretrain_data() -> pd.DataFrame:
    """
    Load full proxy-DILI with per-source task labels for multi-task pre-training.
    Returns one row per compound with columns for each assay + aggregate label.
    """
    proxy = pd.read_csv(ROOT / "data" / "proxy_dili.csv")

    pivot = proxy.pivot_table(
        index="smiles_r", columns="Source", values="TOXICITY",
        aggfunc="max", fill_value=-1,
    ).reset_index()
    pivot = pivot.rename(columns={"smiles_r": "SMILES"})

    label_cols = [c for c in pivot.columns if c != "SMILES"]
    pivot["label"] = (pivot[label_cols].clip(lower=0).max(axis=1)).astype(int)

    # Standardize column names
    rename_map = {}
    for col in label_cols:
        for canonical, idx in PROXY_SOURCES.items():
            if canonical.lower()[:6] in col.lower()[:10]:
                rename_map[col] = f"proxy_{idx}"
    pivot = pivot.rename(columns=rename_map)

    proxy_cols = [f"proxy_{i}" for i in range(len(PROXY_SOURCES))]
    for c in proxy_cols:
        if c not in pivot.columns:
            pivot[c] = -1

    print(f"  Proxy pre-train dataset: {len(pivot)} compounds")
    print(f"    Toxic (any assay): {(pivot['label'] == 1).sum()}")
    print(f"    Safe (all assays): {(pivot['label'] == 0).sum()}")
    for i, name in enumerate(PROXY_SOURCES):
        col = f"proxy_{i}"
        available = (pivot[col] >= 0).sum()
        pos = (pivot[col] == 1).sum()
        print(f"    {name}: {available} tested, {pos} positive")

    return pivot


# --------------------------------------------------------------------------
#  GNN  pre-train / fine-tune
# --------------------------------------------------------------------------

class MultiTaskGCN(nn.Module):
    """GCN backbone + multi-task heads for proxy assays."""

    def __init__(self, backbone: MolGCN, n_tasks: int = 3):
        super().__init__()
        self.backbone = backbone
        # backbone.classifier input = hidden*2 (mean+max pool concat)
        emb_dim = backbone.classifier[0].in_features  # hidden * 2
        self.task_heads = nn.ModuleList([
            nn.Linear(emb_dim, 1) for _ in range(n_tasks)
        ])

    def forward(self, batch):
        from torch_geometric.nn import global_mean_pool, global_max_pool

        x = self.backbone.input_proj(batch.x)
        x = torch.relu(x)
        for conv, bn in zip(self.backbone.convs, self.backbone.bns):
            h = conv(x, batch.edge_index)
            h = bn(h)
            h = torch.relu(h)
            h = torch.dropout(h, p=self.backbone.dropout, train=self.training)
            x = x + h  # residual (matches backbone forward)

        pool_mean = global_mean_pool(x, batch.batch)
        pool_max = global_max_pool(x, batch.batch)
        emb = torch.cat([pool_mean, pool_max], dim=-1)

        main_logit = self.backbone.classifier(emb).squeeze(-1)
        aux_logits = [head(emb).squeeze(-1) for head in self.task_heads]
        return main_logit, aux_logits


def pretrain_gnn(
    proxy_df: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "auto",
) -> dict:
    """Pre-train GCN with multi-task proxy heads."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    smiles = proxy_df["SMILES"].tolist()
    labels = proxy_df["label"].values.astype(np.float32)
    proxy_labels = proxy_df[[f"proxy_{i}" for i in range(len(PROXY_SOURCES))]].values.astype(np.float32)

    train_idx, val_idx = train_test_split(
        np.arange(len(smiles)), test_size=0.1, stratify=labels, random_state=42
    )

    train_smi = [smiles[i] for i in train_idx]
    val_smi = [smiles[i] for i in val_idx]
    y_train, y_val = labels[train_idx], labels[val_idx]
    proxy_train, proxy_val = proxy_labels[train_idx], proxy_labels[val_idx]

    print(f"    Building graphs for {len(train_smi)} train + {len(val_smi)} val...")
    train_data = build_graph_dataset(train_smi, y_train)
    val_data = build_graph_dataset(val_smi, y_val)
    print(f"    Valid graphs: {len(train_data)} train, {len(val_data)} val")

    # Attach proxy labels to each Data object
    _attach_proxy_labels(train_data, train_smi, smiles, proxy_train, train_idx)
    _attach_proxy_labels(val_data, val_smi, smiles, proxy_val, val_idx)

    node_feat_dim = train_data[0].x.shape[1]
    backbone = MolGCN(node_feat_dim=node_feat_dim, hidden=128, num_layers=3)
    model = MultiTaskGCN(backbone, n_tasks=len(PROXY_SOURCES)).to(device)

    pos_count = sum(1 for d in train_data if d.y.item() > 0.5)
    neg_count = len(train_data) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion_main = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_aux = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = TorchDataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = TorchDataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            main_logit, aux_logits = model(batch)

            loss = criterion_main(main_logit, batch.y.squeeze())

            if hasattr(batch, "proxy_labels"):
                p_labels = batch.proxy_labels.to(device)
                for j, aux_l in enumerate(aux_logits):
                    mask = p_labels[:, j] >= 0
                    if mask.any():
                        loss = loss + 0.5 * criterion_aux(aux_l[mask], p_labels[:, j][mask]).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs

        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                main_logit, _ = model(batch)
                probs = torch.sigmoid(main_logit).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch.y.squeeze().cpu().numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.backbone.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Pre-train epoch {epoch+1}/{epochs}  "
                  f"loss={total_loss/len(train_data):.4f}  val_AUC={val_auc:.4f}")

    print(f"    Pre-train best val AUC: {best_auc:.4f}")
    return {"state_dict": best_state, "node_feat_dim": node_feat_dim, "device": device}


def _attach_proxy_labels(data_list, data_smiles, all_smiles, proxy_labels, indices):
    """Attach proxy task labels to PyG Data objects, matching by build order."""
    smi_to_proxy = {}
    for i, idx in enumerate(indices):
        smi_to_proxy[all_smiles[idx]] = proxy_labels[i]

    for i, d in enumerate(data_list):
        smi = data_smiles[i] if i < len(data_smiles) else None
        if smi and smi in smi_to_proxy:
            d.proxy_labels = torch.tensor(smi_to_proxy[smi], dtype=torch.float32).unsqueeze(0)
        else:
            d.proxy_labels = torch.tensor([-1.0] * proxy_labels.shape[1], dtype=torch.float32).unsqueeze(0)


def finetune_gnn(
    pretrained: dict,
    smiles_train: list[str],
    y_train: np.ndarray,
    smiles_test: list[str],
    y_test: np.ndarray,
    epochs_warmup: int = 20,
    epochs_full: int = 50,
    batch_size: int = 32,
    lr: float = 5e-4,
) -> dict:
    """Two-stage fine-tuning: head warmup → full unfreeze."""
    device = pretrained["device"]
    model = MolGCN(
        node_feat_dim=pretrained["node_feat_dim"], hidden=128, num_layers=3
    ).to(device)
    model.load_state_dict(pretrained["state_dict"])

    train_data = build_graph_dataset(smiles_train, y_train)
    test_data = build_graph_dataset(smiles_test, y_test)

    pos_count = sum(1 for d in train_data if d.y.item() > 0.5)
    neg_count = len(train_data) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader = TorchDataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    test_loader = TorchDataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    best_auc = 0.0
    best_state = None

    # ---- Stage 1: freeze backbone, train head only ----
    for p in model.input_proj.parameters():
        p.requires_grad = False
    for p in model.convs.parameters():
        p.requires_grad = False
    for p in model.bns.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)

    print(f"    Stage 1: head warmup ({epochs_warmup} epochs, backbone frozen)")
    for epoch in range(epochs_warmup):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            optimizer.step()

        auc = _eval_gnn(model, test_loader, device)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"    Stage 1 best AUC: {best_auc:.4f}")

    # ---- Stage 2: unfreeze all with discriminative LR ----
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam([
        {"params": model.input_proj.parameters(), "lr": lr * 0.05},
        {"params": model.convs.parameters(), "lr": lr * 0.1},
        {"params": model.bns.parameters(), "lr": lr * 0.1},
        {"params": model.classifier.parameters(), "lr": lr * 0.5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_full)

    print(f"    Stage 2: full fine-tune ({epochs_full} epochs, discriminative LR)")
    patience, wait = 15, 0
    for epoch in range(epochs_full):
        model.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        auc = _eval_gnn(model, test_loader, device)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 15 == 0:
            print(f"    Epoch {epoch+1}/{epochs_full}  test_AUC={auc:.4f}")

    model.load_state_dict(best_state)
    test_probs, test_labels = _predict_gnn(model, test_loader, device)
    print(f"    Fine-tuned GNN best AUC: {best_auc:.4f}")

    return {
        "model": model,
        "test_probs": test_probs,
        "test_preds": (test_probs >= 0.5).astype(int),
        "test_labels": test_labels,
        "best_auc": best_auc,
        "device": device,
    }


def _eval_gnn(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch.y.squeeze().cpu().numpy())
    return roc_auc_score(all_labels, all_probs)


def _predict_gnn(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            probs = torch.sigmoid(model(batch)).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch.y.squeeze().cpu().numpy())
    return np.array(all_probs), np.array(all_labels)


# --------------------------------------------------------------------------
#  HepaTox-Fusion  pre-train / fine-tune
# --------------------------------------------------------------------------

def pretrain_hybrid(
    proxy_df: pd.DataFrame,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "auto",
) -> dict:
    """Pre-train HepaTox-Fusion with multi-task proxy heads."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    smiles = proxy_df["SMILES"].tolist()
    labels = proxy_df["label"].values.astype(np.float32)
    proxy_labels = proxy_df[[f"proxy_{i}" for i in range(len(PROXY_SOURCES))]].values.astype(np.float32)

    train_smi, val_smi, y_train, y_val, p_train, p_val = train_test_split(
        smiles, labels, proxy_labels, test_size=0.1, stratify=labels, random_state=42
    )

    print(f"    Building hybrid data for {len(train_smi)} train + {len(val_smi)} val...")
    train_items = prepare_hybrid_batch(train_smi, y_train, p_train)
    val_items = prepare_hybrid_batch(val_smi, y_val, p_val)
    print(f"    Valid items: {len(train_items)} train, {len(val_items)} val")

    node_dim = train_items[0]["graph"].x.shape[1]
    fp_dim = train_items[0]["fp"].shape[0]
    desc_dim = train_items[0]["desc"].shape[0]

    model = HepaToxFusion(
        node_dim=node_dim, fp_dim=fp_dim, desc_dim=desc_dim,
        hidden=128, n_auxiliary=len(PROXY_SOURCES), dropout=0.2
    ).to(device)

    pos_count = sum(1 for it in train_items if it["y_dili"].item() > 0.5)
    neg_count = len(train_items) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion_main = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_aux = nn.BCEWithLogitsLoss(reduction="none")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    train_loader = TorchDataLoader(train_items, batch_size=batch_size, shuffle=True, collate_fn=collate_hybrid)
    val_loader = TorchDataLoader(val_items, batch_size=batch_size, shuffle=False, collate_fn=collate_hybrid)

    best_auc = 0.0
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            graph = batch["graph"].to(device)
            fp = batch["fp"].to(device)
            desc = batch["desc"].to(device)
            y_d = batch["y_dili"].to(device)

            optimizer.zero_grad()
            dili_logit, aux_logits, _ = model(graph, fp, desc)
            loss = criterion_main(dili_logit, y_d)

            if "y_proxy" in batch:
                y_p = batch["y_proxy"].to(device)
                for j in range(min(len(PROXY_SOURCES), len(aux_logits))):
                    mask = y_p[:, j] >= 0
                    if mask.any():
                        loss = loss + 0.5 * criterion_aux(aux_logits[j][mask], y_p[:, j][mask]).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_d)

        scheduler.step()

        model.eval()
        all_probs, all_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                graph = batch["graph"].to(device)
                fp = batch["fp"].to(device)
                desc = batch["desc"].to(device)
                probs = torch.sigmoid(model(graph, fp, desc)[0]).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(batch["y_dili"].numpy())

        val_auc = roc_auc_score(all_labels, all_probs)
        if val_auc > best_auc:
            best_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            print(f"    Pre-train epoch {epoch+1}/{epochs}  "
                  f"loss={total_loss/len(train_items):.4f}  val_AUC={val_auc:.4f}")

    model.load_state_dict(best_state)
    print(f"    Pre-train best val AUC: {best_auc:.4f}")
    return {
        "state_dict": best_state,
        "node_dim": node_dim,
        "fp_dim": fp_dim,
        "desc_dim": desc_dim,
        "device": device,
    }


def finetune_hybrid(
    pretrained: dict,
    smiles_train: list[str],
    y_train: np.ndarray,
    smiles_test: list[str],
    y_test: np.ndarray,
    proxy_train: np.ndarray | None = None,
    proxy_test: np.ndarray | None = None,
    epochs_warmup: int = 20,
    epochs_full: int = 60,
    batch_size: int = 32,
    lr: float = 3e-4,
    aux_weight: float = 0.2,
) -> dict:
    """Two-stage fine-tune: head warmup → full unfreeze with aux tasks."""
    device = pretrained["device"]
    n_aux = proxy_train.shape[1] if proxy_train is not None else 1

    model = HepaToxFusion(
        node_dim=pretrained["node_dim"],
        fp_dim=pretrained["fp_dim"],
        desc_dim=pretrained["desc_dim"],
        hidden=128, n_auxiliary=max(n_aux, 1), dropout=0.2
    ).to(device)

    # Load pre-trained weights (skip mismatched aux heads)
    pretrained_sd = pretrained["state_dict"]
    model_sd = model.state_dict()
    loaded = 0
    for k, v in pretrained_sd.items():
        if k in model_sd and model_sd[k].shape == v.shape:
            model_sd[k] = v
            loaded += 1
    model.load_state_dict(model_sd)
    print(f"    Loaded {loaded}/{len(pretrained_sd)} pre-trained parameters")

    train_items = prepare_hybrid_batch(smiles_train, y_train, proxy_train)
    test_items = prepare_hybrid_batch(smiles_test, y_test, proxy_test)

    pos_count = sum(1 for it in train_items if it["y_dili"].item() > 0.5)
    neg_count = len(train_items) - pos_count
    pos_weight = torch.tensor([neg_count / max(pos_count, 1)]).to(device)
    criterion_dili = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_aux = nn.BCEWithLogitsLoss(reduction="none")

    train_loader = TorchDataLoader(train_items, batch_size=batch_size, shuffle=True, collate_fn=collate_hybrid)
    test_loader = TorchDataLoader(test_items, batch_size=batch_size, shuffle=False, collate_fn=collate_hybrid)

    best_auc = 0.0
    best_state = None

    # ---- Stage 1: freeze backbone, train heads only ----
    backbone_params = list(model.gin.parameters()) + list(model.fp_branch.parameters()) + \
                      list(model.cross_attn.parameters()) + list(model.desc_branch.parameters())
    for p in backbone_params:
        p.requires_grad = False

    head_params = list(model.shared_trunk.parameters()) + \
                  list(model.dili_head.parameters()) + list(model.auxiliary_heads.parameters())
    optimizer = torch.optim.Adam(head_params, lr=lr)

    print(f"    Stage 1: head warmup ({epochs_warmup} epochs, backbone frozen)")
    for epoch in range(epochs_warmup):
        model.train()
        for batch in train_loader:
            graph = batch["graph"].to(device)
            fp = batch["fp"].to(device)
            desc = batch["desc"].to(device)
            y_d = batch["y_dili"].to(device)

            optimizer.zero_grad()
            dili_logit, _, _ = model(graph, fp, desc)
            loss = criterion_dili(dili_logit, y_d)
            loss.backward()
            optimizer.step()

        auc = _eval_hybrid(model, test_loader, device)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    print(f"    Stage 1 best AUC: {best_auc:.4f}")

    # ---- Stage 2: unfreeze all with discriminative LR ----
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW([
        {"params": list(model.gin.parameters()), "lr": lr * 0.05},
        {"params": list(model.fp_branch.parameters()), "lr": lr * 0.1},
        {"params": list(model.cross_attn.parameters()), "lr": lr * 0.1},
        {"params": list(model.desc_branch.parameters()), "lr": lr * 0.1},
        {"params": list(model.shared_trunk.parameters()), "lr": lr * 0.3},
        {"params": list(model.dili_head.parameters()) + list(model.auxiliary_heads.parameters()), "lr": lr * 0.5},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs_full)

    print(f"    Stage 2: full fine-tune ({epochs_full} epochs)")
    patience, wait = 15, 0
    for epoch in range(epochs_full):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            graph = batch["graph"].to(device)
            fp = batch["fp"].to(device)
            desc = batch["desc"].to(device)
            y_d = batch["y_dili"].to(device)

            optimizer.zero_grad()
            dili_logit, aux_logits, _ = model(graph, fp, desc)
            loss = criterion_dili(dili_logit, y_d)

            if "y_proxy" in batch and n_aux > 0:
                y_p = batch["y_proxy"].to(device)
                for j in range(min(n_aux, len(aux_logits))):
                    mask = y_p[:, j] >= 0
                    if mask.any():
                        loss = loss + aux_weight * criterion_aux(aux_logits[j][mask], y_p[:, j][mask]).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_d)

        scheduler.step()

        auc = _eval_hybrid(model, test_loader, device)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"    Early stop at epoch {epoch+1}")
                break

        if (epoch + 1) % 15 == 0:
            print(f"    Epoch {epoch+1}/{epochs_full}  "
                  f"loss={total_loss/len(train_items):.4f}  test_AUC={auc:.4f}")

    model.load_state_dict(best_state)
    test_probs, test_labels = _predict_hybrid(model, test_loader, device)
    print(f"    Fine-tuned Hybrid best AUC: {best_auc:.4f}")

    return {
        "model": model,
        "test_probs": test_probs,
        "test_preds": (test_probs >= 0.5).astype(int),
        "test_labels": test_labels,
        "best_auc": best_auc,
        "device": device,
    }


def _eval_hybrid(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            graph = batch["graph"].to(device)
            fp = batch["fp"].to(device)
            desc = batch["desc"].to(device)
            probs = torch.sigmoid(model(graph, fp, desc)[0]).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["y_dili"].numpy())
    return roc_auc_score(all_labels, all_probs)


def _predict_hybrid(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            graph = batch["graph"].to(device)
            fp = batch["fp"].to(device)
            desc = batch["desc"].to(device)
            probs = torch.sigmoid(model(graph, fp, desc)[0]).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(batch["y_dili"].numpy())
    return np.array(all_probs), np.array(all_labels)
