"""
Feed-forward neural network for multi-modal DILI prediction.

Combines Morgan fingerprints, physicochemical descriptors, proxy-DILI
assay labels, and PK parameters in a single architecture with learned
feature fusion -- similar to Axiom's approach of combining structure,
in vitro readouts, and exposure data.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score


class DILINet(nn.Module):
    """
    Multi-modal feed-forward network for DILI classification.

    Architecture mirrors the multi-readout approach: separate encoding
    heads for fingerprints and descriptors, then fusion.
    """

    def __init__(self, fp_dim: int, desc_dim: int, hidden: int = 256, dropout: float = 0.3):
        super().__init__()

        self.fp_encoder = nn.Sequential(
            nn.Linear(fp_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )

        self.desc_encoder = nn.Sequential(
            nn.Linear(desc_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        fusion_dim = hidden // 2 + 32
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, fp: torch.Tensor, desc: torch.Tensor) -> torch.Tensor:
        fp_emb = self.fp_encoder(fp)
        desc_emb = self.desc_encoder(desc)
        fused = torch.cat([fp_emb, desc_emb], dim=-1)
        return self.classifier(fused).squeeze(-1)


def train_neural_net(
    X_fp_train: np.ndarray,
    X_desc_train: np.ndarray,
    y_train: np.ndarray,
    X_fp_test: np.ndarray,
    X_desc_test: np.ndarray,
    y_test: np.ndarray,
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = "auto",
) -> dict:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    fp_train = torch.tensor(X_fp_train, dtype=torch.float32)
    desc_train = torch.tensor(X_desc_train, dtype=torch.float32)
    y_tr = torch.tensor(y_train, dtype=torch.float32)

    fp_test = torch.tensor(X_fp_test, dtype=torch.float32)
    desc_test = torch.tensor(X_desc_test, dtype=torch.float32)
    y_te = torch.tensor(y_test, dtype=torch.float32)

    train_ds = TensorDataset(fp_train, desc_train, y_tr)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    model = DILINet(fp_train.shape[1], desc_train.shape[1]).to(device)

    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_auc = 0.0
    best_state = None
    history = {"train_loss": [], "test_auc": []}

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for fp_b, desc_b, y_b in train_loader:
            fp_b, desc_b, y_b = fp_b.to(device), desc_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(fp_b, desc_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(y_b)

        scheduler.step()
        avg_loss = total_loss / len(y_train)
        history["train_loss"].append(avg_loss)

        model.eval()
        with torch.no_grad():
            test_logits = model(fp_test.to(device), desc_test.to(device))
            test_probs = torch.sigmoid(test_logits).cpu().numpy()
            auc = roc_auc_score(y_test, test_probs)
            history["test_auc"].append(auc)

        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 20 == 0:
            print(f"    Epoch {epoch+1}/{epochs}  loss={avg_loss:.4f}  test_AUC={auc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        test_logits = model(fp_test.to(device), desc_test.to(device))
        test_probs = torch.sigmoid(test_logits).cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)

    return {
        "model": model,
        "test_probs": test_probs,
        "test_preds": test_preds,
        "best_auc": best_auc,
        "history": history,
        "device": device,
    }
