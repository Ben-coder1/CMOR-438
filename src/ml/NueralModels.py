import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# Dataset wrappers
# -------------------------
class TabularDataset(Dataset):
    """Wraps a DataFrame or tensors for dense models."""
    def __init__(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = torch.tensor(X.values, dtype=torch.float32)
        if y is not None and isinstance(y, (pd.Series, pd.DataFrame)):
            y = torch.tensor(y.values, dtype=torch.long)
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


class ImageDataset(Dataset):
    """Wraps a DataFrame with 'image' and 'label' columns for CNNs."""
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = self.df.loc[idx, "image"]
        label = self.df.loc[idx, "label"]
        if self.transform:
            img = self.transform(img)
        return img, label


# -------------------------
# Base class
# -------------------------
class BaseModel(nn.Module):
    """
    Base class for classification models.
    Subclasses must define self.net.
    Provides training, evaluation, and prediction utilities.
    """

    def __init__(self, num_classes, device="cpu", class_weights=None):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        # Loss
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None
        )

        # Optimizer placeholders
        self.optimizer = None
        self.scheduler = None

        # Training state
        self.val_history = []
        self.best_state = None
        self.best_val_acc = 0.0

        self.to(device)

    def configure_optimizer(self, optimizer_name="adamw", lr=1e-3):
        """Attach optimizer and scheduler after net is defined."""
        opt = optimizer_name.lower()
        if opt == "adamw":
            self.optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        elif opt == "adam":
            self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        elif opt == "sgd":
            self.optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=0.9)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2, min_lr=5e-5
        )

    def forward(self, x):
        return self.net(x)

    # -------------------------
    # Training / Evaluation
    # -------------------------
    def train_model(self, train_loader, valid_loader, num_epochs=30, patience=5):
        self.val_history = []
        self.best_state = None
        self.best_val_acc = 0.0
        wait = 0

        for epoch in range(num_epochs):
            self.train()
            running_loss = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                preds = self(xb)
                loss = self.criterion(preds, yb)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * xb.size(0)
            train_loss = running_loss / len(train_loader.dataset)

            val_acc = self._accuracy_on_loader(valid_loader)
            self.val_history.append(val_acc)
            self.scheduler.step(val_acc)

            print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_acc={val_acc:.4f}")

            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = self.state_dict()
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if self.best_state is not None:
            self.load_state_dict(self.best_state)
            print(f"Restored best model with val_acc={self.best_val_acc:.4f}")

    def _accuracy_on_loader(self, loader):
        self.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        return correct / max(total, 1)

    def evaluate_model(self, test_loader, target_names=None):
        self.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                preds = self(xb).argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_true.extend(yb.cpu().numpy())

        all_preds, all_true = np.array(all_preds), np.array(all_true)
        acc = (all_preds == all_true).mean()
        print(f"Test accuracy: {acc:.4f}")

        if target_names is not None:
            print("\nClassification report:")
            print(classification_report(all_true, all_preds, target_names=target_names, zero_division=0))
            print("\nConfusion matrix:")
            print(confusion_matrix(all_true, all_preds))

        return acc

    def predict(self, X_loader):
        self.eval()
        preds_out = []
        with torch.no_grad():
            for batch in X_loader:
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                xb = xb.to(self.device)
                preds = self(xb).argmax(dim=1).cpu().numpy()
                preds_out.extend(preds)
        return np.array(preds_out)

    def predict_proba(self, X_loader):
        self.eval()
        probs_out = []
        with torch.no_grad():
            for batch in X_loader:
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                xb = xb.to(self.device)
                logits = self(xb)
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                probs_out.extend(probs)
        return np.array(probs_out)


# -------------------------
# CNN subclass
# -------------------------
class CNNModel(BaseModel):
    def __init__(self, input_channels, img_size, conv_channels, fc_units,
                 num_classes, dropout=0.5, device="cpu", class_weights=None,
                 optimizer_name="adamw", lr=1e-3):
        super().__init__(num_classes=num_classes, device=device,
                         class_weights=class_weights)
        # Build CNN
        layers = []
        in_channels = input_channels
        for out_channels in conv_channels:
            layers += [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            ]
            in_channels = out_channels
        cnn = nn.Sequential(*layers)

        conv_output_size = img_size // (2 ** len(conv_channels))
        flattened_size = conv_output_size ** 2 * conv_channels[-1]

        fc_layers = []
        in_features = flattened_size
        for units in fc_units:
            fc_layers += [
                nn.Linear(in_features, units),
                nn.ReLU(),
                nn.Dropout(dropout)
            ]
            in_features = units
        fc_layers.append(nn.Linear(in_features, num_classes))
        fc = nn.Sequential(*fc_layers)

        self.net = nn.Sequential(cnn, nn.Flatten(), fc)

        # Configure optimizer
        self.configure_optimizer(optimizer_name, lr)


# -------------------------
# Dense/MLP subclass
# -------------------------
class DenseModel(BaseModel):
    def __init__(self, input_dim, hidden_units, num_classes,
                 activation="relu", dropout=0.2, use_batchnorm=False,
                 device="cpu", class_weights=None,
                 optimizer_name="adamw", lr=1e-3):
        super().__init__(num_classes=num_classes, device=device,
                         class_weights=class_weights)

        act = self._get_activation(activation)
        layers = []
        in_dim = input_dim
        for h in hidden_units:
            layers.append(nn.Linear(in_dim, h))
            if use_batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(act)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.net = nn.Sequential
