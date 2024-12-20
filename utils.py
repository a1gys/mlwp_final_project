import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score


def train(model: nn.Module,
          optimizer: torch.optim.Optimizer,
          train_loader: DataLoader,
          device: torch.device):
    model.train()

    train_loss = 0
    for data in train_loader:
        optimizer.zero_grad()
        out = model(data.to(device))
        loss = F.nll_loss(out, data.y.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def validate(model: nn.Module,
             val_loader: DataLoader,
             device: torch.device):
    model.eval()

    val_loss = 0
    with torch.inference_mode():
        for data in val_loader:
            out = model(data.to(device))
            loss = F.nll_loss(out, data.y.to(device))
            val_loss += loss.item()
    return val_loss / len(val_loader)


def test(model: nn.Module,
         test_loader: DataLoader,
         device: torch.device):
    pred = []
    true = []
    model.eval()
    with torch.inference_mode():
        for data in test_loader:
            out = model(data.to(device))
            preds = torch.argmax(F.softmax(out, dim=1), dim=-1)
            pred.append(preds.detach().cpu().numpy())
            true.append(data.y.detach().cpu().numpy())

    pred = np.concatenate(pred)
    true = np.concatenate(true)
    acc = accuracy_score(true, pred)
    recall = recall_score(true, pred, zero_division=0)
    precision = precision_score(true, pred, zero_division=0)
    f1 = f1_score(true, pred)
    roc_auc = roc_auc_score(true, pred)

    return acc, recall, precision, f1, roc_auc


def train_multi(model: nn.Module,
                optimizer: torch.optim.Optimizer,
                train_loader: DataLoader,
                device: torch.device):
    model.train()

    train_loss = 0
    for train_data in train_loader:
        optimizer.zero_grad()
        out = model([data.to(device) for data in train_data])
        loss = F.nll_loss(out, train_data[0].y.to(device))
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)


def validate_multi(model: nn.Module,
                   val_loader: DataLoader,
                   device: torch.device):
    model.eval()

    val_loss = 0
    with torch.inference_mode():
        for val_data in val_loader:
            out = model([data.to(device) for data in val_data])
            loss = F.nll_loss(out, val_data[0].y.to(device))
            val_loss += loss.item()
    return val_loss / len(val_loader)


def test_multi(model: nn.Module,
               test_loader: DataLoader,
               device: torch.device):
    pred = []
    true = []
    model.eval()
    with torch.inference_mode():
        for test_data in test_loader:
            out = model([data.to(device) for data in test_data])
            preds = torch.argmax(F.softmax(out, dim=1), dim=-1)
            pred.append(preds.detach().cpu().numpy())
            true.append(test_data[0].y.detach().cpu().numpy())

    pred = np.concatenate(pred)
    true = np.concatenate(true)
    acc = accuracy_score(true, pred)
    recall = recall_score(true, pred, zero_division=0)
    precision = precision_score(true, pred, zero_division=0)
    f1 = f1_score(true, pred)
    roc_auc = roc_auc_score(true, pred)

    return acc, recall, precision, f1, roc_auc
