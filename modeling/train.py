import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

def train(
        model, 
        train_loader, 
        val_loader=None, 
        num_epochs:int=5, 
        lr:float=1e-3,
        lmbda_l2:float=1e-4,
        device:str="cuda",
        **kwargs
    ):
    """
    Train classifier and report loss, accuracy, and AUC per epoch.
    Assumes binary classification with 2 logits per sample.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=lmbda_l2)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        all_labels = []
        all_probs = []

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)  # shape (batch_size, 2)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)

            # store labels and predicted probabilities for AUC
            probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()  # positive class prob
            all_probs.extend(probs)
            all_labels.extend(labels.detach().cpu().numpy())

        # compute metrics
        train_loss = running_loss / len(train_loader.dataset)
        train_acc = ((torch.tensor(all_probs) > 0.5).numpy() == all_labels).mean()
        train_auc = roc_auc_score(all_labels, all_probs)

        print(f"Epoch {epoch+1}/{num_epochs} "
              f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, AUC: {train_auc:.4f}")

        # ----- Validation metrics -----
        if val_loader is not None:
            model.eval()
            val_labels = []
            val_probs = []
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    probs = F.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
                    val_probs.extend(probs)
                    val_labels.extend(labels.detach().cpu().numpy())

            val_acc = ((torch.tensor(val_probs) > 0.5).numpy() == val_labels).mean()
            val_auc = roc_auc_score(val_labels, val_probs)
            print(f"Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}\n")

        os.makedirs('models', exist_ok=True)
        torch.save(model.state_dict(), f"models/model_{epoch+1}_epoch.pth")


    return model