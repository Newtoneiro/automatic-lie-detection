import torch
from torch.utils.data import DataLoader, TensorDataset
from utils.model_functions.model_statistics import ModelStatsTracker

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32


def eval_torch_model_multiclass(model, criterion, X_test, y_test):
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    stats = ModelStatsTracker()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.argmax(dim=1)

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            _, predicted = outputs.max(1)

            stats.log_val_stats(
                predicted.cpu().numpy(), y_batch.cpu().numpy(), loss.item()
            )

    stats.summary(val_only=True)


def eval_torch_model_binary(model, criterion, X_test, y_test):
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    stats = ModelStatsTracker()

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            outputs = model(X_batch).squeeze(dim=1)
            loss = criterion(outputs, y_batch)

            predicted = (torch.sigmoid(outputs) > 0.5).long()

            stats.log_val_stats(
                predicted.cpu().numpy(), y_batch.cpu().numpy(), loss.item()
            )

        stats.summary(val_only=True)
