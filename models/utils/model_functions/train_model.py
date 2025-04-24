import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler
from utils.model_functions.model_statistics import ModelStatsTracker

EPOCHS = 300
BATCH_SIZE = 32
PREDICTION_THRESHOLD = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_torch_model_multiclass(
    model,
    criterion,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    writer=None,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    lr_scheduler=None
):
    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    stats = ModelStatsTracker()
    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_batch = y_batch.argmax(dim=1)

            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = outputs.max(1)

            stats.log_train_stats(
                predicted.cpu().numpy(), y_batch.cpu().numpy(), loss.item()
            )

        # Validation
        model.eval()

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.argmax(dim=1)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                _, predicted = outputs.max(1)

                stats.log_val_stats(
                    predicted.cpu().numpy(), y_batch.cpu().numpy(), loss.item()
                )

        if lr_scheduler is not None:
            lr_scheduler.step(stats.val_loss)

        stats.summary(writer)

    if writer:
        writer.close()


def train_torch_model_binary(
    model,
    criterion,
    optimizer,
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    unbalanced=False,
    writer=None,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    prediction_threshold=PREDICTION_THRESHOLD,
    show_prediction_stats=False,
    lr_scheduler=None
):
    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    if not unbalanced:
        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    else:
        y_train_np = (
            y_train.numpy().astype(np.int64)
            if hasattr(y_train, "numpy")
            else np.array(y_train)
        )
        class_sample_counts = np.bincount(y_train_np)
        class_sample_counts = np.maximum(class_sample_counts, 1)
        class_weights = 1.0 / class_sample_counts
        class_weights = class_weights / np.sum(class_weights) * len(class_weights)
        sample_weights = class_weights[y_train_np]
        sampler = WeightedRandomSampler(
            weights=sample_weights, num_samples=len(sample_weights), replacement=True
        )
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler)

    stats = ModelStatsTracker()
    for epoch in range(epochs):
        model.train()

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            # Forward pass
            outputs = model(X_batch).squeeze(dim=1)
            if show_prediction_stats:
                probs = torch.sigmoid(outputs).detach()
                print(
                    f"Prediction stats: Min={probs.min():.3f}, Max={probs.max():.3f}, Mean={probs.mean():.3f}"
                )
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            predicted = (torch.sigmoid(outputs).detach() > prediction_threshold).long()

            stats.log_train_stats(
                predicted.cpu().numpy(), y_batch.cpu().numpy(), loss.item()
            )

        # Validation
        model.eval()

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

                outputs = model(X_batch).squeeze(dim=1)
                loss = criterion(outputs, y_batch)

                predicted = (torch.sigmoid(outputs) > prediction_threshold).long()

                stats.log_val_stats(
                    predicted.cpu().numpy(), y_batch.cpu().numpy(), loss.item()
                )

        if lr_scheduler is not None:
            lr_scheduler.step(stats.val_loss)

        stats.summary(writer)

    if writer:
        writer.close()


def overfit_model(
    model,
    criterion,
    optimizer,
    X_train,
    y_train,
    *,
    writer=None,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    prediction_threshold=PREDICTION_THRESHOLD,
):
    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)

    # Get a single batch and keep reusing it
    X_debug, y_debug = next(iter(train_loader))
    X_debug, y_debug = X_debug.to(device), y_debug.to(device).float()

    stats = ModelStatsTracker()
    for epoch in range(epochs):
        model.train()

        outputs = model(X_debug).squeeze(1)
        loss = criterion(outputs, y_debug)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predicted = (torch.sigmoid(outputs).detach() > prediction_threshold).long()
        stats.log_val_stats(
            predicted.cpu().numpy(),
            y_debug.cpu().numpy(),
            loss.item()
        )

        stats.summary(writer, val_only=True)

    if writer:
        writer.close()
