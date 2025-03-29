import torch
from torch.utils.data import DataLoader, TensorDataset

EPOCHS = 300
BATCH_SIZE = 32
PREDICTION_TRESHOLD = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def train_torch_model_multiclass(model, criterion, optimizer, X_train, y_train,
                                 X_val, y_val, *, writer=None, batch_size=BATCH_SIZE, epochs=EPOCHS):
    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

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

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(y_batch).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                y_batch = y_batch.argmax(dim=1)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                correct += predicted.eq(y_batch).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total

        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.8f}, Train Acc: {train_acc:.8f}, "
              f"Val Loss: {val_loss:.8f}, Val Acc: {val_acc:.8f}")

    if writer:
        writer.close()


def train_torch_model_binary(model, criterion, optimizer, X_train, y_train,
                             X_val, y_val, *, writer=None, batch_size=BATCH_SIZE,
                             epochs=EPOCHS, prediction_treshold=PREDICTION_TRESHOLD):
    model = model.to(device)

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

            # Forward pass
            outputs = model(X_batch).squeeze(dim=1)
            probs = torch.sigmoid(outputs).detach()
            print(f"Prediction stats: Min={probs.min():.3f}, Max={probs.max():.3f}, Mean={probs.mean():.3f}")
            loss = criterion(outputs, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            predicted = (torch.sigmoid(outputs) > prediction_treshold).long()
            correct += (predicted == y_batch.long()).sum().item()
            total += y_batch.size(0)

        train_acc = correct / total

        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).float()

                # Forward pass
                outputs = model(X_batch).squeeze(dim=1)
                loss = criterion(outputs, y_batch)

                val_loss += loss.item()
                predicted = (torch.sigmoid(outputs) > prediction_treshold).long()
                correct += (predicted == y_batch.long()).sum().item()
                total += y_batch.size(0)

        val_acc = correct / total

        if writer:
            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Train Loss: {train_loss:.8f}, Train Acc: {train_acc:.8f}, "
              f"Val Loss: {val_loss:.8f}, Val Acc: {val_acc:.8f}")

    if writer:
        writer.close()


def overfit_model(model, criterion, optimizer, X_train, y_train, prediction_treshold=PREDICTION_TRESHOLD):
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, 32, shuffle=True)
    X_debug, y_debug = next(iter(train_loader))
    X_debug, y_debug = X_debug.to(device), y_debug.to(device).float()

    model = model.to(device)

    print("\n=== Debug Mode ===")
    print(f"Input shape: {X_debug.shape}")
    print(f"Label distribution: {torch.mean(y_debug).item():.2f} (1s)")

    for step in range(1000):
        optimizer.zero_grad()
        outputs = model(X_debug).squeeze(1)

        # Convert outputs to probabilities and predictions
        probs = torch.sigmoid(outputs)
        preds = (probs > prediction_treshold).float()

        # Calculate metrics
        loss = criterion(outputs, y_debug)
        acc = (preds == y_debug).float().mean()

        # Backpropagation
        loss.backward()
        optimizer.step()

        # Print diagnostics every 100 steps
        if step % 2 == 0 or step == 999:
            print(f"\nStep {step}:")
            print(f"Loss: {loss.item():.4f}")
            print(f"Accuracy: {acc.item():.2%}")
            print(f"Predictions (5 samples): {probs[:5].detach().cpu().numpy().round(4)}")
            print(f"Labels (5 samples): {y_debug[:5].cpu().numpy()}")

            # Gradient monitoring
            total_grad = 0
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    total_grad += grad_norm
                    if step == 0:  # Print parameter shapes on first step
                        print(f"Param {name}: shape {tuple(param.shape)} | grad norm: {grad_norm:.6f}")

            print(f"Total gradient norm: {total_grad:.4f}")

    print("\n=== Final Prediction Distribution ===")
    final_probs = torch.sigmoid(model(X_debug).squeeze(1))
    print(f"Min: {final_probs.min().item():.4f}")
    print(f"Max: {final_probs.max().item():.4f}")
    print(f"Mean: {final_probs.mean().item():.4f}")
    print(f"Std: {final_probs.std().item():.4f}")
