import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from torch.utils.tensorboard import SummaryWriter


class ModelStatsTracker:
    def __init__(self):
        self.epoch = 0
        self._reset_backlogs()

    def _reset_backlogs(self):
        """Initializes variables for internal model tracking"""
        self.train_loss = 0
        self.train_preds = []
        self.train_labels = []
        self.val_loss = 0
        self.val_preds = []
        self.val_labels = []

    def _new_epoch(self):
        """Clear cache for new epoch stats"""
        self.epoch += 1
        self._reset_backlogs()

    def log_train_stats(self, predicted: np.array, y_batch: np.array, loss: float):
        """Registers variables for single training epoch"""
        self.train_preds.extend(predicted)
        self.train_labels.extend(y_batch)
        self.train_loss += loss

    def log_val_stats(self, predicted: np.array, y_batch: np.array, loss: float):
        """Registers variables for single validating epoch"""
        self.val_preds.extend(predicted)
        self.val_labels.extend(y_batch)
        self.val_loss += loss

    def calculate_statistics(self, val_only: bool = False):
        """Calculates the statistics from given backlog"""
        if not val_only:
            self.train_cm = confusion_matrix(self.train_labels, self.train_preds)
            self.train_acc = self.train_cm.diagonal().sum() / self.train_cm.sum()
            self.train_precision = precision_score(
                self.train_labels, self.train_preds, average="macro", zero_division=0
            )
            self.train_recall = recall_score(
                self.train_labels, self.train_preds, average="macro", zero_division=0
            )
            self.train_f1 = f1_score(
                self.train_labels, self.train_preds, average="macro", zero_division=0
            )

        self.val_cm = confusion_matrix(self.val_labels, self.val_preds)
        self.val_acc = self.val_cm.diagonal().sum() / self.val_cm.sum()
        self.val_precision = precision_score(
            self.val_labels, self.val_preds, average="macro", zero_division=0
        )
        self.val_recall = recall_score(
            self.val_labels, self.val_preds, average="macro", zero_division=0
        )
        self.val_f1 = f1_score(
            self.val_labels, self.val_preds, average="macro", zero_division=0
        )

        if len(np.unique(self.val_labels)) == 2:  # Binary classification
            tn, fp, fn, tp = self.val_cm.ravel()
            self.tpr = tp / (tp + fn)
            self.fpr = fp / (fp + tn)
        else:  # Multi-class classification
            self.tpr = None
            self.fpr = None

    def write(self, writer: SummaryWriter, val_only: bool = False):
        """Register calculated data in writer"""
        writer.add_scalar("Loss/Validation", self.val_loss, self.epoch)
        writer.add_scalar("Accuracy/Validation", self.val_acc, self.epoch)
        writer.add_scalar("Precision/Validation", self.val_precision, self.epoch)
        writer.add_scalar("Recall/Validation", self.val_recall, self.epoch)
        writer.add_scalar("F1/Validation", self.val_f1, self.epoch)

        if not val_only:
            writer.add_scalar("Loss/Train", self.train_loss, self.epoch)
            writer.add_scalar("Accuracy/Train", self.train_acc, self.epoch)
            writer.add_scalar("Precision/Train", self.train_precision, self.epoch)
            writer.add_scalar("Recall/Train", self.train_recall, self.epoch)
            writer.add_scalar("F1/Train", self.train_f1, self.epoch)

        if self.tpr is not None and self.fpr is not None:
            writer.add_scalar("TPR/Validation", self.tpr, self.epoch)
            writer.add_scalar("FPR/Validation", self.fpr, self.epoch)

    def print(self, val_only: bool = False):
        """Prints calculated statistics in console"""
        print("\n" + "=" * 100)
        print(f"{'EPOCH STATISTICS':^100}")
        print("=" * 100)
        print(f"{'Epoch':<12}: {self.epoch + 1}")

        if val_only:
            print("-" * 100)
            print(f"{'VALIDATION':^100}")
            print("-" * 100)
            print(f"{'Loss':<12}: {self.val_loss:.4f}")
            print(f"{'Accuracy':<12}: {self.val_acc:.4f}")
            print(f"{'Precision':<12}: {self.val_precision:.4f}")
            print(f"{'Recall':<12}: {self.val_recall:.4f}")
            print(f"{'F1 Score':<12}: {self.val_f1:.4f}")
        else:
            print("-" * 100)
            print(f"{'TRAINING':^50}{'VALIDATION':^50}")
            print("-" * 100)
            print(
                f"{'Loss':<12}: {self.train_loss:.4f}{'':36}{'Loss':<12}: {self.val_loss:.4f}"
            )
            print(
                f"{'Accuracy':<12}: {self.train_acc:.4f}{'':36}{'Accuracy':<12}: {self.val_acc:.4f}"
            )
            print(
                f"{'Precision':<12}: {self.train_precision:.4f}{'':36}{'Precision':<12}: {self.val_precision:.4f}"
            )
            print(
                f"{'Recall':<12}: {self.train_recall:.4f}{'':36}{'Recall':<12}: {self.val_recall:.4f}"
            )
            print(
                f"{'F1 Score':<12}: {self.train_f1:.4f}{'':36}{'F1 Score':<12}: {self.val_f1:.4f}"
            )

        if self.tpr is not None:
            print("-" * 100)
            print(f"{'VALIDATION EXTRA':^100}")
            print(
                f"{'TP Rate':<12}: {self.tpr:.4f}{'':36}{'FP Rate':<12}: {self.fpr:.4f}"
            )

        print("=" * 100 + "\n")

    def summary(self, writer: SummaryWriter = None, val_only: bool = False):
        """Summarise epoch"""
        self.calculate_statistics(val_only)

        if writer:
            self.write(writer, val_only)

        self.print(val_only)
        self._new_epoch()
