import click
import os
from pprint import pprint
from matplotlib import pyplot as plt

LOGS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "src", "external", "TodyNet", "log"
)


@click.command()
@click.argument("logfile", nargs=1)
def main(logfile):
    if logfile not in os.listdir(LOGS_DIR):
        pprint(f"File {logfile} not found in log dir.")

    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    with open(os.path.join(LOGS_DIR, logfile)) as f:
        _ = f.readline()
        for train_line, val_line in zip(f, f):
            _, _, tloss, tacc, _ = train_line.split(',')
            train_loss.append(float(tloss.split()[1]))
            train_acc.append(float(tacc.split('[')[1].strip(']')))
            _, vloss, vacc, _ = val_line.split(',')
            val_loss.append(float(vloss.split()[1]))
            val_acc.append(float(vacc.split('[')[1].strip(']')))

    epochs = range(1, len(train_loss) + 1)

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    # Training Loss
    axs[0, 0].plot(epochs, train_loss, 'b-o')
    axs[0, 0].set_title('Training Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].grid(True)

    # Validation Loss
    axs[0, 1].plot(epochs, val_loss, 'r-o')
    axs[0, 1].set_title('Validation Loss')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Loss')
    axs[0, 1].grid(True)

    # Training Accuracy
    axs[1, 0].plot(epochs, train_acc, 'b-o')
    axs[1, 0].set_title('Training Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].grid(True)

    # Validation Accuracy
    axs[1, 1].plot(epochs, val_acc, 'r-o')
    axs[1, 1].set_title('Validation Accuracy')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Accuracy')
    axs[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
