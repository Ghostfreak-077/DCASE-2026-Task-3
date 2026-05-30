import matplotlib.pyplot as plt
import os

def plot_loss_curves(train_losses, val_losses, out_path):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Train Loss", color="blue")
    plt.plot(val_losses, label="Val Loss", color="red")
    plt.title("System Convergence History")
    plt.xlabel("Epochs")
    plt.ylabel("Loss Criterion Scale")
    plt.grid(True)
    plt.legend()
    plt.savefig(out_path)
    plt.close()