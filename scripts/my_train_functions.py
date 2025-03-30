import torch
import torch.nn as nn

import lightning as L

from torchmetrics import MetricCollection
from torchmetrics.classification import (
    MulticlassAccuracy, MulticlassF1Score
)

import numpy as np
import statistics
from sklearn.metrics import confusion_matrix

import seaborn as sns
import matplotlib.pyplot as plt

class LModel(L.LightningModule):
    def __init__(self, model, lr=0.001, gamma=0.9):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # for optimizer and shaduler
        self.lr = lr
        self.gamma = gamma

        # model
        self.model = model
        self.criterion = nn.CrossEntropyLoss()

        # metrics
        self.metrics = MetricCollection([
            MulticlassAccuracy(num_classes=6,),
            MulticlassF1Score(num_classes=6,)

        ])
        self.train_metrics = self.metrics.clone(postfix='/train')
        self.val_metrics = self.metrics.clone(postfix='/val')

    def configure_optimizers(self):
        # set optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "monitor": "loss"
            },
        }

    def training_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        loss = self.criterion(out, y)
        self.train_metrics.update(out, y)
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        self.val_metrics.update(out, y)

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

        val_metrics = self.val_metrics.compute()
        self.log_dict(val_metrics)
        self.val_metrics.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.model(x)
        self.metrics.update(out, y)

    def on_test_epoch_end(self):
        self.log_dict(self.metrics.compute())
        self.metrics.reset()    

def plot_confusion_matrix(model_best, test_loader, test_set, n_shift=10):
    """
    Evaluate the model on the test set and plot the confusion matrix.
    
    Args:
        model_best: Trained model to be evaluated.
        test_loader: DataLoader for the test set.
        test_set: Test dataset containing class labels.
        n_shift (int): Number of predictions to aggregate. Default is 10.
    """
    model_best.eval()
    
    all_labels = []
    all_predictions = []
    
    av_labels = []
    av_predictions = []
    
    samples_counter = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            outputs = model_best(inputs)
            predictions = torch.argmax(outputs, axis=1).cpu().numpy()
    
            av_labels.extend(labels.numpy())
            av_predictions.extend(predictions)
    
            samples_counter += 1
            if samples_counter == n_shift:
                samples_counter = 0
    
                l = statistics.mode(av_labels)
                p = statistics.mode(av_predictions)
                all_labels.extend([l])
                all_predictions.extend([p])
                av_labels = []
                av_predictions = []

    # Compute the confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)

    # Plot confusion matrix
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, annot_kws={"size": 12})
    plt.title('Confusion Matrix', fontsize=16, pad=15)
    plt.xlabel('Predicted Label', fontsize=13, labelpad=10)
    plt.ylabel('True Label', fontsize=13)

    el_2_img_mapping = {
        "0_Sector": "Segment",
        "1_Part of helicoid": "Zigzag",
        "2_Disk": "Circle",
        "3_Helicoid": "Spiral",
        "4_Enneper": "Tennis ball",
        "5_Complex structure": "Serpent"
    }
    img_labels = [el_2_img_mapping[label] for label in test_set.classes]
    plt.xticks(np.arange(6) + 0.5, labels=img_labels, rotation=0, fontsize=11)
    plt.yticks(np.arange(6) + 0.5, labels=img_labels, rotation=45, fontsize=11)

    plt.show()